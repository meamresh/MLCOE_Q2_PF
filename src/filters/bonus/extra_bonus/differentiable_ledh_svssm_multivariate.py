"""
Multivariate canonical SVSSM filter (V1 multivariate).

Extends :class:`DifferentiableLEDHLogLikelihoodSVSSM` from
``differentiable_ledh_svssm.py`` (which is univariate, d=1) to vector
state h_t in R^d and vector observation y_t in R^d with component-wise
multiplicative noise.

Model
~~~~~
    h_t      = mu + Phi (h_{t-1} - mu) + Sigma_eta^{1/2} eta_t,
                eta_t ~ N(0, I_d)
    y_{t,i}  = exp(h_{t,i} / 2) eps_{t,i},  eps_{t,i} ~ N(0, 1)  (component-wise)

This first pass supports **diagonal Phi and diagonal Sigma_eta**, which
means the d components are independent dynamics-wise (equivalent to d
parallel univariate filters), but the code uses matrix-form tensors
throughout so the non-diagonal extension is a small change later.

Boundary transform (same as the d=1 filter, applied component-wise):
    z_{t,i} = log(y_{t,i}^2 + delta) = h_{t,i} + mu_z + e_{t,i}
where mu_z = -1.2704 and e_{t,i} approximately N(0, sigma_z^2) under the
Harvey-Ruiz-Shephard Gaussian quasi-likelihood, with sigma_z^2 = 4.93.

Per-particle state covariance is stored as (N, d, d) full matrix so
the same code path generalises to non-diagonal Sigma_eta. In the
diagonal-Phi diagonal-Sigma_eta case this matrix simply stays diagonal
under predict + LEDH-flow updates.

API
~~~
    ll = DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=2, num_particles=64, n_lambda=10, ...
    )
    log_p = ll(mu, phi_diag, sigma_eta_diag_sq, y_obs)

where:
    mu:                  shape (d,)         level parameter
    phi_diag:            shape (d,)         persistence per component
    sigma_eta_diag_sq:   shape (d,)         process-noise variance per component
    y_obs:               shape (T, d)       observation series

Currently the only argument shapes supported are the diagonal-parameter
case above. A future extension will add `Phi: (d, d)` and
`Sigma_eta: (d, d)` overloads for the full-matrix case.
"""

from __future__ import annotations

import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    _as_real_log_scalar,
    LOG_CHI2_MEAN,
    LOG_CHI2_VAR,
    _EPS,
    _CLAMP,
)
from src.filters.dpf.resampling import det_resample


def _safe_nd(x):
    """Replace non-finite entries with zero and clip to a safe range."""
    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    return tf.clip_by_value(x, -_CLAMP, _CLAMP)


def expm_2x2_batch(A: tf.Tensor) -> tf.Tensor:
    """Closed-form matrix exponential for batched 2x2 matrices, XLA-safe.

    A has shape (..., 2, 2). Returns expm(A) with the same shape.

    Derivation: write A = m I + N with m = tr(A)/2; then N is traceless
    and by Cayley-Hamilton N^2 = -det(N) I = xi I where
    xi = ((a-d)/2)^2 + b c. The series exp(N) splits cleanly:
        exp(N) = C(xi) I + S(xi) N
    where
        C(xi) = cosh(sqrt(xi))     if xi >= 0   else cos(sqrt(-xi))
        S(xi) = sinh(sqrt(xi))/sqrt(xi)  if xi >= 0
                else sin(sqrt(-xi))/sqrt(-xi)
    Both branches share the limit C(0) = S(0) = 1.

    Replaces tf.linalg.expm in the LEDH flow when state_dim = 2. The
    stock op uses an unbounded while_loop that XLA cannot statically
    size; this closed form is pure elementwise ops + a couple of
    tf.where branches, so jit_compile=True works.
    """
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]

    m = 0.5 * (a + d)
    half_diff = 0.5 * (a - d)
    xi = half_diff * half_diff + b * c  # = -det(A - m I)

    abs_xi = tf.abs(xi)
    # +eps so sqrt is differentiable at xi=0 and sinhc denominator is safe.
    s = tf.sqrt(abs_xi + tf.constant(1e-30, dtype=A.dtype))

    cosh_pos = tf.cosh(s)
    cos_neg = tf.cos(s)
    C = tf.where(xi >= 0.0, cosh_pos, cos_neg)

    sinhc_pos = tf.sinh(s) / s
    sinc_neg = tf.sin(s) / s
    S = tf.where(xi >= 0.0, sinhc_pos, sinc_neg)

    exp_m = tf.exp(m)
    # exp(A) = exp(m) * (C I + S N) where N = A - m I:
    #   N00 =  half_diff,  N01 = b
    #   N10 =  c,          N11 = -half_diff
    e00 = exp_m * (C + S * half_diff)
    e01 = exp_m * (S * b)
    e10 = exp_m * (S * c)
    e11 = exp_m * (C - S * half_diff)

    row0 = tf.stack([e00, e01], axis=-1)
    row1 = tf.stack([e10, e11], axis=-1)
    return tf.stack([row0, row1], axis=-2)


def expm_pade_batch(A: tf.Tensor, s: int) -> tf.Tensor:
    """Pade-[6/6] matrix exponential with FIXED scaling-and-squaring, XLA-safe.

    A has shape (..., d, d) for any d. Returns expm(A) with the same shape.

    Why not tf.linalg.expm: the stock op implements the same Higham
    scaling-and-squaring algorithm but chooses the scaling count from
    the matrix norm AT RUNTIME via an unbounded while_loop, which XLA
    cannot statically size. Here the caller supplies a static ``s``
    derived from an a-priori norm bound, so every loop unrolls at trace
    time and ``jit_compile=True`` works at any d.

    Algorithm (Higham 2005, fixed-s variant):
        1. Scale:   B = A / 2^s            (target ||B|| <= ~0.5)
        2. Pade-[6/6]:  exp(B) ~= (V - U)^{-1} (V + U)  with
              V = b0 I + b2 B^2 + b4 B^4 + b6 B^6   (even part)
              U = B (b1 I + b3 B^2 + b5 B^4)        (odd part)
           b = [1, 1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280]
           Order-13 accurate; near machine precision for ||B|| <= 0.5.
        3. Square s times:  exp(A) = exp(B)^(2^s)

    Cost: 4 matmuls + 1 batched solve + s matmuls. Autodiff flows
    through matmul/solve natively (no custom gradient needed).

    Choosing s: in the LEDH flow A is clipped entrywise to [-10, 10]
    and the largest pseudo-time substep is eps ~= 0.2, so
    ||A*eps||_F <= 2d. To reach ||B|| <= 0.5 use
    ``s = ceil(log2(4d))`` (e.g. d=3 -> 4, d=8 -> 5). Computed by the
    caller at trace time since d is a static Python int.
    """
    dtype = A.dtype
    d = A.shape[-1]
    I = tf.eye(d, dtype=dtype)
    # Broadcast identity over the batch dims of A.
    I = tf.broadcast_to(I, tf.shape(A))

    b = [1.0, 1.0 / 2.0, 5.0 / 44.0, 1.0 / 66.0,
         1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0]

    B = A * tf.constant(2.0 ** (-s), dtype=dtype)

    B2 = tf.matmul(B, B)
    B4 = tf.matmul(B2, B2)
    B6 = tf.matmul(B4, B2)

    V = b[0] * I + b[2] * B2 + b[4] * B4 + b[6] * B6
    U = tf.matmul(B, b[1] * I + b[3] * B2 + b[5] * B4)

    # exp(B) ~= (V - U)^{-1} (V + U); batched LU solve is XLA-safe.
    E = tf.linalg.solve(V - U, V + U)

    # Square s times (unrolled at trace time -- s is a Python int).
    for _ in range(s):
        E = tf.matmul(E, E)
    return E


def pade_scaling_for_dim(d: int, entry_bound: float = 10.0,
                          eps_max: float = 0.2) -> int:
    """Static scaling count for expm_pade_batch given the LEDH clip bounds.

    ||A * eps||_F <= entry_bound * eps_max * d; we want the scaled norm
    below 0.5, so s = ceil(log2(bound / 0.5)) (floored at 0).
    """
    import math
    bound = entry_bound * eps_max * d
    return max(0, math.ceil(math.log2(max(bound / 0.5, 1.0))))


class DifferentiableLEDHLogLikelihoodSVSSMmulti:
    """V1 multivariate canonical SVSSM filter.

    First-pass scope: diagonal Phi (passed as a vector `phi_diag`) and
    diagonal Sigma_eta (passed as `sigma_eta_diag_sq`). Per-particle
    state covariance is stored as a (N, d) vector since it stays
    diagonal under these dynamics --- exposes the code path that will
    later generalise to (N, d, d) for non-diagonal cases.

    The per-step body uses `@tf.function(jit_compile=True)` for the
    same reasons as the d=1 filter (Phase 1).
    """

    LOG_CHI2_MEAN = LOG_CHI2_MEAN
    LOG_CHI2_VAR = LOG_CHI2_VAR

    def __init__(
        self,
        state_dim: int = 2,
        num_particles: int = 64,
        n_lambda: int = 10,
        sinkhorn_epsilon: float = 1.0,
        sinkhorn_iters: int = 10,
        resample_threshold: float = 0.5,
        grad_window: int = 4,
        jit_compile: bool = True,
        log_y_sq_offset: float = 1e-8,
        init_type: str = "stationary",
        diffuse_var: float = 100.0,
        mat_phi_ridge: float = 1e-3,
        mat_phi_clip_particle: float = 50.0,
        mat_phi_clip_P: float = 1e3,
    ):
        if init_type not in {"stationary", "fixed_mu", "diffuse"}:
            raise ValueError(
                f"init_type must be 'stationary', 'fixed_mu', or 'diffuse', "
                f"got {init_type!r}"
            )
        # Numerical guards for the full-matrix-Phi (call_mat_phi) path ONLY.
        # Defaults reproduce the original heavy guards (needed when HMC drives
        # phi_off large -> ill-conditioned P). These are TRUE constants for the
        # object's lifetime, so reading them inside the JIT'd per-step body is
        # safe (config, not per-call state). Lower the ridge / loosen the clips
        # for cleaner gradients at d=1 / near-diagonal Phi.
        self.mat_phi_ridge = float(mat_phi_ridge)
        self.mat_phi_clip_particle = float(mat_phi_clip_particle)
        self.mat_phi_clip_P = float(mat_phi_clip_P)
        self.state_dim = int(state_dim)
        self.num_particles = int(num_particles)
        self.n_lambda = int(n_lambda)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.resample_threshold = float(resample_threshold)
        self.grad_window = int(grad_window)
        self.jit_compile = bool(jit_compile)
        self.log_y_sq_offset = float(log_y_sq_offset)
        self.init_type = init_type
        self.diffuse_var = float(diffuse_var)

        # Geometric pseudo-time substep sizes (same recipe as d=1 filter).
        q = 1.2
        self.epsilon_1 = (1.0 - q) / (1.0 - q ** self.n_lambda)
        self.epsilons = [self.epsilon_1 * q ** j for j in range(self.n_lambda)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        mu: tf.Tensor,
        phi_diag: tf.Tensor,
        sigma_eta_diag_sq: tf.Tensor,
        y_obs: tf.Tensor,
    ) -> tf.Tensor:
        """Return log p_hat(z | theta) for the multivariate SVSSM.

        Parameters
        ----------
        mu                : (d,)    level vector
        phi_diag          : (d,)    persistence per component, |phi_i| < 1
        sigma_eta_diag_sq : (d,)    process-noise variance per component
        y_obs             : (T, d)  observation series
        """
        d = self.state_dim
        y_obs = tf.cast(y_obs, tf.float32)
        # Component-wise log-y^2 transform
        z_obs = tf.math.log(tf.square(y_obs) +
                            tf.constant(self.log_y_sq_offset, tf.float32))

        mu_t = tf.cast(mu, tf.float32)
        phi_t = tf.cast(phi_diag, tf.float32)
        sigma_eta_sq_t = tf.maximum(tf.cast(sigma_eta_diag_sq, tf.float32), _EPS)

        # Initial-state distribution (diagonal Phi + diagonal Sigma_eta).
        if self.init_type == "stationary":
            one_minus_phi_sq = tf.maximum(1.0 - phi_t ** 2, _EPS)   # (d,)
            init_var = sigma_eta_sq_t / one_minus_phi_sq            # (d,)
            init_mean = mu_t                                         # (d,)
        elif self.init_type == "fixed_mu":
            init_var = sigma_eta_sq_t
            init_mean = mu_t
        else:  # diffuse
            init_var = tf.fill([d], tf.constant(self.diffuse_var, tf.float32))
            init_mean = tf.zeros([d], tf.float32)

        log_ml = self._run_nd(
            mu_t, phi_t, sigma_eta_sq_t, z_obs, init_var, init_mean,
        )
        return _as_real_log_scalar(log_ml)

    # ------------------------------------------------------------------
    # Multivariate outer loop (Python for-loop calling per-step body)
    # ------------------------------------------------------------------

    def _run_nd(
        self,
        mu: tf.Tensor,                  # (d,)
        phi_diag: tf.Tensor,            # (d,)
        sigma_eta_sq_diag: tf.Tensor,   # (d,)
        z_obs: tf.Tensor,               # (T, d)
        init_var_diag: tf.Tensor,       # (d,)
        init_mean: tf.Tensor,           # (d,)
    ):
        """Outer loop. Per-step body is JIT'd."""
        N = self.num_particles
        d = self.state_dim
        T = int(z_obs.shape[0])
        N_f = tf.cast(N, tf.float32)
        R_diag = tf.fill([d], tf.constant(self.LOG_CHI2_VAR, tf.float32))   # (d,)
        mu_z = tf.constant(self.LOG_CHI2_MEAN, tf.float32)

        # Initial particles: shape (N, d).
        # Sample component-wise: particle_i = init_mean + diag(sqrt(init_var)) * noise
        particles = init_mean[tf.newaxis, :] + (
            tf.random.normal([N, d]) * tf.sqrt(init_var_diag)[tf.newaxis, :]
        )                                                                    # (N, d)

        # Per-particle state covariance diagonal: (N, d). Stays diagonal
        # under diagonal Phi + diagonal Sigma_eta.
        P_diag = tf.tile(init_var_diag[tf.newaxis, :], [N, 1])               # (N, d)

        log_w = tf.fill([N], -tf.math.log(N_f))                              # (N,)
        log_ml = tf.constant(0.0)

        for t_int in range(1, T + 1):
            if self.grad_window > 0 and t_int > 1 and (t_int - 1) % self.grad_window == 0:
                particles = tf.stop_gradient(particles)
                P_diag = tf.stop_gradient(P_diag)
                log_w = tf.stop_gradient(log_w)

            z_t = z_obs[t_int - 1]                                           # (d,)
            do_predict = tf.constant(t_int >= 2)

            particles, P_diag, log_w, log_ev = self._timestep_nd(
                particles, P_diag, log_w,
                mu, phi_diag, sigma_eta_sq_diag, R_diag, mu_z,
                z_t, do_predict,
            )
            log_ml = log_ml + tf.where(
                tf.math.is_finite(log_ev), log_ev, tf.constant(-10.0)
            )

        return log_ml

    # ------------------------------------------------------------------
    # Per-step body (predict + LEDH flow + weight update + resample)
    # ------------------------------------------------------------------

    def _timestep_nd_impl(
        self,
        particles, P_diag, log_w,
        mu, phi_diag, sigma_eta_sq_diag, R_diag, mu_z,
        z_t, do_predict,
    ):
        """One timestep of the multivariate filter (diagonal-Phi case)."""
        N = self.num_particles
        d = self.state_dim
        N_f = tf.cast(N, tf.float32)

        # ----- Predict (only for t >= 2) -----
        # x_det = mu + Phi (particles - mu), Phi diagonal: x_det_i = mu_i + phi_i * (x_i - mu_i)
        # noise ~ N(0, Sigma_eta), diagonal: per-component std
        def _predict():
            x_det = mu[tf.newaxis, :] + phi_diag[tf.newaxis, :] * (particles - mu[tf.newaxis, :])
            noise = tf.random.normal([N, d]) * tf.sqrt(sigma_eta_sq_diag)[tf.newaxis, :]
            p_new = _safe_nd(x_det + noise)
            # P_diag predict: P_ii ← phi_i^2 P_ii + sigma_eta_i^2
            P_new = tf.clip_by_value(
                phi_diag[tf.newaxis, :] ** 2 * P_diag + sigma_eta_sq_diag[tf.newaxis, :],
                _EPS, _CLAMP,
            )
            return p_new, P_new

        particles, P_diag = tf.cond(do_predict, _predict, lambda: (particles, P_diag))

        # ----- LEDH flow (component-wise under diagonal Sigma_eta + diagonal Phi) -----
        eta = tf.identity(particles)                                         # (N, d)
        log_det_jac = tf.zeros([N])                                          # (N,)
        lam_cum = 0.0

        # R is diagonal with all entries = LOG_CHI2_VAR.
        R_inv_diag = 1.0 / R_diag                                            # (d,)

        # Per-component innovation: z_t - mu_z - h (the H = I, e = mu_z*1 case).
        innov_const = tf.clip_by_value(z_t - mu_z, -100.0, 100.0)            # (d,)

        for j in range(self.n_lambda):
            eps_j = self.epsilons[j]
            lam_k = lam_cum + eps_j / 2.0
            lam_cum += eps_j

            # All component-wise (diagonal case): everything below has shape (N, d) or (d,).
            S_diag = tf.maximum(lam_k * P_diag + R_diag[tf.newaxis, :], _EPS)  # (N, d)
            A_diag = tf.clip_by_value(-0.5 * P_diag / S_diag, -10.0, 0.0)      # (N, d)
            lam_A = lam_k * A_diag
            I_lam_A = 1.0 + lam_A
            I_2lam_A = 1.0 + 2.0 * lam_A

            # b_vec component-wise: b_i = (I+2λA)_i [ (I+λA)_i P_i / R_i (z_i - μ_z) + A_i η_i ]
            b_vec = I_2lam_A * (
                I_lam_A * P_diag * R_inv_diag[tf.newaxis, :] * innov_const[tf.newaxis, :]
                + A_diag * eta
            )
            b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)

            # Exp-integrator substep (component-wise).
            Az = A_diag * eps_j
            exp_Az = tf.exp(Az)
            A_safe = tf.where(tf.abs(A_diag) > 1e-8, A_diag, tf.fill(tf.shape(A_diag), 1e-8))
            phi_A = tf.math.expm1(Az) / A_safe                                # (N, d)

            particles = _safe_nd(particles * exp_Az + b_vec * phi_A)
            eta = _safe_nd(eta * exp_Az + b_vec * phi_A)
            # log|det J| for the diagonal case: sum_i log|exp(A_ii eps)| = sum_i A_ii eps.
            log_det_jac = log_det_jac + tf.reduce_sum(Az, axis=1)

        # ----- Weight update (multivariate Gaussian under diagonal R) -----
        # resid: (N, d).  log-lik per particle = -0.5 sum_i resid_i^2 / R_ii - 0.5 log|R| - (d/2) log(2π)
        resid = z_t[tf.newaxis, :] - (particles + mu_z)                       # (N, d)
        log_lik = (
            -0.5 * tf.reduce_sum(R_inv_diag[tf.newaxis, :] * resid ** 2, axis=1)
            - 0.5 * tf.reduce_sum(tf.math.log(R_diag))
            - 0.5 * tf.cast(d, tf.float32) * tf.math.log(2.0 * 3.141592653589793)
        )                                                                     # (N,)
        log_lik = _safe_nd(log_lik)

        log_w_incr = log_lik + log_det_jac
        log_w_incr = tf.where(
            tf.math.is_finite(log_w_incr),
            log_w_incr,
            tf.constant(-100.0, dtype=tf.float32),
        )
        log_w_t = log_w + log_w_incr
        log_ev = tf.reduce_logsumexp(log_w_t)
        log_w_t = log_w_t - log_ev

        # ----- Sinkhorn OT resample on the (N, d) particle cloud -----
        # Normalise per-component before passing to det_resample.
        p_mean = tf.reduce_mean(particles, axis=0, keepdims=True)             # (1, d)
        p_std = tf.math.reduce_std(particles, axis=0, keepdims=True) + _EPS   # (1, d)
        p_norm = (particles - p_mean) / p_std                                  # (N, d)

        particles_new_norm, _ = det_resample(
            p_norm, log_w_t,
            epsilon=self.sinkhorn_epsilon, n_iters=self.sinkhorn_iters,
        )
        particles_new_norm = tf.cast(tf.math.real(particles_new_norm), tf.float32)
        particles = particles_new_norm * p_std + p_mean                        # de-normalise

        # Reset P_diag to its mean across particles (matches d=1 filter design).
        P_mean = tf.reduce_mean(P_diag, axis=0, keepdims=True)                 # (1, d)
        P_diag = tf.tile(P_mean, [N, 1])                                        # (N, d)
        log_w = tf.fill([N], -tf.math.log(N_f))

        return particles, P_diag, log_w, log_ev

    # Wrap _timestep_nd_impl with JIT at construction time (similar to d=1).
    def _timestep_nd(self, *args, **kwargs):
        if not hasattr(self, "_jit_step"):
            if self.jit_compile:
                self._jit_step = tf.function(self._timestep_nd_impl,
                                              jit_compile=True)
            else:
                self._jit_step = self._timestep_nd_impl
        return self._jit_step(*args, **kwargs)

    # ==================================================================
    # FULL-MATRIX code path (non-diagonal Sigma_eta).
    # ==================================================================
    #
    # Same model as above (diagonal Phi kept), but Sigma_eta is now a
    # full d×d PSD matrix passed as its Cholesky factor.  Per-particle
    # state covariance is stored as (N, d, d) and the LEDH substep uses
    # tf.linalg.expm + tf.linalg.solve.

    def call_full(
        self,
        mu: tf.Tensor,               # (d,)
        phi_diag: tf.Tensor,         # (d,)  -- diagonal Phi (vector)
        sigma_eta_chol: tf.Tensor,   # (d, d) lower-triangular Cholesky of Sigma_eta
        y_obs: tf.Tensor,            # (T, d)
    ) -> tf.Tensor:
        """Full-matrix Sigma_eta entry point.

        Sigma_eta = sigma_eta_chol @ sigma_eta_chol^T.  Pass a (d, d)
        lower-triangular factor (e.g.\ from tf.linalg.cholesky on a PSD
        matrix) to enforce positivity.
        """
        d = self.state_dim
        y_obs = tf.cast(y_obs, tf.float32)
        z_obs = tf.math.log(tf.square(y_obs) +
                            tf.constant(self.log_y_sq_offset, tf.float32))

        mu_t = tf.cast(mu, tf.float32)
        phi_t = tf.cast(phi_diag, tf.float32)
        L_eta = tf.cast(sigma_eta_chol, tf.float32)            # (d, d) lower-triangular
        Sigma_eta = L_eta @ tf.transpose(L_eta)                # (d, d) PSD

        # Stationary initial covariance for diagonal Phi + full Sigma_eta:
        # Sigma_h satisfies  Sigma_h = Phi Sigma_h Phi^T + Sigma_eta.
        # For diagonal Phi: Sigma_h[i,j] = phi_i phi_j Sigma_h[i,j] + Sigma_eta[i,j]
        # → Sigma_h[i,j] = Sigma_eta[i,j] / (1 - phi_i phi_j).
        if self.init_type == "stationary":
            phi_outer = phi_t[:, tf.newaxis] * phi_t[tf.newaxis, :]   # (d, d)
            init_cov = Sigma_eta / tf.maximum(1.0 - phi_outer, _EPS)
            init_mean = mu_t
        elif self.init_type == "fixed_mu":
            init_cov = Sigma_eta
            init_mean = mu_t
        else:  # diffuse
            init_cov = tf.constant(self.diffuse_var, tf.float32) * tf.eye(d, dtype=tf.float32)
            init_mean = tf.zeros([d], tf.float32)

        log_ml = self._run_nd_full(
            mu_t, phi_t, Sigma_eta, L_eta, z_obs, init_cov, init_mean,
        )
        return _as_real_log_scalar(log_ml)

    def _run_nd_full(
        self,
        mu, phi_diag, Sigma_eta, L_eta,
        z_obs, init_cov, init_mean,
    ):
        """Outer loop for the full-matrix path. Per-step body is JIT'd."""
        N = self.num_particles
        d = self.state_dim
        T = int(z_obs.shape[0])
        N_f = tf.cast(N, tf.float32)
        R_scalar = tf.constant(self.LOG_CHI2_VAR, tf.float32)
        mu_z = tf.constant(self.LOG_CHI2_MEAN, tf.float32)

        # Initial particles: shape (N, d).
        # h_0 ~ N(init_mean, init_cov), sample via L_init @ noise
        L_init = tf.linalg.cholesky(init_cov +
                                    tf.constant(1e-6, tf.float32) * tf.eye(d, dtype=tf.float32))
        noise0 = tf.random.normal([N, d])
        particles = init_mean[tf.newaxis, :] + tf.einsum("ij,nj->ni", L_init, noise0)

        # Per-particle state covariance: (N, d, d). Initially same for all particles.
        P = tf.tile(init_cov[tf.newaxis, :, :], [N, 1, 1])

        log_w = tf.fill([N], -tf.math.log(N_f))
        log_ml = tf.constant(0.0)

        for t_int in range(1, T + 1):
            if self.grad_window > 0 and t_int > 1 and (t_int - 1) % self.grad_window == 0:
                particles = tf.stop_gradient(particles)
                P = tf.stop_gradient(P)
                log_w = tf.stop_gradient(log_w)

            z_t = z_obs[t_int - 1]                           # (d,)
            do_predict = tf.constant(t_int >= 2)

            particles, P, log_w, log_ev = self._timestep_nd_full(
                particles, P, log_w,
                mu, phi_diag, Sigma_eta, L_eta, R_scalar, mu_z,
                z_t, do_predict,
            )
            log_ml = log_ml + tf.where(
                tf.math.is_finite(log_ev), log_ev, tf.constant(-10.0)
            )

        return log_ml

    def _timestep_nd_full_impl(
        self,
        particles, P, log_w,
        mu, phi_diag, Sigma_eta, L_eta, R_scalar, mu_z,
        z_t, do_predict,
    ):
        """One timestep of the full-matrix multivariate filter."""
        N = self.num_particles
        d = self.state_dim
        N_f = tf.cast(N, tf.float32)
        I_d = tf.eye(d, dtype=tf.float32)
        I_d_batch = tf.tile(I_d[tf.newaxis, :, :], [N, 1, 1])  # (N, d, d)

        # ----- Predict -----
        def _predict():
            x_det = mu[tf.newaxis, :] + phi_diag[tf.newaxis, :] * (particles - mu[tf.newaxis, :])
            noise = tf.einsum("ij,nj->ni", L_eta, tf.random.normal([N, d]))
            p_new = _safe_nd(x_det + noise)
            # P predict: P_new = Phi P Phi^T + Sigma_eta.
            # For diagonal Phi: (Phi P Phi^T)[i,j] = phi_i phi_j P[i,j]
            phi_outer = phi_diag[:, tf.newaxis] * phi_diag[tf.newaxis, :]   # (d, d)
            P_new = (phi_outer[tf.newaxis, :, :] * P) + Sigma_eta[tf.newaxis, :, :]
            P_new = tf.clip_by_value(P_new, -_CLAMP, _CLAMP)
            return p_new, P_new

        particles, P = tf.cond(do_predict, _predict, lambda: (particles, P))

        # ----- LEDH flow (matrix form) -----
        # Under H = I and diagonal R = R_scalar * I:
        #   S = lam_k * P + R_scalar * I        shape (N, d, d)
        #   A = -0.5 * P @ S^{-1}               shape (N, d, d)
        # Note: tf.linalg.solve(S, P) computes S^{-1} @ P; we want P @ S^{-1}.
        # Use: P @ S^{-1} = (S^{-T} @ P^T)^T = solve(S^T, P^T)^T.
        #
        # Innovation under H = I: innov = z_t - mu_z * 1 - h.

        eta = tf.identity(particles)                                       # (N, d)
        log_det_jac = tf.zeros([N])                                        # (N,)
        lam_cum = 0.0

        innov_const = tf.clip_by_value(z_t - mu_z, -100.0, 100.0)          # (d,)
        innov_const_batch = tf.tile(innov_const[tf.newaxis, :], [N, 1])    # (N, d)

        # Pre-compute R inverse (scalar diag is trivial).
        R_inv_scalar = 1.0 / R_scalar

        for j in range(self.n_lambda):
            eps_j = self.epsilons[j]
            lam_k = lam_cum + eps_j / 2.0
            lam_cum += eps_j

            S = lam_k * P + R_scalar * I_d_batch                            # (N, d, d)
            # Tiny ridge to keep S well-conditioned.
            S_ridge = S + tf.constant(1e-6, tf.float32) * I_d_batch
            # A = -0.5 P S^{-1}
            # Compute via solve(S_ridge^T, P^T)^T  → batched.
            S_T = tf.linalg.matrix_transpose(S_ridge)
            P_T = tf.linalg.matrix_transpose(P)
            A_T = tf.linalg.solve(S_T, P_T)                                # (N, d, d)
            A = -0.5 * tf.linalg.matrix_transpose(A_T)                     # (N, d, d)
            A = tf.clip_by_value(A, -10.0, 10.0)

            # I + lam_k A and I + 2 lam_k A
            I_lam_A   = I_d_batch + lam_k * A                              # (N, d, d)
            I_2lam_A  = I_d_batch + 2.0 * lam_k * A                        # (N, d, d)

            # inner_v = (I + lam_k A) @ P / R @ innov   (R is scalar)
            # First: P @ innov:  (N, d, d) @ (N, d) → (N, d)
            P_innov = tf.einsum("nij,nj->ni", P, innov_const_batch)        # (N, d)
            # Then (I + lam_k A) @ (P_innov / R):
            ILA_P_innov = tf.einsum("nij,nj->ni", I_lam_A, P_innov) * R_inv_scalar
            # A @ eta (per particle):
            A_eta = tf.einsum("nij,nj->ni", A, eta)                        # (N, d)
            # b = (I + 2 lam_k A) @ ( ILA_P_innov + A_eta )
            b_vec = tf.einsum("nij,nj->ni", I_2lam_A, ILA_P_innov + A_eta) # (N, d)
            b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)

            # Exp integrator: h ← exp(A eps) h + phi_A(eps) @ b
            # where  phi_A(eps) = ∫_0^eps exp(A s) ds = A^{-1} (exp(A eps) - I).
            # So phi_A(eps) @ b = solve(A, (exp(A eps) - I) @ b).
            # NOTE: denominator is A, not A*eps; the eps factor is inside exp(A*eps).
            A_eps = A * eps_j                                              # (N, d, d)
            if d == 2:
                exp_Aeps = expm_2x2_batch(A_eps)                           # XLA-safe closed form
            elif d == 1:
                exp_Aeps = tf.exp(A_eps)                                   # XLA-safe for 1x1 matrix
            else:
                exp_Aeps = expm_pade_batch(                                # XLA-safe fixed-s Pade
                    A_eps, s=pade_scaling_for_dim(d))
            A_ridge = A + tf.constant(1e-6, tf.float32) * I_d_batch        # (N, d, d)
            exp_minus_I_b = tf.einsum("nij,nj->ni",
                                        exp_Aeps - I_d_batch, b_vec)       # (N, d)
            phi_Ab = tf.linalg.solve(A_ridge,
                                      exp_minus_I_b[..., tf.newaxis])[..., 0]   # (N, d)

            # Update particles and eta:
            particles_new = tf.einsum("nij,nj->ni", exp_Aeps, particles) + phi_Ab
            eta_new       = tf.einsum("nij,nj->ni", exp_Aeps, eta) + phi_Ab
            particles = _safe_nd(particles_new)
            eta = _safe_nd(eta_new)

            # log|det J| for the matrix case:  log|det exp(A eps)| = tr(A eps).
            log_det_jac = log_det_jac + tf.linalg.trace(A_eps)              # (N,)

        # ----- Weight update -----
        # Under H = I and R = R_scalar * I_d:
        #   log_lik[n] = -0.5/R * ||z - mu_z - h||^2 - 0.5 d log(R) - 0.5 d log(2 pi)
        resid = z_t[tf.newaxis, :] - (particles + mu_z)                    # (N, d)
        log_lik = (
            -0.5 * R_inv_scalar * tf.reduce_sum(resid ** 2, axis=1)
            - 0.5 * tf.cast(d, tf.float32) * tf.math.log(R_scalar)
            - 0.5 * tf.cast(d, tf.float32) * tf.math.log(2.0 * 3.141592653589793)
        )
        log_lik = _safe_nd(log_lik)

        log_w_incr = log_lik + log_det_jac
        log_w_incr = tf.where(
            tf.math.is_finite(log_w_incr),
            log_w_incr,
            tf.constant(-100.0, dtype=tf.float32),
        )
        log_w_t = log_w + log_w_incr
        log_ev = tf.reduce_logsumexp(log_w_t)
        log_w_t = log_w_t - log_ev

        # ----- Sinkhorn OT resample -----
        p_mean = tf.reduce_mean(particles, axis=0, keepdims=True)          # (1, d)
        p_std = tf.math.reduce_std(particles, axis=0, keepdims=True) + _EPS
        p_norm = (particles - p_mean) / p_std
        particles_new_norm, _ = det_resample(
            p_norm, log_w_t,
            epsilon=self.sinkhorn_epsilon, n_iters=self.sinkhorn_iters,
        )
        particles_new_norm = tf.cast(tf.math.real(particles_new_norm), tf.float32)
        particles = particles_new_norm * p_std + p_mean

        # Reset per-particle P to its mean across particles.
        P_mean = tf.reduce_mean(P, axis=0, keepdims=True)                  # (1, d, d)
        P = tf.tile(P_mean, [N, 1, 1])                                      # (N, d, d)
        log_w = tf.fill([N], -tf.math.log(N_f))

        return particles, P, log_w, log_ev

    def _timestep_nd_full(self, *args, **kwargs):
        # XLA-safe at EVERY d: d=2 uses the closed-form expm_2x2_batch,
        # d=1 uses tf.exp, d>2 uses the fixed-s Pade expm
        # (expm_pade_batch) -- all static control flow.
        if not hasattr(self, "_jit_step_full"):
            self._jit_step_full = tf.function(self._timestep_nd_full_impl,
                                               jit_compile=self.jit_compile)
        return self._jit_step_full(*args, **kwargs)

    # ==================================================================
    # FULL-MATRIX-PHI code path (non-diagonal Phi, cross-asset persistence)
    # ==================================================================
    #
    # Same model as call_full, but Phi is now a full (d, d) matrix —
    # captures cross-asset persistence h_{t,i} depends on lagged h_{t-1,j}
    # for j != i. Required by the V1 multivariate identifiability question
    # (rotation indeterminacy of the latent volatilities).
    #
    # Restrictions: |eigval(Phi)| < 1 for stationarity. Caller responsibility.
    # We compute stationary initial covariance via discrete Lyapunov solve
    #   Sigma_h = Phi Sigma_h Phi^T + Sigma_eta
    # equivalently vec(Sigma_h) = (I - kron(Phi, Phi))^{-1} vec(Sigma_eta).
    # For d <= 10 this is trivial.

    @staticmethod
    def _discrete_lyapunov_solve(Phi: tf.Tensor, Sigma_eta: tf.Tensor,
                                   n_doublings: int = 15) -> tf.Tensor:
        """Solve Sigma_h = Phi Sigma_h Phi^T + Sigma_eta for Sigma_h (d, d).

        Smith's doubling algorithm: starting from
            X_0 = Sigma_eta,   A_0 = Phi
        iterate
            X_{k+1} = X_k + A_k @ X_k @ A_k^T,
            A_{k+1} = A_k @ A_k.
        Then X_k -> sum_{j=0}^{2^k - 1} Phi^j Sigma_eta (Phi^j)^T, which
        converges to the true Sigma_h at rate spec_rad(Phi)^{2^k} — i.e.
        each doubling SQUARES the error. At n_doublings=15 the residual
        is spec_rad^{2^15} = spec_rad^{32768} — numerically exact for any
        stationary Phi (spec_rad < 1). Survives gracefully on near-
        boundary Phi (spec_rad → 1) by returning the partial sum, large
        but finite. Gradient-friendly throughout.

        Why not the closed-form vec(I - Phi ⊗ Phi)^{-1} solve: that
        matrix becomes near-singular when Phi has eigenvalues approaching
        the unit circle (which HMC can wander into between leapfrog
        steps), and tf.linalg.solve then raises "matrix not invertible"
        with no gradient signal. Smith's iteration handles this case
        without a hard failure.
        """
        X = tf.identity(Sigma_eta)
        A = tf.identity(Phi)
        for _ in range(n_doublings):
            X = X + A @ X @ tf.transpose(A)
            A = A @ A
        return 0.5 * (X + tf.transpose(X))

    def call_mat_phi(
        self,
        mu: tf.Tensor,               # (d,)
        Phi: tf.Tensor,              # (d, d)  -- FULL matrix Phi (e.g.\ upper-triangular)
        sigma_eta_chol: tf.Tensor,   # (d, d)  -- lower-triangular Cholesky of Sigma_eta
        y_obs: tf.Tensor,            # (T, d)
    ) -> tf.Tensor:
        """Full-matrix Phi entry point.

        Phi may be any (d, d) matrix with eigenvalues inside the unit disk.
        Sigma_eta = sigma_eta_chol @ sigma_eta_chol^T. Both Phi and
        Sigma_eta can be non-diagonal.
        """
        d = self.state_dim
        y_obs = tf.cast(y_obs, tf.float32)
        z_obs = tf.math.log(tf.square(y_obs) +
                            tf.constant(self.log_y_sq_offset, tf.float32))

        mu_t = tf.cast(mu, tf.float32)
        Phi_t = tf.cast(Phi, tf.float32)
        L_eta = tf.cast(sigma_eta_chol, tf.float32)
        Sigma_eta = L_eta @ tf.transpose(L_eta)

        if self.init_type == "stationary":
            init_cov = self._discrete_lyapunov_solve(Phi_t, Sigma_eta)
            # Defensive clamp: a near-non-stationary Phi can make the
            # Lyapunov solution arbitrarily large.
            init_cov = tf.clip_by_value(init_cov, -1e3, 1e3)
            init_mean = mu_t
        elif self.init_type == "fixed_mu":
            init_cov = Sigma_eta
            init_mean = mu_t
        else:  # diffuse
            init_cov = tf.constant(self.diffuse_var, tf.float32) * tf.eye(d, dtype=tf.float32)
            init_mean = tf.zeros([d], tf.float32)

        log_ml = self._run_nd_full_phi(
            mu_t, Phi_t, Sigma_eta, L_eta, z_obs, init_cov, init_mean,
        )
        return _as_real_log_scalar(log_ml)

    def _run_nd_full_phi(
        self,
        mu, Phi, Sigma_eta, L_eta,
        z_obs, init_cov, init_mean,
    ):
        """Outer loop for the full-matrix-Phi path. Per-step body is tf.function'd."""
        N = self.num_particles
        d = self.state_dim
        T = int(z_obs.shape[0])
        N_f = tf.cast(N, tf.float32)
        R_scalar = tf.constant(self.LOG_CHI2_VAR, tf.float32)
        mu_z = tf.constant(self.LOG_CHI2_MEAN, tf.float32)

        L_init = tf.linalg.cholesky(init_cov +
                                    tf.constant(1e-6, tf.float32) * tf.eye(d, dtype=tf.float32))
        noise0 = tf.random.normal([N, d])
        particles = init_mean[tf.newaxis, :] + tf.einsum("ij,nj->ni", L_init, noise0)

        P = tf.tile(init_cov[tf.newaxis, :, :], [N, 1, 1])

        log_w = tf.fill([N], -tf.math.log(N_f))
        log_ml = tf.constant(0.0)

        for t_int in range(1, T + 1):
            if self.grad_window > 0 and t_int > 1 and (t_int - 1) % self.grad_window == 0:
                particles = tf.stop_gradient(particles)
                P = tf.stop_gradient(P)
                log_w = tf.stop_gradient(log_w)

            z_t = z_obs[t_int - 1]
            do_predict = tf.constant(t_int >= 2)

            particles, P, log_w, log_ev = self._timestep_nd_mat_phi(
                particles, P, log_w,
                mu, Phi, Sigma_eta, L_eta, R_scalar, mu_z,
                z_t, do_predict,
            )
            log_ml = log_ml + tf.where(
                tf.math.is_finite(log_ev), log_ev, tf.constant(-10.0)
            )

        return log_ml

    def _timestep_nd_mat_phi_impl(
        self,
        particles, P, log_w,
        mu, Phi, Sigma_eta, L_eta, R_scalar, mu_z,
        z_t, do_predict,
    ):
        """One timestep of the full-matrix-Phi multivariate filter.

        Identical to _timestep_nd_full_impl except for the predict step:
          x_det = mu + (particles - mu) @ Phi^T
          P_new = Phi P Phi^T + Sigma_eta
        The LEDH flow / weight update / Sinkhorn resample don't see Phi.
        """
        N = self.num_particles
        d = self.state_dim
        N_f = tf.cast(N, tf.float32)
        I_d = tf.eye(d, dtype=tf.float32)
        I_d_batch = tf.tile(I_d[tf.newaxis, :, :], [N, 1, 1])
        # Configurable numerical guards (true constants; safe under JIT).
        _ridge = self.mat_phi_ridge
        _clipp = self.mat_phi_clip_particle
        _clipP = self.mat_phi_clip_P

        # ----- Predict (full matrix Phi) -----
        # Stronger numerical guards than the diagonal-Phi path: when HMC
        # explores phi_off values away from 0, matrix Phi can have large
        # spectral norm even at stable eigenvalues, amplifying particles
        # and P transiently. Tight clamping prevents NaN propagation into
        # the LEDH flow's matrix solves.
        def _predict():
            x_det = mu[tf.newaxis, :] + tf.einsum(
                "nj,ij->ni", particles - mu[tf.newaxis, :], Phi,
            )
            noise = tf.einsum("ij,nj->ni", L_eta, tf.random.normal([N, d]))
            # Clip particles tighter than _CLAMP (1e4) — h values larger
            # than ~50 mean exp(h/2) > 1e10, which kills the log-y² loss.
            p_new = tf.clip_by_value(_safe_nd(x_det + noise), -_clipp, _clipp)
            # P_new = Phi P Phi^T + Sigma_eta. Batched over N particles.
            PhiP = tf.einsum("ik,nkl->nil", Phi, P)
            PhiPPhiT = tf.einsum("nil,jl->nij", PhiP, Phi)
            P_new = PhiPPhiT + Sigma_eta[tf.newaxis, :, :]
            # Clamp P. P entries too large push the LEDH flow's
            # S = lam_k * P + R*I into ill-conditioning.
            P_new = tf.clip_by_value(P_new, -_clipP, _clipP)
            return p_new, P_new

        particles, P = tf.cond(do_predict, _predict, lambda: (particles, P))

        # ----- LEDH flow (matrix form; identical to _timestep_nd_full_impl) -----
        eta = tf.identity(particles)
        log_det_jac = tf.zeros([N])
        lam_cum = 0.0

        innov_const = tf.clip_by_value(z_t - mu_z, -100.0, 100.0)
        innov_const_batch = tf.tile(innov_const[tf.newaxis, :], [N, 1])

        R_inv_scalar = 1.0 / R_scalar

        # Sanitise P entering the LEDH flow: replace any non-finite with
        # zero, then symmetrise (the predict step's clamping can leave
        # numerical asymmetry). This is the entry to a cascade of matrix
        # solves; one NaN here propagates everywhere downstream.
        P = tf.where(tf.math.is_finite(P), P, tf.zeros_like(P))
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))

        for j in range(self.n_lambda):
            eps_j = self.epsilons[j]
            lam_k = lam_cum + eps_j / 2.0
            lam_cum += eps_j

            S = lam_k * P + R_scalar * I_d_batch
            # Bigger ridge for the matrix-Phi path: the HMC can take P
            # into ill-conditioned regions that 1e-6 ridge doesn't rescue.
            S_ridge = S + _ridge * I_d_batch
            S_T = tf.linalg.matrix_transpose(S_ridge)
            P_T = tf.linalg.matrix_transpose(P)
            A_T = tf.linalg.solve(S_T, P_T)
            A = -0.5 * tf.linalg.matrix_transpose(A_T)
            A = tf.clip_by_value(A, -10.0, 10.0)

            I_lam_A   = I_d_batch + lam_k * A
            I_2lam_A  = I_d_batch + 2.0 * lam_k * A

            P_innov = tf.einsum("nij,nj->ni", P, innov_const_batch)
            ILA_P_innov = tf.einsum("nij,nj->ni", I_lam_A, P_innov) * R_inv_scalar
            A_eta = tf.einsum("nij,nj->ni", A, eta)
            b_vec = tf.einsum("nij,nj->ni", I_2lam_A, ILA_P_innov + A_eta)
            b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)

            A_eps = A * eps_j
            if d == 2:
                exp_Aeps = expm_2x2_batch(A_eps)                           # XLA-safe closed form
            elif d == 1:
                exp_Aeps = tf.exp(A_eps)                                   # XLA-safe for 1x1 matrix
            else:
                exp_Aeps = expm_pade_batch(                                # XLA-safe fixed-s Pade
                    A_eps, s=pade_scaling_for_dim(d))
            # Same bigger ridge for the phi_A solve.
            A_ridge = A + _ridge * I_d_batch
            exp_minus_I_b = tf.einsum("nij,nj->ni",
                                        exp_Aeps - I_d_batch, b_vec)
            phi_Ab = tf.linalg.solve(A_ridge,
                                      exp_minus_I_b[..., tf.newaxis])[..., 0]

            particles_new = tf.einsum("nij,nj->ni", exp_Aeps, particles) + phi_Ab
            eta_new       = tf.einsum("nij,nj->ni", exp_Aeps, eta) + phi_Ab
            particles = tf.clip_by_value(_safe_nd(particles_new), -_clipp, _clipp)
            eta = tf.clip_by_value(_safe_nd(eta_new), -_clipp, _clipp)

            log_det_jac = log_det_jac + tf.linalg.trace(A_eps)

        # ----- Weight update -----
        resid = z_t[tf.newaxis, :] - (particles + mu_z)
        log_lik = (
            -0.5 * R_inv_scalar * tf.reduce_sum(resid ** 2, axis=1)
            - 0.5 * tf.cast(d, tf.float32) * tf.math.log(R_scalar)
            - 0.5 * tf.cast(d, tf.float32) * tf.math.log(2.0 * 3.141592653589793)
        )
        log_lik = _safe_nd(log_lik)

        log_w_incr = log_lik + log_det_jac
        log_w_incr = tf.where(
            tf.math.is_finite(log_w_incr),
            log_w_incr,
            tf.constant(-100.0, dtype=tf.float32),
        )
        log_w_t = log_w + log_w_incr
        log_ev = tf.reduce_logsumexp(log_w_t)
        log_w_t = log_w_t - log_ev

        # ----- Sinkhorn OT resample -----
        p_mean = tf.reduce_mean(particles, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(particles, axis=0, keepdims=True) + _EPS
        p_norm = (particles - p_mean) / p_std
        particles_new_norm, _ = det_resample(
            p_norm, log_w_t,
            epsilon=self.sinkhorn_epsilon, n_iters=self.sinkhorn_iters,
        )
        particles_new_norm = tf.cast(tf.math.real(particles_new_norm), tf.float32)
        particles = particles_new_norm * p_std + p_mean

        P_mean = tf.reduce_mean(P, axis=0, keepdims=True)
        P = tf.tile(P_mean, [N, 1, 1])
        log_w = tf.fill([N], -tf.math.log(N_f))

        return particles, P, log_w, log_ev

    def _timestep_nd_mat_phi(self, *args, **kwargs):
        # XLA-safe at EVERY d: d=2 closed-form expm_2x2_batch, d=1
        # tf.exp, d>2 fixed-s Pade expm -- all static control flow.
        if not hasattr(self, "_jit_step_mat_phi"):
            self._jit_step_mat_phi = tf.function(
                self._timestep_nd_mat_phi_impl, jit_compile=self.jit_compile,
            )
        return self._jit_step_mat_phi(*args, **kwargs)
