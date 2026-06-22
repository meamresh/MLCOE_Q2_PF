"""Multivariate (full-Phi) SVSSM LEDH filter with neural OT resampling.

Drop-in replacement for the Sinkhorn-resampling step in
:class:`DifferentiableLEDHLogLikelihoodSVSSMmulti` (the
upper-triangular Phi V1 multivariate filter from
``differentiable_ledh_svssm_multivariate.py``). All filter math
(matrix-form predict, LEDH flow with dimension-dispatched XLA-safe
expm, Smith doubling Lyapunov for h_0) is inherited unchanged. The only
substitution is the OT-resample block, which now calls a trained
DeepONet neural operator instead of Sinkhorn.

Why this matters
~~~~~~~~~~~~~~~~
Section 2's univariate NN-OT (Phase 6-12) measured a 1.3-2.3x speedup
over Sinkhorn at N=64-256. At higher N the speedup grows because
Sinkhorn is O(N^2 K) per resample while DeepONet is O(N * branch_dim).
The multivariate filter has the same Sinkhorn bottleneck at the
resample boundary, so the same neural-operator substitution should
yield comparable speedups.

Context vector for the multivariate operator (any d)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The operator is conditioned on all unconstrained theta parameters
(the same vector HMC samples in) plus per-step context. Generic
layout, dim = ``svssm_multi_ctx_dim(d) = 3d + d(d-1)/2 + 3 + d``::

    ctx = [ mu (d),
            phi_diag_raw (d),
            phi_off upper-tri row-major (d(d-1)/2),
            log_sigma_eta_sq (d),
            t / T_max,
            z_t (d),
            ESS,
            epsilon ]

d=1 gives the 7-D univariate layout, d=2 the 12-D Phase 16 layout
(both orderings preserved exactly, so trained checkpoints remain
loadable). This matches the HMC unconstrained parameterisation, so
the operator's training distribution can cover the HMC posterior
support uniformly (same Phase 2 design choice as univariate).
"""

from __future__ import annotations

import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
    expm_2x2_batch,
    expm_pade_batch,
    pade_scaling_for_dim,
    _safe_nd,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    _EPS,
)


# Legacy aliases (= svssm_multi_ctx_dim(2) and (1)); kept because the
# phase16 training/experiment scripts import them by name.
SVSSM_MULTI_CTX_DIM_D2 = 12
SVSSM_MULTI_CTX_DIM_D1 = 7


def svssm_multi_ctx_dim(d: int) -> int:
    """Context dim for state_dim d:
    mu(d) + phi_diag_raw(d) + phi_off(d*(d-1)/2) + log_sigma_eta_sq(d)
    + t/T_max(1) + z_t(d) + ESS(1) + epsilon(1).
    """
    return 3 * d + d * (d - 1) // 2 + 3 + d


def build_svssm_multi_context_scalars(
    mu: tf.Tensor,                # (d,)
    Phi: tf.Tensor,               # (d, d)  upper-triangular
    sigma_eta_sq_diag: tf.Tensor, # (d,)
    t,                            # scalar tensor or float
    z_t: tf.Tensor,               # (d,)
    ess,                          # scalar tensor or float (XLA-safe: pass tensor!)
    epsilon,                      # scalar tensor or float
    T_max,                        # scalar tensor or float
    d: int = 2,
) -> tf.Tensor:
    """Build the SVSSM multivariate operator context vector for any d.

    Returns shape ``(svssm_multi_ctx_dim(d),)`` in HMC's unconstrained
    parameterisation:

        [mu (d), phi_diag_raw (d), phi_off upper-tri row-major (d(d-1)/2),
         log_sigma_eta_sq (d), t/T_max, z_t (d), ESS, epsilon]
    """
    phi_diag = tf.linalg.diag_part(Phi)
    phi_diag_clipped = tf.clip_by_value(phi_diag, -0.9999, 0.9999)
    phi_diag_raw = tf.atanh(phi_diag_clipped)
    log_sigma_eta_sq = tf.math.log(tf.maximum(sigma_eta_sq_diag, _EPS))

    t_norm = tf.cast(t, tf.float32) / tf.cast(T_max, tf.float32)
    ess_t = tf.cast(ess, tf.float32)
    eps_t = tf.cast(epsilon, tf.float32)

    # Generic layout for any d (matches svssm_multi_ctx_dim(d)):
    #   [mu (d), phi_diag_raw (d), phi_off upper-tri row-major (d(d-1)/2),
    #    log_sigma_eta_sq (d), t/T_max, z_t (d), ESS, epsilon]
    # d is a static Python int at trace time, so the index loops unroll
    # into static gathers -- fully XLA-safe. For d=1 the phi_off block
    # is empty (7-D context); for d=2 this reproduces the original
    # 12-D Phase 16 layout exactly.
    entries = [mu[i] for i in range(d)]
    entries += [phi_diag_raw[i] for i in range(d)]
    entries += [Phi[i, j] for i in range(d) for j in range(i + 1, d)]
    entries += [log_sigma_eta_sq[i] for i in range(d)]
    entries += [t_norm]
    entries += [z_t[i] for i in range(d)]
    entries += [ess_t, eps_t]
    return tf.stack(entries)


def _compute_ess(weights_normalised: tf.Tensor) -> tf.Tensor:
    return 1.0 / tf.reduce_sum(weights_normalised ** 2)


class DifferentiableLEDHNeuralOTSVSSMmulti(
        DifferentiableLEDHLogLikelihoodSVSSMmulti):
    """Multivariate full-Phi SVSSM filter with neural-OT resampling.

    Inherits all the math from
    :class:`DifferentiableLEDHLogLikelihoodSVSSMmulti` (matrix predict,
    LEDH flow with dimension-dispatched XLA-safe expm, Smith doubling
    Lyapunov, weight update). Overrides only the OT-resample block
    inside ``_timestep_nd_mat_phi_impl`` to call a trained neural
    operator.

    Parameters
    ----------
    neural_ot_model
        Pre-trained DeepONet (or compatible) with batched signature
        ``model((B, N, d) particles, (B, N) weights, (B, ctx_dim) ctx)
        -> (B, N, d)`` where ``ctx_dim = svssm_multi_ctx_dim(d)``.
    All other args
        Forwarded to :class:`DifferentiableLEDHLogLikelihoodSVSSMmulti`.
        ``jit_compile=True`` is fully supported at ANY d: the Keras
        forward fuses into the XLA cluster (Phase 6 lesson — the JIT
        blockers are Python-side scalar conversions, not Keras), and
        the LEDH-flow expm is XLA-static at every d (closed-form 2x2
        at d=2, tf.exp at d=1, fixed-s Pade otherwise).
    """

    def __init__(self, neural_ot_model, **kwargs):
        super().__init__(**kwargs)
        self.neural_ot_model = neural_ot_model

        # ONLY truly call-static state lives on self: T_max, the series
        # length used for the t/T_max context entry. Do NOT stash
        # per-HMC-call theta here — self.* attributes are frozen into
        # the XLA trace as constants on first call (see memory:
        # feedback_tf_function_self_state). Theta reaches the per-step
        # context builder through the function arguments instead.
        self._current_T_max = None

    # ------------------------------------------------------------------
    # Public entry point: same signature as call_mat_phi, but stash
    # T_max so the per-step body can build the t/T_max context entry.
    # ------------------------------------------------------------------
    def call_mat_phi(self, mu, Phi, sigma_eta_chol, y_obs):
        self._current_T_max = float(int(y_obs.shape[0]))
        return super().call_mat_phi(mu, Phi, sigma_eta_chol, y_obs)

    # ------------------------------------------------------------------
    # Override the per-timestep body: same as parent up to the resample,
    # then swap Sinkhorn for the neural operator call.
    # ------------------------------------------------------------------
    def _timestep_nd_mat_phi_impl(
        self,
        particles, P, log_w,
        mu, Phi, Sigma_eta, L_eta, R_scalar, mu_z,
        z_t, do_predict,
    ):
        """Same as parent's _timestep_nd_mat_phi_impl up to the resample.

        We reproduce the predict + LEDH-flow + weight-update logic
        inline (copy from parent) rather than refactor the parent
        because the resample is in the middle of the function. After
        the weight update we call the neural operator instead of
        Sinkhorn.
        """
        N = self.num_particles
        d = self.state_dim
        N_f = tf.cast(N, tf.float32)
        I_d = tf.eye(d, dtype=tf.float32)
        I_d_batch = tf.tile(I_d[tf.newaxis, :, :], [N, 1, 1])

        # ----- Predict (same as parent's full-Phi predict) -----
        def _predict():
            x_det = mu[tf.newaxis, :] + tf.einsum(
                "nj,ij->ni", particles - mu[tf.newaxis, :], Phi,
            )
            noise = tf.einsum("ij,nj->ni", L_eta, tf.random.normal([N, d]))
            p_new = tf.clip_by_value(_safe_nd(x_det + noise), -50.0, 50.0)
            PhiP = tf.einsum("ik,nkl->nil", Phi, P)
            PhiPPhiT = tf.einsum("nil,jl->nij", PhiP, Phi)
            P_new = PhiPPhiT + Sigma_eta[tf.newaxis, :, :]
            P_new = tf.clip_by_value(P_new, -1e3, 1e3)
            return p_new, P_new

        particles, P = tf.cond(do_predict, _predict, lambda: (particles, P))

        # ----- LEDH flow (matrix form; identical to parent) -----
        eta = tf.identity(particles)
        log_det_jac = tf.zeros([N])
        lam_cum = 0.0

        innov_const = tf.clip_by_value(z_t - mu_z, -100.0, 100.0)
        innov_const_batch = tf.tile(innov_const[tf.newaxis, :], [N, 1])

        R_inv_scalar = 1.0 / R_scalar

        P = tf.where(tf.math.is_finite(P), P, tf.zeros_like(P))
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))

        for j in range(self.n_lambda):
            eps_j = self.epsilons[j]
            lam_k = lam_cum + eps_j / 2.0
            lam_cum += eps_j

            S = lam_k * P + R_scalar * I_d_batch
            S_ridge = S + tf.constant(1e-3, tf.float32) * I_d_batch
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
                exp_Aeps = expm_2x2_batch(A_eps)                # XLA-safe closed form
            elif d == 1:
                exp_Aeps = tf.exp(A_eps)                        # XLA-safe for 1x1 matrix
            else:
                exp_Aeps = expm_pade_batch(                     # XLA-safe fixed-s Pade
                    A_eps, s=pade_scaling_for_dim(d))
            A_ridge = A + tf.constant(1e-3, tf.float32) * I_d_batch
            exp_minus_I_b = tf.einsum("nij,nj->ni",
                                        exp_Aeps - I_d_batch, b_vec)
            phi_Ab = tf.linalg.solve(A_ridge,
                                      exp_minus_I_b[..., tf.newaxis])[..., 0]

            particles_new = tf.einsum("nij,nj->ni", exp_Aeps, particles) + phi_Ab
            eta_new       = tf.einsum("nij,nj->ni", exp_Aeps, eta) + phi_Ab
            particles = tf.clip_by_value(_safe_nd(particles_new), -50.0, 50.0)
            eta = tf.clip_by_value(_safe_nd(eta_new), -50.0, 50.0)

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

        # ----- Neural-OT resample (THE ONLY DIFFERENCE from parent) -----
        # Same normalise -> map -> de-normalise -> uniform-weight pattern
        # as the univariate NN-OT filter, but on (N, d) particles.
        p_mean = tf.reduce_mean(particles, axis=0, keepdims=True)   # (1, d)
        p_std = tf.math.reduce_std(particles, axis=0, keepdims=True) + _EPS
        p_norm = (particles - p_mean) / p_std                        # (N, d)

        w = tf.nn.softmax(log_w_t, axis=0)                           # (N,)
        ess = _compute_ess(w)

        # CRITICAL: use mu / Phi / Sigma_eta from THIS function's args,
        # NOT from self._current_* attributes. The self.* attributes
        # become frozen constants when @tf.function(jit_compile=True)
        # traces this body for the first time, causing every subsequent
        # HMC call (with a different theta) to use the FIRST theta's
        # context vector. Phase-16-bisect bug, fixed.
        sigma_eta_sq_diag = tf.linalg.diag_part(Sigma_eta)
        ctx = build_svssm_multi_context_scalars(
            mu=mu,                      # function arg, retraced per call
            Phi=Phi,                    # function arg
            sigma_eta_sq_diag=sigma_eta_sq_diag,
            t=tf.constant(0.0, tf.float32),
            z_t=z_t,
            ess=ess,                    # tf.Tensor; no .numpy()
            epsilon=tf.constant(self.sinkhorn_epsilon, tf.float32),
            T_max=tf.constant(self._current_T_max, tf.float32),  # python float -> constant; same across HMC calls (y_obs len doesn't change)
            d=d,
        )
        ctx = tf.stop_gradient(ctx)

        # Add a batch dim of 1 for the operator (the DeepONet expects
        # batched (B, N, d) particles + (B, N) weights + (B, ctx)).
        p_resampled_norm = self.neural_ot_model(
            p_norm[tf.newaxis, ...],     # (1, N, d)
            w[tf.newaxis, :],            # (1, N)
            ctx[tf.newaxis, :],          # (1, ctx_dim)
        )                                 # (1, N, d)
        p_resampled_norm = tf.cast(tf.math.real(p_resampled_norm),
                                     tf.float32)[0]   # (N, d)
        particles = p_resampled_norm * p_std + p_mean

        P_mean = tf.reduce_mean(P, axis=0, keepdims=True)
        P = tf.tile(P_mean, [N, 1, 1])
        log_w = tf.fill([N], -tf.math.log(N_f))

        return particles, P, log_w, log_ev
