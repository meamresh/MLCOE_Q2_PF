"""
SVSSM differentiable LEDH filter with **neural OT** resampling.

Drop-in replacement for
:class:`~src.filters.bonus.differentiable_ledh_svssm.DifferentiableLEDHLogLikelihoodSVSSM`
that substitutes every Sinkhorn resampling call with a single forward pass
through a pre-trained, parameter-conditioned mGradNet (or DeepONet /
Hyper-DeepONet) neural operator.

Pipeline
~~~~~~~~
- The state-process equations, log-squared observation transform, LEDH flow,
  weight increment, and stationary initialisation are *identical* to
  :class:`DifferentiableLEDHLogLikelihoodSVSSM`.
- The OT resampling step (``_ot_resample_1d``) is replaced by
  ``_neural_ot_resample_1d``, which normalises particles, builds a 7-D
  context vector, calls the neural OT model, and de-normalises.

Context vector (passed to the network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
7 scalars in the SAME unconstrained parameterisation HMC samples in:

    ctx = [ mu, phi_raw=atanh(phi), log_sigma_eta_sq,
            t / T_max, z_t, ESS, epsilon ]

This matches what HMC sees, so the network's training distribution can
cover the HMC posterior support uniformly.

JIT / efficiency note
~~~~~~~~~~~~~~~~~~~~~
The keras model call is kept eager (mirrors the
:class:`~src.filters.bonus.differentiable_ledh_neural_ot.DifferentiableLEDHNeuralOT`
design choice; XLA-compiling a Keras forward inside the per-step kernel
is fragile in TF 2.16). The per-step LEDH kernel itself is still
``@tf.function(jit_compile=True)`` for the predict + flow + weight ops;
the eager NN call replaces the Sinkhorn block at the resampling boundary.
The trade-off: we lose Sinkhorn's O(N^2 K) inner cost (good); we gain a
small Python-dispatch tax around the keras call (small). Net wall-time
win grows with N (matches the §6 cost-decomposition crossover).

Usage
~~~~~
::

    from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
    from src.filters.bonus.differentiable_ledh_neural_ot_svssm import (
        DifferentiableLEDHNeuralOTSVSSM,
        build_svssm_context_scalars,
    )

    model = ConditionalMGradNet(num_particles=64, n_scalar_ctx=7, ...)
    # ... train model (Phase 2) ...

    ll = DifferentiableLEDHNeuralOTSVSSM(
        neural_ot_model=model,
        num_particles=64,
        sinkhorn_epsilon=1.0,   # must match training epsilon
    )
    log_p_z = ll(mu, phi, sigma_eta_sq, y_obs)
"""

from __future__ import annotations

import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    _safe_scalar,
    _as_real_log_scalar,
    LOG_CHI2_MEAN,
    LOG_CHI2_VAR,
    _EPS,
    _CLAMP,
)

# Module-level constants matching DifferentiableLEDHLogLikelihoodSVSSM.
_SVSSM_CTX_DIM = 7


def build_svssm_context_scalars(
    mu: tf.Tensor,
    phi: tf.Tensor,
    sigma_eta_sq: tf.Tensor,
    t: float,
    z_t: float,
    ess: float,
    epsilon: float,
    T_max: float = 50.0,
) -> tf.Tensor:
    """Build the 7-D scalar context vector for the SVSSM neural operator.

    Returns shape ``(7,)`` in HMC's unconstrained parameterisation::

        [ mu,
          phi_raw = atanh(phi),
          log_sigma_eta_sq,
          t / T_max,
          z_t,
          ESS,
          epsilon ]
    """
    phi_clipped = tf.clip_by_value(tf.cast(phi, tf.float32), -0.9999, 0.9999)
    phi_raw = tf.atanh(phi_clipped)
    log_sigma_eta_sq = tf.math.log(tf.maximum(tf.cast(sigma_eta_sq, tf.float32), _EPS))
    return tf.stack([
        tf.cast(mu, tf.float32),
        phi_raw,
        log_sigma_eta_sq,
        tf.cast(t / T_max, tf.float32),
        tf.cast(z_t, tf.float32),
        tf.cast(ess, tf.float32),
        tf.cast(epsilon, tf.float32),
    ])


def _compute_ess(weights: tf.Tensor) -> tf.Tensor:
    """ESS = 1 / sum(w_i^2). `weights` must be normalised."""
    return 1.0 / tf.reduce_sum(weights ** 2)


class DifferentiableLEDHNeuralOTSVSSM:
    """SVSSM LEDH log-likelihood with neural OT resampling.

    Identical interface to
    :class:`~src.filters.bonus.differentiable_ledh_svssm.DifferentiableLEDHLogLikelihoodSVSSM`
    but replaces every Sinkhorn call with the pre-trained ``neural_ot_model``.

    Parameters
    ----------
    neural_ot_model
        Pre-trained mGradNet / DeepONet / Hyper-DeepONet conforming to the
        ``model(particles_1d_normalised, weights, ctx_7)`` interface.
    num_particles : int
        Number of particles N.
    n_lambda : int
        LEDH flow pseudo-time substeps. Default 10 (matches the
        exp-integrator recommendation from §6).
    sinkhorn_epsilon : float
        Must match the epsilon used during neural OT training (the
        network's amortisation target was the Sinkhorn map at this eps).
    resample_threshold : float
        ESS / N threshold. (Currently UNUSED; resampling is unconditional
        every step, matching DifferentiableLEDHLogLikelihoodSVSSM.)
    grad_window : int
        Insert tf.stop_gradient on the particle, P, log_w state every
        `grad_window` timesteps, to truncate the backward chain.
    jit_compile : bool
        If True, compile the LEDH per-timestep block with XLA. The
        Keras neural-OT call remains eager regardless (TF 2.16 keras +
        XLA inter-op is fragile).
    integrator : {"exp", "euler"}
        LEDH-substep discretisation, same semantics as the SVSSM filter.
    log_y_sq_offset : float
        Delta inside log(y^2 + delta) (boundary observation transform).
    init_type : {"stationary", "fixed_mu", "diffuse"}
        Initial-state choice; same semantics as the SVSSM filter
        (see §7.4 of the writeup).
    diffuse_var : float
        Variance for init_type="diffuse".
    """

    LOG_CHI2_MEAN = LOG_CHI2_MEAN
    LOG_CHI2_VAR = LOG_CHI2_VAR

    def __init__(
        self,
        neural_ot_model,
        num_particles: int = 64,
        n_lambda: int = 10,
        sinkhorn_epsilon: float = 1.0,
        resample_threshold: float = 0.5,
        grad_window: int = 4,
        jit_compile: bool = False,
        integrator: str = "exp",
        log_y_sq_offset: float = 1e-8,
        init_type: str = "stationary",
        diffuse_var: float = 100.0,
    ):
        if integrator not in {"euler", "exp"}:
            raise ValueError(f"integrator must be 'euler' or 'exp', got {integrator!r}")
        if init_type not in {"stationary", "fixed_mu", "diffuse"}:
            raise ValueError(
                f"init_type must be 'stationary', 'fixed_mu', or 'diffuse', "
                f"got {init_type!r}"
            )
        self.neural_ot_model = neural_ot_model
        self.num_particles = int(num_particles)
        self.n_lambda = int(n_lambda)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.resample_threshold = float(resample_threshold)
        self.grad_window = int(grad_window)
        self.jit_compile = bool(jit_compile)
        self.integrator = integrator
        self.log_y_sq_offset = float(log_y_sq_offset)
        self.init_type = init_type
        self.diffuse_var = float(diffuse_var)

        # Geometric pseudo-time sizes (same recipe as SVSSM filter).
        q = 1.2
        self.epsilon_1 = (1.0 - q) / (1.0 - q ** self.n_lambda)
        self.epsilons = [self.epsilon_1 * q ** j for j in range(self.n_lambda)]

        # State holders the per-step routine reads back during a __call__.
        self._current_theta_raw: tf.Tensor | None = None
        self._current_T_max: float = 50.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        mu: tf.Tensor,
        phi: tf.Tensor,
        sigma_eta_sq: tf.Tensor,
        y_obs: tf.Tensor,
    ) -> tf.Tensor:
        """Return differentiable log p_hat(z | mu, phi, sigma_eta_sq).

        Same boundary transform as :class:`DifferentiableLEDHLogLikelihoodSVSSM`:
        observations enter as ``y_t``, are transformed to
        ``z_t = log(y_t^2 + delta)`` once at the boundary, and the filter
        operates on z. The y -> z Jacobian is omitted (theta-independent,
        cancels in the HMC acceptance ratio).
        """
        y_obs = tf.reshape(tf.cast(y_obs, tf.float32), [-1])
        z_obs = tf.math.log(tf.square(y_obs) + tf.constant(self.log_y_sq_offset, tf.float32))

        mu_t = tf.cast(mu, tf.float32)
        phi_t = tf.cast(phi, tf.float32)
        sigma_eta_sq_t = tf.maximum(tf.cast(sigma_eta_sq, tf.float32), _EPS)

        # Store theta in HMC's unconstrained representation for the
        # context-building inside the per-step routine.
        phi_clipped = tf.clip_by_value(phi_t, -0.9999, 0.9999)
        self._current_theta_raw = tf.stack([
            mu_t, tf.atanh(phi_clipped), tf.math.log(sigma_eta_sq_t),
        ])
        self._current_T_max = float(z_obs.shape[0])

        # Initial-state distribution (mirrors SVSSM filter exactly).
        if self.init_type == "stationary":
            one_minus_phi_sq = tf.maximum(1.0 - phi_t ** 2, _EPS)
            init_var = sigma_eta_sq_t / one_minus_phi_sq
            init_mean = mu_t
        elif self.init_type == "fixed_mu":
            init_var = sigma_eta_sq_t
            init_mean = mu_t
        else:
            init_var = tf.constant(self.diffuse_var, tf.float32)
            init_mean = tf.constant(0.0, tf.float32)

        log_ml = self._run_1d(mu_t, phi_t, sigma_eta_sq_t, z_obs, init_var, init_mean)
        return _as_real_log_scalar(log_ml)

    # ------------------------------------------------------------------
    # 1-D outer loop (eager: lets us call the keras model at resample)
    # ------------------------------------------------------------------

    def _run_1d(
        self,
        mu: tf.Tensor,
        phi: tf.Tensor,
        sigma_eta_sq: tf.Tensor,
        z_obs: tf.Tensor,
        init_var: tf.Tensor,
        init_mean: tf.Tensor,
    ):
        """Eager outer loop with neural-OT resampling at each timestep."""
        N = self.num_particles
        T = int(z_obs.shape[0])
        N_f = tf.cast(N, tf.float32)
        R_val = tf.constant(self.LOG_CHI2_VAR, tf.float32)
        R_inv_val = 1.0 / R_val
        log_det_R = tf.math.log(R_val)
        log_norm_const = -0.5 * (log_det_R + tf.math.log(2.0 * 3.141592653589793))
        mu_z = tf.constant(self.LOG_CHI2_MEAN, tf.float32)

        particles = init_mean + tf.random.normal([N]) * tf.sqrt(init_var)
        P = tf.fill([N], init_var)
        log_w = tf.fill([N], -tf.math.log(N_f))
        log_ml = tf.constant(0.0)

        use_exp = self.integrator == "exp"

        for t_int in range(1, T + 1):
            if self.grad_window > 0 and t_int > 1 and (t_int - 1) % self.grad_window == 0:
                particles = tf.stop_gradient(particles)
                P = tf.stop_gradient(P)
                log_w = tf.stop_gradient(log_w)

            t_f = tf.cast(t_int, tf.float32)
            z_val = z_obs[t_int - 1]

            # --- Predict (do_predict only for t >= 2) ---
            if t_int >= 2:
                x_det = _safe_scalar(mu + phi * (particles - mu))
                particles = x_det + tf.random.normal([N]) * tf.sqrt(sigma_eta_sq)
                particles = _safe_scalar(particles)
                P = tf.clip_by_value(phi ** 2 * P + sigma_eta_sq, _EPS, _CLAMP)

            # --- LEDH flow (SVSSM simplifications: H=1, e=mu_z) ---
            eta = tf.identity(particles)
            log_det_jac = tf.zeros([N])
            lam_cum = 0.0
            for j in range(self.n_lambda):
                eps_j = self.epsilons[j]
                lam_k = lam_cum + eps_j / 2.0
                lam_cum += eps_j

                # H = 1, e = mu_z (constants under SVSSM transform).
                S = tf.maximum(lam_k * P + R_val, _EPS)
                A = tf.clip_by_value(-0.5 * P / S, -10.0, 0.0)
                lam_A = lam_k * A
                I_lam_A = 1.0 + lam_A
                I_2lam_A = 1.0 + 2.0 * lam_A
                innov = tf.clip_by_value(z_val - mu_z, -100.0, 100.0)
                b_vec = I_2lam_A * (I_lam_A * P * R_inv_val * innov + A * eta)
                b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)

                if use_exp:
                    Az = A * eps_j
                    exp_Az = tf.exp(Az)
                    expm1_Az = tf.math.expm1(Az)
                    A_safe = tf.where(tf.abs(A) > 1e-8, A, tf.fill(tf.shape(A), 1e-8))
                    phi_A = expm1_Az / A_safe
                    particles = _safe_scalar(particles * exp_Az + b_vec * phi_A)
                    eta = _safe_scalar(eta * exp_Az + b_vec * phi_A)
                    log_det_jac = log_det_jac + Az
                else:
                    vel = tf.clip_by_value(A * particles + b_vec, -50.0, 50.0)
                    particles = _safe_scalar(particles + eps_j * vel)
                    vel_eta = tf.clip_by_value(A * eta + b_vec, -50.0, 50.0)
                    eta = _safe_scalar(eta + eps_j * vel_eta)
                    J_val = tf.maximum(tf.abs(1.0 + eps_j * A), _EPS)
                    log_det_jac = log_det_jac + tf.math.log(J_val)

            # --- Weight increment (Gaussian quasi-likelihood) ---
            resid = z_val - (particles + mu_z)
            log_lik = -0.5 * R_inv_val * resid ** 2 + log_norm_const
            log_lik = _safe_scalar(log_lik)
            log_w_incr = log_lik + log_det_jac
            log_w_incr = tf.where(
                tf.math.is_finite(log_w_incr),
                log_w_incr,
                tf.constant(-100.0, dtype=tf.float32),
            )
            log_w_t = log_w + log_w_incr
            log_ev = tf.reduce_logsumexp(log_w_t)
            log_w_t = log_w_t - log_ev

            # --- Neural OT resampling (replaces Sinkhorn) ---
            particles, log_w = self._neural_ot_resample_1d(
                particles, log_w_t, t_f, z_val, mu, phi, sigma_eta_sq,
            )
            P_mean = tf.reduce_mean(P)
            P = tf.fill([N], P_mean)

            log_ml = log_ml + tf.where(
                tf.math.is_finite(log_ev), log_ev, tf.constant(-10.0)
            )

        return log_ml

    # ------------------------------------------------------------------
    # Neural OT resampling (the only methodological change vs the SVSSM filter)
    # ------------------------------------------------------------------

    def _neural_ot_resample_1d(
        self,
        particles_1d: tf.Tensor,
        log_w: tf.Tensor,
        t_f: tf.Tensor,
        z_t: tf.Tensor,
        mu: tf.Tensor,
        phi: tf.Tensor,
        sigma_eta_sq: tf.Tensor,
    ):
        """Neural OT resampling for 1-D particles (N,) -> (N,).

        Mirrors the existing
        :meth:`DifferentiableLEDHLogLikelihoodSVSSM._ot_resample_1d`
        boundary contract (normalise -> map -> de-normalise -> uniform weights)
        but uses one neural-net forward pass in place of Sinkhorn iterations.
        """
        p2d = particles_1d[:, tf.newaxis]
        p_mean = tf.reduce_mean(p2d, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(p2d, axis=0, keepdims=True) + _EPS
        p_norm = (p2d - p_mean) / p_std

        w = tf.nn.softmax(log_w, axis=0)
        ess = _compute_ess(w)

        ctx = build_svssm_context_scalars(
            mu=mu, phi=phi, sigma_eta_sq=sigma_eta_sq,
            t=float(t_f.numpy()) if hasattr(t_f, "numpy") else float(t_f),
            z_t=float(z_t.numpy()) if hasattr(z_t, "numpy") else float(z_t),
            ess=float(ess.numpy()) if hasattr(ess, "numpy") else float(ess),
            epsilon=self.sinkhorn_epsilon,
            T_max=self._current_T_max,
        )
        ctx = tf.stop_gradient(ctx)

        # Single forward pass through the trained neural operator.
        p_resampled_norm = self.neural_ot_model(p_norm[:, 0], w, ctx)
        p_resampled_norm = tf.cast(tf.math.real(p_resampled_norm), tf.float32)

        p_resampled = p_resampled_norm[:, tf.newaxis] * p_std + p_mean
        N = self.num_particles
        w_uniform = tf.fill([N], 1.0 / tf.cast(N, tf.float32))
        return p_resampled[:, 0], tf.math.log(w_uniform + 1e-20)
