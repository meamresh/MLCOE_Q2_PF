"""
Differentiable LEDH particle filter with **neural OT** resampling.

Drop-in replacement for :class:`DifferentiableLEDHLogLikelihood` that
substitutes every Sinkhorn resampling call with a single forward pass
through a pre-trained conditional mGradNet.

The filter logic (predict -> LEDH flow -> weight) is *identical* to the
original in :mod:`src.filters.bonus.differentiable_ledh`; only the
``_ot_resample_*`` methods are swapped.

Usage
-----
::

    from src.filters.bonus.neural_ot_resampling import NeuralOTTrainer
    from src.filters.bonus.differentiable_ledh_neural_ot import (
        DifferentiableLEDHNeuralOT,
    )

    # 1. Train the neural OT model (one-time cost)
    trainer = NeuralOTTrainer(state_dim=1, num_particles=50)
    ot_model, diag = trainer.train(y_obs, sinkhorn_epsilon=2.0)

    # 2. Build the filter with the trained model
    filt = DifferentiableLEDHNeuralOT(
        neural_ot_model=ot_model,
        num_particles=50,
        sinkhorn_epsilon=2.0,   # must match training epsilon
    )

    # 3. Use exactly like DifferentiableLEDHLogLikelihood
    log_ml = filt(ssm, y_obs)
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.differentiable_ledh import (
    _safe_scalar,
    _as_real_log_scalar,
    _slogdet_logabsdet_real_batched,
    _CLAMP,
    _EPS,
    _SLOGDET_RIDGE,
)
from src.filters.pfpf_filter import (
    _compute_flow_matrix_A,
    _compute_flow_vector_b_batch,
    _compute_mahalanobis_batch,
    _compute_gaussian_log_prob,
)
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.neural_ot_resampling import (
    build_context_scalars,
    _compute_ess,
)

tfd = tfp.distributions


class DifferentiableLEDHNeuralOT:
    """Differentiable LEDH log-likelihood with neural OT resampling.

    Identical interface to
    :class:`~src.filters.bonus.differentiable_ledh.DifferentiableLEDHLogLikelihood`
    but replaces every Sinkhorn call with the pre-trained ``neural_ot_model``.

    Parameters
    ----------
    neural_ot_model : ConditionalMGradNet
        Pre-trained conditional mGradNet (from :class:`NeuralOTTrainer`).
    num_particles : int
    n_lambda : int
        LEDH flow pseudo-time steps.
    sinkhorn_epsilon : float
        Must match the epsilon used during neural OT training.
    resample_threshold : float
        ESS / N threshold (unused — kept for API compatibility).
    grad_window : int
        ``tf.stop_gradient`` every ``grad_window`` timesteps.
    jit_compile : bool
        Whether to XLA-compile the per-timestep kernel.
        Note: the neural OT forward pass is always eager because
        keras model calls inside XLA can be problematic. Set to False
        for safety.
    """

    def __init__(
        self,
        neural_ot_model: ConditionalMGradNet,
        num_particles: int = 50,
        n_lambda: int = 5,
        sinkhorn_epsilon: float = 2.0,
        sinkhorn_iters: int = 20,      # ignored (kept for API compat)
        resample_threshold: float = 0.5,
        grad_window: int = 1,
        jit_compile: bool = False,
    ):
        """Initialise with a pre-trained neural OT model; see class docstring."""
        self.neural_ot_model = neural_ot_model
        self.num_particles = num_particles
        self.n_lambda = n_lambda
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.resample_threshold = resample_threshold
        self.grad_window = grad_window
        self.jit_compile = jit_compile

        # Geometric pseudo-time sizes (same as original LEDH).
        q = 1.2
        self.epsilon_1 = (1.0 - q) / (1.0 - q ** n_lambda)
        self.epsilons = [self.epsilon_1 * q ** j for j in range(n_lambda)]

        # Will be set per-call so the 1-D kernel can access them.
        self._current_theta: tf.Tensor = tf.zeros([2])
        self._current_T_max: float = 50.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        ssm,
        y_obs: tf.Tensor,
        *,
        theta: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Run the filter and return scalar log marginal likelihood.

        Parameters
        ----------
        ssm
            State-space model (same interface as
            :class:`~src.models.ssm_katigawa.PMCMCNonlinearSSM`).
        y_obs
            Observations, shape ``(T,)`` or ``(T, obs_dim)``.
        theta
            Current model parameters ``[log_sv2, log_sw2]`` for the
            neural OT context.  If *None*, inferred from ``ssm.Q / ssm.R``.
        """
        if len(y_obs.shape) == 1:
            y_obs = y_obs[:, tf.newaxis]

        Q_val = tf.maximum(ssm.Q[0, 0], _EPS)
        R_val = tf.maximum(ssm.R[0, 0], _EPS)
        init_var = tf.cast(
            getattr(ssm, "initial_var", float(Q_val)), tf.float32,
        )

        if theta is not None:
            self._current_theta = theta
        else:
            self._current_theta = tf.stack([
                tf.math.log(Q_val), tf.math.log(R_val),
            ])
        self._current_T_max = float(y_obs.shape[0])

        if ssm.state_dim == 1:
            return _as_real_log_scalar(
                self._run_1d(Q_val, R_val, y_obs, init_var)
            )
        return _as_real_log_scalar(self._run_nd(ssm, y_obs))

    # ------------------------------------------------------------------
    # 1-D path (eager, with neural OT resampling)
    # ------------------------------------------------------------------

    def _run_1d(self, Q_val, R_val, y_obs, init_var):
        """Scalar Kitagawa path: full filter loop with neural OT resampling."""
        N = self.num_particles
        T = int(y_obs.shape[0])
        N_f = tf.cast(N, tf.float32)
        R_inv_val = 1.0 / R_val
        log_det_R = tf.math.log(R_val)
        log_norm_const = -0.5 * (
            log_det_R + tf.math.log(2.0 * 3.141592653589793)
        )

        particles = tf.random.normal([N]) * tf.sqrt(init_var)
        P = tf.fill([N], init_var)
        log_w = tf.fill([N], -tf.math.log(N_f))
        log_ml = tf.constant(0.0)

        for t_int in range(1, T + 1):
            if (self.grad_window > 0 and t_int > 1
                    and (t_int - 1) % self.grad_window == 0):
                particles = tf.stop_gradient(particles)
                P = tf.stop_gradient(P)
                log_w = tf.stop_gradient(log_w)

            t_f = tf.cast(t_int, tf.float32)
            z_val = y_obs[t_int - 1, 0]

            # ---- Predict ----
            if t_int >= 2:
                x_det = _safe_scalar(
                    0.5 * particles
                    + 25.0 * particles / (1.0 + particles ** 2)
                    + 8.0 * tf.cos(1.2 * t_f)
                )
                particles = x_det + tf.random.normal([N]) * tf.sqrt(Q_val)
                particles = _safe_scalar(particles)
                F = _safe_scalar(
                    0.5 + 25.0 * (1.0 - particles ** 2)
                    / tf.maximum((1.0 + particles ** 2) ** 2, _EPS)
                )
                P = tf.clip_by_value(F ** 2 * P + Q_val, _EPS, _CLAMP)

            # ---- LEDH flow ----
            eta = tf.identity(particles)
            log_det_jac = tf.zeros([N])
            lam_cum = 0.0

            for j in range(self.n_lambda):
                eps_j = self.epsilons[j]
                lam_k = lam_cum + eps_j / 2.0
                lam_cum += eps_j

                H = _safe_scalar(eta / 10.0)
                h_eta = _safe_scalar(eta ** 2 / 20.0)
                e = h_eta - H * eta

                S = tf.maximum(lam_k * H ** 2 * P + R_val, _EPS)
                A = tf.clip_by_value(-0.5 * P * H ** 2 / S, -10.0, 0.0)

                I_lam_A = 1.0 + lam_k * A
                I_2lam_A = 1.0 + 2.0 * lam_k * A

                innov = tf.clip_by_value(z_val - e, -100.0, 100.0)
                b_vec = I_2lam_A * (
                    (I_lam_A * P * H * R_inv_val * innov) + A * eta
                )
                b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)

                vel = tf.clip_by_value(
                    A * particles + b_vec, -50.0, 50.0,
                )
                particles = _safe_scalar(particles + eps_j * vel)

                vel_eta = tf.clip_by_value(
                    A * eta + b_vec, -50.0, 50.0,
                )
                eta = _safe_scalar(eta + eps_j * vel_eta)

                J_val = tf.maximum(tf.abs(1.0 + eps_j * A), _EPS)
                log_det_jac = log_det_jac + tf.math.log(J_val)

            # ---- Weights ----
            y_pred = _safe_scalar(particles ** 2 / 20.0)
            resid = z_val - y_pred
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
            log_w_t = log_w_t - tf.reduce_logsumexp(log_w_t)

            # ---- Neural OT resampling (replaces Sinkhorn) ----
            particles, log_w_t = self._neural_ot_resample_1d(
                particles, log_w_t, t_f, z_val,
            )
            P_mean = tf.reduce_mean(P)
            P = tf.fill([N], P_mean)

            log_w = log_w_t
            log_ml = log_ml + tf.where(
                tf.math.is_finite(log_ev), log_ev, tf.constant(-10.0),
            )

        return log_ml

    # ------------------------------------------------------------------
    # N-D path
    # ------------------------------------------------------------------

    def _run_nd(self, ssm, y_obs):
        """General N-D path (same LEDH flow, neural OT resampling)."""
        N = self.num_particles
        T = int(y_obs.shape[0])
        state_dim = ssm.state_dim
        N_f = tf.cast(N, tf.float32)

        if len(y_obs.shape) == 1:
            y_obs = y_obs[:, tf.newaxis]
        obs_dim = int(y_obs.shape[1]) if y_obs.shape[1] is not None else 1

        Q_scale = tf.sqrt(tf.linalg.diag_part(ssm.Q) + _EPS)
        particles = tf.random.normal([N, state_dim]) * Q_scale[tf.newaxis, :]
        P_preds = tf.tile(ssm.Q[tf.newaxis, :, :], [N, 1, 1])

        R = ssm.full_measurement_cov(1)
        eye_obs = tf.eye(obs_dim, dtype=R.dtype)
        R_reg = R + eye_obs * (1e-4 + _SLOGDET_RIDGE)
        R_inv = tf.linalg.inv(R_reg)
        _, log_det_R = tf.linalg.slogdet(R_reg)
        log_det_R = tf.cast(tf.math.real(log_det_R), tf.float32)
        obs_dim_f = tf.cast(obs_dim, tf.float32)

        log_w = tf.fill([N], -tf.math.log(N_f))
        log_ml = tf.constant(0.0)
        I_sd = tf.eye(state_dim)

        for t_int in range(1, T + 1):
            if (self.grad_window > 0 and t_int > 1
                    and (t_int - 1) % self.grad_window == 0):
                particles = tf.stop_gradient(particles)
                P_preds = tf.stop_gradient(P_preds)
                log_w = tf.stop_gradient(log_w)

            t_f = tf.cast(t_int, tf.float32)
            control = tf.fill([N, 1], t_f)
            preds = ssm.motion_model(particles, control)
            L_Q = tf.linalg.cholesky(ssm.Q + I_sd * _EPS)
            noise = tf.random.normal([N, state_dim])
            particles = preds + noise @ tf.transpose(L_Q)

            F_batch = ssm.motion_jacobian(particles, control)
            F_T = tf.transpose(F_batch, [0, 2, 1])
            P_preds = (
                tf.einsum("bij,bjk,bkl->bil", F_batch, P_preds, F_T)
                + ssm.Q[tf.newaxis, :, :]
            )
            P_preds = tf.clip_by_value(P_preds, _EPS, _CLAMP)

            eta_bars = tf.identity(particles)
            log_det_jac = tf.zeros([N])
            lam_cum = 0.0
            z_t = y_obs[t_int - 1]
            z_t_batch = tf.broadcast_to(z_t[tf.newaxis, :], [N, obs_dim])

            for j in range(self.n_lambda):
                eps_j = self.epsilons[j]
                lam_k = lam_cum + eps_j / 2.0
                lam_cum += eps_j

                H_batch = tf.reshape(
                    ssm.measurement_jacobian(eta_bars, None),
                    [N, obs_dim, state_dim],
                )
                H_T = tf.transpose(H_batch, [0, 2, 1])
                h_eta = tf.reshape(
                    ssm.measurement_model(eta_bars, None), [N, obs_dim],
                )
                H_eta = tf.squeeze(
                    tf.matmul(H_batch, eta_bars[:, :, tf.newaxis]), axis=2,
                )
                e_lambda = h_eta - H_eta

                HPH = tf.einsum("bij,bjk,bkl->bil", H_batch, P_preds, H_T)
                S_lambda = (
                    lam_k * HPH
                    + R_reg[tf.newaxis, :, :]
                    + tf.eye(obs_dim)[tf.newaxis, :, :] * 1e-4
                )
                S_lambda = tf.maximum(S_lambda, _EPS)
                S_lambda = (
                    S_lambda
                    + tf.eye(obs_dim, dtype=S_lambda.dtype)[tf.newaxis, :, :]
                    * tf.cast(_SLOGDET_RIDGE, S_lambda.dtype)
                )
                S_inv = tf.linalg.inv(S_lambda)

                A_batch = _compute_flow_matrix_A(
                    P_preds, H_T, S_inv, H_batch,
                )
                A_batch = tf.clip_by_value(A_batch, -10.0, 0.0)

                I_batch = tf.tile(I_sd[tf.newaxis, :, :], [N, 1, 1])
                I_lam_A = I_batch + lam_k * A_batch
                I_2lam_A = I_batch + 2.0 * lam_k * A_batch
                z_minus_e = tf.clip_by_value(
                    z_t_batch - e_lambda, -100.0, 100.0,
                )

                b_batch = tf.clip_by_value(
                    _compute_flow_vector_b_batch(
                        I_lam_A, I_2lam_A, P_preds, H_T,
                        R_inv, z_minus_e, A_batch, eta_bars,
                    ),
                    -100.0, 100.0,
                )

                vel = tf.clip_by_value(
                    tf.einsum("bij,bj->bi", A_batch, particles) + b_batch,
                    -50.0, 50.0,
                )
                particles = particles + eps_j * vel

                vel_bar = tf.clip_by_value(
                    tf.einsum("bij,bj->bi", A_batch, eta_bars) + b_batch,
                    -50.0, 50.0,
                )
                eta_bars = eta_bars + eps_j * vel_bar

                J_step = (
                    I_batch + eps_j * A_batch
                    + I_sd[tf.newaxis, :, :] * _EPS
                )
                log_dets = _slogdet_logabsdet_real_batched(J_step, I_batch)
                log_det_jac = log_det_jac + log_dets

            y_pred = tf.reshape(
                ssm.measurement_model(particles, None), [N, obs_dim],
            )
            innov = z_t_batch - y_pred
            mahal = _compute_mahalanobis_batch(innov, R_inv)
            log_lik = _compute_gaussian_log_prob(mahal, log_det_R, obs_dim_f)

            log_w_incr = log_lik + log_det_jac
            log_w_incr = tf.where(
                tf.math.is_finite(log_w_incr),
                log_w_incr,
                tf.constant(-100.0, dtype=tf.float32),
            )

            log_w = log_w + log_w_incr
            log_ev = tf.reduce_logsumexp(log_w)
            log_ml = log_ml + tf.where(
                tf.math.is_finite(log_ev), log_ev, tf.constant(-10.0),
            )
            log_w = log_w - tf.reduce_logsumexp(log_w)

            # ---- Neural OT resampling ----
            particles, log_w = self._neural_ot_resample_nd(
                particles, log_w, t_f, z_t[0],
            )
            P_mean = tf.reduce_mean(P_preds, axis=0, keepdims=True)
            P_preds = tf.tile(P_mean, [N, 1, 1])

        return log_ml

    # ------------------------------------------------------------------
    # Neural OT resampling (replaces Sinkhorn)
    # ------------------------------------------------------------------

    def _neural_ot_resample_1d(
        self,
        particles_1d: tf.Tensor,
        log_w: tf.Tensor,
        t_f: tf.Tensor,
        y_t: tf.Tensor,
    ):
        """Neural OT resampling for 1-D particles (N,) -> (N,)."""
        p2d = particles_1d[:, tf.newaxis]  # (N, 1)
        p_mean = tf.reduce_mean(p2d, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(p2d, axis=0, keepdims=True) + _EPS
        p_norm = (p2d - p_mean) / p_std

        w = tf.nn.softmax(log_w, axis=0)
        ess = _compute_ess(w)

        ctx = build_context_scalars(
            self._current_theta,
            float(t_f),
            float(y_t),
            float(ess),
            self.sinkhorn_epsilon,
            self._current_T_max,
        )
        ctx = tf.stop_gradient(ctx)

        # Single forward pass (no iterations!)
        p_resampled_norm = self.neural_ot_model(
            p_norm[:, 0], w, ctx,
        )  # (N,)

        # Ensure real float32
        p_resampled_norm = tf.cast(
            tf.math.real(p_resampled_norm), tf.float32,
        )

        p_resampled = p_resampled_norm[:, tf.newaxis] * p_std + p_mean
        N = self.num_particles
        w_uniform = tf.fill(
            [N], 1.0 / tf.cast(N, tf.float32),
        )
        return p_resampled[:, 0], tf.math.log(w_uniform + 1e-20)

    def _neural_ot_resample_nd(
        self,
        particles: tf.Tensor,
        log_w: tf.Tensor,
        t_f: tf.Tensor,
        y_t: tf.Tensor,
    ):
        """General N-D neural OT resampling (N, d) -> (N, d)."""
        p_mean = tf.reduce_mean(particles, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(particles, axis=0, keepdims=True) + _EPS
        p_norm = (particles - p_mean) / p_std

        w = tf.nn.softmax(log_w, axis=0)
        ess = _compute_ess(w)

        ctx = build_context_scalars(
            self._current_theta,
            float(t_f),
            float(y_t),
            float(ess),
            self.sinkhorn_epsilon,
            self._current_T_max,
        )
        ctx = tf.stop_gradient(ctx)

        p_resampled_norm = self.neural_ot_model(p_norm, w, ctx)
        p_resampled_norm = tf.cast(
            tf.math.real(p_resampled_norm), tf.float32,
        )

        p_resampled = p_resampled_norm * p_std + p_mean
        N = self.num_particles
        w_uniform = tf.fill(
            [N], 1.0 / tf.cast(N, tf.float32),
        )
        return p_resampled, w_uniform
