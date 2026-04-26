"""
Particle Flow Particle Filter with Dai & Daum Stochastic Flow (PFPF-Dai-Daum).

This filter combines:
- Particle Filter framework (sampling, resampling)
- Dai & Daum (2021) stochastic particle flow with stiffness mitigation
- Optimal homotopy schedule β*(λ) computed once per time step at reference point
- Removed stochastic diffusion Q to align with Li and Coates deterministic map

Reference:
    Dai & Daum (2021), "Stiffness Mitigation in Stochastic Particle Flow Filters"
    Li & Coates (2017), "Particle Filtering with Invertible Particle Flow"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    HAS_TQDM = False

from src.filters.spf_dai_daum import (
    DTYPE,
    REGULARIZATION,
    _tf_interp_1d,
)
from src.filters.resampling import systematic_resample, compute_ess

tfd = tfp.distributions

# =============================================================================
# Configuration (PFPF-specific: MAX_PARTICLE_DISTANCE; SPFConfig with use_hutchinson)
# =============================================================================

MAX_PARTICLE_DISTANCE = tf.constant(200.0, dtype=DTYPE)


@dataclass
class SPFConfig:
    """Configuration for Dai-Daum stochastic particle flow (PFPF variant)."""

    n_particles: int = 200
    lambda_steps: int = 200
    bvp_mesh_points: int = 50
    bvp_max_iter: int = 1000
    bvp_learning_rate: float = 1e-3
    mu: float = 0.2
    use_mu_continuation: bool = True
    mu_steps: int = 8
    verbose: bool = False
    use_hutchinson: bool = True
    hutchinson_samples: int = 5
    # Early stopping: halt a μ stage when loss hasn’t improved by more than
    # `bvp_early_stop_tol` for `bvp_early_stop_patience` consecutive checks
    # (checks happen every 50 iterations).  Set tol=0 to disable.
    bvp_early_stop_tol: float = 1e-5
    bvp_early_stop_patience: int = 3

    @property
    def dlambda(self) -> tf.Tensor:
        """Pseudo-time step size Δλ = 1 / lambda_steps."""
        return tf.constant(1.0 / self.lambda_steps, dtype=DTYPE)


def _use_hutchinson(config: SPFConfig) -> bool:
    """Compatibility: support both use_hutchinson and use_hutchinson_trace."""
    return getattr(config, "use_hutchinson", getattr(config, "use_hutchinson_trace", True))


# =============================================================================
# Dai & Daum core (kept local to avoid double regularization vs spf_dai_daum)
# =============================================================================


def compute_M(
    ssm: Any, x_ref: tf.Tensor, beta: tf.Tensor, z: tf.Tensor
) -> tf.Tensor:
    """Compute metric tensor M(β) = -∇²log π(x,λ)."""
    beta = tf.cast(beta, dtype=DTYPE)
    x_ref = tf.cast(x_ref, dtype=DTYPE)
    z = tf.cast(z, dtype=DTYPE)
    state_dim = tf.shape(x_ref)[0]
    alpha = tf.constant(1.0, dtype=DTYPE) - beta
    H0 = tf.cast(ssm.hessian_log_prior(x_ref), DTYPE)
    Hh = tf.cast(ssm.hessian_log_likelihood(x_ref, z), DTYPE)
    H = alpha * H0 + beta * Hh
    M = -H
    M_reg = M + REGULARIZATION * tf.eye(state_dim, dtype=DTYPE)
    M_reg = 0.5 * (M_reg + tf.transpose(M_reg))
    return M_reg


def cond_number(M: tf.Tensor, use_hutchinson: bool = True, n_samples: int = 5) -> tf.Tensor:
    """Nuclear-norm condition number κ*(M) = tr(M) tr(M⁻¹). Hutchinson optional."""
    M = tf.cast(M, dtype=DTYPE)
    state_dim = tf.shape(M)[0]
    M_reg = M + REGULARIZATION * tf.eye(state_dim, dtype=DTYPE)
    M_reg = 0.5 * (M_reg + tf.transpose(M_reg))
    if use_hutchinson:
        v_samples = tf.random.uniform(
            [n_samples, state_dim], minval=0, maxval=2, dtype=tf.int32
        )
        v_samples = tf.cast(v_samples * 2 - 1, dtype=DTYPE)
        Mv = tf.matmul(v_samples, M_reg, transpose_b=False)
        tr_M = tf.reduce_mean(tf.reduce_sum(v_samples * Mv, axis=1))
        M_inv_v_T = tf.linalg.solve(M_reg, tf.transpose(v_samples))
        tr_M_inv = tf.reduce_mean(tf.reduce_sum(v_samples * tf.transpose(M_inv_v_T), axis=1))
        return tr_M * tr_M_inv
    M_inv = tf.linalg.inv(M_reg)
    return tf.linalg.trace(M_reg) * tf.linalg.trace(M_inv)


@tf.function(reduce_retracing=True)
def compute_objective_J(
    beta_vals: tf.Tensor,
    beta_dot_vals: tf.Tensor,
    x_ref: tf.Tensor,
    z: tf.Tensor,
    mu: tf.Tensor,
    ssm: Any,
    use_hutchinson: bool = True,
    hutchinson_samples: int = 5,
) -> tf.Tensor:
    """Objective J(β) = ∫ [½(β')² + μ κ*(M)] dλ."""
    beta_vals = tf.cast(beta_vals, dtype=DTYPE)
    beta_dot_vals = tf.cast(beta_dot_vals, dtype=DTYPE)
    x_ref = tf.cast(x_ref, dtype=DTYPE)
    z = tf.cast(z, dtype=DTYPE)
    mu = tf.cast(mu, dtype=DTYPE)
    n_points = tf.shape(beta_vals)[0]
    kappa_values = tf.TensorArray(dtype=DTYPE, size=n_points, dynamic_size=False)
    for i in tf.range(n_points):
        M = compute_M(ssm, x_ref, beta_vals[i], z)
        kappa_values = kappa_values.write(
            i, cond_number(M, use_hutchinson=use_hutchinson, n_samples=hutchinson_samples)
        )
    kappa_vals = kappa_values.stack()
    integrand = tf.constant(0.5, dtype=DTYPE) * beta_dot_vals**2 + mu * kappa_vals
    dlambda = tf.constant(1.0, dtype=DTYPE) / tf.cast(n_points - 1, DTYPE)
    return tf.reduce_sum((integrand[:-1] + integrand[1:]) * 0.5) * dlambda


# =============================================================================
# BVP and flow (PFPF-specific: solve_optimal_beta_tf, flow with Jacobian trace)
# =============================================================================


def _beta_dot_centered(beta_full: tf.Tensor, lam_vals: tf.Tensor) -> tf.Tensor:
    """Centered-difference derivative of beta w.r.t. lambda (same length as beta_full)."""
    return tf.concat([
        (beta_full[1:2] - beta_full[0:1]) / (lam_vals[1] - lam_vals[0]),
        (beta_full[2:] - beta_full[:-2]) / (lam_vals[2:] - lam_vals[:-2]),
        (beta_full[-1:] - beta_full[-2:-1]) / (lam_vals[-1] - lam_vals[-2])
    ], axis=0)


def solve_optimal_beta_tf(
    x_ref: tf.Tensor,
    z: tf.Tensor,
    config: SPFConfig,
    ssm: Any,
    beta_interior_init: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Solve for optimal homotopy β*(λ) via gradient descent on J(β).

    Parameters
    ----------
    beta_interior_init : tf.Tensor, optional
        Warm-start for the interior β values (length bvp_mesh_points - 2).
        When reuse_beta_across_steps=True the caller can pass the previous
        solution so the optimizer converges in far fewer iterations.

    Returns beta_opt_lam, beta_opt, beta_opt_dot, J_opt.
    """
    n_mesh = config.bvp_mesh_points
    mu_target = tf.constant(config.mu, dtype=DTYPE)
    mu_target_val = float(config.mu)
    lam_vals = tf.linspace(
        tf.constant(0.0, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE), n_mesh
    )

    if beta_interior_init is not None:
        init_vals = tf.cast(beta_interior_init, DTYPE)
    else:
        init_vals = lam_vals[1:-1]
    beta_interior = tf.Variable(init_vals, dtype=DTYPE, trainable=True)

    if config.use_mu_continuation:
        mu_schedule_vals = [
            mu_target_val * (2.0 ** i) for i in range(config.mu_steps - 1, -1, -1)
        ]
    else:
        mu_schedule_vals = [mu_target_val]

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.bvp_learning_rate)
    check_every = 50          # how often to test for plateau
    tol = config.bvp_early_stop_tol
    patience = config.bvp_early_stop_patience

    for mu_idx, mu_val in enumerate(mu_schedule_vals):
        mu_val_tf = tf.constant(float(mu_val), dtype=DTYPE)

        iter_range = range(config.bvp_max_iter)
        if config.verbose:
            iter_range = tqdm(iter_range, desc=f"    Optimizing β (μ={mu_val:.3f})", leave=False)

        best_loss = float("inf")
        no_improve_count = 0
        stopped_early = False

        for iter_step in iter_range:
            with tf.GradientTape() as tape:
                beta_full = tf.concat([
                    tf.constant([0.0], dtype=DTYPE),
                    beta_interior,
                    tf.constant([1.0], dtype=DTYPE)
                ], axis=0)
                beta_dot = _beta_dot_centered(beta_full, lam_vals)
                J = compute_objective_J(
                    beta_full, beta_dot, x_ref, z, mu_val_tf, ssm,
                    use_hutchinson=config.use_hutchinson,
                    hutchinson_samples=config.hutchinson_samples
                )

            grads = tape.gradient(J, [beta_interior])
            if grads[0] is not None:
                optimizer.apply_gradients(zip(grads, [beta_interior]))
                beta_interior.assign(tf.clip_by_value(beta_interior, 0.0, 1.0))

            # Early stopping check every `check_every` iterations
            if tol > 0 and (iter_step + 1) % check_every == 0:
                current_loss = float(J.numpy())
                if best_loss - current_loss > tol:
                    best_loss = current_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        if config.verbose:
                            print(
                                f"      Early stop at iter {iter_step + 1} "
                                f"(no improvement > {tol:.2e} for "
                                f"{patience * check_every} iterations)"
                            )
                        stopped_early = True
                        break

    beta_opt = tf.concat([
        tf.constant([0.0], dtype=DTYPE),
        beta_interior,
        tf.constant([1.0], dtype=DTYPE)
    ], axis=0)
    beta_opt_dot = _beta_dot_centered(beta_opt, lam_vals)
    J_final = compute_objective_J(
        beta_opt, beta_opt_dot, x_ref, z, mu_target, ssm,
        use_hutchinson=config.use_hutchinson,
        hutchinson_samples=config.hutchinson_samples
    )

    # Return the interior values so the caller can use them as a warm-start
    return lam_vals, beta_opt, beta_opt_dot, J_final


def _compute_flow_field(
    x: tf.Tensor,
    beta: tf.Tensor,
    beta_dot: tf.Tensor,
    z: tf.Tensor,
    ssm: Any,
) -> tf.Tensor:
    """
    Compute deterministic flow field f(x, β) without trace computation.
    """
    x = tf.cast(x, dtype=DTYPE)
    beta = tf.cast(beta, dtype=DTYPE)
    beta_dot = tf.cast(beta_dot, dtype=DTYPE)
    z = tf.cast(z, dtype=DTYPE)

    state_dim = tf.shape(x)[0]
    alpha = tf.constant(1.0, dtype=DTYPE) - beta
    Q = tf.cast(ssm.Q, DTYPE)
    I = tf.eye(state_dim, dtype=DTYPE)

    g0 = tf.cast(ssm.gradient_log_prior(x), DTYPE)
    gh = tf.cast(ssm.gradient_log_likelihood(x, z), DTYPE)
    H0 = tf.cast(ssm.hessian_log_prior(x), DTYPE)
    Hh = tf.cast(ssm.hessian_log_likelihood(x, z), DTYPE)

    H = alpha * H0 + beta * Hh
    gp = alpha * g0 + beta * gh

    H_reg = H + REGULARIZATION * I
    H_reg = 0.5 * (H_reg + tf.transpose(H_reg))
    H_inv = tf.linalg.inv(H_reg)

    K2_gh = -beta_dot * tf.linalg.matvec(H_inv, gh)
    H_inv_gp = tf.linalg.matvec(H_inv, gp)
    Hh_H_inv_gp = tf.linalg.matvec(Hh, H_inv_gp)
    H_inv_Hh_H_inv_gp = tf.linalg.matvec(H_inv, Hh_H_inv_gp)
    K1_gp = 0.5 * tf.linalg.matvec(Q, gp) - 0.5 * beta_dot * H_inv_Hh_H_inv_gp

    f = K1_gp + K2_gh

    f_norm = tf.norm(f)
    f = tf.cond(
        f_norm > tf.constant(100.0, dtype=DTYPE),
        lambda: f * (tf.constant(100.0, dtype=DTYPE) / f_norm),
        lambda: f,
    )

    return f


@tf.function(reduce_retracing=True)
def f_flow_deterministic_tf(
    x: tf.Tensor,
    beta: tf.Tensor,
    beta_dot: tf.Tensor,
    z: tf.Tensor,
    ssm: Any,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Deterministic flow field for Li-Coates PFPF with Dai optimized homotopy.
    Returns f, tr_df_dx (trace of Jacobian for log-det accumulation).
    """
    f = _compute_flow_field(x, beta, beta_dot, z, ssm)

    state_dim = tf.shape(x)[0]
    n_samples = 3

    v_samples = tf.random.uniform(
        [n_samples, state_dim],
        minval=0,
        maxval=2,
        dtype=tf.int32
    )
    v_samples = tf.cast(v_samples * 2 - 1, dtype=DTYPE)

    tr_estimates = []
    for i in range(n_samples):
        v = v_samples[i]
        with tf.GradientTape() as tape:
            tape.watch(x)
            f_val = _compute_flow_field(x, beta, beta_dot, z, ssm)
            f_dot_v = tf.reduce_sum(f_val * v)
        df_dx_v = tape.gradient(f_dot_v, x)
        if df_dx_v is not None:
            tr_estimates.append(tf.reduce_sum(v * df_dx_v))

    if tr_estimates:
        tr_df_dx = tf.reduce_mean(tr_estimates)
    else:
        x_c = tf.cast(x, dtype=DTYPE)
        beta_c = tf.cast(beta, dtype=DTYPE)
        alpha = tf.constant(1.0, dtype=DTYPE) - beta_c
        Q = tf.cast(ssm.Q, DTYPE)
        H0 = tf.cast(ssm.hessian_log_prior(x_c), DTYPE)
        Hh = tf.cast(ssm.hessian_log_likelihood(x_c, z), DTYPE)
        H = alpha * H0 + beta_c * Hh
        H_reg = H + REGULARIZATION * tf.eye(state_dim, dtype=DTYPE)
        H_reg = 0.5 * (H_reg + tf.transpose(H_reg))
        H_inv = tf.linalg.inv(H_reg)
        QH = 0.5 * Q @ H_reg
        H_inv_Hh = H_inv @ Hh
        beta_dot_c = tf.cast(beta_dot, dtype=DTYPE)
        tr_df_dx = tf.linalg.trace(QH) + beta_dot_c * (
            tf.linalg.trace(H_inv_Hh @ H_inv_Hh) - tf.linalg.trace(H_inv_Hh)
        )

    return f, tr_df_dx


def propagate_particle_deterministic_tf(
    x0: tf.Tensor,
    beta_grid: tf.Tensor,
    beta_dot_grid: tf.Tensor,
    z: tf.Tensor,
    config: SPFConfig,
    ssm: Any,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Propagate one particle through DETERMINISTIC flow (Li-Coates PFPF).
    Returns x_final, log_det_J.
    """
    state_dim = tf.shape(x0)[0]
    x = tf.Variable(tf.cast(x0, dtype=DTYPE), trainable=False)
    dlambda = config.dlambda

    log_det_J = tf.constant(0.0, dtype=DTYPE)

    for k in range(config.lambda_steps):
        x.assign(tf.clip_by_value(x.value(), -MAX_PARTICLE_DISTANCE, MAX_PARTICLE_DISTANCE))

        beta_k = tf.cast(beta_grid[k], dtype=DTYPE)
        beta_dot_k = tf.cast(beta_dot_grid[k], dtype=DTYPE)

        f, tr_df_dx = f_flow_deterministic_tf(x.value(), beta_k, beta_dot_k, z, ssm)

        x_new = x + f * dlambda
        x.assign(tf.clip_by_value(x_new, -MAX_PARTICLE_DISTANCE, MAX_PARTICLE_DISTANCE))

        log_det_J = log_det_J + tr_df_dx * dlambda

    return x.value(), log_det_J


# =============================================================================
# Main Filter Class: PFPF with Dai-Daum Flow
# =============================================================================


def _apply_uniform_weights(
    num_particles: int,
    weights: tf.Variable,
    log_weights: tf.Variable,
) -> None:
    """Set weights and log_weights to uniform 1/N."""
    uniform = 1.0 / tf.cast(num_particles, tf.float32)
    w = tf.ones(num_particles, dtype=tf.float32) * uniform
    weights.assign(w)
    log_weights.assign(tf.math.log(w))


class PFPFDaiDaumFilter:
    """
    Particle Flow Particle Filter with Dai optimized homotopy (Li-Coates framework).
    Uses EKF/UKF base filter for prediction; deterministic flow with Jacobian weighting.
    """

    def __init__(
        self,
        ssm: Any,
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor,
        num_particles: int = 200,
        config: Optional[SPFConfig] = None,
        resample_threshold: float = 0.5,
        homotopy_mode: str = "optimal",
        filter_type: str = "ekf",
        ukf_alpha: float = 0.1,
        ukf_beta: float = 2.0,
        show_progress: bool = False,
        reuse_beta_across_steps: bool = False,
    ):
        """Initialise the PFPF-Dai-Daum filter.

        Parameters
        ----------
        ssm : StateSpaceModel
            State-space model providing motion/measurement models and
            gradient/Hessian methods required by the Dai-Daum flow.
        initial_state : tf.Tensor
            Prior mean, shape ``(state_dim,)``.
        initial_covariance : tf.Tensor
            Prior covariance, shape ``(state_dim, state_dim)``.
        num_particles : int
            Number of particles.
        config : SPFConfig, optional
            Stochastic particle flow configuration.
        resample_threshold : float
            ESS fraction below which systematic resampling is triggered.
        homotopy_mode : str
            ``"optimal"`` (Dai-Daum BVP) or ``"linear"`` (β = λ).
        filter_type : str
            Base filter used for prediction: ``"ekf"`` or ``"ukf"``.
        ukf_alpha, ukf_beta : float
            UKF sigma-point parameters (ignored when *filter_type* is ``"ekf"``).
        show_progress : bool
            If True, display tqdm bars during β optimisation and particle flow.
        reuse_beta_across_steps : bool
            If True, reuse the previous optimal β(λ) as a warm-start.
        """
        self.ssm = ssm
        self.num_particles = num_particles
        self.config = config or SPFConfig()
        self.resample_threshold = resample_threshold
        self.homotopy_mode = homotopy_mode
        self.filter_type = filter_type
        self.show_progress = show_progress and HAS_TQDM
        self.reuse_beta_across_steps = reuse_beta_across_steps

        self.state_dim = int(tf.shape(initial_state)[0])

        if filter_type == "ekf":
            from src.filters.ekf import ExtendedKalmanFilter
            self.base_filter = ExtendedKalmanFilter(
                ssm, initial_state, initial_covariance
            )
        elif filter_type == "ukf":
            from src.filters.ukf import UnscentedKalmanFilter
            self.base_filter = UnscentedKalmanFilter(
                ssm, initial_state, initial_covariance,
                alpha=ukf_alpha, beta=ukf_beta, kappa=0.0
            )
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        dist = tfd.MultivariateNormalTriL(
            loc=tf.cast(initial_state, tf.float32),
            scale_tril=tf.linalg.cholesky(
                tf.cast(initial_covariance, tf.float32) +
                tf.eye(self.state_dim, dtype=tf.float32) * 1e-6
            )
        )
        self.particles = tf.Variable(
            dist.sample(num_particles), trainable=False, dtype=tf.float32
        )

        self.weights = tf.Variable(
            tf.ones(num_particles, dtype=tf.float32) / num_particles,
            trainable=False
        )
        self.log_weights = tf.Variable(
            tf.math.log(self.weights), trainable=False
        )

        self.particles_prev = tf.Variable(
            tf.identity(self.particles), trainable=False
        )

        self._beta_opt_lam: Optional[tf.Tensor] = None
        self._beta_opt: Optional[tf.Tensor] = None
        self._beta_opt_dot: Optional[tf.Tensor] = None

        self.ess_before_resample = tf.Variable(
            float(num_particles), trainable=False, dtype=tf.float32
        )

        self.control_prev = None

    @property
    def state(self) -> tf.Tensor:
        """Weighted mean of the particle ensemble, shape ``(state_dim,)``."""
        return tf.reduce_sum(
            self.particles * self.weights[:, tf.newaxis], axis=0
        )

    @property
    def covariance(self) -> tf.Tensor:
        """Weighted sample covariance, shape ``(state_dim, state_dim)``."""
        x_hat = self.state
        diff = self.particles - x_hat[tf.newaxis, :]
        weighted_diff = diff * tf.sqrt(self.weights[:, tf.newaxis])
        return tf.matmul(weighted_diff, weighted_diff, transpose_a=True)

    def predict(self, control: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Propagate particles and base filter through the motion model.

        Returns the base-filter predicted state and covariance.
        """
        self.control_prev = control

        x_pred, P_pred = self.base_filter.predict(control)

        self.particles_prev.assign(tf.identity(self.particles))

        if control is not None:
            control_batch = tf.tile(
                tf.reshape(control, [1, -1]),
                [self.num_particles, 1]
            )
        else:
            control_batch = None

        particles_pred = self.ssm.motion_model(self.particles, control_batch)
        if isinstance(particles_pred, (tuple, list)):
            particles_pred = particles_pred[0]
        particles_pred = tf.reshape(particles_pred, [self.num_particles, self.state_dim])

        noise_samples = self.ssm.sample_process_noise(
            shape=self.num_particles, use_gen=False
        )
        if len(noise_samples.shape) > 2:
            noise_samples = tf.reshape(noise_samples, [self.num_particles, -1])
        elif len(noise_samples.shape) == 2 and noise_samples.shape[1] != self.state_dim:
            noise_samples = tf.reshape(noise_samples, [self.num_particles, self.state_dim])

        self.particles.assign(
            tf.cast(particles_pred + noise_samples, tf.float32)
        )

        return x_pred, P_pred

    def update(
        self,
        measurement: tf.Tensor,
        landmarks: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Incorporate a measurement via deterministic particle flow.

        Solves for the optimal homotopy β*(λ), flows each particle through
        the Dai-Daum drift field, accumulates Jacobian log-determinants for
        weight correction, and resamples when ESS drops below threshold.

        Returns ``(state, covariance, residual)``.
        """
        measurement_original = measurement
        measurement_dtype = measurement.dtype
        measurement_dtype64 = tf.cast(measurement, DTYPE)

        _, P_base, _ = self.base_filter.update(measurement_original, landmarks)
        x_prior = tf.cast(self.base_filter.state, DTYPE)
        P_prior = tf.cast(P_base, DTYPE)

        if hasattr(self.ssm, "prior_mean") and hasattr(self.ssm, "prior_cov"):
            self.ssm.prior_mean = x_prior
            self.ssm.prior_cov = P_prior

        x_ref = tf.cast(
            tf.reduce_sum(self.particles * self.weights[:, tf.newaxis], axis=0),
            DTYPE
        )

        if self.homotopy_mode == "optimal":
            if self.reuse_beta_across_steps and self._beta_opt is not None:
                beta_opt_lam = self._beta_opt_lam
                beta_opt = self._beta_opt
                beta_opt_dot = self._beta_opt_dot
            else:
                if self.show_progress:
                    print("  Solving for β*(λ)...")
                # Pass previous interior values as warm-start when available
                warm_start = (
                    self._beta_opt[1:-1] if self._beta_opt is not None else None
                )
                beta_opt_lam, beta_opt, beta_opt_dot, _ = solve_optimal_beta_tf(
                    x_ref, measurement_dtype64, self.config, self.ssm,
                    beta_interior_init=warm_start,
                )
                if self.reuse_beta_across_steps:
                    self._beta_opt_lam = beta_opt_lam
                    self._beta_opt = beta_opt
                    self._beta_opt_dot = beta_opt_dot

        lam_grid = tf.linspace(
            tf.constant(0.0, dtype=DTYPE),
            tf.constant(1.0, dtype=DTYPE),
            self.config.lambda_steps + 1,
        )

        if self.homotopy_mode == "linear":
            beta_grid = lam_grid
            beta_dot_grid = tf.ones_like(lam_grid, dtype=DTYPE)
        elif self.homotopy_mode == "optimal":
            beta_grid = _tf_interp_1d(lam_grid, beta_opt_lam, beta_opt)
            beta_dot_grid = _tf_interp_1d(lam_grid, beta_opt_lam, beta_opt_dot)
        else:
            raise ValueError(f"Unknown homotopy_mode: {self.homotopy_mode}")

        particles_after_list = []
        log_det_jacobians = []
        iterator = range(self.num_particles)
        if self.show_progress:
            iterator = tqdm(iterator, desc="  Flowing particles", leave=False)

        for i in iterator:
            measurement_flow = tf.cast(measurement_original, DTYPE)
            particle_final, log_det_J = propagate_particle_deterministic_tf(
                self.particles[i], beta_grid, beta_dot_grid,
                measurement_flow, self.config, self.ssm
            )
            particles_after_list.append(particle_final)
            log_det_jacobians.append(log_det_J)

        particles_after = tf.stack(particles_after_list, axis=0)
        log_det_J_batch = tf.stack(log_det_jacobians, axis=0)
        self.particles.assign(tf.cast(particles_after, tf.float32))

        log_likelihoods = self._compute_log_likelihood_batch(
            self.particles, measurement_original, landmarks
        )
        log_likelihoods = tf.clip_by_value(log_likelihoods, -100.0, 100.0)

        log_det_J = tf.cast(log_det_J_batch, tf.float32)
        log_det_J = tf.clip_by_value(log_det_J, -50.0, 50.0)

        self.log_weights.assign(
            self.log_weights + log_likelihoods - log_det_J
        )

        self._normalize_weights()
        self._resample_if_needed()

        out = self.ssm.measurement_model(
            tf.cast(self.state, getattr(self.ssm, "dtype", tf.float32)), landmarks
        )
        h_pred = out[0] if isinstance(out, (tuple, list)) and len(out) >= 1 else out
        residual = measurement_original - tf.cast(h_pred, measurement_dtype)

        return (
            tf.cast(self.state, tf.float32),
            tf.cast(self.covariance, tf.float32),
            tf.cast(residual, tf.float32)
        )

    def _compute_log_likelihood_batch(
        self,
        particles: tf.Tensor,
        measurement: tf.Tensor,
        landmarks: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Per-particle Gaussian log-likelihood of *measurement* given predicted observations."""
        particles_dtype = particles.dtype
        measurement = tf.cast(measurement, particles_dtype)
        meas_size = tf.shape(tf.reshape(measurement, [-1]))[0]

        if hasattr(self.ssm, "full_measurement_cov"):
            R = tf.cast(self.ssm.full_measurement_cov(meas_size), particles_dtype)
        else:
            R = tf.cast(self.ssm.R, particles_dtype)
            if tf.shape(R)[0] != meas_size:
                R = tf.eye(meas_size, dtype=particles_dtype) * (tf.linalg.diag_part(R)[0] if R.shape.rank == 2 else R)
        R_reg = R + tf.eye(tf.shape(R)[0], dtype=particles_dtype) * 1e-6
        R_inv = tf.linalg.inv(R_reg)
        log_det_R = tf.linalg.slogdet(R_reg)[1]

        z_preds = self.ssm.measurement_model(particles, landmarks)
        if len(z_preds.shape) > 2:
            z_preds = tf.reshape(z_preds, [tf.shape(particles)[0], -1])
        z_preds = tf.cast(z_preds, particles_dtype)

        z_meas = tf.reshape(measurement, [1, -1])
        innovations = z_meas - z_preds

        weighted = tf.matmul(innovations, R_inv)
        mahalanobis_dists = tf.reduce_sum(weighted * innovations, axis=1)

        meas_dim = tf.cast(tf.shape(z_meas)[1], particles_dtype)
        log_probs = -0.5 * (
            mahalanobis_dists + log_det_R +
            meas_dim * tf.math.log(2.0 * 3.14159)
        )

        return log_probs

    def _normalize_weights(self) -> None:
        """Log-domain stabilisation; set ``weights`` to softmax of finite log-weights (or uniform if degenerate)."""
        finite_mask = tf.math.is_finite(self.log_weights)

        if tf.reduce_any(finite_mask):
            max_log_weight = tf.reduce_max(tf.where(
                finite_mask, self.log_weights,
                tf.constant(-1e10, dtype=tf.float32)
            ))
            self.log_weights.assign(self.log_weights - max_log_weight)

            weights_unnorm = tf.where(
                finite_mask, tf.exp(self.log_weights),
                tf.zeros_like(self.log_weights)
            )
            weight_sum = tf.reduce_sum(weights_unnorm)

            if weight_sum > 1e-10:
                self.weights.assign(weights_unnorm / weight_sum)
                self.log_weights.assign(tf.math.log(self.weights + 1e-10))
            else:
                _apply_uniform_weights(self.num_particles, self.weights, self.log_weights)
        else:
            _apply_uniform_weights(self.num_particles, self.weights, self.log_weights)

    def _resample_if_needed(self) -> None:
        """Systematic resample and jitter when ESS is below the threshold."""
        ess = compute_ess(self.weights)
        self.ess_before_resample.assign(ess)

        threshold = self.resample_threshold * tf.cast(self.num_particles, tf.float32)

        if ess < threshold:
            indices = systematic_resample(self.weights)
            self.particles.assign(tf.gather(self.particles, indices))
            _apply_uniform_weights(self.num_particles, self.weights, self.log_weights)

            particle_std = tf.math.reduce_std(self.particles, axis=0)
            jitter_scale = tf.maximum(particle_std * 0.01, 0.01)
            jitter = tf.random.normal(
                [self.num_particles, self.state_dim],
                mean=0.0, stddev=1.0, dtype=tf.float32
            ) * jitter_scale[tf.newaxis, :]
            self.particles.assign_add(jitter)


__all__ = [
    "PFPFDaiDaumFilter",
    "SPFConfig",
    "compute_M",
    "cond_number",
    "solve_optimal_beta_tf",
    "f_flow_deterministic_tf",
    "propagate_particle_deterministic_tf",
]
