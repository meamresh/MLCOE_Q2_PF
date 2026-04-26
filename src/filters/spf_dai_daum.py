"""
Structure follows the same pattern as other filters: a single main class
DaiDaumStochasticParticleFlow(BaseFilter) that takes an SSM and implements
predict() and update(). All numerical helpers take the SSM instance and use
its prior, Q, R, and gradient/Hessian methods.

Reference:
    Dai & Daum (2021), "Stiffness Mitigation in Stochastic Particle Flow Filters"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.base import BaseFilter, StateSpaceModel


tfd = tfp.distributions
DTYPE = tf.float64

# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------

MAX_PARTICLE_DISTANCE = tf.constant(50.0, dtype=DTYPE)
REGULARIZATION = tf.constant(1e-6, dtype=DTYPE)
MIN_EIGENVALUE_RATIO = tf.constant(1e-8, dtype=DTYPE)  # For eigenvalue clipping


@dataclass
class SPFConfig:
    """Configuration for the Dai–Daum stochastic particle flow."""

    n_particles: int = 200
    lambda_steps: int = 200
    bvp_mesh_points: int = 50
    bvp_max_iter: int = 1000
    bvp_learning_rate: float = 1e-3
    mu: float = 0.2
    use_mu_continuation: bool = True
    mu_steps: int = 8
    verbose: bool = True

    @property
    def dlambda(self) -> tf.Tensor:
        """Pseudo-time step size Δλ = 1 / lambda_steps."""
        return tf.constant(1.0 / self.lambda_steps, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Main filter class (same structure as EDH, LEDH, etc.)
# ---------------------------------------------------------------------------


class DaiDaumStochasticParticleFlow(BaseFilter):
    """
    Stochastic particle flow filter with stiffness mitigation (Dai & Daum, 2021).

    Takes a state-space model that implements the StateSpaceModel protocol and
    (for this method) provides prior and log-density helpers: prior_mean,
    prior_cov, log_prior, gradient_log_prior, hessian_log_prior, log_likelihood,
    gradient_log_likelihood, hessian_log_likelihood. Typically used with
    DaiDaumBearingSSM.

    Parameters
    ----------
    ssm : StateSpaceModel
        State-space model (e.g. DaiDaumBearingSSM) with Q, R, motion_model,
        measurement_model, and the log-density methods above.
    initial_state : tf.Tensor
        Initial state (state_dim,).
    initial_covariance : tf.Tensor
        Initial covariance (state_dim, state_dim).
    config : SPFConfig, optional
        Filter/config for particle count, λ steps, BVP, etc.
    homotopy_mode : str
        'linear' (β(λ)=λ) or 'optimal' (β* from BVP).
    """

    def __init__(
        self,
        ssm: StateSpaceModel,
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor,
        config: Optional[SPFConfig] = None,
        homotopy_mode: str = "optimal",
    ) -> None:
        """Initialise SPF; see class docstring for parameters."""
        super().__init__(
            ssm=ssm,
            initial_state=initial_state,
            initial_covariance=initial_covariance,
        )
        self.config = config or SPFConfig()
        self.homotopy_mode = homotopy_mode
        self._beta_opt_lam: Optional[tf.Tensor] = None
        self._beta_opt: Optional[tf.Tensor] = None
        self._beta_opt_dot: Optional[tf.Tensor] = None

    def predict(
        self, control: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """No dynamics: return current state and covariance."""
        return self.state, self.covariance

    def update(
        self,
        measurement: tf.Tensor,
        beta_opt_lam: Optional[tf.Tensor] = None,
        beta_opt: Optional[tf.Tensor] = None,
        beta_opt_dot: Optional[tf.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        One stochastic particle flow update.

        Optional beta_opt_lam, beta_opt, beta_opt_dot can be passed when
        homotopy_mode='optimal' to reuse a precomputed β* (e.g. from the experiment).
        """
        measurement = tf.convert_to_tensor(measurement, dtype=DTYPE)
        ssm = self.ssm
        x_ref = ssm.prior_mean

        if self.homotopy_mode == "optimal":
            if beta_opt_lam is not None and beta_opt is not None:
                b_lam, b, b_dot = beta_opt_lam, beta_opt, beta_opt_dot
            else:
                if self._beta_opt is None:
                    (
                        self._beta_opt_lam,
                        self._beta_opt,
                        self._beta_opt_dot,
                        _,
                    ) = solve_optimal_beta_tf(x_ref, measurement, self.config, ssm)
                b_lam = self._beta_opt_lam
                b = self._beta_opt
                b_dot = self._beta_opt_dot
        else:
            b_lam = b = b_dot = None

        x_hat, P_hat, _ = run_particle_flow(
            self.homotopy_mode,
            measurement,
            self.config,
            ssm,
            beta_opt_lam=b_lam,
            beta_opt=b,
            beta_opt_dot=b_dot,
        )
        self.state = tf.cast(x_hat, tf.float32)
        self.covariance = tf.cast(P_hat, tf.float32)
        state_dtype = getattr(ssm, "R", tf.constant(0.0)).dtype
        h_pred, _ = ssm.measurement_model(tf.cast(self.state, state_dtype))
        residual = measurement - tf.cast(h_pred, DTYPE)
        return self.state, self.covariance, residual


# ---------------------------------------------------------------------------
# Helpers (all take ssm for prior / Q / R / gradients)
# ---------------------------------------------------------------------------


def compute_M(
    ssm: Any, x_ref: tf.Tensor, beta: tf.Tensor, z: tf.Tensor
) -> tf.Tensor:
    """M(β) = -∇² log π(x, λ) where log π = (1-β) log p₀ + β log p(z|x)."""
    beta = tf.cast(beta, dtype=DTYPE)
    x_ref = tf.cast(x_ref, dtype=DTYPE)
    z = tf.cast(z, dtype=DTYPE)
    state_dim = tf.shape(x_ref)[0]
    alpha = tf.constant(1.0, dtype=DTYPE) - beta
    H0 = tf.cast(ssm.hessian_log_prior(x_ref), DTYPE)
    Hh = tf.cast(ssm.hessian_log_likelihood(x_ref, z), DTYPE)
    H = alpha * H0 + beta * Hh          # <-- fixed: was (alpha+beta)*H0
    M = -H
    M_reg = M + REGULARIZATION * tf.eye(state_dim, dtype=DTYPE)
    M_reg = 0.5 * (M_reg + tf.transpose(M_reg))
    return M_reg


def cond_number(M: tf.Tensor) -> tf.Tensor:
    """κ*(M) = tr(M) tr(M^{-1})."""
    M = tf.cast(M, dtype=DTYPE)
    state_dim = tf.shape(M)[0]
    M_reg = M + REGULARIZATION * tf.eye(state_dim, dtype=DTYPE)
    M_reg = 0.5 * (M_reg + tf.transpose(M_reg))
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
) -> tf.Tensor:
    """J(β) = ∫ [½(β')² + μ κ*(M)] dλ; uses ssm via compute_M(ssm, ...)."""
    beta_vals = tf.cast(beta_vals, dtype=DTYPE)
    beta_dot_vals = tf.cast(beta_dot_vals, dtype=DTYPE)
    x_ref = tf.cast(x_ref, dtype=DTYPE)
    z = tf.cast(z, dtype=DTYPE)
    mu = tf.cast(mu, dtype=DTYPE)
    n_points = tf.shape(beta_vals)[0]
    kappa_values = tf.TensorArray(dtype=DTYPE, size=n_points, dynamic_size=False)
    for i in tf.range(n_points):
        M = compute_M(ssm, x_ref, beta_vals[i], z)
        kappa_values = kappa_values.write(i, cond_number(M))
    kappa_vals = kappa_values.stack()
    integrand = tf.constant(0.5, dtype=DTYPE) * beta_dot_vals**2 + mu * kappa_vals
    dlambda = tf.constant(1.0, dtype=DTYPE) / tf.cast(n_points - 1, DTYPE)
    return tf.reduce_sum((integrand[:-1] + integrand[1:]) * 0.5) * dlambda


def solve_optimal_beta_tf(
    x_ref: tf.Tensor,
    z: tf.Tensor,
    config: SPFConfig,
    ssm: Any,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Solve for β*(λ); uses ssm in compute_objective_J."""
    n_mesh = config.bvp_mesh_points
    mu_target = tf.constant(config.mu, dtype=DTYPE)
    lam_vals = tf.linspace(
        tf.constant(0.0, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE), n_mesh
    )
    beta_interior = tf.Variable(lam_vals[1:-1], dtype=DTYPE, trainable=True)
    mu_values = (
        [mu_target * (i + 1) / config.mu_steps for i in range(config.mu_steps)]
        if config.use_mu_continuation
        else [mu_target]
    )

    def loss_fn(mu_val: tf.Tensor) -> tf.Tensor:
        beta_vals = tf.concat(
            [
                tf.constant([0.0], dtype=DTYPE),
                beta_interior,
                tf.constant([1.0], dtype=DTYPE),
            ],
            axis=0,
        )
        dlambda = tf.constant(1.0, dtype=DTYPE) / tf.cast(n_mesh - 1, DTYPE)
        beta_dot_vals = (beta_vals[1:] - beta_vals[:-1]) / dlambda
        beta_dot_at_points = tf.concat(
            [
                beta_dot_vals[:1],
                (beta_dot_vals[:-1] + beta_dot_vals[1:]) * 0.5,
                beta_dot_vals[-1:],
            ],
            axis=0,
        )
        J = compute_objective_J(
            beta_vals, beta_dot_at_points, x_ref, z, mu_val, ssm
        )
        # If b_dot < 0, penalize, ensures monotonicity
        penalty = tf.reduce_sum(tf.nn.relu(-beta_dot_vals)) * tf.constant(
            10.0, dtype=DTYPE
        )
        return J + penalty

    for mu_step, mu_current in enumerate(mu_values, start=1):
        if config.verbose:
            print(
                f"\nμ-continuation step {mu_step}/{len(mu_values)}: "
                f"μ = {float(mu_current):.4f}"
            )
        optimizer = tf.optimizers.Adam(learning_rate=config.bvp_learning_rate)
        for iteration in range(config.bvp_max_iter):
            with tf.GradientTape() as tape:
                loss_val = loss_fn(mu_current)
            grads = tape.gradient(loss_val, beta_interior)
            optimizer.apply_gradients([(grads, beta_interior)])
            if config.verbose and (
                iteration % 200 == 0 or iteration == config.bvp_max_iter - 1
            ):
                print(
                    f"  Iteration {iteration}, "
                    f"loss (J + penalty) = {float(loss_val):.6f}"
                )

    beta_vals = tf.concat(
        [
            tf.constant([0.0], dtype=DTYPE),
            beta_interior,
            tf.constant([1.0], dtype=DTYPE),
        ],
        axis=0,
    )
    dlambda = tf.constant(1.0, dtype=DTYPE) / tf.cast(n_mesh - 1, DTYPE)
    beta_dot_vals = (beta_vals[1:] - beta_vals[:-1]) / dlambda
    beta_dot_at_points = tf.concat(
        [
            beta_dot_vals[:1],
            (beta_dot_vals[:-1] + beta_dot_vals[1:]) * 0.5,
            beta_dot_vals[-1:],
        ],
        axis=0,
    )
    J_optimal = compute_objective_J(
        beta_vals, beta_dot_at_points, x_ref, z, mu_target, ssm
    )
    return lam_vals, beta_vals, beta_dot_at_points, J_optimal


def _condition_matrix(
    H: tf.Tensor, min_eigenvalue_ratio: tf.Tensor = MIN_EIGENVALUE_RATIO
) -> tf.Tensor:
    """
    Lightweight conditioning: just add regularization and symmetrize.
    Full eigenvalue clipping is too expensive (O(d³)) to call per-particle.
    """
    H_sym = 0.5 * (H + tf.transpose(H))
    return H_sym


def _compute_adaptive_trust_region(
    f: tf.Tensor, prior_cov: tf.Tensor, base_radius: tf.Tensor = tf.constant(100.0, dtype=DTYPE)
) -> tf.Tensor:
    """
    Adaptive trust region radius based on prior covariance scale.
    Returns clipped drift f with radius proportional to sqrt(trace(P)).
    """
    trace_P = tf.linalg.trace(prior_cov)
    scale = tf.sqrt(trace_P + tf.constant(1e-8, dtype=DTYPE))
    radius = base_radius * scale
    f_norm = tf.norm(f)
    return tf.cond(
        f_norm > radius,
        lambda: f * (radius / f_norm),
        lambda: f,
    )


@tf.function(reduce_retracing=True)
def f_flow_tf_optimized(
    x: tf.Tensor,
    beta: tf.Tensor,
    beta_dot: tf.Tensor,
    z: tf.Tensor,
    ssm: Any,
    H0_ref: Optional[tf.Tensor] = None,
    Hh_ref: Optional[tf.Tensor] = None,
    use_gauss_newton_h: bool = False,
    use_adaptive_clip: bool = False,
) -> tf.Tensor:
    """
    Optimized drift f(x, λ) with Hessian reuse and improved conditioning.
    
    Parameters
    ----------
    x : tf.Tensor (state_dim,)
        Current particle state.
    beta, beta_dot : tf.Tensor
        Homotopy schedule values.
    z : tf.Tensor
        Measurement.
    ssm : StateSpaceModel
        State-space model.
    H0_ref : tf.Tensor, optional (state_dim, state_dim)
        Precomputed reference prior Hessian. If None, computed at x.
    Hh_ref : tf.Tensor, optional (state_dim, state_dim)
        Precomputed reference likelihood Hessian. If None, computed at x.
        If use_gauss_newton_h=True, this is ignored and Gauss-Newton is used.
    use_gauss_newton_h : bool
        If True, approximate Hh using Gauss-Newton: Hh ≈ -J^T R^-1 J.
    use_adaptive_clip : bool
        If True, use adaptive trust region based on prior covariance.
    
    Returns
    -------
    f : tf.Tensor (state_dim,)
        Drift vector.
    """
    x = tf.cast(x, dtype=DTYPE)
    beta = tf.cast(beta, dtype=DTYPE)
    beta_dot = tf.cast(beta_dot, dtype=DTYPE)
    z = tf.cast(z, dtype=DTYPE)
    state_dim = tf.shape(x)[0]
    I = tf.eye(state_dim, dtype=DTYPE)
    Q = tf.cast(ssm.Q, DTYPE)
    alpha = tf.constant(1.0, dtype=DTYPE) - beta

    # Gradients: always computed per-particle (x-dependent)
    g0 = tf.cast(ssm.gradient_log_prior(x), DTYPE)
    gh = tf.cast(ssm.gradient_log_likelihood(x, z), DTYPE)

    # Hessians: reuse reference if provided, otherwise compute
    if H0_ref is None:
        H0 = tf.cast(ssm.hessian_log_prior(x), DTYPE)
    else:
        H0 = tf.cast(H0_ref, DTYPE)

    if use_gauss_newton_h:
        # Gauss-Newton approximation: Hh ≈ -J^T R^-1 J
        # NOTE: This is expensive per-particle. Better to precompute Hh_ref once.
        # Only use this if Hh_ref is not available.
        with tf.GradientTape() as tape:
            tape.watch(x)
            h_pred = ssm.measurement_model(x)
            # Handle tuple return (some SSMs return (h, jac))
            if isinstance(h_pred, (tuple, list)):
                h_pred = h_pred[0]
        J = tape.jacobian(h_pred, x)  # (meas_dim, state_dim)
        if J.shape.rank == 1:
            J = J[tf.newaxis, :]
        elif J.shape.rank == 0:
            # Scalar measurement
            J = tf.reshape(J, [1, 1])
        R_inv = tf.cast(tf.linalg.inv(ssm.R), DTYPE)
        Hh = -tf.linalg.matmul(tf.transpose(J), tf.linalg.matmul(R_inv, J))
    elif Hh_ref is None:
        Hh = tf.cast(ssm.hessian_log_likelihood(x, z), DTYPE)
    else:
        Hh = tf.cast(Hh_ref, DTYPE)

    # Homotopy Hessian and gradient
    H = alpha * H0 + beta * Hh
    gp = alpha * g0 + beta * gh

    # Regularization and symmetrization (lightweight - no expensive eigenvalue decomposition)
    H_reg = H + REGULARIZATION * I
    H_reg = 0.5 * (H_reg + tf.transpose(H_reg))

    # Drift computation (same as before)
    K2_gh = -beta_dot * tf.linalg.solve(H_reg, gh[:, tf.newaxis])[:, 0]
    H_inv_gp = tf.linalg.solve(H_reg, gp[:, tf.newaxis])[:, 0]
    Hh_H_inv_gp = tf.linalg.matvec(Hh, H_inv_gp)
    H_inv_Hh_H_inv_gp = tf.linalg.solve(H_reg, Hh_H_inv_gp[:, tf.newaxis])[:, 0]
    K1_gp = 0.5 * tf.linalg.matvec(Q, gp) - 0.5 * beta_dot * H_inv_Hh_H_inv_gp

    f = K1_gp + K2_gh

    # Adaptive clipping
    if use_adaptive_clip and hasattr(ssm, "prior_cov"):
        prior_cov = tf.cast(ssm.prior_cov, DTYPE)
        f = _compute_adaptive_trust_region(f, prior_cov)
    else:
        f_norm = tf.norm(f)
        f = tf.cond(
            f_norm > tf.constant(100.0, dtype=DTYPE),
            lambda: f * (tf.constant(100.0, dtype=DTYPE) / f_norm),
            lambda: f,
        )

    return f


@tf.function
def f_flow_tf_batched(
    x_batch: tf.Tensor,
    beta: tf.Tensor,
    beta_dot: tf.Tensor,
    z: tf.Tensor,
    ssm: Any,
    H0_ref: Optional[tf.Tensor] = None,
    Hh_ref: Optional[tf.Tensor] = None,
    use_gauss_newton_h: bool = False,
    use_adaptive_clip: bool = False,
) -> tf.Tensor:
    """
    Batched version of f_flow_tf_optimized for multiple particles.
    
    Parameters
    ----------
    x_batch : tf.Tensor (N, state_dim)
        Batch of particle states.
    beta, beta_dot : tf.Tensor
        Homotopy schedule values (scalar or broadcastable).
    z : tf.Tensor
        Measurement.
    ssm : StateSpaceModel
        State-space model.
    H0_ref, Hh_ref : tf.Tensor, optional (state_dim, state_dim)
        Precomputed reference Hessians.
    use_gauss_newton_h, use_adaptive_clip : bool
        See f_flow_tf_optimized.
    
    Returns
    -------
    f_batch : tf.Tensor (N, state_dim)
        Drift vectors for all particles.
    """
    N = tf.shape(x_batch)[0]
    
    # Vectorized map over particles
    def flow_single(x_i):
        return f_flow_tf_optimized(
            x_i, beta, beta_dot, z, ssm, H0_ref, Hh_ref, use_gauss_newton_h, use_adaptive_clip
        )
    
    f_batch = tf.map_fn(
        flow_single,
        x_batch,
        fn_output_signature=tf.TensorSpec(shape=(None,), dtype=DTYPE),
        parallel_iterations=10,
    )
    return f_batch


def propagate_particles_batched_tf(
    x_batch0: tf.Tensor,
    beta_grid: tf.Tensor,
    beta_dot_grid: tf.Tensor,
    z: tf.Tensor,
    config: SPFConfig,
    ssm: Any,
) -> tf.Tensor:
    """Euler–Maruyama propagation for a batch of particles; uses ssm.Q."""
    N = tf.shape(x_batch0)[0]
    state_dim = tf.shape(x_batch0)[1]
    x_batch = tf.Variable(tf.cast(x_batch0, dtype=DTYPE), trainable=False)
    Lq = tf.linalg.cholesky(tf.cast(ssm.Q, DTYPE))
    dlambda = config.dlambda
    sqrt_dlambda = tf.sqrt(dlambda)
    
    for k in range(config.lambda_steps):
        x_batch.assign(
            tf.clip_by_value(
                x_batch, -MAX_PARTICLE_DISTANCE, MAX_PARTICLE_DISTANCE
            )
        )
        beta_k = tf.cast(beta_grid[k], dtype=DTYPE)
        beta_dot_k = tf.cast(beta_dot_grid[k], dtype=DTYPE)
        
        f_batch = f_flow_tf_batched(x_batch.value(), beta_k, beta_dot_k, z, ssm)
        
        dW = tf.random.normal([N, state_dim], mean=0.0, stddev=1.0, dtype=DTYPE) * sqrt_dlambda
        # Lq * dW per particle -> dW @ Lq^T
        diffusion = tf.matmul(dW, Lq, transpose_b=True)
        
        x_new = x_batch + f_batch * dlambda + diffusion
        x_batch.assign(
            tf.clip_by_value(
                x_new, -MAX_PARTICLE_DISTANCE, MAX_PARTICLE_DISTANCE
            )
        )
    return x_batch.value()


def _tf_interp_1d(
    x_new: tf.Tensor, x_old: tf.Tensor, y_old: tf.Tensor
) -> tf.Tensor:
    """Pure TF 1D linear interpolation."""
    x_new = tf.cast(tf.convert_to_tensor(x_new), dtype=DTYPE)
    x_old = tf.cast(tf.convert_to_tensor(x_old), dtype=DTYPE)
    y_old = tf.cast(tf.convert_to_tensor(y_old), dtype=DTYPE)
    indices = tf.searchsorted(x_old, x_new, side="right")
    indices = tf.clip_by_value(indices, 1, tf.shape(x_old)[0] - 1)
    idx_low = indices - 1
    idx_high = indices
    x_low = tf.gather(x_old, idx_low)
    x_high = tf.gather(x_old, idx_high)
    y_low = tf.gather(y_old, idx_low)
    y_high = tf.gather(y_old, idx_high)
    dx = x_high - x_low
    dx = tf.where(tf.abs(dx) < 1e-12, tf.ones_like(dx) * 1e-12, dx)
    weights = (x_new - x_low) / dx
    return y_low + (y_high - y_low) * weights


def run_particle_flow(
    homotopy_mode: str,
    z: tf.Tensor,
    config: SPFConfig,
    ssm: Any,
    beta_opt_lam: Optional[tf.Tensor] = None,
    beta_opt: Optional[tf.Tensor] = None,
    beta_opt_dot: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    One SPF update: sample from prior, propagate with β(λ) or β*(λ), return
    posterior mean, covariance, and particles. Uses ssm.prior_mean, ssm.prior_cov, ssm.Q.
    """
    z_tf = tf.cast(z, dtype=DTYPE)
    prior_mean = tf.cast(ssm.prior_mean, DTYPE)
    prior_cov = tf.cast(ssm.prior_cov, DTYPE)
    dist = tfd.MultivariateNormalTriL(
        loc=prior_mean,
        scale_tril=tf.linalg.cholesky(prior_cov),
    )
    particles = dist.sample(config.n_particles)
    lam_grid = tf.linspace(
        tf.constant(0.0, dtype=DTYPE),
        tf.constant(1.0, dtype=DTYPE),
        config.lambda_steps + 1,
    )
    if homotopy_mode == "linear":
        beta_grid = lam_grid
        beta_dot_grid = tf.ones_like(lam_grid, dtype=DTYPE)
    elif homotopy_mode == "optimal":
        if beta_opt_lam is None or beta_opt is None:
            raise ValueError(
                "beta_opt_lam and beta_opt required for homotopy_mode='optimal'"
            )
        beta_grid = _tf_interp_1d(lam_grid, beta_opt_lam, beta_opt)
        beta_dot_grid = (
            _tf_interp_1d(lam_grid, beta_opt_lam, beta_opt_dot)
            if beta_opt_dot is not None
            else tf.concat(
                [
                    (beta_grid[1:2] - beta_grid[0:1]) / config.dlambda,
                    (beta_grid[2:] - beta_grid[:-2]) / (2.0 * config.dlambda),
                    (beta_grid[-1:] - beta_grid[-2:-1]) / config.dlambda,
                ],
                axis=0,
            )
        )
    else:
        raise ValueError(f"Unknown homotopy_mode: {homotopy_mode}")

    # Batched particle propagation using tf.map_fn under the hood
    final_particles = propagate_particles_batched_tf(
        particles, beta_grid, beta_dot_grid, z_tf, config, ssm
    )

    final_particles = tf.clip_by_value(
        final_particles, -MAX_PARTICLE_DISTANCE, MAX_PARTICLE_DISTANCE
    )
    x_hat = tf.reduce_mean(final_particles, axis=0)
    diff = final_particles - x_hat
    P_hat = tf.matmul(diff, diff, transpose_a=True) / tf.cast(
        config.n_particles - 1, DTYPE
    )
    return x_hat, P_hat, final_particles


__all__ = [
    "SPFConfig",
    "DaiDaumStochasticParticleFlow",
    "compute_M",
    "cond_number",
    "compute_objective_J",
    "solve_optimal_beta_tf",
    "f_flow_tf_optimized",
    "f_flow_tf_batched",
    "propagate_particles_batched_tf",
    "run_particle_flow",
]
