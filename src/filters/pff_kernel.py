"""
Implementation of Particle Flow Filter (PFF) following Hu & van Leeuwen (2021).

Paper: "A particle flow filter for high-dimensional system applications"
       Quarterly Journal of the Royal Meteorological Society, 2021
       DOI: 10.1002/qj.4028

Key equations:
- Flow field: Eq. (6) with D = B (localized prior covariance)
- Integration: Eq. (7) - forward Euler in pseudo-time
- Scalar kernel: Eq. (16)-(19)
- Matrix kernel: Eq. (20)-(23)
- Localization: Eq. (29) - Gaussian C_ij = exp(-d_ij^2 / r_in^2)
"""

from typing import Optional, Tuple, Dict
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.resampling import systematic_resample
from src.utils.linalg import localization_matrix

tfd = tfp.distributions


# ============================================================================
# KERNEL FLOWS (Paper Eq. 6, 16-23)
# ============================================================================

@tf.function
def scalar_kernel_flow(particles: tf.Tensor,
                      grads: tf.Tensor,
                      B: tf.Tensor,
                      B_inv: tf.Tensor,
                      alpha: tf.Tensor) -> tf.Tensor:
    """
    Scalar kernel particle flow (Paper Eq. 16-19, 6).

    Kernel: k(x,z) = exp(-0.5 * (x-z)^T A (x-z)) where A = (αB)^(-1)
    Flow: f_i = B @ (1/N) Σ_j [ k(x_i,x_j) ∇log p(x_j|y) + ∇_{x_j} k(x_j, x_i) ]

    For Gaussian kernel: ∇_{x_j} k(x_j, x_i) = k(x_i, x_j) * A (x_i - x_j)

    Args:
        particles: [N, d] particle positions
        grads: [N, d] full posterior gradients ∇log p(x|y)
        B: [d, d] localized prior covariance (D in paper)
        B_inv: [d, d] inverse of B (full matrix, preserves localized structure)
        alpha: scalar kernel width α (paper uses α ≈ 1/Np)

    Returns:
        Flow vectors [N, d]
    """
    x = tf.cast(particles, tf.float32)
    grads = tf.cast(grads, tf.float32)
    B = tf.cast(B, tf.float32)
    B_inv = tf.cast(B_inv, tf.float32)
    alpha = tf.maximum(tf.cast(alpha, tf.float32), 1e-12)

    N = tf.cast(tf.shape(x)[0], tf.float32)
    d = tf.shape(x)[1]

    delta = x[:, None, :] - x[None, :, :]
    A = B_inv / alpha
    Adelta = tf.einsum("ijc,cd->ijd", delta, A)
    quad = tf.reduce_sum(delta * Adelta, axis=-1)
    k = tf.exp(-0.5 * quad)
    attract = tf.matmul(k, grads)
    repulse = tf.reduce_sum(k[:, :, tf.newaxis] * Adelta, axis=1)
    I_f = (attract + repulse) / N
    return tf.matmul(I_f, B)


@tf.function
def diagonal_kernel_flow(particles: tf.Tensor,
                         grads: tf.Tensor,
                         B: tf.Tensor,
                         alpha: tf.Tensor) -> tf.Tensor:
    """
    Diagonal (matrix-valued) kernel particle flow (Paper Eq. 20-23, 6).

    Kernel per dimension: K^(a)(x,z) = exp(-0.5 * (x^a - z^a)^2 / (α σ_a^2))
    where σ_a^2 = diag(B)_a (prior variance of component a).

    Args:
        particles: [N, d] particle positions
        grads: [N, d] full posterior gradients ∇log p(x|y)
        B: [d, d] localized prior covariance (D in paper)
        alpha: scalar kernel width α (paper uses α ≈ 1/Np)

    Returns:
        Flow vectors [N, d]
    """
    x = tf.cast(particles, tf.float32)
    grads = tf.cast(grads, tf.float32)
    B = tf.cast(B, tf.float32)
    alpha = tf.maximum(tf.cast(alpha, tf.float32), 1e-12)

    N = tf.cast(tf.shape(x)[0], tf.float32)
    delta = x[:, None, :] - x[None, :, :]
    sq = delta ** 2
    sigma2 = tf.maximum(tf.linalg.diag_part(B), 1e-12)
    h2 = alpha * sigma2
    k_a = tf.exp(-sq / (2.0 * tf.reshape(h2, [1, 1, -1])))
    attract = tf.reduce_sum(k_a * grads[tf.newaxis, :, :], axis=1)
    h2_reshaped = tf.reshape(h2, [1, 1, -1])
    delta_scaled = delta / h2_reshaped
    repulse = tf.reduce_sum(k_a * delta_scaled, axis=1)
    I_f = (attract + repulse) / N
    return tf.matmul(I_f, B)


# ============================================================================
# GRADIENTS (Paper Section 2.2)
# ============================================================================

def compute_log_likelihood_gradient(particles: tf.Tensor,
                                    obs: tf.Tensor,
                                    obs_idx: tf.Tensor,
                                    obs_noise_std: float,
                                    ssm=None,
                                    landmarks: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    Log-likelihood gradient: ∇_x log p(y|x) = H^T R^{-1} (y - h(x))

    Paper Eq. (13) for linear observations, or general form for nonlinear.

    Args:
        particles: [N, d] particle positions
        obs: [n_obs] observations
        obs_idx: [n_obs] observed dimension indices (for direct observations)
        obs_noise_std: observation noise standard deviation
        ssm: state space model (for nonlinear observations)
        landmarks: sensor/landmark positions (for nonlinear observations)

    Returns:
        Log-likelihood gradients [N, d]
    """
    x = tf.cast(particles, tf.float32)
    N = tf.shape(x)[0]
    d = tf.shape(x)[1]
    obs = tf.cast(obs, tf.float32)

    if ssm is not None and landmarks is not None and hasattr(ssm, 'measurement_model'):
        landmarks_tf = tf.cast(landmarks, tf.float32)
        pred = ssm.measurement_model(x, landmarks_tf)
        pred = tf.cast(pred, tf.float32)
        pred_flat = tf.reshape(pred, [N, -1])
        obs_vec = tf.reshape(obs, [-1])
        residual = obs_vec[tf.newaxis, :] - pred_flat

        if len(pred.shape) == 3 and pred.shape[-1] == 2:
            r3 = tf.reshape(residual, [N, -1, 2])
            bearings = r3[:, :, 1]
            wrapped = tf.math.atan2(tf.sin(bearings), tf.cos(bearings))
            r3 = tf.concat([r3[:, :, 0:1], wrapped[:, :, tf.newaxis]], axis=2)
            residual = tf.reshape(r3, [N, -1])

        meas_dim = tf.shape(pred_flat)[1]
        if hasattr(ssm, 'full_measurement_cov'):
            num_landmarks = tf.shape(landmarks_tf)[0]
            R_full = tf.cast(ssm.full_measurement_cov(num_landmarks), tf.float32)
        else:
            sigma2 = float(obs_noise_std) ** 2
            R_full = tf.eye(meas_dim, dtype=tf.float32) * sigma2

        R_full = 0.5 * (R_full + tf.transpose(R_full))
        R_full = R_full + 1e-6 * tf.eye(meas_dim, dtype=tf.float32)
        chol = tf.linalg.cholesky(R_full)
        tmp = tf.linalg.cholesky_solve(chol, tf.transpose(residual))
        tmp = tf.transpose(tmp)

        if hasattr(ssm, 'measurement_jacobian'):
            H = ssm.measurement_jacobian(x, landmarks_tf)
            H = tf.cast(H, tf.float32)
            H = tf.reshape(H, [N, -1, d])
        else:
            def compute_H(particle):
                with tf.GradientTape() as tape:
                    particle = tf.reshape(particle, [1, -1])
                    tape.watch(particle)
                    p = ssm.measurement_model(particle, landmarks_tf)
                    p = tf.reshape(p, [-1])
                J = tape.jacobian(p, particle)
                return tf.squeeze(J, axis=1)
            H = tf.vectorized_map(compute_H, x)

        return tf.einsum("bmd,bm->bd", H, tmp)

    obs_idx = tf.cast(obs_idx, tf.int32)
    sigma2 = float(obs_noise_std) ** 2
    obs_vec = tf.reshape(obs, [-1])
    x_obs = tf.gather(x, obs_idx, axis=1)
    residuals = obs_vec[tf.newaxis, :] - x_obs
    updates = residuals / sigma2
    n_obs = tf.shape(obs_vec)[0]
    rows = tf.repeat(tf.range(N, dtype=tf.int32), repeats=n_obs)
    cols = tf.tile(obs_idx, [N])
    idxs = tf.stack([rows, cols], axis=1)
    upd = tf.reshape(updates, [-1])
    return tf.tensor_scatter_nd_add(tf.zeros((N, d), dtype=tf.float32), idxs, upd)


@tf.function
def compute_log_prior_gradient(particles: tf.Tensor,
                               prior_mean: tf.Tensor,
                               prior_cov_inv: tf.Tensor) -> tf.Tensor:
    """
    Prior gradient: ∇_x log p(x) = -B^{-1} (x - x̄0)

    Paper Eq. (15). Uses ensemble mean x̄0 and localized B.
    """
    x = tf.cast(particles, tf.float32)
    prior_mean = tf.cast(prior_mean, tf.float32)
    prior_cov_inv = tf.cast(prior_cov_inv, tf.float32)
    diff = x - tf.reshape(prior_mean, [1, -1])
    return -tf.matmul(diff, prior_cov_inv)


# ============================================================================
# PARTICLE FLOW FILTER (Paper Algorithm 1)
# ============================================================================

def assimilate_pff(particles: tf.Tensor,
                   obs: tf.Tensor,
                   obs_idx: tf.Tensor,
                   obs_noise_std: float,
                   prior_mean: tf.Tensor,
                   prior_cov: tf.Tensor,
                   kernel_type: str = 'scalar',
                   alpha: Optional[float] = None,
                   max_steps: int = 500,
                   initial_step_size: float = 0.05,
                   convergence_tol: float = 1e-5,
                   ssm=None,
                   landmarks: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, Dict]:
    """
    Particle Flow Filter - Algorithm 1 from Hu & van Leeuwen (2021).

    Flows particles from prior to posterior using pseudo-time integration.
    All particles maintain equal weight throughout.

    Args:
        particles: [N, d] initial particles (prior ensemble)
        obs: [n_obs] observations
        obs_idx: [n_obs] observed dimension indices
        obs_noise_std: observation noise standard deviation
        prior_mean: [d] ensemble mean x̄0
        prior_cov: [d, d] localized prior covariance B (caller applies Eq. 29)
        kernel_type: 'scalar' or 'diagonal' (matrix-valued)
        alpha: kernel width α (None = 1/Np, paper default)
        max_steps: maximum pseudo-time iterations
        initial_step_size: initial Δs (paper: 0.05 for linear, 0.001 for nonlinear)
        convergence_tol: convergence threshold on ||f||
        ssm: state space model (for nonlinear observations)
        landmarks: sensor/landmark positions (for nonlinear observations)

    Returns:
        particles_final: [N, d] flowed particles (posterior ensemble)
        trajectory: [n_snapshots, N, d] trajectory snapshots
        diagnostics: dict with convergence info
    """
    X = tf.identity(tf.cast(particles, tf.float32))
    prior_mean = tf.cast(prior_mean, tf.float32)
    B = tf.cast(prior_cov, tf.float32)

    B = 0.5 * (B + tf.transpose(B))
    mean_var = tf.reduce_mean(tf.linalg.diag_part(B))
    B = B + tf.eye(tf.shape(B)[0], dtype=tf.float32) * (1e-6 * mean_var + 1e-8)

    try:
        chol_B = tf.linalg.cholesky(B)
        B_inv = tf.linalg.cholesky_solve(chol_B, tf.eye(tf.shape(B)[0], dtype=tf.float32))
    except Exception:
        reg = tf.reduce_mean(tf.linalg.diag_part(B)) * 1e-4
        B_reg = B + tf.eye(tf.shape(B)[0], dtype=tf.float32) * reg
        B_inv = tf.linalg.inv(B_reg)

    Np = tf.cast(tf.shape(X)[0], tf.float32)
    if alpha is None:
        alpha_used = tf.maximum(1.0 / tf.maximum(Np, 1.0), 1e-12)
    else:
        alpha_used = tf.maximum(tf.cast(alpha, tf.float32), 1e-12)

    if kernel_type == 'scalar':
        flow_fn = lambda x, g: scalar_kernel_flow(x, g, B, B_inv, alpha_used)
    elif kernel_type == 'diagonal':
        flow_fn = lambda x, g: diagonal_kernel_flow(x, g, B, alpha_used)
    else:
        raise ValueError(f"kernel_type must be 'scalar' or 'diagonal', got {kernel_type}")

    def compute_gradients(x):
        lik = compute_log_likelihood_gradient(x, obs, obs_idx, obs_noise_std, ssm=ssm, landmarks=landmarks)
        prior_g = compute_log_prior_gradient(x, prior_mean, B_inv)
        return lik + prior_g

    def flow_step(x, dt):
        g = compute_gradients(x)
        f = flow_fn(x, g)
        f_sq_sum = tf.reduce_sum(f ** 2, axis=1)
        f_mag = tf.sqrt(tf.reduce_mean(f_sq_sum))
        inc = dt * f
        return inc, f_mag

    trajectory = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    trajectory = trajectory.write(0, X)
    flow_mags = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    dt_hist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    step = tf.constant(0)
    converged = tf.constant(False)
    traj_idx = tf.constant(1)
    dt = tf.cast(initial_step_size, tf.float32)
    prev_flow_mag = tf.constant(1e30, dtype=tf.float32)
    dec_streak = tf.constant(0, dtype=tf.int32)
    adapt_factor = tf.constant(1.4, dtype=tf.float32)
    dt_min = tf.constant(1e-6, dtype=tf.float32)
    dt_max = tf.constant(0.05, dtype=tf.float32)

    prior_scale = tf.sqrt(tf.maximum(tf.reduce_mean(tf.linalg.diag_part(B)), 1e-12))
    max_step_norm = tf.maximum(0.35 * prior_scale, 1e-3)

    def loop_condition(step, X, converged, trajectory, flow_mags, dt_hist, traj_idx, dt, prev_flow_mag, dec_streak):
        return tf.logical_and(step < max_steps, tf.logical_not(converged))

    def loop_body(step, X, converged, trajectory, flow_mags, dt_hist, traj_idx, dt, prev_flow_mag, dec_streak):
        inc, f_mag = flow_step(X, dt)
        raw_step_mag = tf.sqrt(tf.reduce_mean(tf.reduce_sum(inc ** 2, axis=1)))
        scale = tf.minimum(1.0, max_step_norm / (raw_step_mag + 1e-12))
        inc = inc * scale
        flow_mags = flow_mags.write(step, f_mag)
        dt_hist = dt_hist.write(step, dt)
        converged = f_mag < convergence_tol
        X = X + inc
        decreased = f_mag < prev_flow_mag
        dec_streak_new = tf.cond(decreased, lambda: dec_streak + 1, lambda: tf.constant(0, dtype=tf.int32))
        clipped_hard = scale < 0.5
        should_shrink = tf.logical_or(tf.logical_not(decreased), clipped_hard)
        dt_new = tf.cond(should_shrink, lambda: tf.maximum(dt / adapt_factor, dt_min), lambda: dt)
        should_grow = dec_streak_new >= 20
        dt_final = tf.cond(should_grow, lambda: tf.minimum(dt_new * adapt_factor, dt_max), lambda: dt_new)
        dec_streak_final = tf.cond(should_grow, lambda: tf.constant(0, dtype=tf.int32), lambda: dec_streak_new)
        should_store = tf.equal(step % 10, 0)
        traj_idx_new, trajectory_new = tf.cond(
            should_store,
            lambda: (traj_idx + 1, trajectory.write(traj_idx, X)),
            lambda: (traj_idx, trajectory)
        )
        return step + 1, X, converged, trajectory_new, flow_mags, dt_hist, traj_idx_new, dt_final, f_mag, dec_streak_final

    _, X, converged, trajectory, flow_mags, dt_hist, traj_idx, dt, prev_flow_mag, dec_streak = tf.while_loop(
        loop_condition,
        loop_body,
        [step, X, converged, trajectory, flow_mags, dt_hist, traj_idx, dt, prev_flow_mag, dec_streak]
    )

    trajectory = trajectory.write(traj_idx, X)
    flow_mags_tensor = flow_mags.stack()
    dt_hist_tensor = dt_hist.stack()
    trajectory_tensor = trajectory.stack()

    diagnostics = {
        'n_steps': tf.shape(flow_mags_tensor)[0],
        'flow_magnitudes': flow_mags_tensor,
        'dt_history': dt_hist_tensor,
        'alpha_used': alpha_used,
        'converged': tf.cond(
            tf.shape(flow_mags_tensor)[0] > 0,
            lambda: flow_mags_tensor[-1] < convergence_tol,
            lambda: tf.constant(False)
        ),
        'B_eigs': tf.linalg.eigvalsh(B)
    }

    return X, trajectory_tensor, diagnostics


# ============================================================================
# SEQUENTIAL FILTERS (for online state estimation)
# ============================================================================

def _measurements_to_flat(measurements) -> tf.Tensor:
    """Convert measurements (list, tuple, or tensor) to flat float32 tensor."""
    t = tf.convert_to_tensor(measurements, dtype=tf.float32)
    return tf.reshape(t, [-1])


def _obs_indices_for_landmarks(landmarks_or_sensors) -> tf.Tensor:
    """Build observation indices [0,1,2,3,...] for range-bearing: 2 per landmark."""
    landmarks_tf = tf.convert_to_tensor(landmarks_or_sensors, dtype=tf.float32)
    n_landmarks = tf.shape(landmarks_tf)[0]
    pairs = tf.stack([2 * tf.range(n_landmarks), 2 * tf.range(n_landmarks) + 1], axis=1)
    return tf.reshape(pairs, [-1])


class ScalarPFF:
    """
    Scalar kernel Particle Flow Filter for sequential estimation.

    Uses paper's scalar kernel (Eq. 16-19) with D=B preconditioning.
    """

    def __init__(self, ssm, initial_state, initial_covariance, num_particles=500,
                 step_size=0.05, alpha=None, localization_radius=4.0,
                 convergence_tol=1e-5, max_steps=500, show_progress=False):
        self.ssm = ssm
        self.num_particles = num_particles
        self.step_size = step_size
        self.alpha = alpha
        self.localization_radius = localization_radius
        self.convergence_tol = convergence_tol
        self.max_steps = max_steps
        self.show_progress = show_progress

        self.state = tf.constant(initial_state, dtype=tf.float32)
        P0 = tf.constant(initial_covariance, dtype=tf.float32)

        dist = tf.random.normal([num_particles, tf.shape(self.state)[0]], dtype=tf.float32)
        L = tf.linalg.cholesky(P0)
        self.particles = self.state + tf.matmul(dist, L, transpose_b=True)

        self.weights = tf.ones(num_particles, dtype=tf.float32) / num_particles
        self.ess_before_resample = float(num_particles)
        self.P = P0

    def predict(self, control):
        """Prediction step: propagate particles through motion model."""
        if hasattr(self.ssm, 'N_s'):
            particles_pred = self.ssm.motion_model(self.particles)
            noise = self.ssm.sample_process_noise((self.num_particles,), use_gen=False)
            noise_flat = tf.reshape(noise, [self.num_particles, -1])
            self.particles = particles_pred + noise_flat
        else:
            control = tf.constant(control, dtype=tf.float32)
            control_batch = tf.tile(tf.reshape(control, [1, -1]), [self.num_particles, 1])
            particles_pred = self.ssm.motion_model(self.particles, control_batch)
            Q = self.ssm.Q
            noise = tf.random.normal([self.num_particles, tf.shape(self.state)[0]], dtype=tf.float32)
            L_Q = tf.linalg.cholesky(Q)
            self.particles = particles_pred + tf.matmul(noise, L_Q, transpose_b=True)

        self.state = tf.reduce_mean(self.particles, axis=0)
        particles_centered = self.particles - self.state
        self.P = tf.matmul(particles_centered, particles_centered, transpose_a=True) / float(self.num_particles)

    def update(self, measurements, landmarks_or_sensors):
        """Update step: assimilate observations via particle flow."""
        import warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings('ignore')

        measurements_flat = _measurements_to_flat(measurements)

        if hasattr(self.ssm, 'N_s'):
            obs_indices = tf.constant([0], dtype=tf.int32)
            obs_noise_std = float(self.ssm.sigma_w) if hasattr(self.ssm, 'sigma_w') else 1.0
        else:
            obs_indices = _obs_indices_for_landmarks(landmarks_or_sensors)
            obs_noise_std = float(tf.sqrt(self.ssm.R[0, 0]))

        prior_mean = tf.reduce_mean(self.particles, axis=0)
        particles_centered = self.particles - prior_mean
        prior_cov = tf.matmul(particles_centered, particles_centered, transpose_a=True) / float(self.num_particles - 1)

        if self.localization_radius > 0:
            d = tf.shape(prior_cov)[0]
            idx = tf.range(d, dtype=tf.float32)
            dist = tf.abs(idx[:, None] - idx[None, :])
            C = tf.exp(-(dist ** 2) / (self.localization_radius ** 2))
            prior_cov = prior_cov * C

        prior_cov = 0.5 * (prior_cov + tf.transpose(prior_cov))
        mean_var = tf.reduce_mean(tf.linalg.diag_part(prior_cov))
        prior_cov = prior_cov + tf.eye(tf.shape(prior_cov)[0], dtype=tf.float32) * (1e-6 * mean_var + 1e-8)

        try:
            particles_new, _, _ = assimilate_pff(
                self.particles,
                measurements_flat,
                obs_indices,
                obs_noise_std,
                prior_mean,
                prior_cov,
                kernel_type='scalar',
                alpha=self.alpha,
                max_steps=self.max_steps,
                initial_step_size=self.step_size,
                convergence_tol=self.convergence_tol,
                ssm=self.ssm,
                landmarks=tf.convert_to_tensor(landmarks_or_sensors, dtype=tf.float32)
            )
            self.particles = particles_new
        except Exception as e:
            if self.show_progress:
                print("Warning: ScalarPFF update failed, using fallback")
            weights = self._compute_weights(measurements, landmarks_or_sensors)
            ess = 1.0 / (tf.reduce_sum(weights**2) + 1e-15)
            if ess < self.num_particles * 0.5:
                self.particles = self._resample(self.particles, weights)
                self.weights = tf.ones(self.num_particles, dtype=tf.float32) / self.num_particles
        finally:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            warnings.filterwarnings('default')

        self.state = tf.reduce_mean(self.particles, axis=0)
        particles_centered = self.particles - self.state
        self.P = tf.matmul(particles_centered, particles_centered, transpose_a=True) / float(self.num_particles)
        self.ess_before_resample = float(self.num_particles)

    def _compute_weights(self, measurements, landmarks_or_sensors):
        """Compute importance weights (fallback)."""
        if hasattr(self.ssm, 'N_s'):
            meas_pred = self.ssm.measurement_model(self.particles)
            meas_true = tf.constant(measurements, dtype=tf.float32)
            diff = meas_pred - meas_true
            log_lik = -0.5 * tf.reduce_sum(diff**2 / self.ssm.sigma_w**2, axis=1)
        else:
            landmarks_tf = tf.constant(landmarks_or_sensors, dtype=tf.float32)
            meas_pred = self.ssm.measurement_model(self.particles, landmarks_tf)
            meas_true = _measurements_to_flat(measurements)
            meas_pred_flat = tf.reshape(meas_pred, [self.num_particles, -1])
            diff = meas_pred_flat - meas_true
            R_inv_diag = 1.0 / tf.linalg.diag_part(self.ssm.R)
            log_lik = -0.5 * tf.reduce_sum(diff**2 * R_inv_diag, axis=1)
        weights = tf.exp(log_lik - tf.reduce_max(log_lik))
        return weights / (tf.reduce_sum(weights) + 1e-15)

    def _resample(self, particles, weights):
        """Systematic resampling using shared utility."""
        indices = systematic_resample(weights)
        return tf.gather(particles, indices)


class MatrixPFF:
    """
    Matrix-valued (diagonal) kernel Particle Flow Filter.

    Uses paper's diagonal kernel (Eq. 20-23) with D=B preconditioning.
    """

    def __init__(self, ssm, initial_state, initial_covariance, num_particles=500,
                 step_size=0.05, alpha=None, localization_radius=4.0,
                 convergence_tol=1e-5, max_steps=500, show_progress=False):
        self.ssm = ssm
        self.num_particles = num_particles
        self.step_size = step_size
        self.alpha = alpha
        self.localization_radius = localization_radius
        self.convergence_tol = convergence_tol
        self.max_steps = max_steps
        self.show_progress = show_progress

        self.state = tf.constant(initial_state, dtype=tf.float32)
        P0 = tf.constant(initial_covariance, dtype=tf.float32)

        dist = tf.random.normal([num_particles, tf.shape(self.state)[0]], dtype=tf.float32)
        L = tf.linalg.cholesky(P0)
        self.particles = self.state + tf.matmul(dist, L, transpose_b=True)

        self.weights = tf.ones(num_particles, dtype=tf.float32) / self.num_particles
        self.ess_before_resample = float(num_particles)
        self.P = P0

    def predict(self, control):
        """Prediction step: propagate particles through motion model."""
        if hasattr(self.ssm, 'N_s'):
            particles_pred = self.ssm.motion_model(self.particles)
            noise = self.ssm.sample_process_noise((self.num_particles,), use_gen=False)
            noise_flat = tf.reshape(noise, [self.num_particles, -1])
            self.particles = particles_pred + noise_flat
        else:
            control = tf.constant(control, dtype=tf.float32)
            control_batch = tf.tile(tf.reshape(control, [1, -1]), [self.num_particles, 1])
            particles_pred = self.ssm.motion_model(self.particles, control_batch)
            Q = self.ssm.Q
            noise = tf.random.normal([self.num_particles, tf.shape(self.state)[0]], dtype=tf.float32)
            L_Q = tf.linalg.cholesky(Q)
            self.particles = particles_pred + tf.matmul(noise, L_Q, transpose_b=True)

        self.state = tf.reduce_mean(self.particles, axis=0)
        particles_centered = self.particles - self.state
        self.P = tf.matmul(particles_centered, particles_centered, transpose_a=True) / float(self.num_particles)

    def update(self, measurements, landmarks_or_sensors):
        """Update step: assimilate observations via particle flow."""
        import warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings('ignore')

        measurements_flat = _measurements_to_flat(measurements)

        if hasattr(self.ssm, 'N_s'):
            obs_indices = tf.constant([0], dtype=tf.int32)
            obs_noise_std = float(self.ssm.sigma_w) if hasattr(self.ssm, 'sigma_w') else 1.0
        else:
            obs_indices = _obs_indices_for_landmarks(landmarks_or_sensors)
            obs_noise_std = float(tf.sqrt(self.ssm.R[0, 0]))

        prior_mean = tf.reduce_mean(self.particles, axis=0)
        particles_centered = self.particles - prior_mean
        prior_cov = tf.matmul(particles_centered, particles_centered, transpose_a=True) / float(self.num_particles - 1)

        if self.localization_radius > 0:
            d = tf.shape(prior_cov)[0]
            idx = tf.range(d, dtype=tf.float32)
            dist = tf.abs(idx[:, None] - idx[None, :])
            C = tf.exp(-(dist ** 2) / (self.localization_radius ** 2))
            prior_cov = prior_cov * C

        prior_cov = 0.5 * (prior_cov + tf.transpose(prior_cov))
        mean_var = tf.reduce_mean(tf.linalg.diag_part(prior_cov))
        prior_cov = prior_cov + tf.eye(tf.shape(prior_cov)[0], dtype=tf.float32) * (1e-6 * mean_var + 1e-8)

        try:
            particles_new, _, _ = assimilate_pff(
                self.particles,
                measurements_flat,
                obs_indices,
                obs_noise_std,
                prior_mean,
                prior_cov,
                kernel_type='diagonal',
                alpha=self.alpha,
                max_steps=self.max_steps,
                initial_step_size=self.step_size,
                convergence_tol=self.convergence_tol,
                ssm=self.ssm,
                landmarks=tf.convert_to_tensor(landmarks_or_sensors, dtype=tf.float32)
            )
            self.particles = particles_new
        except Exception as e:
            if self.show_progress:
                print("Warning: MatrixPFF update failed, using fallback")
            weights = self._compute_weights(measurements, landmarks_or_sensors)
            ess = 1.0 / (tf.reduce_sum(weights**2) + 1e-15)
            if ess < self.num_particles * 0.5:
                self.particles = self._resample(self.particles, weights)
                self.weights = tf.ones(self.num_particles, dtype=tf.float32) / self.num_particles
        finally:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            warnings.filterwarnings('default')

        self.state = tf.reduce_mean(self.particles, axis=0)
        particles_centered = self.particles - self.state
        self.P = tf.matmul(particles_centered, particles_centered, transpose_a=True) / float(self.num_particles)
        self.ess_before_resample = float(self.num_particles)

    def _compute_weights(self, measurements, landmarks_or_sensors):
        """Compute importance weights (fallback)."""
        if hasattr(self.ssm, 'N_s'):
            meas_pred = self.ssm.measurement_model(self.particles)
            meas_true = tf.constant(measurements, dtype=tf.float32)
            diff = meas_pred - meas_true
            log_lik = -0.5 * tf.reduce_sum(diff**2 / self.ssm.sigma_w**2, axis=1)
        else:
            landmarks_tf = tf.constant(landmarks_or_sensors, dtype=tf.float32)
            meas_pred = self.ssm.measurement_model(self.particles, landmarks_tf)
            meas_true = _measurements_to_flat(measurements)
            meas_pred_flat = tf.reshape(meas_pred, [self.num_particles, -1])
            diff = meas_pred_flat - meas_true
            R_inv_diag = 1.0 / tf.linalg.diag_part(self.ssm.R)
            log_lik = -0.5 * tf.reduce_sum(diff**2 * R_inv_diag, axis=1)
        weights = tf.exp(log_lik - tf.reduce_max(log_lik))
        return weights / (tf.reduce_sum(weights) + 1e-15)

    def _resample(self, particles, weights):
        """Systematic resampling using shared utility."""
        indices = systematic_resample(weights)
        return tf.gather(particles, indices)
