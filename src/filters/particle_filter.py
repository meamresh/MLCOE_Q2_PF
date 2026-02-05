"""
Particle Filter for nonlinear state estimation.

This module implements the Bootstrap Particle Filter (BPF) using
Sequential Importance Resampling (SIR) for nonlinear state-space models.

Algorithm Overview
------------------
The particle filter approximates the posterior distribution p(x_k|y_{1:k})
using a weighted set of samples (particles):

    p(x_k|y_{1:k}) ≈ Σᵢ wᵢ δ(x - xᵢ)

Each time step:
1. Prediction: Propagate particles through motion model + process noise
   xᵢ_k ~ p(x_k|x_{k-1}, u_k)

2. Update: Compute importance weights from likelihood
   wᵢ ∝ p(y_k|xᵢ_k)

3. Resample: When ESS < threshold, resample to avoid weight degeneracy

Key equations:
    Log-likelihood: log p(y|x) = -0.5 * [(y-h(x))ᵀR⁻¹(y-h(x)) + log|R| + d*log(2π)]
    ESS: N_eff = 1 / Σᵢ(wᵢ)²
    
References
----------
- Doucet, A., de Freitas, N., & Gordon, N. (2001). Sequential Monte Carlo Methods
- Arulampalam, M. S., et al. (2002). A tutorial on particle filters

"""

from __future__ import annotations

import tensorflow as tf

from src.metrics.particle_filter_metrics import compute_effective_sample_size
from src.utils.linalg import regularize_covariance, sample_from_gaussian
from src.filters.resampling import systematic_resample, compute_ess


# =============================================================================
# JIT-compiled weight computation for performance
# =============================================================================


class ParticleFilter:
    """
    Particle Filter for nonlinear state estimation.
    Uses Sequential Importance Resampling (SIR) / Bootstrap Filter.
    
    Parameters
    ----------
    ssm : RangeBearingSSM
        State space model with motion_model and measurement_model methods.
    initial_state : tf.Tensor
        Initial state estimate of shape (state_dim,).
    initial_covariance : tf.Tensor
        Initial uncertainty matrix of shape (state_dim, state_dim).
    num_particles : int, optional
        Number of particles. Defaults to 3000.
    resample_threshold : float, optional
        Effective sample size threshold for resampling (0-1). Defaults to 0.3.
    
    Attributes
    ----------
    ssm : RangeBearingSSM
        State-space model.
    num_particles : int
        Number of particles.
    resample_threshold : float
        Resampling threshold.
    state_dim : int
        State dimension.
    particles : tf.Tensor
        Particle states of shape (num_particles, state_dim).
    weights : tf.Tensor
        Particle weights of shape (num_particles,).
    state : tf.Tensor
        Current state estimate (weighted mean).
    covariance : tf.Tensor
        Current covariance estimate (weighted covariance).
    """

    def __init__(self, ssm, initial_state: tf.Tensor, initial_covariance: tf.Tensor,
                 num_particles: int = 3000, resample_threshold: float = 0.3):
        """
        Initialize the Particle Filter.

        Args:
            ssm: State space model (RangeBearingSSM)
            initial_state: Initial state estimate [state_dim]
            initial_covariance: Initial uncertainty [state_dim, state_dim]
            num_particles: Number of particles
            resample_threshold: Effective sample size threshold for resampling (0-1)
        """
        self.ssm = ssm
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self.state_dim = ssm.state_dim

        # Initialize particles around initial state
        initial_state = tf.cast(initial_state, tf.float32)
        initial_covariance = tf.cast(initial_covariance, tf.float32)

        # Sample particles from initial distribution
        self.particles = self._sample_particles_from_gaussian(
            initial_state, initial_covariance, num_particles
        )

        # Initialize uniform weights
        self.weights = tf.ones(num_particles, dtype=tf.float32) / tf.cast(num_particles, tf.float32)

        # State estimate (weighted mean)
        self.state = self._compute_state_estimate()
        self.covariance = self._compute_covariance_estimate()

        # ESS before resampling (for diagnostics / plotting)
        self.ess_before_resample = tf.cast(num_particles, tf.float32)

    def _sample_particles_from_gaussian(self, mean: tf.Tensor, covariance: tf.Tensor,
                                        n_samples: int) -> tf.Tensor:
        """Sample n particles from multivariate Gaussian using shared utility."""
        return sample_from_gaussian(mean, covariance, n_samples)

    def _compute_state_estimate(self) -> tf.Tensor:
        """Compute weighted mean of particles."""
        weights_expanded = tf.expand_dims(self.weights, axis=1)
        state_estimate = tf.reduce_sum(weights_expanded * self.particles, axis=0)
        return state_estimate

    def _compute_covariance_estimate(self) -> tf.Tensor:
        """Compute weighted covariance of particles."""
        state_mean = self._compute_state_estimate()
        diff = self.particles - state_mean

        weights_expanded = tf.expand_dims(self.weights, axis=1)
        weighted_diff = weights_expanded * diff

        covariance = tf.matmul(weighted_diff, diff, transpose_a=True)

        # Use shared regularization utility
        return regularize_covariance(covariance, eps=1e-6)

    def _effective_sample_size(self) -> tf.Tensor:
        """
        Compute effective sample size (ESS) for resampling decision.

        Uses the standard formula ESS = 1 / sum(w_i^2) on normalized weights
        via the shared metrics utility, then clamps to [1, N] for numerical
        robustness, matching the typical particle filtering literature.
        """
        ess = compute_effective_sample_size(self.weights)
        num_p = tf.cast(self.num_particles, tf.float32)
        ess = tf.clip_by_value(ess, 1.0, num_p)
        return ess
    
    def _systematic_resample(self) -> None:
        """Systematic resampling using shared utility."""
        indices = systematic_resample(self.weights)
        self.particles = tf.gather(self.particles, indices)
        # Reset weights to uniform
        N = self.num_particles
        self.weights = tf.ones(N, dtype=tf.float32) / tf.cast(N, tf.float32)
    
    def predict(self, control: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Particle Filter Prediction Step.

        Parameters
        ----------
        control : tf.Tensor
            Control input of shape (2,).

        Returns
        -------
        state_pred : tf.Tensor
            Predicted state estimate of shape (state_dim,).
        covariance_pred : tf.Tensor
            Predicted covariance estimate of shape (state_dim, state_dim).
        """
        control = tf.cast(control, tf.float32)

        # Propagate each particle through motion model with process noise
        N = self.num_particles
        control_tiled = tf.tile(tf.reshape(control, [1, -1]), [N, 1])

        # Deterministic propagation
        particles_pred = self.ssm.motion_model(self.particles, control_tiled)

        # Add process noise
        # Sample from process noise distribution
        L_Q = tf.linalg.cholesky(self.ssm.Q)
        noise_samples = tf.random.normal([N, self.state_dim], dtype=tf.float32)
        process_noise = tf.linalg.matvec(L_Q, noise_samples, transpose_a=True)

        self.particles = particles_pred + process_noise

        # Update state estimate
        self.state = self._compute_state_estimate()
        self.covariance = self._compute_covariance_estimate()

        return self.state, self.covariance
 
    def update(self, measurement: tf.Tensor, landmarks: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, bool]:
        """
        Particle Filter Update Step (Weight Update) - VECTORIZED & OPTIMIZED.

        Parameters
        ----------
        measurement : tf.Tensor
            Measurement vector of shape (num_landmarks, 2) or flattened.
        landmarks : tf.Tensor
            Landmark positions of shape (num_landmarks, 2).

        Returns
        -------
        state_updated : tf.Tensor
            Updated state estimate of shape (state_dim,).
        covariance_updated : tf.Tensor
            Updated covariance estimate of shape (state_dim, state_dim).
        residual : tf.Tensor
            Mean residual (for visualization).
        did_resample : bool
            Whether resampling occurred.
        """
        landmarks = tf.cast(landmarks, tf.float32)
        measurement = tf.cast(measurement, tf.float32)
        num_landmarks = tf.shape(landmarks)[0]

        # 1. Predict measurements for all particles [N, M, 2]
        measurements_pred = self.ssm.measurement_model(self.particles, landmarks)
        
        # 2. Flatten measurements to [N, 2*M]
        measurements_pred_flat = tf.reshape(measurements_pred, [self.num_particles, -1])
        measurement_flat = tf.reshape(measurement, [-1])

        # 3. Compute residuals [N, 2*M]
        residuals = measurement_flat - measurements_pred_flat

        # 4. Wrap bearing residuals (Odd indices: 1, 3, 5...) - only if range-bearing format
        # Only wrap bearings if meas_per_landmark == 2 (range-bearing format)
        if hasattr(self.ssm, "meas_per_landmark") and self.ssm.meas_per_landmark == 2:
            # Create a mask for bearing indices
            bearing_indices = tf.range(1, 2 * num_landmarks, 2, dtype=tf.int32)

            # Extract bearings, wrap them, and put them back
            bearings = tf.gather(residuals, bearing_indices, axis=1)
            wrapped_bearings = tf.math.atan2(tf.sin(bearings), tf.cos(bearings))

            # Construct mask: 0 for range, 1 for bearing
            mask = tf.scatter_nd(
                tf.expand_dims(bearing_indices, 1),
                tf.ones_like(bearing_indices, dtype=tf.float32),
                [2 * num_landmarks],
            )
            mask = tf.expand_dims(mask, 0)  # Broadcast to [1, 2*M]

            # Reconstruct residuals: keep range as is, use wrapped bearings
            residuals_wrapped = residuals * (1.0 - mask) + tf.scatter_nd(
                tf.stack(
                    [
                        tf.tile(tf.range(self.num_particles)[:, None], [1, num_landmarks]),
                        tf.tile(bearing_indices[None, :], [self.num_particles, 1]),
                    ],
                    axis=2,
                ),
                wrapped_bearings,
                [self.num_particles, 2 * num_landmarks],
            )
        else:
            # No bearings to wrap (e.g., acoustic measurements)
            residuals_wrapped = residuals

        # 5. Compute Likelihoods (Vectorized Mahalanobis Distance)
        R_full = self.ssm.full_measurement_cov(num_landmarks)
        # Add slight regularization to R inverse for stability
        eye_dim = tf.shape(R_full)[0]
        R_inv = tf.linalg.inv(R_full + 1e-6 * tf.eye(eye_dim, dtype=R_full.dtype))

        # Vectorized calculation: (x-u)^T S^-1 (x-u)
        # [N, 2M] @ [2M, 2M] -> [N, 2M]
        weighted_residuals = tf.matmul(residuals_wrapped, R_inv) 
        # Row-wise dot product: sum( [N, 2M] * [N, 2M], axis=1 ) -> [N]
        mahalanobis_dist = tf.reduce_sum(weighted_residuals * residuals_wrapped, axis=1)

        # Log-weights (include normalization constant for correct likelihood)
        # log p(z|x) = -0.5 * [(z-h(x))^T R^{-1} (z-h(x)) + log|R| + d*log(2π)]
        # Since we normalize weights, the constant terms cancel, but we include
        # log|R| for completeness (though it doesn't affect ESS after normalization)
        log_det_R = tf.linalg.slogdet(R_full)[1]
        meas_dim = tf.cast(tf.shape(R_full)[0], tf.float32)
        log_weights = -0.5 * (
            mahalanobis_dist + log_det_R + meas_dim * tf.math.log(2.0 * 3.141592653589793)
        )

        # 6. Normalize Weights (Log-Sum-Exp Trick for Stability)
        max_log_weight = tf.reduce_max(log_weights)
        # Subtract max to avoid overflow/underflow
        weights_unnormalized = tf.exp(log_weights - max_log_weight)
        
        weights_sum = tf.reduce_sum(weights_unnormalized)
        
        # Safe division
        self.weights = tf.math.divide_no_nan(weights_unnormalized, weights_sum)
        
        # Fallback if weights collapse (e.g. outlier measurement)
        if weights_sum < 1e-10:
             self.weights = tf.ones(self.num_particles, dtype=tf.float32) / tf.cast(self.num_particles, tf.float32)

        # 7. Resample
        did_resample = False

        ess = self._effective_sample_size()
        self.ess_before_resample = ess  # Store ESS before resampling
        ess_threshold = self.resample_threshold * tf.cast(self.num_particles, tf.float32)

        if ess < ess_threshold:
            self._systematic_resample()
            # Add small random noise to prevent particle deprivation
            small_noise = tf.random.normal(self.particles.shape, stddev=0.01)
            self.particles += small_noise
            did_resample = True

        # 8. Update Estimates
        self.state = self._compute_state_estimate()
        self.covariance = self._compute_covariance_estimate()
        
        # Return residual of the mean state (for plotting consistency)
        # Note: This is just for visualization, doesn't affect the filter
        mean_residual = tf.reduce_mean(residuals_wrapped, axis=0)
        return self.state, self.covariance, mean_residual, did_resample

