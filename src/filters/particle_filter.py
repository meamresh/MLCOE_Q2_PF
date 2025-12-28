"""
Particle Filter for nonlinear state estimation.

This module implements the Particle Filter using Sequential Importance Resampling (SIR)
/ Bootstrap Filter for nonlinear state-space models.
"""

from __future__ import annotations

import tensorflow as tf

from src.metrics.particle_filter_metrics import compute_effective_sample_size


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

    def _sample_particles_from_gaussian(self, mean: tf.Tensor, covariance: tf.Tensor,
                                        n_samples: int) -> tf.Tensor:
        """Sample n particles from multivariate Gaussian."""
        # Use Cholesky decomposition for sampling
        try:
            L = tf.linalg.cholesky(covariance)
        except:
            # If Cholesky fails, add regularization
            covariance = covariance + 1e-4 * tf.eye(self.state_dim, dtype=covariance.dtype)
            L = tf.linalg.cholesky(covariance)

        # Standard normal samples
        standard_samples = tf.random.normal([n_samples, self.state_dim], dtype=tf.float32)

        # Transform to desired distribution
        samples = mean + tf.linalg.matvec(L, standard_samples, transpose_a=True)

        return samples

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

        # Ensure symmetry and add small regularization
        covariance = 0.5 * (covariance + tf.transpose(covariance))
        covariance = covariance + 1e-6 * tf.eye(self.state_dim, dtype=covariance.dtype)

        return covariance

    def _effective_sample_size(self) -> tf.Tensor:
        """Compute effective sample size (ESS) for resampling decision."""
        return compute_effective_sample_size(self.weights)
    
    def _systematic_resample(self) -> None:
        """Systematic resampling (lower variance than multinomial)."""
        N = self.num_particles

        # Compute cumulative sum of weights
        cumsum = tf.cumsum(self.weights)

        # Generate systematic samples
        u = tf.random.uniform([], dtype=tf.float32) / tf.cast(N, tf.float32)
        positions = u + tf.cast(tf.range(N), tf.float32) / tf.cast(N, tf.float32)

        # Find indices using searchsorted
        indices = tf.searchsorted(cumsum, positions, side='right')
        indices = tf.minimum(indices, N - 1)

        # Resample particles
        self.particles = tf.gather(self.particles, indices)

        # Reset weights to uniform
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

        # 4. Wrap bearing residuals (Odd indices: 1, 3, 5...)
        # Create a mask for bearing indices
        bearing_indices = tf.range(1, 2 * num_landmarks, 2, dtype=tf.int32)
        
        # Extract bearings, wrap them, and put them back
        bearings = tf.gather(residuals, bearing_indices, axis=1)
        wrapped_bearings = tf.math.atan2(tf.sin(bearings), tf.cos(bearings))
        
        # Reconstruct residuals with wrapped bearings
        # Use a simpler approach: create a new tensor and update bearing positions
        residuals_wrapped = tf.identity(residuals)
        
        # Update bearing positions using a loop-free approach
        # Create indices for scatter_nd_update: [N, num_landmarks, 2]
        particle_indices = tf.range(self.num_particles, dtype=tf.int32)
        particle_indices_expanded = tf.expand_dims(particle_indices, 1)  # [N, 1]
        bearing_indices_expanded = tf.expand_dims(bearing_indices, 0)   # [1, num_landmarks]
        
        # Create index pairs: [N, num_landmarks, 2]
        indices = tf.stack([
            tf.tile(particle_indices_expanded, [1, num_landmarks]),
            tf.tile(bearing_indices_expanded, [self.num_particles, 1])
        ], axis=2)
        
        # Flatten indices and values for scatter_nd_update
        indices_flat = tf.reshape(indices, [-1, 2])
        wrapped_bearings_flat = tf.reshape(wrapped_bearings, [-1])
        
        # Update residuals with wrapped bearings
        residuals_wrapped = tf.tensor_scatter_nd_update(
            residuals_wrapped,
            indices_flat,
            wrapped_bearings_flat
        )

        # 5. Compute Likelihoods (Vectorized Mahalanobis Distance)
        R_full = self.ssm.full_measurement_cov(num_landmarks)
        # Add slight regularization to R inverse for stability
        R_inv = tf.linalg.inv(R_full + 1e-6 * tf.eye(2 * num_landmarks, dtype=R_full.dtype))

        # Vectorized calculation: (x-u)^T S^-1 (x-u)
        # [N, 2M] @ [2M, 2M] -> [N, 2M]
        weighted_residuals = tf.matmul(residuals_wrapped, R_inv) 
        # Row-wise dot product: sum( [N, 2M] * [N, 2M], axis=1 ) -> [N]
        mahalanobis_dist = tf.reduce_sum(weighted_residuals * residuals_wrapped, axis=1)

        # Log-weights
        log_weights = -0.5 * mahalanobis_dist

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
        ess_threshold = self.resample_threshold * tf.cast(self.num_particles, tf.float32)

        if ess < ess_threshold:
            self._systematic_resample()
            # Add small random noise to prevent particle deprivation
            small_noise = tf.random.normal(tf.shape(self.particles), stddev=0.01, dtype=tf.float32)
            self.particles = self.particles + small_noise
            did_resample = True

        # 8. Update Estimates
        self.state = self._compute_state_estimate()
        self.covariance = self._compute_covariance_estimate()
        
        # Return residual of the mean state (for plotting consistency)
        # Note: This is just for visualization, doesn't affect the filter
        mean_residual = tf.reduce_mean(residuals_wrapped, axis=0)
        return self.state, self.covariance, mean_residual, did_resample

