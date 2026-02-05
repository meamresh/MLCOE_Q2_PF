"""
Local Exact Daum-Huang Filter (LEDH) - Algorithm 2 from Daum & Huang.

This module implements the Local EDH particle flow filter, which uses
per-particle linearization for more accurate flow computation in
highly nonlinear systems.

The flow equation for each particle i is: dx_i/dλ = A_i(λ)x_i + b_i(λ)

Key equations (local linearization at each particle):
    A_i(λ) = -½ P H_i^T (λ H_i P H_i^T + R)^{-1} H_i
    b_i(λ) = (I + 2λA_i)[(I + λA_i) P H_i^T R^{-1} (z - e_i) + A_i x̄]
    
where:
    - P: Common prior covariance (from EKF/UKF)
    - H_i: Local measurement Jacobian at particle i
    - R: Measurement noise covariance
    - z: Measurement vector  
    - e_i: Local linearization offset h(x_i) - H_i @ x_i

LEDH vs EDH:
    - EDH: Global linearization (all particles share same A, b)
    - LEDH: Local linearization (each particle has its own A_i, b_i)
    
LEDH is more accurate for non-Gaussian posteriors but computationally
more expensive (O(N) matrix operations vs O(1) for EDH).

All per-particle computations are vectorized using tf.einsum for efficiency.

References
----------
- Daum, F., & Huang, J. (2008, 2011). Particle flow papers.
- Li, Y., & Coates, M. (2017). "Particle Filtering with Invertible Particle Flow."

"""

from __future__ import annotations

import tensorflow as tf
from typing import Literal, Optional

from src.utils.linalg import regularize_covariance, sample_from_gaussian

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False




class LEDH:
    """Corrected Local Exact Flow Filter with proper λ-dependence and vectorization."""

    def __init__(
        self,
        ssm,
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor,
        num_particles: int = 1000,
        n_lambda: int = 50,
        filter_type: Literal['ekf', 'ukf'] = 'ukf',
        ukf_alpha: float = 0.01,
        show_progress: bool = False,
        redraw_particles: bool = False
    ):
        self.ssm = ssm
        self.num_particles = num_particles
        self.n_lambda = n_lambda
        # Paper uses exponentially spaced steps with q=1.2 (same as PFPF filters)
        # ε₁ = (1-q)/(1-q^Nλ) ≈ 0.001, εⱼ = q * εⱼ₋₁
        self.q = 1.2
        self.epsilon_1 = (1.0 - self.q) / (1.0 - self.q ** self.n_lambda)
        self.filter_type = filter_type
        self.ukf_alpha = ukf_alpha
        self.state_dim = ssm.state_dim
        self.show_progress = show_progress and HAS_TQDM
        self.redraw_particles = redraw_particles
        self.step_count = 0

        # Initialize
        initial_state = tf.cast(initial_state, tf.float32)
        initial_covariance = tf.cast(initial_covariance, tf.float32)
        initial_covariance = self._regularize_covariance(initial_covariance)

        # Sample particles using shared utility
        self.particles = sample_from_gaussian(
            initial_state, initial_covariance, num_particles
        )

        # Per-particle covariances for LEDH (local linearization)
        # NOTE: This is a variant from the paper - paper uses common P_pred from EKF/UKF
        # This variant provides better local fidelity but uses more memory/compute
        # and requires careful regularization to avoid numerical issues
        self.particle_covs = tf.Variable(
            tf.tile(initial_covariance[tf.newaxis, :, :], [num_particles, 1, 1]),
            trainable=False,
            dtype=tf.float32
        )

        self.x_hat = initial_state
        self.state = initial_state
        self.m = initial_state
        self.P = initial_covariance

        # Cache for Cholesky decompositions
        self._L_Q = None

        self._create_base_filter()

    def _regularize_covariance(self, P: tf.Tensor) -> tf.Tensor:
        """Fast covariance regularization using shared utility."""
        return regularize_covariance(P, eps=1e-6)

    def _create_base_filter(self):
        """Create EKF or UKF."""
        from .ekf import ExtendedKalmanFilter
        from .ukf import UnscentedKalmanFilter

        if self.filter_type == 'ekf':
            self.base_filter = ExtendedKalmanFilter(self.ssm, self.m, self.P)
        else:
            self.base_filter = UnscentedKalmanFilter(
                self.ssm, self.m, self.P, alpha=self.ukf_alpha
            )

    def _compute_particle_mean(self) -> tf.Tensor:
        """Calculate particle mean."""
        return tf.reduce_mean(self.particles, axis=0)

    def _compute_local_flow_matrices_vectorized(
        self,
        particles: tf.Tensor,
        x_bar: tf.Tensor,
        P_pred: tf.Tensor,  # Common P_pred from EKF/UKF (paper's formulation)
        measurement: tf.Tensor,
        lam: tf.Tensor,
        landmarks: Optional[tf.Tensor] = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        CORRECTED: Compute A_i(λ) and b_i(λ) for all particles with proper λ-dependence.

        Equations for each particle i:
        A_i(λ) = -½ P H_i^T (λ H_i P H_i^T + R)^{-1} H_i
        b_i(λ) = (I + 2λA_i)[(I + λA_i) P H_i^T R^{-1} z + A_i x̄]

        Local linearization at each particle x^i using common P_pred (paper's formulation).
        Note: H_i differs per particle (local), but P is common (from EKF/UKF).

        Returns
        -------
        A_batch : tf.Tensor
            Shape [N, state_dim, state_dim].
        b_batch : tf.Tensor
            Shape [N, state_dim].
        """
        N = tf.shape(particles)[0]

        # Check for empty landmarks
        if landmarks is not None:
            num_landmarks = tf.shape(landmarks)[0]
            if num_landmarks == 0:
                # No landmarks - return zero matrices
                A_batch = tf.zeros([N, self.state_dim, self.state_dim], dtype=tf.float32)
                b_batch = tf.zeros([N, self.state_dim], dtype=tf.float32)
                return A_batch, b_batch

        # Batch compute Jacobians for all particles (local linearization)
        if landmarks is not None:
            H_batch = self.ssm.measurement_jacobian(particles, landmarks)
            gamma_batch = self.ssm.measurement_model(particles, landmarks)
            # Handle different measurement formats:
            # - Range-bearing: [N, num_landmarks, 2] -> [N, 2*num_landmarks]
            # - Acoustic: [N, N_s] (already flat)
            if hasattr(self.ssm, 'meas_per_landmark') and self.ssm.meas_per_landmark == 2:
                gamma_batch = tf.reshape(gamma_batch, [tf.shape(gamma_batch)[0], -1])
        else:
            H_batch = self.ssm.measurement_jacobian(particles)
            gamma_batch = self.ssm.measurement_model(particles)

        # Flatten measurement
        z_k = tf.reshape(measurement, [-1])
        meas_dim = tf.shape(z_k)[0]

        # Compute residuals for all particles: [N, meas_dim]
        residual_batch = z_k[tf.newaxis, :] - gamma_batch

        # Handle bearing wrapping for range-bearing SSM only (fully tensor-based)
        if landmarks is not None and hasattr(self.ssm, 'meas_per_landmark') and self.ssm.meas_per_landmark == 2:
            meas_dim_local = tf.shape(residual_batch)[1]
            is_even = tf.equal(meas_dim_local % 2, 0)
            is_valid = tf.greater_equal(meas_dim_local, 2)
            should_wrap = tf.logical_and(is_even, is_valid)

            def do_wrap():
                bearing_indices = tf.range(1, meas_dim_local, 2)
                bearing_res = tf.gather(residual_batch, bearing_indices, axis=1)
                bearing_wrapped = tf.math.atan2(tf.sin(bearing_res), tf.cos(bearing_res))

                N_val = tf.shape(residual_batch)[0]
                num_bearings = tf.shape(bearing_indices)[0]
                indices = tf.stack([
                    tf.repeat(tf.range(N_val), num_bearings),
                    tf.tile(bearing_indices, [N_val])
                ], axis=1)
                return tf.tensor_scatter_nd_update(
                    residual_batch, indices, tf.reshape(bearing_wrapped, [-1])
                )

            residual_batch = tf.cond(should_wrap, do_wrap, lambda: residual_batch)

        # Get measurement covariance
        R = self.ssm.R
        if landmarks is not None:
            num_landmarks = tf.shape(landmarks)[0]
            R = self.ssm.full_measurement_cov(num_landmarks)
        elif len(R.shape) == 0 or R.shape[0] != meas_dim:
            R = tf.eye(meas_dim, dtype=tf.float32) * R

        # Regularize R
        R_reg = R + 1e-6 * tf.eye(meas_dim, dtype=R.dtype)
        cholR = tf.linalg.cholesky(R_reg)
        R_inv = tf.linalg.cholesky_solve(cholR, tf.eye(meas_dim, dtype=R_reg.dtype))

        # Compute e_lambda for each particle: e_i = h(x^i) - H_i @ x^i
        H_particles = tf.einsum('nij,nj->ni', H_batch, particles)
        e_lambda_batch = gamma_batch - H_particles

        # Compute z - e for each particle: z_minus_e_i = z_k - e_i
        z_minus_e_batch = z_k[tf.newaxis, :] - e_lambda_batch

        # Vectorized computation of A_i(λ) for all particles using COMMON P_pred
        H_T_batch = tf.transpose(H_batch, [0, 2, 1])
        HPHt_batch = tf.einsum('nij,jk,nkl->nil', H_batch, P_pred, H_T_batch)

        # For each particle: S_i(λ) = λ H_i P H_i^T + R
        inner = lam * HPHt_batch + R_reg[tf.newaxis, :, :]

        # Symmetrize for numerical stability
        inner = 0.5 * (inner + tf.transpose(inner, [0, 2, 1]))

        # Adaptive regularization per particle (check eigenvalues)
        eigs = tf.linalg.eigvalsh(inner)
        min_eigs = tf.reduce_min(eigs, axis=1)
        base_eps = tf.constant(1e-6, dtype=inner.dtype)
        needed = tf.nn.relu(base_eps - min_eigs)
        safety = tf.constant(1e-2, dtype=inner.dtype)
        reg_per = tf.reshape(tf.maximum(base_eps, needed * (1.0 + safety)), [-1, 1, 1])

        inner_reg = inner + reg_per * tf.eye(meas_dim, dtype=inner.dtype)[tf.newaxis, :, :]

        # Batched Cholesky and solve
        chol = tf.linalg.cholesky(inner_reg)
        inner_inv_H = tf.linalg.cholesky_solve(chol, H_batch)

        # Compute M = H_i^T @ inner_inv_H
        M = tf.einsum('nij,njk->nik', H_T_batch, inner_inv_H)

        # A_batch = -0.5 * P @ M (using common P_pred)
        A_batch = -0.5 * tf.einsum('ij,njk->nik', P_pred, M)

        # Equation: b_i = (I + 2λA_i)[(I + λA_i) P H_i^T R^{-1} (z - e_i) + A_i x̄]
        I = tf.eye(self.state_dim, dtype=tf.float32)

        # Compute (I + λA_i) for all particles
        I_plus_lambda_A = I[tf.newaxis, :, :] + lam * A_batch

        # Compute P H_i^T R^{-1} (z - e_i) for all particles (using common P_pred)
        PHt_batch = tf.einsum('ij,njk->nik', P_pred, H_T_batch)
        z_minus_e_col = z_minus_e_batch[:, :, tf.newaxis]
        Rinv_z_minus_e_batch = tf.linalg.cholesky_solve(cholR, z_minus_e_col)
        Rinv_z_minus_e_batch = tf.squeeze(Rinv_z_minus_e_batch, axis=2)
        PHt_Rinv_z_minus_e = tf.einsum('nij,nj->ni', PHt_batch, Rinv_z_minus_e_batch)

        # First term: (I + λA_i) P H_i^T R^{-1} (z - e_i)
        term1 = tf.einsum('nij,nj->ni', I_plus_lambda_A, PHt_Rinv_z_minus_e)

        # Second term: A_i x̄ (use predicted mean, not current mean)
        term2 = tf.einsum('nij,j->ni', A_batch, x_bar)

        # Combine: (I + 2λA_i)[term1 + term2]
        I_plus_2lambda_A = I[tf.newaxis, :, :] + 2.0 * lam * A_batch
        b_batch = tf.einsum('nij,nj->ni', I_plus_2lambda_A, term1 + term2)

        return A_batch, b_batch

    def predict(self, control: Optional[tf.Tensor] = None) -> tuple[tf.Tensor, tf.Tensor]:
        """Vectorized prediction step."""
        self.step_count += 1
        N = self.num_particles

        # Propagate all particles at once
        if control is not None:
            control = tf.cast(control, tf.float32)
            control_batch = tf.tile(tf.reshape(control, [1, -1]), [N, 1])
            particles_pred = self.ssm.motion_model(self.particles, control_batch)
        else:
            particles_pred = self.ssm.motion_model(self.particles)

        # Add process noise - cache Cholesky decomposition
        if self._L_Q is None:
            try:
                self._L_Q = tf.linalg.cholesky(self.ssm.Q)
            except Exception:
                Q_reg = self.ssm.Q + 1e-6 * tf.eye(self.state_dim, dtype=self.ssm.Q.dtype)
                self._L_Q = tf.linalg.cholesky(Q_reg)

        # Add process noise to particles
        noise = tf.random.normal([N, self.state_dim], dtype=tf.float32)
        process_noise = noise @ tf.transpose(self._L_Q)
        self.particles = particles_pred + process_noise

        self.x_bar = self._compute_particle_mean()

        # EKF/UKF prediction - use paper's formulation with EKF/UKF P_pred
        try:
            if control is not None:
                m_pred, P_pred = self.base_filter.predict(control)
            else:
                m_pred, P_pred = self.base_filter.predict()

            self.m_pred = m_pred
            self.P_pred = self._regularize_covariance(P_pred)
        except Exception:
            self.m_pred = self.x_bar
            # Fallback: use particle covariance if EKF/UKF fails
            particles_centered = self.particles - self.x_bar[tf.newaxis, :]
            P_particle = tf.reduce_mean(
                particles_centered[:, :, tf.newaxis] * particles_centered[:, tf.newaxis, :],
                axis=0
            )
            self.P_pred = self._regularize_covariance(P_particle + self.ssm.Q)

        self.x_hat = self.x_bar
        self.state = self.x_hat
        return self.x_hat, self.P_pred

    def update(
        self,
        measurement: tf.Tensor,
        landmarks: Optional[tf.Tensor] = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        CORRECTED: Vectorized homotopy migration with proper λ-dependent flow.

        All particles processed in parallel with local linearization.
        """
        measurement = tf.cast(measurement, tf.float32)
        if landmarks is not None:
            landmarks = tf.cast(landmarks, tf.float32)

            # Check if landmarks is empty
            num_landmarks = tf.shape(landmarks)[0]
            if num_landmarks == 0 or tf.size(measurement) == 0:
                # No landmarks visible - skip update
                self.x_hat = self.x_bar
                self.state = self.x_hat
                self.m = self.x_bar
                self.P = self.P_pred
                return self.x_hat, self.P

        # Compute epsilon steps for exponential spacing (same as PFPF filters)
        epsilon_steps = []
        for j in range(1, self.n_lambda + 1):
            epsilon_j = self.epsilon_1 * (self.q ** (j - 1))
            epsilon_steps.append(epsilon_j)

        lambda_iterator = range(self.n_lambda)
        if self.show_progress:
            lambda_iterator = tqdm(
                lambda_iterator,
                desc=f"  LEDH λ-steps (t={self.step_count})",
                leave=False,
                ncols=80
            )

        # Homotopy loop - only loop over lambda, not particles
        x_bar_pred = self.m_pred

        lambda_cumulative = 0.0
        for j in lambda_iterator:
            epsilon_j = epsilon_steps[j]
            lam = tf.constant(lambda_cumulative + epsilon_j / 2.0, dtype=tf.float32)
            lambda_cumulative += epsilon_j

            # Vectorized computation for ALL particles at once (local linearization)
            A_batch, b_batch = self._compute_local_flow_matrices_vectorized(
                self.particles, x_bar_pred, self.P_pred, measurement, lam, landmarks
            )

            # Vectorized migration: dx/dλ = A_i @ x^i + b_i for all particles
            dx_dlambda = tf.einsum('nij,nj->ni', A_batch, self.particles) + b_batch

            # Update all particles at once
            self.particles = self.particles + epsilon_j * dx_dlambda

            # Update mean
            self.x_bar = self._compute_particle_mean()

        # After flow, particles represent posterior
        self.x_hat = self._compute_particle_mean()

        # EKF/UKF update (for covariance refinement, but use post-flow mean as state)
        try:
            self.base_filter.state.assign(self.x_hat)
            self.base_filter.covariance.assign(self.P_pred)

            if landmarks is not None:
                m_updated, P_updated, _ = self.base_filter.update(measurement, landmarks)
            else:
                m_updated, P_updated, _ = self.base_filter.update(measurement)

            self.P = self._regularize_covariance(P_updated)
            self.m = m_updated
        except Exception:
            # Fallback: use particle covariance
            particles_centered = self.particles - self.x_hat[tf.newaxis, :]
            P_particle = tf.reduce_mean(
                particles_centered[:, :, tf.newaxis] * particles_centered[:, tf.newaxis, :],
                axis=0
            )
            self.P = self._regularize_covariance(P_particle)
            self.m = self.x_hat

        # Final state is particle mean (post-flow)
        self.state = self.x_hat
        self.m = self.x_hat

        if self.redraw_particles:
            self.particles = sample_from_gaussian(
                self.x_hat, self.P, self.num_particles
            )

        self._create_base_filter()

        return self.x_hat, self.P
