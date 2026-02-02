"""
Corrected Algorithm 1: Exact Flow Daum-Huang Filter (EDH).

Key corrections:
- Proper λ-dependence in A matrix: A = -½ P H^T (λ H P H^T + R)^{-1} H
- Proper b vector: b = (I + 2λA)[(I + λA) P H^T R^{-1} z + A x̄]
- Handle empty landmarks gracefully
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


class EDH:
    """Corrected Exact Flow Filter with proper λ-dependence."""

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

    def _compute_flow_matrices(
        self,
        x_bar: tf.Tensor,
        P_pred: tf.Tensor,
        measurement: tf.Tensor,
        lam: tf.Tensor,
        landmarks: Optional[tf.Tensor] = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        CORRECTED: Calculate A and b with proper λ-dependence.

        Equations (8)-(9) from Daum-Huang paper:
        A = -½ P H^T (λ H P H^T + R)^{-1} H
        b = (I + 2λA)[(I + λA) P H^T R^{-1} z + A x̄]

        Global linearization at x̄_k (particle mean).
        """
        # Calculate H and gamma at particle mean
        if landmarks is not None:
            # Check if landmarks is empty
            num_landmarks = tf.shape(landmarks)[0]
            if num_landmarks == 0:
                # No landmarks visible - return zero matrices
                A = tf.zeros([self.state_dim, self.state_dim], dtype=tf.float32)
                b = tf.zeros([self.state_dim], dtype=tf.float32)
                return A, b

            H = self.ssm.measurement_jacobian(x_bar[tf.newaxis, :], landmarks)[0]
            gamma_x_bar = self.ssm.measurement_model(x_bar[tf.newaxis, :], landmarks)[0]
            # Handle different measurement formats:
            # - Range-bearing: [num_landmarks, 2] -> [2*num_landmarks]
            # - Acoustic: [N_s] (already flat)
            if hasattr(self.ssm, 'meas_per_landmark') and self.ssm.meas_per_landmark == 2:
                gamma_x_bar = tf.reshape(gamma_x_bar, [-1])
        else:
            H = self.ssm.measurement_jacobian(x_bar[tf.newaxis, :])[0]
            gamma_x_bar = self.ssm.measurement_model(x_bar[tf.newaxis, :])[0]

        # Compute residual
        z_k = tf.reshape(measurement, [-1])
        residual = z_k - gamma_x_bar

        # Handle bearing wrapping for range-bearing SSM only (fully tensor-based)
        meas_dim = tf.shape(z_k)[0]
        if landmarks is not None and hasattr(self.ssm, 'meas_per_landmark') and self.ssm.meas_per_landmark == 2:
            def wrap_bearings_tensor():
                is_even = tf.equal(meas_dim % 2, 0)
                is_valid = tf.greater_equal(meas_dim, 2)
                should_wrap = tf.logical_and(is_even, is_valid)

                def do_wrap():
                    bearing_indices = tf.range(1, meas_dim, 2)
                    bearing_res = tf.gather(residual, bearing_indices)
                    bearing_wrapped = tf.math.atan2(tf.sin(bearing_res), tf.cos(bearing_res))
                    return tf.tensor_scatter_nd_update(
                        residual, tf.expand_dims(bearing_indices, 1), bearing_wrapped
                    )

                return tf.cond(should_wrap, do_wrap, lambda: residual)

            residual = wrap_bearings_tensor()

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

        # Compute e_lambda = h(x̄) - H @ x̄ (linearization error at linearization point)
        H_x_bar = tf.linalg.matvec(H, x_bar)
        e_lambda = gamma_x_bar - H_x_bar

        # Compute z - e = z - (h(x̄) - H @ x̄) = z - h(x̄) + H @ x̄
        # residual = z_k - gamma_x_bar = z - h(x̄)
        # So z_minus_e = residual + H @ x_bar = z - h(x̄) + H @ x̄ = z - e
        z_minus_e = residual + H_x_bar

        # Compute H P H^T
        HPHt = H @ P_pred @ tf.transpose(H)

        # Compute inner = λ H P H^T + R
        inner = lam * HPHt + R_reg

        # Symmetrize for numerical stability
        inner = 0.5 * (inner + tf.transpose(inner))

        # Adaptive regularization based on eigenvalues
        eigs = tf.linalg.eigvalsh(inner)
        min_eig = tf.reduce_min(eigs)
        base_eps = tf.constant(1e-6, dtype=inner.dtype)
        needed = tf.nn.relu(base_eps - min_eig)
        safety = tf.constant(1e-2, dtype=inner.dtype)
        reg = tf.maximum(base_eps, needed * (1.0 + safety))

        inner_reg = inner + reg * tf.eye(meas_dim, dtype=inner.dtype)

        # Use Cholesky solve instead of inverse: compute inner_reg^{-1} @ H
        chol = tf.linalg.cholesky(inner_reg)
        inner_inv_H = tf.linalg.cholesky_solve(chol, H)

        # Equation (8): A = -½ P H^T (λ H P H^T + R)^{-1} H
        M = tf.transpose(H) @ inner_inv_H
        A = -0.5 * P_pred @ M

        # Equation (9): b = (I + 2λA)[(I + λA) P H^T R^{-1} (z - e) + A x̄]
        I = tf.eye(self.state_dim, dtype=tf.float32)

        # First term: (I + λA) P H^T R^{-1} (z - e)
        I_plus_lambda_A = I + lam * A
        PHt = P_pred @ tf.transpose(H)
        Rinv_z_minus_e = tf.linalg.cholesky_solve(cholR, tf.reshape(z_minus_e, [-1, 1]))
        Rinv_z_minus_e = tf.reshape(Rinv_z_minus_e, [-1])
        PHt_Rinv_z_minus_e = tf.linalg.matvec(PHt, Rinv_z_minus_e)
        term1 = tf.linalg.matvec(I_plus_lambda_A, PHt_Rinv_z_minus_e)

        # Second term: A x̄
        term2 = tf.linalg.matvec(A, x_bar)

        # Combine: b = (I + 2λA)[term1 + term2]
        I_plus_2lambda_A = I + 2.0 * lam * A
        b = tf.linalg.matvec(I_plus_2lambda_A, term1 + term2)

        return A, b

    def predict(self, control: Optional[tf.Tensor] = None) -> tuple[tf.Tensor, tf.Tensor]:
        """Optimized prediction step."""
        self.step_count += 1
        N = self.num_particles

        # Propagate all particles at once
        if control is not None:
            control = tf.cast(control, tf.float32)
            control_batch = tf.tile(tf.reshape(control, [1, -1]), [N, 1])
            particles_pred = self.ssm.motion_model(self.particles, control_batch)
        else:
            particles_pred = self.ssm.motion_model(self.particles)

        # Add process noise - cache Cholesky
        if self._L_Q is None:
            try:
                self._L_Q = tf.linalg.cholesky(self.ssm.Q)
            except Exception:
                Q_reg = self.ssm.Q + 1e-6 * tf.eye(self.state_dim, dtype=self.ssm.Q.dtype)
                self._L_Q = tf.linalg.cholesky(Q_reg)

        noise = tf.random.normal([N, self.state_dim], dtype=tf.float32)
        process_noise = noise @ tf.transpose(self._L_Q)

        self.particles = particles_pred + process_noise
        self.x_bar = self._compute_particle_mean()

        # EKF/UKF prediction - use paper's formulation with EKF/UKF P_pred
        try:
            if control is not None:
                m_pred, P_pred_ekf = self.base_filter.predict(control)
            else:
                m_pred, P_pred_ekf = self.base_filter.predict()
            self.m_pred = m_pred
            self.P_pred = self._regularize_covariance(P_pred_ekf)
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
        CORRECTED: Homotopy migration with proper λ-dependent flow equations.

        Uses corrected equations (8)-(9) from Daum-Huang paper.
        """
        measurement = tf.cast(measurement, tf.float32)
        if landmarks is not None:
            landmarks = tf.cast(landmarks, tf.float32)

            # Check if landmarks is empty - skip update if no measurements
            num_landmarks = tf.shape(landmarks)[0]
            if num_landmarks == 0 or tf.size(measurement) == 0:
                # No landmarks visible - skip update, keep prediction
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
                desc=f"  EDH λ-steps (t={self.step_count})",
                leave=False,
                ncols=80
            )

        # Homotopy loop
        # Use linearization point that flows with particles (same as PFPF_EDH)
        eta_bar = tf.Variable(self.m_pred, trainable=False)

        lambda_cumulative = 0.0
        for j in lambda_iterator:
            epsilon_j = epsilon_steps[j]
            lam = tf.constant(lambda_cumulative + epsilon_j / 2.0, dtype=tf.float32)
            lambda_cumulative += epsilon_j

            # Calculate A, b at current linearization point with λ-dependence (global linearization)
            A, b = self._compute_flow_matrices(
                eta_bar, self.P_pred, measurement, lam, landmarks
            )

            # Vectorized particle update: dx/dλ = A @ x + b
            # All particles updated simultaneously with same A, b
            self.particles = self.particles + epsilon_j * (
                self.particles @ tf.transpose(A) + b
            )

            # Flow linearization point: dη/dλ = A @ η + b (same as PFPF_EDH)
            velocity_bar = tf.linalg.matvec(A, eta_bar) + b
            eta_bar.assign(eta_bar + epsilon_j * velocity_bar)

            # Wrap angle if needed (for 2D/3D state spaces)
            if self.state_dim > 2:
                angle = eta_bar[2]
                angle_wrapped = tf.math.atan2(tf.sin(angle), tf.cos(angle))
                eta_bar_new = tf.concat([
                    eta_bar[:2],
                    [angle_wrapped],
                    eta_bar[3:] if self.state_dim > 3 else tf.zeros([0], dtype=tf.float32)
                ], axis=0)
                eta_bar.assign(eta_bar_new)

            # Update particle mean
            self.x_bar = self._compute_particle_mean()

        # After flow, particles represent posterior - use post-flow mean
        self.x_hat = self._compute_particle_mean()

        # Compute covariance from particles (posterior covariance)
        particles_centered = self.particles - self.x_hat[tf.newaxis, :]
        P_particle = tf.reduce_mean(
            particles_centered[:, :, tf.newaxis] * particles_centered[:, tf.newaxis, :],
            axis=0
        )
        self.P = self._regularize_covariance(P_particle)

        # Final state is particle mean (post-flow)
        self.state = self.x_hat
        self.m = self.x_hat

        if self.redraw_particles:
            self.particles = sample_from_gaussian(
                self.x_hat, self.P, self.num_particles
            )

        # Update base filter for next prediction step (use particle mean and covariance)
        self._create_base_filter()

        return self.x_hat, self.P
