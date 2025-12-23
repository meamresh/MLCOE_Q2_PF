"""
Unscented Kalman Filter (UKF) for nonlinear state-space models.

This module implements the Unscented Kalman Filter, which uses the unscented
transform to propagate mean and covariance through nonlinear transformations
without linearization.
"""

from __future__ import annotations

import tensorflow as tf


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear state-space models.

    The UKF uses sigma points and the unscented transform to propagate
    statistics through nonlinear transformations, avoiding the need for
    linearization.

    Parameters
    ----------
    ssm : RangeBearingSSM
        Nonlinear state-space model with motion_model and measurement_model
        methods.
    initial_state : tf.Tensor
        Initial state estimate of shape (3,).
    initial_covariance : tf.Tensor
        Initial covariance matrix of shape (3, 3).
    alpha : float, optional
        Spread parameter for sigma points. Defaults to 0.1.
    beta : float, optional
        Parameter for incorporating prior knowledge. Defaults to 1.0.
    kappa : float, optional
        Secondary scaling parameter. Defaults to 0.

    Attributes
    ----------
    ssm : RangeBearingSSM
        State-space model.
    state : tf.Variable
        Current state estimate.
    covariance : tf.Variable
        Current covariance matrix.
    n : int
        State dimension.
    wm : tf.Tensor
        Mean weights for sigma points.
    wc : tf.Tensor
        Covariance weights for sigma points.
    """

    def __init__(self, ssm, initial_state: tf.Tensor,
                 initial_covariance: tf.Tensor,
                 alpha: float = 0.01, beta: float = 1.0,
                 kappa: float | None = None) -> None:
        self.ssm = ssm
        self.state = tf.Variable(tf.cast(initial_state, tf.float32))
        self.covariance = tf.Variable(tf.cast(initial_covariance, tf.float32))

        self.n = ssm.state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 0
        self.lambda_ = alpha ** 2 * (self.n + self.kappa) - self.n

        m = 2 * self.n + 1
        W0m = self.lambda_ / (self.n + self.lambda_)
        W0c = W0m + (1 - alpha ** 2 + beta)
        Wi = 1.0 / (2.0 * (self.n + self.lambda_))

        wm = [W0m] + [Wi] * (m - 1)
        wc = [W0c] + [Wi] * (m - 1)

        self.wm = tf.constant(wm, dtype=tf.float32)
        self.wc = tf.constant(wc, dtype=tf.float32)

    def matrix_sqrt_cholesky(self, matrix: tf.Tensor) -> tf.Tensor:
        """
        Compute matrix square root using Cholesky decomposition.

        More stable than SVD for positive definite matrices.

        Parameters
        ----------
        matrix : tf.Tensor
            Positive definite matrix.

        Returns
        -------
        L : tf.Tensor
            Lower triangular Cholesky factor such that L @ L^T = matrix.
        """
        # Ensure symmetry
        matrix = 0.5 * (matrix + tf.transpose(matrix))

        # Add regularization for numerical stability
        eps = 1e-6
        matrix_reg = matrix + eps * tf.eye(tf.shape(matrix)[0], dtype=matrix.dtype)

        try:
            # Try Cholesky first (fastest and most stable for PD matrices)
            L = tf.linalg.cholesky(matrix_reg)
            return L
        except Exception:
            # Fallback to SVD if Cholesky fails
            return self.matrix_sqrt_svd(matrix_reg)

    def matrix_sqrt_svd(self, matrix: tf.Tensor) -> tf.Tensor:
        """
        Compute matrix square root using SVD (fallback method).

        Parameters
        ----------
        matrix : tf.Tensor
            Symmetric matrix.

        Returns
        -------
        sqrt_matrix : tf.Tensor
            Matrix square root such that sqrt_matrix @ sqrt_matrix^T = matrix.
        """
        # Ensure symmetry
        matrix = 0.5 * (matrix + tf.transpose(matrix))

        # Compute SVD: A = U @ diag(S) @ V^T
        S, U, V = tf.linalg.svd(matrix)

        # Clip small/negative eigenvalues
        S = tf.maximum(S, 1e-10)

        # For symmetric matrices: sqrt(A) = U @ diag(sqrt(S)) @ U^T
        sqrt_S = tf.sqrt(S)
        sqrt_matrix = U @ tf.linalg.diag(sqrt_S) @ tf.transpose(U)

        return sqrt_matrix

    def generate_sigma_points(self, state: tf.Tensor,
                              covariance: tf.Tensor) -> tf.Tensor:
        """
        Generate 2n+1 sigma points using Cholesky decomposition.

        Parameters
        ----------
        state : tf.Tensor
            Mean state of shape (n,).
        covariance : tf.Tensor
            Covariance matrix of shape (n, n).

        Returns
        -------
        sigma_points : tf.Tensor
            Sigma points of shape (2n+1, n).
        """
        state = tf.reshape(state, [self.n])
        covariance = tf.reshape(covariance, [self.n, self.n])

        scale = tf.cast(self.n + self.lambda_, dtype=covariance.dtype)
        scaled_cov = scale * covariance

        # Use Cholesky instead of SVD (more stable)
        L = self.matrix_sqrt_cholesky(scaled_cov)

        sigma_points = [state]

        for i in range(self.n):
            col = L[:, i]
            sigma_points.append(state + col)
            sigma_points.append(state - col)

        return tf.stack(sigma_points, axis=0)

    def unscented_transform(self, sigma_points: tf.Tensor,
                           noise_cov: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Compute mean and covariance from sigma points.

        Parameters
        ----------
        sigma_points : tf.Tensor
            Sigma points of shape (2n+1, n).
        noise_cov : tf.Tensor
            Additional noise covariance to add.

        Returns
        -------
        mean : tf.Tensor
            Weighted mean of shape (n,).
        covariance : tf.Tensor
            Weighted covariance of shape (n, n).
        """
        wm = self.wm[:, tf.newaxis]
        mean = tf.reduce_sum(wm * sigma_points, axis=0)

        diff = sigma_points - mean
        wc = self.wc[:, tf.newaxis, tf.newaxis]
        diff_exp1 = tf.expand_dims(diff, -1)
        diff_exp2 = tf.expand_dims(diff, 1)

        covariance = tf.reduce_sum(wc * (diff_exp1 * diff_exp2), axis=0)

        # Add noise and symmetrize
        covariance = covariance + noise_cov
        covariance = 0.5 * (covariance + tf.transpose(covariance))

        # Light regularization
        eps = 1e-8
        covariance = (covariance +
                     eps * tf.eye(tf.shape(covariance)[0],
                                 dtype=covariance.dtype))

        return mean, covariance

    def predict(self, control: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        UKF prediction step.

        Parameters
        ----------
        control : tf.Tensor
            Control input of shape (2,).

        Returns
        -------
        state_pred : tf.Tensor
            Predicted state of shape (3,).
        covariance_pred : tf.Tensor
            Predicted covariance matrix of shape (3, 3).
        """
        control = tf.cast(control, tf.float32)

        sigma_points = self.generate_sigma_points(self.state, self.covariance)
        m = tf.shape(sigma_points)[0]

        control_tiled = tf.tile(tf.reshape(control, [1, -1]), [m, 1])
        sigma_points_pred = self.ssm.motion_model(sigma_points, control_tiled)

        state_pred, covariance_pred = self.unscented_transform(
            sigma_points_pred, self.ssm.Q)

        self.state.assign(state_pred)
        self.covariance.assign(covariance_pred)

        return state_pred, covariance_pred

    def update(self, measurement: tf.Tensor,
              landmarks: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        UKF update step.

        Parameters
        ----------
        measurement : tf.Tensor
            Measurement vector of shape (num_landmarks, 2) or flattened.
        landmarks : tf.Tensor
            Landmark positions of shape (num_landmarks, 2).

        Returns
        -------
        state_updated : tf.Tensor
            Updated state estimate of shape (3,).
        covariance_updated : tf.Tensor
            Updated covariance matrix of shape (3, 3).
        residual : tf.Tensor
            Measurement residual (innovation).
        """
        landmarks = tf.cast(landmarks, tf.float32)
        measurement = tf.cast(measurement, tf.float32)

        num_landmarks = tf.shape(landmarks)[0]
        meas_dim = 2 * num_landmarks

        sigma_points = self.generate_sigma_points(self.state, self.covariance)
        m = tf.shape(sigma_points)[0]

        measurements_pred = self.ssm.measurement_model(sigma_points, landmarks)
        measurements_pred_flat = tf.reshape(measurements_pred, [m, meas_dim])

        meas_pred, S = self.unscented_transform(
            measurements_pred_flat,
            self.ssm.full_measurement_cov(num_landmarks))

        # Cross covariance
        diff_state = sigma_points - tf.reshape(self.state, [1, self.n])
        diff_meas = measurements_pred_flat - meas_pred

        # Wrap bearing differences
        bearing_indices = tf.range(1, meas_dim, 2, dtype=tf.int32)
        K_bearings = tf.shape(bearing_indices)[0]
        bearing_diffs = tf.gather(diff_meas, bearing_indices, axis=1)
        wrapped = tf.math.atan2(tf.sin(bearing_diffs), tf.cos(bearing_diffs))

        rows = tf.repeat(tf.range(m, dtype=tf.int32), repeats=K_bearings)
        cols = tf.tile(bearing_indices, [m])
        scatter_idx = tf.stack([rows, cols], axis=1)
        updates = tf.reshape(wrapped, [-1])
        diff_meas = tf.tensor_scatter_nd_update(diff_meas, scatter_idx, updates)

        # Compute cross-covariance
        wc = self.wc[:, tf.newaxis, tf.newaxis]
        diff_state_exp = tf.expand_dims(diff_state, -1)
        diff_meas_exp = tf.expand_dims(diff_meas, 1)

        P_xy = tf.reduce_sum(wc * (diff_state_exp * diff_meas_exp), axis=0)

        # Use S (not S_reg) for Kalman gain, but S_reg for inversion
        S_clean = 0.5 * (S + tf.transpose(S))
        eps_inv = 1e-6  # Much smaller regularization
        S_inv = tf.linalg.inv(S_clean +
                              eps_inv * tf.eye(tf.shape(S_clean)[0],
                                              dtype=S_clean.dtype))

        K_gain = P_xy @ S_inv

        # Residual
        meas_vec = tf.reshape(measurement, [-1])
        residual = meas_vec - meas_pred

        bearing_residuals = tf.gather(residual, bearing_indices)
        residual = tf.tensor_scatter_nd_update(
            residual, bearing_indices[:, tf.newaxis],
            tf.math.atan2(tf.sin(bearing_residuals),
                         tf.cos(bearing_residuals)))

        # Update state
        state_updated = self.state + tf.linalg.matvec(K_gain, residual)

        # Use correct covariance update (use S, not S_reg!)
        covariance_updated = (self.covariance -
                             K_gain @ S_clean @ tf.transpose(K_gain))

        # Ensure positive definite
        covariance_updated = 0.5 * (covariance_updated +
                                   tf.transpose(covariance_updated))

        # Check eigenvalues to ensure PSD
        eigvals = tf.linalg.eigvalsh(covariance_updated)
        min_eigval = tf.reduce_min(eigvals)

        if min_eigval < 0:
            # If negative eigenvalues, add correction
            correction = tf.abs(min_eigval) + 1e-6
            covariance_updated = (covariance_updated +
                                 correction * tf.eye(self.n,
                                                    dtype=covariance_updated.dtype))

        # Light regularization
        covariance_updated = (covariance_updated +
                             1e-8 * tf.eye(self.n,
                                          dtype=covariance_updated.dtype))

        self.state.assign(state_updated)
        self.covariance.assign(covariance_updated)

        return state_updated, covariance_updated, residual

