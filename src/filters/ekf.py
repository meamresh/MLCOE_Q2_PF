"""
Extended Kalman Filter (EKF) for nonlinear state-space models.

This module implements the Extended Kalman Filter, which linearizes nonlinear
dynamics and measurement models using first-order Taylor expansion (Jacobians).
"""

from __future__ import annotations

import tensorflow as tf


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state-space models.

    The EKF linearizes the nonlinear motion and measurement models using
    their Jacobians, then applies the standard Kalman filter equations.

    Parameters
    ----------
    ssm : RangeBearingSSM
        Nonlinear state-space model with motion_model, measurement_model,
        motion_jacobian, and measurement_jacobian methods.
    initial_state : tf.Tensor
        Initial state estimate of shape (3,).
    initial_covariance : tf.Tensor
        Initial covariance matrix of shape (3, 3).

    Attributes
    ----------
    ssm : RangeBearingSSM
        State-space model.
    state : tf.Variable
        Current state estimate.
    covariance : tf.Variable
        Current covariance matrix.
    """

    def __init__(self, ssm, initial_state: tf.Tensor,
                 initial_covariance: tf.Tensor) -> None:
        self.ssm = ssm
        self.state = tf.Variable(tf.cast(initial_state, tf.float32))
        self.covariance = tf.Variable(tf.cast(initial_covariance, tf.float32))

    def predict(self, control: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        EKF prediction step.

        Predicts the next state and covariance using the linearized motion
        model.

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

        state_pred = self.ssm.motion_model(self.state, control)[0]
        F = self.ssm.motion_jacobian(self.state, control)[0]

        covariance_pred = (F @ self.covariance @ tf.transpose(F) +
                          self.ssm.Q)

        # Ensure symmetry and add regularization
        covariance_pred = 0.5 * (covariance_pred + tf.transpose(covariance_pred))
        covariance_pred = (covariance_pred +
                          1e-6 * tf.eye(self.ssm.state_dim, dtype=tf.float32))

        self.state.assign(state_pred)
        self.covariance.assign(covariance_pred)

        return state_pred, covariance_pred

    def update(self, measurement: tf.Tensor,
               landmarks: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        EKF update step.

        Updates the state and covariance using the linearized measurement
        model.

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

        # Measurement prediction: handle both batched and unbatched SSMs
        meas_pred_full = self.ssm.measurement_model(self.state, landmarks)
        if len(meas_pred_full.shape) > 1:
            meas_pred = meas_pred_full[0]
        else:
            meas_pred = meas_pred_full

        meas_pred_vec = tf.reshape(meas_pred, [-1])
        meas_vec = tf.reshape(measurement, [-1])

        # Measurement Jacobian: handle both [batch, M, n] and [M, n]
        H_full = self.ssm.measurement_jacobian(self.state, landmarks)
        if len(H_full.shape) == 3:
            H = H_full[0]
        else:
            H = H_full
        R_full = self.ssm.full_measurement_cov(num_landmarks)

        residual = tf.reshape(meas_vec - meas_pred_vec, [-1])

        # Only wrap bearings if meas_per_landmark == 2 (range-bearing format)
        if hasattr(self.ssm, "meas_per_landmark") and self.ssm.meas_per_landmark == 2:
            idx_bearings = tf.range(1, 2 * num_landmarks, 2, dtype=tf.int32)
            bearing_res = tf.gather(residual, idx_bearings)
            bearing_res_wrapped = tf.math.atan2(tf.sin(bearing_res), tf.cos(bearing_res))
            residual = tf.tensor_scatter_nd_update(
                residual, idx_bearings[:, tf.newaxis], bearing_res_wrapped
            )

        S = H @ self.covariance @ tf.transpose(H) + R_full

        # Ensure S is symmetric and well-conditioned
        S = 0.5 * (S + tf.transpose(S))
        S = S + 1e-4 * tf.eye(tf.shape(S)[0], dtype=S.dtype)

        K = self.covariance @ tf.transpose(H) @ tf.linalg.inv(S)

        state_updated = self.state + tf.linalg.matvec(K, residual)

        # Joseph-stabilized covariance update
        I = tf.eye(self.ssm.state_dim, dtype=tf.float32)
        KH = K @ H
        cov_updated = ((I - KH) @ self.covariance @ tf.transpose(I - KH) +
                      K @ R_full @ tf.transpose(K))

        # Ensure symmetry
        cov_updated = 0.5 * (cov_updated + tf.transpose(cov_updated))
        cov_updated = (cov_updated +
                      1e-6 * tf.eye(self.ssm.state_dim, dtype=tf.float32))

        self.state.assign(state_updated)
        self.covariance.assign(cov_updated)

        return state_updated, cov_updated, residual

