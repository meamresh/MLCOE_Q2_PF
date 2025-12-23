"""
Nonlinear state-space model for range-bearing sensor localization.

This module implements a range-bearing sensor model for robot localization
with known landmarks. The state consists of position and orientation, and
measurements are range and bearing to landmarks.
"""

from __future__ import annotations

import tensorflow as tf


class RangeBearingSSM:
    """
    Range-Bearing Sensor Model for Robot Localization.

    This model represents a robot moving in 2D space with range-bearing
    measurements to known landmarks. The state includes position (x, y) and
    orientation (theta), while measurements are range and bearing to each
    landmark.

    Parameters
    ----------
    dt : float
        Time step duration.
    process_noise : tf.Tensor, optional
        Process noise covariance matrix Q of shape (3, 3). Defaults to
        0.01 * I_3.
    meas_noise : tf.Tensor, optional
        Measurement noise covariance matrix R for a single landmark of shape
        (2, 2). Defaults to 0.1 * I_2.

    Attributes
    ----------
    dt : float
        Time step duration.
    state_dim : int
        State dimension (3: x, y, theta).
    meas_per_landmark : int
        Measurements per landmark (2: range, bearing).
    Q : tf.Tensor
        Process noise covariance matrix (3, 3).
    R : tf.Tensor
        Measurement noise covariance matrix per landmark (2, 2).

    Notes
    -----
    State: [x, y, theta]
    Control: [v, omega] (velocity and angular velocity)
    Measurement: [range, bearing] to each landmark
    """

    def __init__(self, dt: float = 0.1,
                 process_noise: tf.Tensor | None = None,
                 meas_noise: tf.Tensor | None = None) -> None:
        self.dt = dt
        self.state_dim = 3
        self.meas_per_landmark = 2
        self.Q = (process_noise if process_noise is not None
                  else tf.eye(3, dtype=tf.float32) * 0.01)
        self.R = (meas_noise if meas_noise is not None
                  else tf.eye(2, dtype=tf.float32) * 0.1)

    def full_measurement_cov(self, num_landmarks: int | tf.Tensor) -> tf.Tensor:
        """
        Build block-diagonal measurement covariance for all landmarks.

        Constructs the full measurement covariance matrix by placing the
        per-landmark covariance R along the diagonal.

        Parameters
        ----------
        num_landmarks : int or tf.Tensor
            Number of landmarks.

        Returns
        -------
        R_full : tf.Tensor
            Full measurement covariance matrix of shape
            (2*num_landmarks, 2*num_landmarks).
        """
        if isinstance(num_landmarks, tf.Tensor):
            static_n = tf.get_static_value(num_landmarks)
            if static_n is not None:
                M = int(static_n)
            else:
                M = int(num_landmarks.numpy())
        else:
            M = int(num_landmarks)

        if M == 0:
            return tf.zeros([0, 0], dtype=self.R.dtype)

        # Try block_diag first (most efficient)
        try:
            mats = [self.R for _ in range(M)]
            return tf.linalg.block_diag(*mats)
        except Exception:
            pass

        # Fallback to Kronecker product
        try:
            I = tf.eye(M, dtype=self.R.dtype)
            return tf.linalg.kron(I, self.R)
        except Exception:
            pass

        # Manual construction as last resort
        size = M * tf.shape(self.R)[0]
        size_int = int(size.numpy()) if isinstance(size, tf.Tensor) else int(size)
        R_full_var = tf.Variable(tf.zeros([size_int, size_int], dtype=self.R.dtype))
        block_size = int(tf.shape(self.R)[0])

        for i in range(M):
            start = i * block_size
            end = start + block_size
            R_full_var[start:end, start:end].assign(self.R)

        return R_full_var.read_value()

    def motion_model(self, state: tf.Tensor, control: tf.Tensor) -> tf.Tensor:
        """
        State transition model: x_{k+1} = f(x_k, u_k) + w_k.

        Implements unicycle motion model with velocity and angular velocity
        control inputs.

        Parameters
        ----------
        state : tf.Tensor
            Current state of shape (batch, 3) or (3,).
        control : tf.Tensor
            Control input [v, omega] of shape (batch, 2) or (2,).

        Returns
        -------
        state_next : tf.Tensor
            Next state of shape (batch, 3).
        """
        state = tf.cast(state, tf.float32)
        control = tf.cast(control, tf.float32)

        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        if len(control.shape) == 1:
            control = control[tf.newaxis, :]

        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        v = control[:, 0]
        omega = control[:, 1]

        x_next = x + v * self.dt * tf.cos(theta)
        y_next = y + v * self.dt * tf.sin(theta)
        theta_next = theta + omega * self.dt

        return tf.stack([x_next, y_next, theta_next], axis=1)

    def measurement_model(self, state: tf.Tensor,
                         landmarks: tf.Tensor) -> tf.Tensor:
        """
        Measurement model: z = h(x).

        Computes range and bearing measurements to all landmarks.

        Parameters
        ----------
        state : tf.Tensor
            State of shape (batch, 3) or (3,).
        landmarks : tf.Tensor
            Landmark positions of shape (num_landmarks, 2).

        Returns
        -------
        measurements : tf.Tensor
            Measurements of shape (batch, num_landmarks, 2), where the last
            dimension is [range, bearing].
        """
        state = tf.cast(state, tf.float32)
        landmarks = tf.cast(landmarks, tf.float32)

        if len(state.shape) == 1:
            state = state[tf.newaxis, :]

        batch_size = tf.shape(state)[0]
        num_landmarks = tf.shape(landmarks)[0]

        state_exp = tf.reshape(state, [batch_size, 1, 3])
        landmarks_exp = tf.reshape(landmarks, [1, num_landmarks, 2])

        dx = landmarks_exp[:, :, 0] - state_exp[:, :, 0]
        dy = landmarks_exp[:, :, 1] - state_exp[:, :, 1]

        ranges = tf.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        bearings = tf.math.atan2(dy, dx) - state_exp[:, :, 2]
        bearings = tf.math.atan2(tf.sin(bearings), tf.cos(bearings))

        return tf.stack([ranges, bearings], axis=-1)

    def motion_jacobian(self, state: tf.Tensor,
                        control: tf.Tensor) -> tf.Tensor:
        """
        Compute motion model Jacobian: F = ∂f/∂x.

        Parameters
        ----------
        state : tf.Tensor
            State of shape (batch, 3) or (3,).
        control : tf.Tensor
            Control input of shape (batch, 2) or (2,).

        Returns
        -------
        F : tf.Tensor
            Jacobian matrix of shape (batch, 3, 3).
        """
        state = tf.cast(state, tf.float32)
        control = tf.cast(control, tf.float32)

        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        if len(control.shape) == 1:
            control = control[tf.newaxis, :]

        batch_size = tf.shape(state)[0]
        theta = state[:, 2]
        v = control[:, 0]

        F = tf.tile(tf.eye(3, dtype=tf.float32)[tf.newaxis, :, :],
                    [batch_size, 1, 1])

        F_x_theta = -v * self.dt * tf.sin(theta)
        F_y_theta = v * self.dt * tf.cos(theta)

        b_idx = tf.range(batch_size, dtype=tf.int32)
        idxs = tf.concat([
            tf.stack([b_idx, tf.zeros(batch_size, dtype=tf.int32),
                     tf.fill([batch_size], 2)], axis=1),
            tf.stack([b_idx, tf.ones(batch_size, dtype=tf.int32),
                     tf.fill([batch_size], 2)], axis=1),
        ], axis=0)

        updates = tf.concat([F_x_theta, F_y_theta], axis=0)
        F = tf.tensor_scatter_nd_update(F, idxs, updates)

        return F

    def measurement_jacobian(self, state: tf.Tensor,
                            landmarks: tf.Tensor) -> tf.Tensor:
        """
        Compute measurement model Jacobian: H = ∂h/∂x.

        Parameters
        ----------
        state : tf.Tensor
            State of shape (batch, 3) or (3,).
        landmarks : tf.Tensor
            Landmark positions of shape (num_landmarks, 2).

        Returns
        -------
        H : tf.Tensor
            Jacobian matrix of shape (batch, 2*num_landmarks, 3).
        """
        state = tf.cast(state, tf.float32)
        landmarks = tf.cast(landmarks, tf.float32)

        if len(state.shape) == 1:
            state = state[tf.newaxis, :]

        batch_size = tf.shape(state)[0]
        num_landmarks = tf.shape(landmarks)[0]

        state_exp = tf.reshape(state, [batch_size, 1, 3])
        landmarks_exp = tf.reshape(landmarks, [1, num_landmarks, 2])

        dx = landmarks_exp[:, :, 0] - state_exp[:, :, 0]
        dy = landmarks_exp[:, :, 1] - state_exp[:, :, 1]

        ranges_sq = dx ** 2 + dy ** 2 + 1e-8
        ranges = tf.sqrt(ranges_sq)

        H = tf.zeros([batch_size, num_landmarks * 2, 3], dtype=tf.float32)

        n_lm = (int(tf.get_static_value(num_landmarks))
                if isinstance(num_landmarks, tf.Tensor)
                and tf.get_static_value(num_landmarks) is not None
                else int(num_landmarks.numpy()))

        for i in range(n_lm):
            dr_dx = -dx[:, i] / ranges[:, i]
            dr_dy = -dy[:, i] / ranges[:, i]
            dphi_dx = dy[:, i] / ranges_sq[:, i]
            dphi_dy = -dx[:, i] / ranges_sq[:, i]
            dphi_dtheta = -tf.ones_like(dphi_dx)

            idx_range = 2 * i
            idx_bearing = 2 * i + 1

            H = tf.tensor_scatter_nd_update(
                H,
                tf.stack([tf.range(batch_size),
                         tf.fill([batch_size], idx_range),
                         tf.zeros([batch_size], dtype=tf.int32)], axis=1),
                dr_dx
            )

            H = tf.tensor_scatter_nd_update(
                H,
                tf.stack([tf.range(batch_size),
                         tf.fill([batch_size], idx_range),
                         tf.ones([batch_size], dtype=tf.int32)], axis=1),
                dr_dy
            )

            H = tf.tensor_scatter_nd_update(
                H,
                tf.stack([tf.range(batch_size),
                         tf.fill([batch_size], idx_bearing),
                         tf.zeros([batch_size], dtype=tf.int32)], axis=1),
                dphi_dx
            )

            H = tf.tensor_scatter_nd_update(
                H,
                tf.stack([tf.range(batch_size),
                         tf.fill([batch_size], idx_bearing),
                         tf.ones([batch_size], dtype=tf.int32)], axis=1),
                dphi_dy
            )

            H = tf.tensor_scatter_nd_update(
                H,
                tf.stack([tf.range(batch_size),
                         tf.fill([batch_size], idx_bearing),
                         tf.fill([batch_size], 2)], axis=1),
                dphi_dtheta
            )

        return H

