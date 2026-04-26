"""
Nonlinear state-space model used in Andrieu et al. PMCMC paper (2010), Eq. 14-15.

x_t = 0.5 * x_{t-1} + 25 * (x_{t-1} / (1 + x_{t-1}^2)) + 8 * cos(1.2 * t) + v_t
y_t = x_t^2 / 20 + w_t

where v_t ~ N(0, Q) and w_t ~ N(0, R).
"""

from __future__ import annotations

import tensorflow as tf


class PMCMCNonlinearSSM:
    """
    1D nonlinear state-space model from PMCMC paper.

    Parameters
    ----------
    sigma_v_sq : float or tf.Tensor
        Process noise variance (Q).
    sigma_w_sq : float or tf.Tensor
        Measurement noise variance (R).
    initial_var : float
        Initial state variance (used by external experiment code).
    """

    def __init__(
        self,
        sigma_v_sq: float | tf.Tensor = 10.0,
        sigma_w_sq: float | tf.Tensor = 1.0,
        initial_var: float = 5.0,
    ) -> None:
        """Set process/measurement variances and initial-state prior scale; see class docstring."""
        self.state_dim = 1
        self.meas_per_landmark = 1
        self.Q = tf.reshape(tf.cast(sigma_v_sq, tf.float32), [1, 1])
        self.R = tf.reshape(tf.cast(sigma_w_sq, tf.float32), [1, 1])
        self.initial_var = float(initial_var)

    def full_measurement_cov(self, num_landmarks: int | tf.Tensor = 1) -> tf.Tensor:
        """Return measurement covariance."""
        del num_landmarks
        return self.R

    def motion_model(self, state: tf.Tensor, control: tf.Tensor) -> tf.Tensor:
        """
        State transition model.

        Parameters
        ----------
        state : tf.Tensor
            Current state shape (batch, 1) or (1,).
        control : tf.Tensor
            Control input representing time t, shape (batch, 1) or (1,).

        Returns
        -------
        tf.Tensor
            Next state shape (batch, 1).
        """
        state = tf.cast(state, tf.float32)
        control = tf.cast(control, tf.float32)

        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        if len(control.shape) == 1:
            control = control[tf.newaxis, :]

        x = state[:, 0]
        t = control[:, 0]
        x_next = 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * tf.cos(1.2 * t)
        return x_next[:, tf.newaxis]

    def measurement_model(self, state: tf.Tensor, landmarks: tf.Tensor = None) -> tf.Tensor:
        """
        Measurement model.

        Parameters
        ----------
        state : tf.Tensor
            State shape (batch, 1) or (1,).
        landmarks : tf.Tensor, optional
            Ignored.

        Returns
        -------
        tf.Tensor
            Measurements shape (batch, 1).
        """
        del landmarks
        state = tf.cast(state, tf.float32)

        if len(state.shape) == 1:
            state = state[tf.newaxis, :]

        x = state[:, 0]
        y = (x**2) / 20.0
        return y[:, tf.newaxis]

    def motion_jacobian(self, state: tf.Tensor, control: tf.Tensor) -> tf.Tensor:
        """
        Motion model Jacobian.

        Parameters
        ----------
        state : tf.Tensor
            State shape (batch, 1) or (1,).
        control : tf.Tensor
            Control input representing time t (unused in derivative).

        Returns
        -------
        tf.Tensor
            Jacobian shape (batch, 1, 1).
        """
        del control
        state = tf.cast(state, tf.float32)

        if len(state.shape) == 1:
            state = state[tf.newaxis, :]

        batch_size = tf.shape(state)[0]
        x = state[:, 0]
        dx = 0.5 + 25.0 * (1.0 - x**2) / tf.square(1.0 + x**2)
        return tf.reshape(dx, [batch_size, 1, 1])

    def measurement_jacobian(self, state: tf.Tensor, landmarks: tf.Tensor = None) -> tf.Tensor:
        """
        Measurement model Jacobian.

        Parameters
        ----------
        state : tf.Tensor
            State shape (batch, 1) or (1,).
        landmarks : tf.Tensor, optional
            Ignored.

        Returns
        -------
        tf.Tensor
            Jacobian shape (batch, 1, 1).
        """
        del landmarks
        state = tf.cast(state, tf.float32)

        if len(state.shape) == 1:
            state = state[tf.newaxis, :]

        batch_size = tf.shape(state)[0]
        x = state[:, 0]
        dy_dx = x / 10.0
        return tf.reshape(dy_dx, [batch_size, 1, 1])

