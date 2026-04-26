"""
Bearing-only localization state-space model for Dai & Daum (2021) SPF example.

Implements a static 2D bearing-only localization problem matching Section 4 of:
    Dai & Daum (2021), "Stiffness Mitigation in Stochastic Particle Flow Filters"

Compatible with the StateSpaceModel protocol (motion_model, measurement_model, Q, R)
and provides extra methods required by the Dai–Daum stochastic particle flow filter
(log_prior, gradient_log_prior, hessian_log_prior, log_likelihood, etc.).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import tensorflow as tf


# Default dtype for this model
DTYPE = tf.float64


class DaiDaumBearingSSM:
    """
    Bearing-only localization SSM for the Dai–Daum (2021) stochastic particle flow example.

    State: x = [x, y] (target position).
    Measurement: z = [bearing_1, bearing_2] from two fixed sensors.
    Prior: Gaussian N(prior_mean, prior_cov). No process dynamics (static problem).

    Parameters
    ----------
    prior_mean : tf.Tensor, optional
        Prior mean (2,). Defaults to paper value [3, 5].
    prior_cov : tf.Tensor, optional
        Prior covariance (2, 2). Defaults to paper value diag(1000, 2).
    Q : tf.Tensor, optional
        Flow diffusion matrix (2, 2). Defaults to paper value diag(4, 0.4).
    R : tf.Tensor, optional
        Measurement noise covariance (2, 2). Defaults to paper value 0.04*I.
    sensor_positions : tf.Tensor, optional
        (2, 2) array of sensor positions; row i = [sx_i, sy_i]. Defaults to S1=[3.5,0], S2=[-3.5,0].
        x_true : tf.Tensor, optional
        Reference true state for experiments (2,). Defaults to [4, 4].
    """

    def __init__(
        self,
        prior_mean: Optional[tf.Tensor] = None,
        prior_cov: Optional[tf.Tensor] = None,
        Q: Optional[tf.Tensor] = None,
        R: Optional[tf.Tensor] = None,
        sensor_positions: Optional[tf.Tensor] = None,
        x_true: Optional[tf.Tensor] = None,
    ) -> None:
        """Assign prior, flow diffusion Q, measurement R, sensors, and optional ground truth."""
        self.state_dim = 2
        self.meas_dim = 2

        self.prior_mean = tf.cast(
            prior_mean if prior_mean is not None else tf.constant([3.0, 5.0], dtype=DTYPE),
            DTYPE,
        )
        self.prior_cov = tf.cast(
            prior_cov
            if prior_cov is not None
            else tf.constant([[1000.0, 0.0], [0.0, 2.0]], dtype=DTYPE),
            DTYPE,
        )
        self.Q = tf.cast(
            Q if Q is not None else tf.constant([[4.0, 0.0], [0.0, 0.4]], dtype=DTYPE),
            DTYPE,
        )
        self.R = tf.cast(
            R if R is not None else tf.constant([[0.04, 0.0], [0.0, 0.04]], dtype=DTYPE),
            DTYPE,
        )
        self._R_inv = tf.linalg.inv(self.R)
        if sensor_positions is not None:
            self._sensors = tf.cast(sensor_positions, DTYPE)  # (2, 2)
        else:
            self._sensors = tf.constant(
                [[3.5, 0.0], [-3.5, 0.0]], dtype=DTYPE
            )  # S1, S2
        self.x_true = tf.cast(
            x_true if x_true is not None else tf.constant([4.0, 4.0], dtype=DTYPE),
            DTYPE,
        )

    def _angle_meas(self, x: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        """Bearing atan2(dy, dx) from sensor *s* to position *x* (broadcasting-friendly)."""
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        s = tf.convert_to_tensor(s, dtype=DTYPE)
        dx = x[..., 0] - s[0]
        dy = x[..., 1] - s[1]
        return tf.math.atan2(dy, dx)

    def measurement_model(
        self, state: tf.Tensor, *args: Any
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Bearing-only measurement h(x) = [h1(x); h2(x)] and its Jacobian.
        """
        state = tf.convert_to_tensor(state, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(state)
            h = self._h_vec(state)
        jac = tape.jacobian(h, state)
        return h, jac

    def _h_vec(self, x: tf.Tensor) -> tf.Tensor:
        """Stacked bearings from *x* to both fixed sensors."""
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        return tf.stack(
            [
                self._angle_meas(x, self._sensors[0]),
                self._angle_meas(x, self._sensors[1]),
            ],
            axis=-1,
        )

    def motion_model(
        self,
        state: tf.Tensor,
        control: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Identity (no dynamics); Jacobian is I.
        """
        state = tf.convert_to_tensor(state, dtype=DTYPE)
        jac = tf.eye(self.state_dim, dtype=DTYPE)
        return state, jac

    def log_prior(self, x: tf.Tensor) -> tf.Tensor:
        """Log-density of N(prior_mean, prior_cov) at *x*."""
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        d = x - self.prior_mean
        P_inv = tf.linalg.inv(self.prior_cov)
        return -0.5 * tf.reduce_sum(d * tf.linalg.matvec(P_inv, d))

    def gradient_log_prior(self, x: tf.Tensor) -> tf.Tensor:
        """Gradient of :meth:`log_prior` w.r.t. *x*."""
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(x)
            lp = self.log_prior(x)
        return tape.gradient(lp, x)

    def hessian_log_prior(self, x: tf.Tensor) -> tf.Tensor:
        """Hessian of :meth:`log_prior`; Gaussian prior gives ``-P_0^{-1}`` (constant in *x*)."""
        _ = tf.convert_to_tensor(x, dtype=DTYPE)
        return -tf.linalg.inv(self.prior_cov)

    def log_likelihood(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Gaussian measurement log p(z | x) with h(x) from :meth:`_h_vec` and R."""
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        z = tf.convert_to_tensor(z, dtype=DTYPE)
        r = self._h_vec(x) - z
        return -0.5 * tf.reduce_sum(r * tf.linalg.matvec(self._R_inv, r))

    def gradient_log_likelihood(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Gradient of :meth:`log_likelihood` w.r.t. *x*."""
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        z = tf.convert_to_tensor(z, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(x)
            ll = self.log_likelihood(x, z)
        return tape.gradient(ll, x)

    def hessian_log_likelihood(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Hessian of :meth:`log_likelihood` w.r.t. *x* (autodiff on gradient)."""
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        z = tf.convert_to_tensor(z, dtype=DTYPE)
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(x)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(x)
                ll = self.log_likelihood(x, z)
            grad = inner_tape.gradient(ll, x)
        return outer_tape.jacobian(grad, x)


# Backward compatibility: module-level names for code that imports them
def _default_ssm() -> DaiDaumBearingSSM:
    return DaiDaumBearingSSM()


S1 = tf.constant([3.5, 0.0], dtype=DTYPE)
S2 = tf.constant([-3.5, 0.0], dtype=DTYPE)
x_true = tf.constant([4.0, 4.0], dtype=DTYPE)
mu_prior = tf.constant([3.0, 5.0], dtype=DTYPE)
P_prior = tf.constant([[1000.0, 0.0], [0.0, 2.0]], dtype=DTYPE)
R = tf.constant([[0.04, 0.0], [0.0, 0.04]], dtype=DTYPE)
R_inv = tf.linalg.inv(R)
Q = tf.constant([[4.0, 0.0], [0.0, 0.4]], dtype=DTYPE)


def h_vec(x: tf.Tensor) -> tf.Tensor:
    """Stacked bearing measurement (uses default SSM geometry)."""
    ssm = _default_ssm()
    return ssm._h_vec(x)


__all__ = [
    "DTYPE",
    "DaiDaumBearingSSM",
    "S1",
    "S2",
    "x_true",
    "mu_prior",
    "P_prior",
    "R",
    "R_inv",
    "Q",
    "h_vec",
]
