"""
Lorenz-96 state-space model (chaotic dynamical system).

Used for high-dimensional data assimilation experiments (e.g. Hu & van Leeuwen 2021).
All computation is TensorFlow; no NumPy in step/derivative.
"""

from __future__ import annotations

import tensorflow as tf


class Lorenz96Model:
    """Lorenz-96 chaotic dynamical system (vectorized TensorFlow)."""

    def __init__(self, dim=40, F=8.0, dt=0.05):
        self.dim = dim
        self.F = F
        self.dt = dt

    @tf.function
    def derivative(self, x):
        """Compute derivative (vectorized)."""
        x = tf.cast(x, tf.float32)
        xm2 = tf.roll(x, shift=2, axis=-1)
        xm1 = tf.roll(x, shift=1, axis=-1)
        xp1 = tf.roll(x, shift=-1, axis=-1)
        return (xp1 - xm2) * xm1 - x + self.F

    @tf.function
    def step(self, x):
        """RK4 integration step."""
        k1 = self.derivative(x)
        k2 = self.derivative(x + 0.5 * self.dt * k1)
        k3 = self.derivative(x + 0.5 * self.dt * k2)
        k4 = self.derivative(x + self.dt * k3)
        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def generate_trajectory(self, x0, n_steps, spinup=1000):
        """Generate trajectory with spinup. Returns tensor of shape (n_steps+1, ...)."""
        x = tf.cast(x0, tf.float32)
        for _ in range(spinup):
            x = self.step(x)
        trajectory = [x]
        for _ in range(n_steps):
            x = self.step(x)
            trajectory.append(x)
        return tf.stack(trajectory)
