"""Differentiable particle filters with Sinkhorn OT and soft resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from .resampling import det_resample, soft_resample

tfd = tfp.distributions


@dataclass
class BootstrapModel:
    """
    Simple interface for a bootstrap particle filter / DPF.

    Users provide:
      - sample_initial: function (N, y1) -> x1, log_w1
      - transition: function (t, x_prev, y_t) -> x_t, log_w_t
        where log_w_t is log g(y_t | x_t), i.e. observation log-likelihood.
    """

    sample_initial: Callable[[int, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
    transition: Callable[[int, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


class BaseParticleFilter(tf.keras.Model):
    """Base class for particle filters using tf.scan for the filtering loop."""

    def __init__(
        self,
        model: BootstrapModel,
        num_particles: int,
        resample_threshold: float = 0.5,
    ) -> None:
        """Initialise with a bootstrap model, particle count, and ESS threshold."""
        super().__init__()
        self.particle_model = model
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold

    def _resample_fn(self, x: tf.Tensor, log_w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Resampling logic to be implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def _effective_sample_size(log_w: tf.Tensor) -> tf.Tensor:
        """ESS = 1 / sum(w^2) from normalised log-weights."""
        w = tf.nn.softmax(log_w, axis=-1)
        return 1.0 / tf.reduce_sum(w**2)

    def call(self, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Run the filter on observations *y*; return ``(log_lik, particles)``."""
        N = self.num_particles
        log_N = tf.math.log(tf.cast(N, dtype=tf.float32))

        # Initial step (t=0)
        x_0, log_w_0 = self.particle_model.sample_initial(N, y[0])
        log_w_0 = tf.squeeze(log_w_0)

        # log p(y_1) = logsumexp(log_w) - log(N)
        loglik_0 = tf.reduce_logsumexp(log_w_0) - log_N
        # Normalize weights: log_w <- log_w - logsumexp(log_w)
        log_w_0 = tf.nn.log_softmax(log_w_0)

        def scan_fn(acc, y_t):
            t, x, log_w, _ = acc

            # 1. Resample if ESS is low
            ess = self._effective_sample_size(log_w)
            should_resample = ess < self.resample_threshold * N

            x, log_w = tf.cond(
                should_resample,
                lambda: self._resample_fn(x, log_w),
                lambda: (x, log_w),
            )

            # 2. Propagate
            x, log_likelihood_t = self.particle_model.transition(t, x, y_t)
            log_w = log_w + tf.squeeze(log_likelihood_t)

            # 3. Compute incremental log-likelihood and normalize
            loglik_t = tf.reduce_logsumexp(log_w)
            log_w = log_w - loglik_t

            return (t + 1, x, log_w, loglik_t)

        # Initialize scan with result of t=0
        # elems = y[1:] (all subsequent observations)
        initializer = (tf.constant(1), x_0, log_w_0, tf.constant(0.0, dtype=tf.float32))

        _, final_x, final_log_w, incremental_logliks = tf.scan(
            scan_fn, y[1:], initializer=initializer
        )

        # Total log-likelihood
        loglik = loglik_0 + tf.reduce_sum(incremental_logliks)

        # final_log_w is a trajectory if we don't aggregate. Pick the last one.
        return loglik, final_log_w[-1]


class StandardParticleFilter(BaseParticleFilter):
    """Standard bootstrap particle filter with multinomial resampling."""

    def _resample_fn(self, x: tf.Tensor, log_w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Multinomial resample; return uniform log-weights."""
        N = self.num_particles
        # Use tfd.Categorical for cleaner sampling
        dist = tfd.Categorical(logits=log_w)
        idx = dist.sample(N)
        x_resampled = tf.gather(x, idx, axis=0)
        log_w_new = tf.fill([N], -tf.math.log(tf.cast(N, dtype=tf.float32)))
        return x_resampled, log_w_new


class DifferentiableParticleFilter(BaseParticleFilter):
    """Differentiable Particle Filter (DPF) using DET resampling."""

    def __init__(
        self,
        model: BootstrapModel,
        num_particles: int,
        epsilon: float = 0.5,
        sinkhorn_iters: int = 20,
        resample_threshold: float = 0.5,
    ) -> None:
        """Initialise DPF with Sinkhorn regularisation and iteration count."""
        super().__init__(model, num_particles, resample_threshold)
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters

    def _resample_fn(self, x: tf.Tensor, log_w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Entropy-regularised deterministic transport (Sinkhorn) resampling."""
        x_resampled, w_uniform = det_resample(
            x, log_w, epsilon=self.epsilon, n_iters=self.sinkhorn_iters
        )
        return x_resampled, tf.math.log(w_uniform + 1e-20)


class SoftResamplingParticleFilter(BaseParticleFilter):
    """
    Baseline DPF using soft-resampling (mixture with uniform).
    Fulfills requirement (a) from arXiv:2302.09639.
    """

    def __init__(
        self,
        model: BootstrapModel,
        num_particles: int,
        alpha: float = 0.5,
        resample_threshold: float = 0.5,
    ) -> None:
        """Initialise with mixture coefficient *alpha* for soft resampling."""
        super().__init__(model, num_particles, resample_threshold)
        self.alpha = alpha

    def _resample_fn(self, x: tf.Tensor, log_w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Mixture of categorical resample and uniform (soft) resampling."""
        return soft_resample(x, log_w, alpha=self.alpha)


class StopGradientParticleFilter(BaseParticleFilter):
    """
    Stop-Gradient DPF (Jonschkowski et al., arXiv:1805.11122).

    Uses standard multinomial resampling but wraps the output with
    tf.stop_gradient, blocking gradient flow through the resampling
    step while preserving gradients through dynamics and observations
    within each time step.

    This serves as a critical ablation baseline to isolate the
    contribution of differentiable resampling (OT/Soft) vs. simply
    having differentiable dynamics/observation models.
    """

    def _resample_fn(self, x: tf.Tensor, log_w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Multinomial resample with ``tf.stop_gradient`` on outputs (ablation baseline)."""
        N = self.num_particles
        dist = tfd.Categorical(logits=log_w)
        idx = dist.sample(N)
        x_resampled = tf.gather(x, idx, axis=0)
        log_w_new = tf.fill([N], -tf.math.log(tf.cast(N, dtype=tf.float32)))
        # Stop gradient: blocks backprop through the resampling step
        return tf.stop_gradient(x_resampled), tf.stop_gradient(log_w_new)


class ParticleTransformerFilter(BaseParticleFilter):
    """
    Particle Transformer DPF (Li et al., arXiv:2004.11938).

    Uses a pre-trained Particle Transformer neural network for resampling.
    The resampling is fully differentiable through the learned network weights.
    The PT must be pre-trained before being passed to this filter.
    """

    def __init__(
        self,
        model: BootstrapModel,
        num_particles: int,
        pt_model,
        resample_threshold: float = 0.5,
    ) -> None:
        """Initialise with a pre-trained ParticleTransformer model."""
        super().__init__(model, num_particles, resample_threshold)
        self.pt_model = pt_model

    def _resample_fn(self, x: tf.Tensor, log_w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Learned resampling via Particle Transformer; uniform log-weights after."""
        N = self.num_particles
        w = tf.nn.softmax(log_w)
        # Run the Particle Transformer (expects (N, d) for single batch)
        x_resampled = self.pt_model(x, w)  # (N, d)
        log_w_new = tf.fill([N], -tf.math.log(tf.cast(N, dtype=tf.float32)))
        return x_resampled, log_w_new

