"""
Training utilities for Differentiable Particle Filters.

Implements Sections IV-B and IV-C of arXiv:2004.11938:
  - Individual pre-training of the Particle Transformer (steps 2-3)
  - End-to-end joint training of PT + SSM parameters (step 4)
"""

from __future__ import annotations

import math
import time
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple

from .diff_particle_filter import (
    StandardParticleFilter,
    ParticleTransformerFilter,
    BootstrapModel,
)
from .particle_transformer import ParticleTransformer
from src.models.ssm_katigawa import PMCMCNonlinearSSM

tfd = tfp.distributions


class _KitagawaBootstrapWrapper:
    """Expose PMCMCNonlinearSSM as (sample_initial, step) for BootstrapModel."""

    def __init__(self, ssm: PMCMCNonlinearSSM, phi: float | tf.Tensor = 1.0):
        """Wrap Kitagawa SSM; *phi* is the trainable dynamics gain coefficient."""
        self.ssm = ssm
        self.phi = tf.Variable(phi, trainable=True, dtype=tf.float32)

    def sample_initial(self, N: int, y1: tf.Tensor):
        """Draw initial states and observation log-weights for *N* particles at *y1*."""
        y1 = tf.cast(y1, tf.float32)
        # Initial state ~ N(0, initial_var)
        x1 = tf.random.normal((N, 1), stddev=tf.sqrt(tf.cast(self.ssm.initial_var, tf.float32)), dtype=y1.dtype)
        log_w1 = self._log_obs_density(x1, y1)
        return x1, log_w1

    def step(self, t: int, x_prev: tf.Tensor, y_t: tf.Tensor):
        """One transition step: propagate *x_prev* to time *t* and return log g(y_t|x_t)."""
        x_prev = tf.cast(x_prev, tf.float32)
        y_t = tf.cast(y_t, tf.float32)
        t_float = tf.cast(t + 1, x_prev.dtype)

        term1 = 0.5 * x_prev
        term2 = 25.0 * self.phi * (x_prev / (1.0 + x_prev**2))
        term3 = 8.0 * tf.cos(1.2 * t_float)
        mean = term1 + term2 + term3

        std_v = tf.sqrt(tf.cast(self.ssm.Q[0, 0], x_prev.dtype))
        v_t = tf.random.normal(tf.shape(x_prev), mean=0.0, stddev=std_v, dtype=x_prev.dtype)
        x_t = mean + v_t

        log_w_t = self._log_obs_density(x_t, y_t)
        return x_t, log_w_t

    def _log_obs_density(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Log observation density for y = x^2/20 + w with w ~ N(0, R) (per particle)."""
        # y = x^2/20 + w, w ~ N(0, R)
        mean_y = (x**2) / 20.0
        var_w = tf.cast(self.ssm.R[0, 0], x.dtype)
        diff = tf.expand_dims(tf.cast(y, x.dtype), axis=0) - mean_y
        loglik = -0.5 * (diff**2) / var_w - 0.5 * tf.math.log(2.0 * math.pi * var_w)
        return tf.reduce_sum(loglik, axis=-1)


def _build_kitagawa_bootstrap(phi: float | tf.Tensor = 1.0) -> tuple[BootstrapModel, tf.Variable]:
    """Create a BootstrapModel + the phi variable used inside step()."""
    ssm = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0, initial_var=10.0)
    wrapped = _KitagawaBootstrapWrapper(ssm=ssm, phi=phi)
    model = BootstrapModel(sample_initial=wrapped.sample_initial, transition=wrapped.step)
    return model, wrapped.phi


def kde_log_prob(particles, eval_points, bandwidth=0.5):
    """Gaussian KDE log-probability."""
    # (B, M, 1, d) - (B, 1, N, d) = (B, M, N, d)
    diff = tf.expand_dims(eval_points, 2) - tf.expand_dims(particles, 1)
    sq_dist = tf.reduce_sum(diff ** 2, axis=-1)           # (B, M, N)
    d = tf.cast(tf.shape(particles)[-1], tf.float32)
    log_kernel = -0.5 * sq_dist / (bandwidth ** 2) - 0.5 * d * tf.math.log(
        2.0 * math.pi * bandwidth ** 2
    )
    N = tf.cast(tf.shape(particles)[1], tf.float32)
    log_probs = tf.reduce_logsumexp(log_kernel, axis=-1) - tf.math.log(N)
    return log_probs


def resampler_loss(pt_model, input_particles, input_weights, bandwidth=0.5):
    """KL-divergence loss from Section IV-B of arXiv:2004.11938 (Eq. 1)."""
    resampled = pt_model(input_particles, input_weights)  # (B, N, d)
    log_q = kde_log_prob(resampled, input_particles, bandwidth)  # (B, N)

    w_norm = input_weights / (tf.reduce_sum(input_weights, axis=-1, keepdims=True) + 1e-20)

    loss = -tf.reduce_mean(tf.reduce_sum(w_norm * log_q, axis=-1))
    return loss


def collect_training_data(y_data, n_episodes=200, N=50, T=50):
    """
    Section IV-C step 2: run a standard PF and collect pre-resampling
    particle sets to train the Particle Transformer individually.
    """
    all_particles = []
    all_weights = []

    for ep in range(n_episodes):
        tf.random.set_seed(5000 + ep)
        model, _ = _build_kitagawa_bootstrap(phi=1.0)

        x, log_w = model.sample_initial(N, y_data[0])
        log_w = tf.squeeze(log_w)
        log_w = tf.nn.log_softmax(log_w)

        for t in range(1, len(y_data)):
            w = tf.nn.softmax(log_w)
            all_particles.append(x)
            all_weights.append(w)

            ess = 1.0 / tf.reduce_sum(w ** 2)
            if ess < 0.5 * N:
                idx = tfd.Categorical(logits=log_w).sample(N)
                x = tf.gather(x, idx, axis=0)
                log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

            x, log_obs = model.transition(t, x, y_data[t])
            log_w = log_w + tf.squeeze(log_obs)
            log_w = log_w - tf.reduce_logsumexp(log_w)

    particles = tf.stack(all_particles, axis=0)   # (M, N, state_dim)
    weights = tf.stack(all_weights, axis=0)       # (M, N)
    return particles, weights


def train_particle_transformer(
    y_data,
    n_particles: int = 50,
    state_dim: int = 1,
    n_episodes: int = 200,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    bandwidth: float = 0.5,
):
    """
    Section IV-C steps 2-3: individually pre-train the Particle Transformer
    on particle sets collected from a standard PF.
    """
    t0 = time.perf_counter()
    print("Collecting training data for Particle Transformer...")
    particles, weights = collect_training_data(
        y_data, n_episodes=n_episodes, N=n_particles
    )
    t_collect = time.perf_counter() - t0
    print(f"  Collected {len(particles)} particle sets.  [{t_collect:.1f}s]")

    pt_model = ParticleTransformer(state_dim=state_dim)

    optimizer = tf.keras.optimizers.Adam(lr)
    n_samples = int(particles.shape[0])
    steps_per_epoch = max(n_samples // batch_size, 1)

    t_train_start = time.perf_counter()
    for epoch in range(n_epochs):
        perm = tf.random.shuffle(tf.range(n_samples))
        particles = tf.gather(particles, perm, axis=0)
        weights = tf.gather(weights, perm, axis=0)

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            s = step * batch_size
            e = s + batch_size
            batch_p = tf.constant(particles[s:e], dtype=tf.float32)
            batch_w = tf.constant(weights[s:e], dtype=tf.float32)

            with tf.GradientTape() as tape:
                loss = resampler_loss(pt_model, batch_p, batch_w, bandwidth)

            grads = tape.gradient(loss, pt_model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            optimizer.apply_gradients(zip(grads, pt_model.trainable_variables))
            epoch_loss += float(loss.numpy())

        avg_loss = epoch_loss / steps_per_epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}")

    t_train = time.perf_counter() - t_train_start
    t_total = time.perf_counter() - t0
    print(f"  Particle Transformer individual pre-training complete.  [train: {t_train:.1f}s | total: {t_total:.1f}s]")
    return pt_model


def end_to_end_train_particle_transformer(
    y_data,
    pt_model: ParticleTransformer,
    n_particles: int = 50,
    n_epochs: int = 10,
    n_seeds_per_epoch: int = 20,
    lr: float = 1e-4,
    clip_norm: float = 10.0,
    train_resampler: bool = True,
):
    """
    Section IV-C step 4: end-to-end joint training of the PT inside a
    differentiable particle filter, maximising the filter's log-likelihood.

    Gradients flow through the PT's resampling step (differentiable) back
    into both the PT weights and the SSM parameter phi.  Following the
    paper's finding (Table I, k=1), we stop gradients across time steps
    to avoid the exploding-gradient issue documented in Appendix B.

    Args:
        y_data: observation sequence, shape (T,) or (T, obs_dim).
        pt_model: a pre-trained ParticleTransformer (from step 3).
        n_particles: number of particles for the filter.
        n_epochs: number of end-to-end training epochs.
        n_seeds_per_epoch: random seeds per epoch (each gives one LL estimate).
        lr: learning rate (lower than individual pre-training for stability).
        clip_norm: global gradient clip norm (paper Appendix B: 10).
        train_resampler: if True, update PT weights jointly (best results
            in paper Table I row "Particle transformer").  If False, freeze
            PT and only update SSM (row "Particle transformer (frozen)").
    Returns:
        pt_model (updated in-place).
    """
    t0 = time.perf_counter()
    print(f"  End-to-end training (epochs={n_epochs}, N={n_particles}, "
          f"train_resampler={train_resampler})...")

    if train_resampler:
        trainable = pt_model.trainable_variables
    else:
        trainable = []

    optimizer = tf.keras.optimizers.Adam(lr)

    for epoch in range(n_epochs):
        epoch_ll = 0.0

        for seed_offset in range(n_seeds_per_epoch):
            tf.random.set_seed(7000 + epoch * 100 + seed_offset)

            model, phi_var = _build_kitagawa_bootstrap(phi=1.0)
            pf = ParticleTransformerFilter(
                model, num_particles=n_particles,
                pt_model=pt_model, resample_threshold=2.0,
            )

            all_vars = list(trainable) + [phi_var]

            # Use default GradientTape (watch_accessed_variables=True) so all
            # variables read during the forward pass are tracked. Explicit
            # tape.watch(all_vars) can fail when trainable includes Keras
            # Variable objects that are not accepted by watch().
            with tf.GradientTape() as tape:
                loglik, _ = pf(y_data)
                loss = -loglik

            grads = tape.gradient(loss, all_vars)
            grads = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(grads, all_vars)]
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)

            if trainable:
                pt_grads = grads[:len(trainable)]
                optimizer.apply_gradients(zip(pt_grads, trainable))

            epoch_ll += float(loglik.numpy())

        avg_ll = epoch_ll / n_seeds_per_epoch
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    E2E Epoch {epoch+1}/{n_epochs}  mean_loglik={avg_ll:.2f}")

    t_elapsed = time.perf_counter() - t0
    print(f"  End-to-end training complete.  [{t_elapsed:.1f}s]")
    return pt_model
