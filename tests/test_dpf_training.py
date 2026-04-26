"""
Unit tests for DPF training utilities.

Covers:
  - kde_log_prob
  - resampler_loss
  - collect_training_data
  - _KitagawaBootstrapWrapper / _build_kitagawa_bootstrap
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.dpf.training import (
    kde_log_prob,
    resampler_loss,
    collect_training_data,
    _build_kitagawa_bootstrap,
)
from src.filters.dpf.particle_transformer import ParticleTransformer


class TestKdeLogProb(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_output_shape(self):
        B, N, M, d = 2, 10, 5, 1
        particles = tf.random.normal([B, N, d])
        eval_pts = tf.random.normal([B, M, d])
        lp = kde_log_prob(particles, eval_pts, bandwidth=0.5)
        self.assertEqual(lp.shape, (B, M))

    def test_finite_outputs(self):
        B, N, d = 3, 8, 1
        particles = tf.random.normal([B, N, d])
        lp = kde_log_prob(particles, particles, bandwidth=1.0)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(lp)))

    def test_higher_at_particle_locations(self):
        """KDE density should be higher at particle locations than far away."""
        B, N, d = 1, 20, 1
        particles = tf.random.normal([B, N, d])
        far_pts = tf.constant([[[100.0]]])
        lp_near = kde_log_prob(particles, particles[:, :1, :], bandwidth=0.5)
        lp_far = kde_log_prob(particles, far_pts, bandwidth=0.5)
        self.assertGreater(float(lp_near[0, 0]), float(lp_far[0, 0]))


class TestResamplerLoss(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.pt = ParticleTransformer(
            state_dim=1, n_seed_bank=16, d_model=8, n_heads=2, d_ff=16
        )

    def test_returns_finite_scalar(self):
        B, N = 2, 10
        particles = tf.random.normal([B, N, 1])
        weights = tf.nn.softmax(tf.random.normal([B, N]), axis=-1)
        loss = resampler_loss(self.pt, particles, weights, bandwidth=0.5)
        self.assertEqual(loss.shape, ())
        self.assertTrue(tf.math.is_finite(loss))

    def test_gradient_flows_to_pt(self):
        B, N = 2, 8
        particles = tf.random.normal([B, N, 1])
        weights = tf.nn.softmax(tf.random.normal([B, N]), axis=-1)
        with tf.GradientTape() as tape:
            loss = resampler_loss(self.pt, particles, weights)
        grads = tape.gradient(loss, self.pt.trainable_variables)
        has_grad = any(g is not None for g in grads)
        self.assertTrue(has_grad)


class TestBuildKitagawaBootstrap(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_returns_model_and_phi(self):
        model, phi = _build_kitagawa_bootstrap(phi=1.0)
        self.assertIsNotNone(model.sample_initial)
        self.assertIsNotNone(model.transition)
        self.assertTrue(phi.trainable)

    def test_sample_initial_shapes(self):
        model, _ = _build_kitagawa_bootstrap()
        x, log_w = model.sample_initial(15, tf.constant([[1.0]]))
        self.assertEqual(x.shape, (15, 1))

    def test_transition_shapes(self):
        model, _ = _build_kitagawa_bootstrap()
        x0, _ = model.sample_initial(10, tf.constant([[0.5]]))
        x1, log_w1 = model.transition(1, x0, tf.constant([[0.5]]))
        self.assertEqual(x1.shape, (10, 1))


class TestCollectTrainingData(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_output_shapes(self):
        y = tf.random.normal([8, 1])
        particles, weights = collect_training_data(
            y, n_episodes=2, N=10, T=8
        )
        self.assertEqual(particles.shape[1], 10)
        self.assertEqual(particles.shape[2], 1)
        self.assertEqual(weights.shape[1], 10)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(particles)))

    def test_weights_nonnegative(self):
        y = tf.random.normal([6, 1])
        _, weights = collect_training_data(y, n_episodes=2, N=8, T=6)
        self.assertTrue(tf.reduce_all(weights >= 0.0))

    def test_weights_sum_to_one_per_step(self):
        """Each time-step's weight vector should sum to ~1 (softmax output)."""
        y = tf.random.normal([6, 1])
        _, weights = collect_training_data(y, n_episodes=1, N=10, T=6)
        sums = tf.reduce_sum(weights, axis=1)
        for s in sums:
            self.assertAlmostEqual(float(s), 1.0, places=4)


class TestResamplerLossDecreases(unittest.TestCase):
    """Training for a few steps should reduce the resampler loss."""

    def setUp(self):
        tf.random.set_seed(42)

    def test_loss_decreases(self):
        pt = ParticleTransformer(
            state_dim=1, n_seed_bank=16, d_model=8, n_heads=2, d_ff=16
        )
        B, N = 4, 10
        particles = tf.random.normal([B, N, 1])
        weights = tf.nn.softmax(tf.random.normal([B, N]), axis=-1)

        loss_before = float(resampler_loss(pt, particles, weights).numpy())
        optimizer = tf.keras.optimizers.Adam(1e-3)
        for _ in range(20):
            with tf.GradientTape() as tape:
                loss = resampler_loss(pt, particles, weights)
            grads = tape.gradient(loss, pt.trainable_variables)
            optimizer.apply_gradients(zip(grads, pt.trainable_variables))

        loss_after = float(resampler_loss(pt, particles, weights).numpy())
        self.assertLess(loss_after, loss_before)


if __name__ == "__main__":
    unittest.main()
