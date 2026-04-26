"""
Unit tests for DPF filter variants and ParticleTransformer.

Covers:
  - BootstrapModel
  - StandardParticleFilter
  - DifferentiableParticleFilter
  - SoftResamplingParticleFilter
  - StopGradientParticleFilter
  - ParticleTransformerFilter
  - WeightedMultiHeadAttention, FeedForward, TransformerBlock
  - ParticleTransformer
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.dpf.diff_particle_filter import (
    BootstrapModel,
    StandardParticleFilter,
    DifferentiableParticleFilter,
    SoftResamplingParticleFilter,
    StopGradientParticleFilter,
    ParticleTransformerFilter,
)
from src.filters.dpf.particle_transformer import (
    WeightedMultiHeadAttention,
    FeedForward,
    TransformerBlock,
    ParticleTransformer,
)


def _make_kitagawa_model():
    """Simple 1D Kitagawa bootstrap model for testing."""
    sigma_v = 3.16  # sqrt(10)
    sigma_w = 1.0

    def sample_initial(N, y0):
        x = tf.random.normal([N, 1]) * 2.0
        diff = (y0 - x ** 2 / 20.0)
        log_w = -0.5 * tf.reduce_sum(diff ** 2, axis=-1) / sigma_w ** 2
        return x, log_w

    def transition(t, x_prev, y_t):
        t_f = tf.cast(t, tf.float32)
        x_mean = 0.5 * x_prev + 25.0 * x_prev / (1.0 + x_prev ** 2) + 8.0 * tf.cos(1.2 * t_f)
        x = x_mean + tf.random.normal(tf.shape(x_prev)) * sigma_v
        diff = (y_t - x ** 2 / 20.0)
        log_w = -0.5 * tf.reduce_sum(diff ** 2, axis=-1) / sigma_w ** 2
        return x, log_w

    return BootstrapModel(sample_initial=sample_initial, transition=transition)


class TestBootstrapModel(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_creation(self):
        model = _make_kitagawa_model()
        self.assertIsNotNone(model.sample_initial)
        self.assertIsNotNone(model.transition)

    def test_sample_initial(self):
        model = _make_kitagawa_model()
        x, log_w = model.sample_initial(20, tf.constant([[1.0]]))
        self.assertEqual(x.shape, (20, 1))
        self.assertEqual(log_w.shape[0], 20)


class TestStandardParticleFilter(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.model = _make_kitagawa_model()
        self.N = 20

    def test_instantiation(self):
        pf = StandardParticleFilter(self.model, self.N)
        self.assertEqual(pf.num_particles, self.N)

    def test_call_output(self):
        pf = StandardParticleFilter(self.model, self.N)
        T = 5
        y = tf.random.normal([T, 1])
        loglik, final_log_w = pf(y)
        self.assertEqual(loglik.shape, ())
        self.assertTrue(tf.math.is_finite(loglik))
        self.assertEqual(final_log_w.shape, (self.N,))

    def test_log_weights_finite(self):
        pf = StandardParticleFilter(self.model, self.N)
        y = tf.random.normal([8, 1])
        _, final_log_w = pf(y)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(final_log_w)))

    def test_longer_sequence(self):
        """Filter should not degenerate over a moderately long sequence."""
        pf = StandardParticleFilter(self.model, 50)
        y = tf.random.normal([20, 1])
        loglik, _ = pf(y)
        self.assertTrue(tf.math.is_finite(loglik))


class TestDifferentiableParticleFilter(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.model = _make_kitagawa_model()
        self.N = 15

    def test_call_output(self):
        dpf = DifferentiableParticleFilter(
            self.model, self.N, epsilon=1.0, sinkhorn_iters=10
        )
        y = tf.random.normal([5, 1])
        loglik, final_log_w = dpf(y)
        self.assertEqual(loglik.shape, ())
        self.assertTrue(tf.math.is_finite(loglik))

    def test_loglik_different_for_different_data(self):
        """Log-likelihood should differ for different observation sequences."""
        dpf1 = DifferentiableParticleFilter(
            self.model, self.N, epsilon=1.0, sinkhorn_iters=10
        )
        dpf2 = DifferentiableParticleFilter(
            self.model, self.N, epsilon=1.0, sinkhorn_iters=10
        )
        tf.random.set_seed(42)
        y1 = tf.random.normal([5, 1])
        tf.random.set_seed(42)
        ll1, _ = dpf1(y1)
        y2 = tf.constant([[100.0]] * 5)
        tf.random.set_seed(42)
        ll2, _ = dpf2(y2)
        self.assertNotAlmostEqual(float(ll1), float(ll2), places=1)


class TestSoftResamplingParticleFilter(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.model = _make_kitagawa_model()

    def test_call_output(self):
        pf = SoftResamplingParticleFilter(self.model, 15, alpha=0.5)
        y = tf.random.normal([5, 1])
        loglik, final_log_w = pf(y)
        self.assertEqual(loglik.shape, ())
        self.assertTrue(tf.math.is_finite(loglik))


class TestStopGradientParticleFilter(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.model = _make_kitagawa_model()

    def test_call_output(self):
        pf = StopGradientParticleFilter(self.model, 15)
        y = tf.random.normal([5, 1])
        loglik, final_log_w = pf(y)
        self.assertEqual(loglik.shape, ())
        self.assertTrue(tf.math.is_finite(loglik))


class TestParticleTransformerFilter(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.model = _make_kitagawa_model()
        self.N = 15

    def test_call_output(self):
        pt = ParticleTransformer(
            state_dim=1, n_seed_bank=16, d_model=8, n_heads=2, d_ff=16
        )
        pf = ParticleTransformerFilter(self.model, self.N, pt_model=pt)
        y = tf.random.normal([5, 1])
        loglik, final_log_w = pf(y)
        self.assertEqual(loglik.shape, ())
        self.assertTrue(tf.math.is_finite(loglik))

    def test_gradient_flows_to_pt_weights(self):
        """Gradient should reach ParticleTransformer weights through the filter."""
        pt = ParticleTransformer(
            state_dim=1, n_seed_bank=16, d_model=8, n_heads=2, d_ff=16
        )
        pf = ParticleTransformerFilter(self.model, self.N, pt_model=pt)
        y = tf.random.normal([5, 1])
        with tf.GradientTape() as tape:
            loglik, _ = pf(y)
            loss = -loglik
        grads = tape.gradient(loss, pt.trainable_variables)
        has_grad = any(g is not None for g in grads)
        self.assertTrue(has_grad, "Gradient should flow to PT weights")


# =========================================================================
# Particle Transformer components
# =========================================================================

class TestWeightedMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.attn = WeightedMultiHeadAttention(d_model=8, n_heads=2)

    def test_output_shape(self):
        B, N = 2, 5
        q = tf.random.normal([B, N, 8])
        k = tf.random.normal([B, N, 8])
        v = tf.random.normal([B, N, 8])
        out = self.attn(q, k, v)
        self.assertEqual(out.shape, (B, N, 8))

    def test_weighted_output_shape(self):
        B, N = 2, 5
        q = tf.random.normal([B, N, 8])
        k = tf.random.normal([B, N, 8])
        v = tf.random.normal([B, N, 8])
        w = tf.nn.softmax(tf.random.normal([B, N]), axis=-1)
        out = self.attn(q, k, v, weights=w)
        self.assertEqual(out.shape, (B, N, 8))

    def test_finite_output(self):
        B, N = 3, 8
        q = tf.random.normal([B, N, 8])
        out = self.attn(q, q, q)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))


class TestFeedForward(unittest.TestCase):

    def test_output_shape(self):
        ff = FeedForward(d_model=8, d_ff=16)
        x = tf.random.normal([2, 5, 8])
        out = ff(x)
        self.assertEqual(out.shape, (2, 5, 8))


class TestParticleTransformer(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.pt = ParticleTransformer(
            state_dim=1, n_seed_bank=16, d_model=8, n_heads=2, d_ff=16
        )

    def test_forward_2d_input(self):
        N = 10
        particles = tf.random.normal([N, 1])
        weights = tf.nn.softmax(tf.random.normal([N]))
        out = self.pt(particles, weights)
        self.assertEqual(out.shape, (N, 1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))

    def test_forward_3d_input(self):
        B, N = 2, 10
        particles = tf.random.normal([B, N, 1])
        weights = tf.nn.softmax(tf.random.normal([B, N]), axis=-1)
        out = self.pt(particles, weights)
        self.assertEqual(out.shape, (B, N, 1))

    def test_resampled_particles_finite(self):
        """All resampled particles should remain finite."""
        N = 20
        particles = tf.random.normal([N, 1])
        weights = tf.nn.softmax(tf.random.normal([N]))
        out = self.pt(particles, weights)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))

    def test_uniform_weights_produce_spread(self):
        """With uniform weights, resampled particles should have nonzero spread."""
        N = 30
        particles = tf.random.normal([N, 1]) * 3.0
        weights = tf.ones([N]) / float(N)
        out = self.pt(particles, weights)
        spread = tf.math.reduce_std(out)
        self.assertGreater(float(spread), 0.0)


if __name__ == "__main__":
    unittest.main()
