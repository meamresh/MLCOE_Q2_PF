"""
Unit tests for resampling algorithms.
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.resampling import (
    systematic_resample,
    multinomial_resample,
    stratified_resample,
    residual_resample,
    resample_particles,
    compute_ess,
    should_resample
)


class TestSystematicResample(unittest.TestCase):
    """Test cases for systematic resampling."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_output_shape(self):
        """Test that output shape matches input."""
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
        indices = systematic_resample(weights)
        
        self.assertEqual(tf.shape(indices)[0], 4)
        self.assertEqual(indices.dtype, tf.int32)

    def test_valid_indices(self):
        """Test that all indices are valid."""
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
        indices = systematic_resample(weights)
        
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < 4))

    def test_high_weight_particle_selected(self):
        """Test that high weight particles are preferentially selected."""
        # One particle has most weight
        weights = tf.constant([0.01, 0.01, 0.01, 0.97], dtype=tf.float32)
        indices = systematic_resample(weights)
        
        # The high-weight particle (index 3) should appear multiple times
        count_3 = tf.reduce_sum(tf.cast(indices == 3, tf.int32))
        self.assertGreater(int(count_3), 2)

    def test_uniform_weights(self):
        """Test resampling with uniform weights."""
        N = 100
        weights = tf.ones(N, dtype=tf.float32) / N
        indices = systematic_resample(weights)
        
        self.assertEqual(tf.shape(indices)[0], N)
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < N))

    def test_unnormalized_weights(self):
        """Test that unnormalized weights are handled correctly."""
        weights = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)  # Sum = 10
        indices = systematic_resample(weights)
        
        self.assertEqual(tf.shape(indices)[0], 4)
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < 4))


class TestMultinomialResample(unittest.TestCase):
    """Test cases for multinomial resampling."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_output_shape(self):
        """Test that output shape matches input."""
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
        indices = multinomial_resample(weights)
        
        self.assertEqual(tf.shape(indices)[0], 4)
        self.assertEqual(indices.dtype, tf.int32)

    def test_valid_indices(self):
        """Test that all indices are valid."""
        weights = tf.constant([0.25, 0.25, 0.25, 0.25], dtype=tf.float32)
        indices = multinomial_resample(weights)
        
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < 4))

    def test_statistical_correctness(self):
        """Test that multinomial sampling is statistically correct."""
        weights = tf.constant([0.5, 0.3, 0.15, 0.05], dtype=tf.float32)
        
        # Run many resamples and check empirical distribution
        counts = tf.zeros(4, dtype=tf.int32)
        n_trials = 1000
        for _ in range(n_trials):
            indices = multinomial_resample(weights)
            for i in range(4):
                counts = counts + tf.one_hot(i, 4, dtype=tf.int32) * tf.reduce_sum(
                    tf.cast(indices == i, tf.int32)
                )
        
        # High weight particle should have highest count
        self.assertEqual(int(tf.argmax(counts)), 0)


class TestStratifiedResample(unittest.TestCase):
    """Test cases for stratified resampling."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_output_shape(self):
        """Test that output shape matches input."""
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
        indices = stratified_resample(weights)
        
        self.assertEqual(tf.shape(indices)[0], 4)
        self.assertEqual(indices.dtype, tf.int32)

    def test_valid_indices(self):
        """Test that all indices are valid."""
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
        indices = stratified_resample(weights)
        
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < 4))


class TestResidualResample(unittest.TestCase):
    """Test cases for residual resampling."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_output_shape(self):
        """Test that output shape matches input."""
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
        indices = residual_resample(weights)
        
        self.assertEqual(tf.shape(indices)[0], 4)

    def test_valid_indices(self):
        """Test that all indices are valid."""
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
        indices = residual_resample(weights)
        
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < 4))


class TestResampleParticles(unittest.TestCase):
    """Test cases for the resample_particles convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.particles = tf.constant([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ], dtype=tf.float32)
        self.weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)

    def test_systematic_method(self):
        """Test resampling with systematic method."""
        new_particles, new_weights = resample_particles(
            self.particles, self.weights, method='systematic'
        )
        
        self.assertEqual(new_particles.shape, self.particles.shape)
        self.assertEqual(new_weights.shape, self.weights.shape)
        tf.debugging.assert_near(tf.reduce_sum(new_weights), 1.0, atol=1e-5)

    def test_multinomial_method(self):
        """Test resampling with multinomial method."""
        new_particles, new_weights = resample_particles(
            self.particles, self.weights, method='multinomial'
        )
        
        self.assertEqual(new_particles.shape, self.particles.shape)
        tf.debugging.assert_near(tf.reduce_sum(new_weights), 1.0, atol=1e-5)

    def test_stratified_method(self):
        """Test resampling with stratified method."""
        new_particles, new_weights = resample_particles(
            self.particles, self.weights, method='stratified'
        )
        
        self.assertEqual(new_particles.shape, self.particles.shape)
        tf.debugging.assert_near(tf.reduce_sum(new_weights), 1.0, atol=1e-5)

    def test_residual_method(self):
        """Test resampling with residual method."""
        new_particles, new_weights = resample_particles(
            self.particles, self.weights, method='residual'
        )
        
        self.assertEqual(new_particles.shape, self.particles.shape)
        tf.debugging.assert_near(tf.reduce_sum(new_weights), 1.0, atol=1e-5)

    def test_uniform_weights_after_resample(self):
        """Test that weights are uniform after resampling."""
        _, new_weights = resample_particles(
            self.particles, self.weights, method='systematic'
        )
        
        expected_weight = 1.0 / 4
        tf.debugging.assert_near(
            new_weights,
            tf.ones(4, dtype=tf.float32) * expected_weight,
            atol=1e-5
        )

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError):
            resample_particles(self.particles, self.weights, method='invalid')


class TestComputeESS(unittest.TestCase):
    """Test cases for ESS computation."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_uniform_weights_max_ess(self):
        """Test that uniform weights give maximum ESS = N."""
        N = 100
        uniform_weights = tf.ones(N, dtype=tf.float32) / N
        ess = compute_ess(uniform_weights)
        
        tf.debugging.assert_near(ess, float(N), atol=1e-3)

    def test_degenerate_weights_min_ess(self):
        """Test that degenerate weights give ESS = 1."""
        N = 100
        degenerate_weights = tf.zeros(N, dtype=tf.float32)
        degenerate_weights = tf.tensor_scatter_nd_update(
            degenerate_weights, [[0]], [1.0]
        )
        ess = compute_ess(degenerate_weights)
        
        tf.debugging.assert_near(ess, 1.0, atol=1e-3)

    def test_ess_bounds(self):
        """Test that ESS is always between 1 and N."""
        N = 50
        random_weights = tf.abs(tf.random.normal([N], dtype=tf.float32))
        random_weights = random_weights / tf.reduce_sum(random_weights)
        ess = compute_ess(random_weights)
        
        self.assertGreaterEqual(float(ess), 1.0)
        self.assertLessEqual(float(ess), float(N))

    def test_unnormalized_weights(self):
        """Test that unnormalized weights are handled correctly."""
        weights = tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)  # Sum = 4
        ess = compute_ess(weights)
        
        # Should normalize and give ESS = 4
        tf.debugging.assert_near(ess, 4.0, atol=1e-3)


class TestShouldResample(unittest.TestCase):
    """Test cases for should_resample function."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_uniform_weights_no_resample(self):
        """Test that uniform weights don't trigger resampling."""
        N = 100
        weights = tf.ones(N, dtype=tf.float32) / N
        
        result = should_resample(weights, threshold=0.5)
        self.assertFalse(result)

    def test_degenerate_weights_resample(self):
        """Test that degenerate weights trigger resampling."""
        N = 100
        weights = tf.zeros(N, dtype=tf.float32)
        weights = tf.tensor_scatter_nd_update(weights, [[0]], [1.0])
        
        result = should_resample(weights, threshold=0.5)
        self.assertTrue(result)

    def test_threshold_effect(self):
        """Test that threshold parameter works correctly."""
        N = 100
        # Create more degenerate weights with low ESS
        weights = tf.ones(N, dtype=tf.float32) * 0.001
        weights = tf.tensor_scatter_nd_update(
            weights, [[0], [1]], [0.9, 0.09]  # Two particles dominate
        )
        weights = weights / tf.reduce_sum(weights)
        
        # ESS should be low (around 1-2)
        ess = compute_ess(weights)
        
        # With threshold 0.5 (need ESS > 50), should resample
        result_high = should_resample(weights, threshold=0.5)
        
        # Result should be True (need resampling)
        self.assertTrue(result_high)


if __name__ == '__main__':
    unittest.main()
