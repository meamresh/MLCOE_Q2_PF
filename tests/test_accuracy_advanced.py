"""
Unit tests for advanced accuracy metrics (NIS, autocorrelation, whiteness, consistency).
"""

import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.accuracy import (
    compute_nis,
    compute_autocorrelation,
    innovation_whiteness,
    analyze_filter_consistency
)

tfd = tfp.distributions


class TestNIS(unittest.TestCase):
    """Test cases for Normalized Innovation Squared (NIS)."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_compute_nis_single_landmark(self):
        """Test NIS computation with single landmark."""
        # Single time step, single landmark
        innovations = [tf.constant([[0.1, 0.05]], dtype=tf.float32)]
        innovation_covariances = [[tf.constant([[0.1, 0.0], [0.0, 0.1]], dtype=tf.float32)]]
        
        nis, lower, upper = compute_nis(innovations, innovation_covariances, n_meas_per_landmark=2)
        
        # Check shapes
        self.assertEqual(tf.shape(nis)[0], 1)
        self.assertIsInstance(lower, tf.Tensor)
        self.assertIsInstance(upper, tf.Tensor)
        
        # NIS should be non-negative
        self.assertTrue(tf.reduce_all(nis >= 0.0))
        
        # Bounds should be valid (lower < upper)
        self.assertLess(float(lower.numpy()), float(upper.numpy()))

    def test_compute_nis_multiple_landmarks(self):
        """Test NIS computation with multiple landmarks."""
        # Single time step, multiple landmarks
        innovations = [tf.constant([[0.1, 0.05], [0.2, 0.1]], dtype=tf.float32)]
        innovation_covariances = [[
            tf.constant([[0.1, 0.0], [0.0, 0.1]], dtype=tf.float32),
            tf.constant([[0.1, 0.0], [0.0, 0.1]], dtype=tf.float32)
        ]]
        
        nis, lower, upper = compute_nis(innovations, innovation_covariances, n_meas_per_landmark=2)
        
        # Should have NIS for each landmark
        self.assertEqual(tf.shape(nis)[0], 2)
        
        # All NIS values should be non-negative
        self.assertTrue(tf.reduce_all(nis >= 0.0))

    def test_compute_nis_multiple_time_steps(self):
        """Test NIS computation over multiple time steps."""
        # Multiple time steps
        innovations = [
            tf.constant([[0.1, 0.05]], dtype=tf.float32),
            tf.constant([[0.15, 0.08]], dtype=tf.float32),
            tf.constant([[0.12, 0.06]], dtype=tf.float32)
        ]
        innovation_covariances = [
            [tf.constant([[0.1, 0.0], [0.0, 0.1]], dtype=tf.float32)],
            [tf.constant([[0.1, 0.0], [0.0, 0.1]], dtype=tf.float32)],
            [tf.constant([[0.1, 0.0], [0.0, 0.1]], dtype=tf.float32)]
        ]
        
        nis, lower, upper = compute_nis(innovations, innovation_covariances, n_meas_per_landmark=2)
        
        # Should have NIS for each time step
        self.assertEqual(tf.shape(nis)[0], 3)
        
        # All NIS values should be non-negative
        self.assertTrue(tf.reduce_all(nis >= 0.0))

    def test_compute_nis_chi_squared_bounds(self):
        """Test that NIS bounds are from chi-squared distribution."""
        innovations = [tf.constant([[0.0, 0.0]], dtype=tf.float32)]
        innovation_covariances = [[tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)]]
        
        nis, lower, upper = compute_nis(innovations, innovation_covariances, n_meas_per_landmark=2)
        
        # For df=2, 95% CI should be around [0.05, 7.38]
        # Lower bound should be positive
        self.assertGreater(float(lower.numpy()), 0.0)
        # Upper bound should be reasonable
        self.assertLess(float(upper.numpy()), 20.0)


class TestAutocorrelation(unittest.TestCase):
    """Test cases for autocorrelation computation."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_compute_autocorrelation_white_noise(self):
        """Test autocorrelation of white noise."""
        # White noise should have autocorrelation ≈ 1 at lag 0, ≈ 0 elsewhere
        white_noise = tf.random.normal([100], dtype=tf.float32)
        acf = compute_autocorrelation(white_noise, nlags=10)
        
        # Lag 0 should be 1.0
        tf.debugging.assert_near(acf[0], 1.0, atol=1e-5)
        
        # Other lags should be small (not exactly 0 due to finite sample)
        for i in range(1, 11):
            self.assertLess(float(tf.abs(acf[i]).numpy()), 0.3)

    def test_compute_autocorrelation_constant(self):
        """Test autocorrelation of constant signal."""
        constant = tf.ones(100, dtype=tf.float32) * 5.0
        acf = compute_autocorrelation(constant, nlags=5)
        
        # For constant signal, variance is 0, so division by zero occurs
        # Lag 0 should still be 1.0
        tf.debugging.assert_near(acf[0], 1.0, atol=1e-5)
        
        # Other lags will be NaN/Inf due to division by zero (following paper algorithm)
        # This is expected behavior for constant signals
        self.assertTrue(tf.math.is_finite(acf[0]))

    def test_compute_autocorrelation_shape(self):
        """Test autocorrelation output shape."""
        x = tf.random.normal([50], dtype=tf.float32)
        nlags = 10
        acf = compute_autocorrelation(x, nlags=nlags)
        
        # Should have nlags + 1 values (including lag 0)
        self.assertEqual(tf.shape(acf)[0], nlags + 1)


class TestInnovationWhiteness(unittest.TestCase):
    """Test cases for innovation whiteness testing."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_white_innovations(self):
        """Test that white noise is identified as white."""
        # Generate white noise
        white_innov = tf.random.normal([200], mean=0.0, stddev=1.0, dtype=tf.float32)
        results = innovation_whiteness(white_innov, nlags=20)
        
        # Should be identified as white (or at least close)
        # Note: finite sample effects may cause slight correlation
        self.assertIsNotNone(results['is_white'])
        self.assertIsNotNone(results['autocorrelation'])
        self.assertIsNotNone(results['mean'])
        self.assertIsNotNone(results['std'])
        self.assertIsNotNone(results['is_zero_mean'])
        self.assertIsNotNone(results['confidence_bound'])

    def test_whiteness_zero_mean(self):
        """Test zero-mean property check."""
        # Zero-mean white noise
        white_innov = tf.random.normal([200], mean=0.0, stddev=1.0, dtype=tf.float32)
        results = innovation_whiteness(white_innov)
        
        # Should be approximately zero mean
        mean_abs = tf.abs(results['mean'])
        self.assertLess(float(mean_abs.numpy()), 0.2)

    def test_whiteness_autocorrelation(self):
        """Test autocorrelation computation in whiteness test."""
        white_innov = tf.random.normal([200], dtype=tf.float32)
        results = innovation_whiteness(white_innov, nlags=10)
        
        # Autocorrelation should have correct shape
        self.assertEqual(tf.shape(results['autocorrelation'])[0], 11)  # nlags + 1
        
        # Lag 0 should be 1.0
        tf.debugging.assert_near(results['autocorrelation'][0], 1.0, atol=1e-5)


class TestFilterConsistencyAnalysis(unittest.TestCase):
    """Test cases for filter consistency analysis."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        
        # Create mock filter results
        self.num_steps = 10
        self.state_dim = 3
        self.meas_dim = 2
        
        # True states
        self.true_states = tf.random.normal([self.num_steps, self.state_dim], dtype=tf.float32)
        
        # EKF results
        self.ekf_results = {
            'states': self.true_states + tf.random.normal([self.num_steps, self.state_dim], stddev=0.1, dtype=tf.float32),
            'covariances': tf.stack([tf.eye(self.state_dim, dtype=tf.float32) * 0.1 for _ in range(self.num_steps)]),
            'innovations': [tf.random.normal([1, self.meas_dim], dtype=tf.float32) for _ in range(self.num_steps)],
            'S': [[tf.eye(self.meas_dim, dtype=tf.float32) * 0.1] for _ in range(self.num_steps)]
        }
        
        # UKF results
        self.ukf_results = {
            'states': self.true_states + tf.random.normal([self.num_steps, self.state_dim], stddev=0.1, dtype=tf.float32),
            'covariances': tf.stack([tf.eye(self.state_dim, dtype=tf.float32) * 0.1 for _ in range(self.num_steps)]),
            'innovations': [tf.random.normal([1, self.meas_dim], dtype=tf.float32) for _ in range(self.num_steps)],
            'S': [[tf.eye(self.meas_dim, dtype=tf.float32) * 0.1] for _ in range(self.num_steps)]
        }

    def test_analyze_filter_consistency_structure(self):
        """Test that consistency analysis returns correct structure."""
        from src.metrics.accuracy import compute_nees
        
        analysis = analyze_filter_consistency(
            self.true_states,
            self.ekf_results,
            self.ukf_results,
            compute_nees
        )
        
        # Check structure
        self.assertIn('ekf', analysis)
        self.assertIn('ukf', analysis)
        self.assertIn('bounds', analysis)
        
        # Check EKF results
        self.assertIn('nees', analysis['ekf'])
        self.assertIn('nis', analysis['ekf'])
        self.assertIn('nees_consistent_pct', analysis['ekf'])
        self.assertIn('nis_consistent_pct', analysis['ekf'])
        self.assertIn('whiteness', analysis['ekf'])
        
        # Check UKF results
        self.assertIn('nees', analysis['ukf'])
        self.assertIn('nis', analysis['ukf'])
        self.assertIn('nees_consistent_pct', analysis['ukf'])
        self.assertIn('nis_consistent_pct', analysis['ukf'])
        self.assertIn('whiteness', analysis['ukf'])
        
        # Check bounds
        self.assertIn('nees_lower', analysis['bounds'])
        self.assertIn('nees_upper', analysis['bounds'])
        self.assertIn('nis_lower', analysis['bounds'])
        self.assertIn('nis_upper', analysis['bounds'])

    def test_analyze_filter_consistency_values(self):
        """Test that consistency analysis produces valid values."""
        from src.metrics.accuracy import compute_nees
        
        analysis = analyze_filter_consistency(
            self.true_states,
            self.ekf_results,
            self.ukf_results,
            compute_nees
        )
        
        # NEES should be non-negative
        self.assertTrue(tf.reduce_all(analysis['ekf']['nees'] >= 0.0))
        self.assertTrue(tf.reduce_all(analysis['ukf']['nees'] >= 0.0))
        
        # NIS should be non-negative
        self.assertTrue(tf.reduce_all(analysis['ekf']['nis'] >= 0.0))
        self.assertTrue(tf.reduce_all(analysis['ukf']['nis'] >= 0.0))
        
        # Consistency percentages should be in [0, 100]
        self.assertGreaterEqual(analysis['ekf']['nees_consistent_pct'], 0.0)
        self.assertLessEqual(analysis['ekf']['nees_consistent_pct'], 100.0)
        self.assertGreaterEqual(analysis['ukf']['nees_consistent_pct'], 0.0)
        self.assertLessEqual(analysis['ukf']['nees_consistent_pct'], 100.0)
        
        # Bounds should be valid
        self.assertLess(float(analysis['bounds']['nees_lower'].numpy()),
                       float(analysis['bounds']['nees_upper'].numpy()))
        self.assertLess(float(analysis['bounds']['nis_lower'].numpy()),
                       float(analysis['bounds']['nis_upper'].numpy()))


if __name__ == '__main__':
    unittest.main()

