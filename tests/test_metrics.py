"""
Unit tests for metrics (accuracy and stability).
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.accuracy import (
    compute_rmse, compute_mae, compute_nees,
    compute_per_dimension_rmse, compute_nll
)
from src.metrics.stability import (
    compute_condition_numbers, check_symmetry,
    check_positive_definite, compute_frobenius_norm_difference,
    compute_trace, compute_log_determinant, enforce_symmetry
)


class TestAccuracyMetrics(unittest.TestCase):
    """Test cases for accuracy metrics."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_compute_rmse(self):
        """Test RMSE computation."""
        estimates = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        ground_truth = tf.constant([[1.1, 2.1], [2.9, 4.1]], dtype=tf.float32)
        
        rmse = compute_rmse(estimates, ground_truth)
        
        # Should be positive
        self.assertGreater(rmse, 0.0)
        # Should be small for these close values
        self.assertLess(rmse, 0.2)

    def test_compute_rmse_3d(self):
        """Test RMSE with 3D tensors."""
        estimates = tf.constant([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=tf.float32)
        ground_truth = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        
        rmse = compute_rmse(estimates, ground_truth)
        self.assertGreaterEqual(rmse, 0.0)

    def test_compute_mae(self):
        """Test MAE computation."""
        estimates = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        ground_truth = tf.constant([[1.1, 2.1], [2.9, 4.1]], dtype=tf.float32)
        
        mae = compute_mae(estimates, ground_truth)
        
        self.assertGreater(mae, 0.0)
        self.assertLess(mae, 0.2)

    def test_compute_nees(self):
        """Test NEES computation."""
        estimates = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        covariances = tf.constant([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ], dtype=tf.float32)
        ground_truth = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        
        nees = compute_nees(estimates, covariances, ground_truth)
        
        # Should have shape (N,)
        self.assertEqual(tf.shape(nees)[0], 2)
        # NEES should be non-negative
        self.assertTrue(tf.reduce_all(nees >= 0))

    def test_compute_per_dimension_rmse(self):
        """Test per-dimension RMSE."""
        estimates = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        ground_truth = tf.constant([[1.1, 2.1], [2.9, 4.1]], dtype=tf.float32)
        
        rmse_per_dim = compute_per_dimension_rmse(estimates, ground_truth)
        
        # Should have shape (n,)
        self.assertEqual(tf.shape(rmse_per_dim)[0], 2)
        self.assertTrue(tf.reduce_all(rmse_per_dim >= 0))


class TestStabilityMetrics(unittest.TestCase):
    """Test cases for stability metrics."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_compute_condition_numbers(self):
        """Test condition number computation."""
        # Well-conditioned matrix
        P = tf.constant([
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ], dtype=tf.float32)
        
        cond = compute_condition_numbers(P)
        
        self.assertEqual(tf.shape(cond)[0], 2)
        # Condition number should be >= 1
        self.assertTrue(tf.reduce_all(cond >= 1.0))

    def test_check_symmetry(self):
        """Test symmetry checking."""
        # Symmetric matrix
        P = tf.constant([
            [[1.0, 0.5], [0.5, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]]
        ], dtype=tf.float32)
        
        sym_error = check_symmetry(P)
        
        self.assertEqual(tf.shape(sym_error)[0], 2)
        # Should be small for symmetric matrices
        self.assertTrue(tf.reduce_all(sym_error < 1e-5))

    def test_check_positive_definite(self):
        """Test positive definiteness check."""
        # Positive definite matrix
        P = tf.constant([
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ], dtype=tf.float32)
        
        min_eigvals, is_pd = check_positive_definite(P)
        
        self.assertEqual(tf.shape(min_eigvals)[0], 2)
        self.assertEqual(tf.shape(is_pd)[0], 2)
        # Should be positive definite
        self.assertTrue(tf.reduce_all(is_pd))

    def test_compute_frobenius_norm_difference(self):
        """Test Frobenius norm difference."""
        P1 = tf.constant([
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]]
        ], dtype=tf.float32)
        
        P2 = tf.constant([
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]]
        ], dtype=tf.float32)
        
        diff = compute_frobenius_norm_difference(P1, P2)
        
        self.assertEqual(tf.shape(diff)[0], 2)
        # Should be zero for identical matrices
        self.assertTrue(tf.reduce_all(diff < 1e-6))

    def test_compute_trace(self):
        """Test trace computation."""
        P = tf.constant([
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]]
        ], dtype=tf.float32)
        
        traces = compute_trace(P)
        
        self.assertEqual(tf.shape(traces)[0], 2)
        # Trace should be sum of diagonal elements
        tf.debugging.assert_near(traces[0], 3.0, rtol=1e-6)
        tf.debugging.assert_near(traces[1], 7.0, rtol=1e-6)

    def test_compute_log_determinant(self):
        """Test log-determinant computation."""
        P = tf.constant([
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ], dtype=tf.float32)
        
        log_dets = compute_log_determinant(P)
        
        self.assertEqual(tf.shape(log_dets)[0], 2)
        # log(2*1) = log(2) ≈ 0.693
        tf.debugging.assert_near(log_dets[0], tf.math.log(2.0), rtol=1e-5)

    def test_enforce_symmetry(self):
        """Test symmetry enforcement."""
        # Slightly asymmetric matrix
        P = tf.constant([[1.0, 0.51], [0.49, 1.0]], dtype=tf.float32)
        
        P_sym = enforce_symmetry(P)
        
        # Should be symmetric
        sym_error = tf.reduce_max(tf.abs(P_sym - tf.transpose(P_sym)))
        self.assertLess(float(sym_error), 1e-6)


if __name__ == '__main__':
    unittest.main()

