"""
Unit tests for linear algebra utilities.
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.linalg import (
    regularize_covariance,
    sample_from_gaussian,
    compute_condition_number,
    localization_matrix,
    nearest_psd,
    ensure_symmetric,
    safe_cholesky,
    compute_particle_covariance
)


class TestRegularizeCovariance(unittest.TestCase):
    """Test cases for covariance regularization."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_symmetric_output(self):
        """Test that output is symmetric."""
        # Slightly asymmetric input
        P = tf.constant([[1.0, 0.501], [0.499, 1.0]], dtype=tf.float32)
        P_reg = regularize_covariance(P)
        
        # Check symmetry
        sym_error = tf.reduce_max(tf.abs(P_reg - tf.transpose(P_reg)))
        self.assertLess(float(sym_error), 1e-6)

    def test_positive_definite_output(self):
        """Test that output is positive definite."""
        P = tf.constant([[1.0, 0.5], [0.5, 1.0]], dtype=tf.float32)
        P_reg = regularize_covariance(P)
        
        eigvals = tf.linalg.eigvalsh(P_reg)
        self.assertTrue(tf.reduce_all(eigvals > 0))

    def test_diagonal_boost(self):
        """Test that diagonal is boosted by epsilon."""
        P = tf.eye(3, dtype=tf.float32)
        eps = 1e-4
        P_reg = regularize_covariance(P, eps=eps)
        
        diag = tf.linalg.diag_part(P_reg)
        tf.debugging.assert_near(diag, tf.ones(3, dtype=tf.float32) + eps, atol=1e-8)

    def test_preserves_shape(self):
        """Test that shape is preserved."""
        P = tf.random.normal([5, 5], dtype=tf.float32)
        P = tf.matmul(P, P, transpose_b=True)  # Make PSD
        P_reg = regularize_covariance(P)
        
        self.assertEqual(P_reg.shape, (5, 5))


class TestSampleFromGaussian(unittest.TestCase):
    """Test cases for Gaussian sampling."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_output_shape(self):
        """Test that output has correct shape."""
        mean = tf.constant([0.0, 0.0], dtype=tf.float32)
        cov = tf.eye(2, dtype=tf.float32)
        samples = sample_from_gaussian(mean, cov, n_samples=100)
        
        self.assertEqual(samples.shape, (100, 2))

    def test_sample_statistics(self):
        """Test that samples have approximately correct mean."""
        mean = tf.constant([1.0, 2.0], dtype=tf.float32)
        cov = tf.eye(2, dtype=tf.float32) * 0.01
        samples = sample_from_gaussian(mean, cov, n_samples=10000)
        
        sample_mean = tf.reduce_mean(samples, axis=0)
        tf.debugging.assert_near(sample_mean, mean, atol=0.1)

    def test_identity_covariance(self):
        """Test sampling with identity covariance."""
        mean = tf.zeros(3, dtype=tf.float32)
        cov = tf.eye(3, dtype=tf.float32)
        samples = sample_from_gaussian(mean, cov, n_samples=1000)
        
        sample_cov = tf.linalg.matmul(
            samples - tf.reduce_mean(samples, axis=0),
            samples - tf.reduce_mean(samples, axis=0),
            transpose_a=True
        ) / 999.0
        
        # Should be close to identity
        tf.debugging.assert_near(sample_cov, cov, atol=0.2)

    def test_handles_1d_mean(self):
        """Test that 1D mean vector is handled correctly."""
        mean = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        cov = tf.eye(3, dtype=tf.float32)
        samples = sample_from_gaussian(mean, cov, n_samples=10)
        
        self.assertEqual(samples.shape, (10, 3))


class TestComputeConditionNumber(unittest.TestCase):
    """Test cases for condition number computation."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_identity_condition_number(self):
        """Test that identity matrix has condition number 1."""
        I = tf.eye(5, dtype=tf.float32)
        cond = compute_condition_number(I)
        
        self.assertAlmostEqual(cond, 1.0, places=5)

    def test_diagonal_condition_number(self):
        """Test condition number of diagonal matrix."""
        D = tf.linalg.diag([1.0, 2.0, 4.0])
        cond = compute_condition_number(D)
        
        # Condition number should be max/min = 4/1 = 4
        self.assertAlmostEqual(cond, 4.0, places=4)

    def test_ill_conditioned_matrix(self):
        """Test condition number of ill-conditioned matrix."""
        D = tf.linalg.diag([1.0, 1e-8])
        cond = compute_condition_number(D)
        
        # Condition number should be very large
        self.assertGreater(cond, 1e7)

    def test_handles_3d_tensor(self):
        """Test that 3D tensors are handled (uses first element)."""
        batch = tf.stack([tf.eye(3, dtype=tf.float32) for _ in range(5)])
        cond = compute_condition_number(batch)
        
        self.assertAlmostEqual(cond, 1.0, places=5)


class TestLocalizationMatrix(unittest.TestCase):
    """Test cases for localization matrix computation."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_output_shape(self):
        """Test that output has correct shape."""
        d = 10
        C = localization_matrix(d, r_in=3.0)
        
        self.assertEqual(C.shape, (d, d))

    def test_symmetric(self):
        """Test that output is symmetric."""
        C = localization_matrix(20, r_in=5.0)
        
        sym_error = tf.reduce_max(tf.abs(C - tf.transpose(C)))
        self.assertLess(float(sym_error), 1e-6)

    def test_diagonal_ones(self):
        """Test that diagonal entries are 1."""
        C = localization_matrix(10, r_in=3.0)
        
        diag = tf.linalg.diag_part(C)
        tf.debugging.assert_near(diag, tf.ones(10, dtype=tf.float32), atol=1e-6)

    def test_values_in_range(self):
        """Test that all values are in [0, 1]."""
        C = localization_matrix(15, r_in=4.0)
        
        self.assertTrue(tf.reduce_all(C >= 0.0))
        self.assertTrue(tf.reduce_all(C <= 1.0))

    def test_decay_with_distance(self):
        """Test that values decay with distance."""
        C = localization_matrix(10, r_in=2.0, periodic=False)
        
        # C[0, 1] should be larger than C[0, 5]
        self.assertGreater(float(C[0, 1]), float(C[0, 5]))

    def test_periodic_wrapping(self):
        """Test periodic boundary conditions."""
        d = 10
        C = localization_matrix(d, r_in=3.0, periodic=True)
        
        # For periodic, C[0, 9] should equal C[0, 1] (distance 1 from both ends)
        tf.debugging.assert_near(C[0, 9], C[0, 1], atol=1e-6)


class TestNearestPSD(unittest.TestCase):
    """Test cases for nearest PSD projection."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_already_psd_unchanged(self):
        """Test that PSD matrix is approximately unchanged."""
        P = tf.constant([[2.0, 0.5], [0.5, 2.0]], dtype=tf.float32)
        P_psd = nearest_psd(P)
        
        tf.debugging.assert_near(P_psd, P, atol=1e-5)

    def test_non_psd_becomes_psd(self):
        """Test that non-PSD matrix becomes PSD."""
        # Matrix with negative eigenvalue (eigenvalues are 3 and -1)
        P = tf.constant([[1.0, 2.0], [2.0, 1.0]], dtype=tf.float32)
        P_psd = nearest_psd(P, epsilon=1e-6)
        
        eigvals = tf.linalg.eigvalsh(P_psd)
        # All eigenvalues should be positive (>= epsilon)
        self.assertTrue(tf.reduce_all(eigvals >= 1e-8))

    def test_symmetric_output(self):
        """Test that output is symmetric."""
        P = tf.constant([[1.0, 0.6], [0.4, 1.0]], dtype=tf.float32)
        P_psd = nearest_psd(P)
        
        sym_error = tf.reduce_max(tf.abs(P_psd - tf.transpose(P_psd)))
        self.assertLess(float(sym_error), 1e-6)


class TestEnsureSymmetric(unittest.TestCase):
    """Test cases for symmetry enforcement."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_symmetric_output(self):
        """Test that output is symmetric."""
        P = tf.constant([[1.0, 0.6], [0.4, 1.0]], dtype=tf.float32)
        P_sym = ensure_symmetric(P)
        
        sym_error = tf.reduce_max(tf.abs(P_sym - tf.transpose(P_sym)))
        self.assertLess(float(sym_error), 1e-6)

    def test_already_symmetric(self):
        """Test that symmetric matrix is unchanged."""
        P = tf.constant([[1.0, 0.5], [0.5, 1.0]], dtype=tf.float32)
        P_sym = ensure_symmetric(P)
        
        tf.debugging.assert_near(P_sym, P, atol=1e-6)

    def test_averaging(self):
        """Test that off-diagonal is averaged."""
        P = tf.constant([[1.0, 0.6], [0.4, 1.0]], dtype=tf.float32)
        P_sym = ensure_symmetric(P)
        
        # Off-diagonal should be average of 0.6 and 0.4 = 0.5
        tf.debugging.assert_near(P_sym[0, 1], 0.5, atol=1e-6)
        tf.debugging.assert_near(P_sym[1, 0], 0.5, atol=1e-6)


class TestSafeCholesky(unittest.TestCase):
    """Test cases for safe Cholesky decomposition."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_psd_matrix(self):
        """Test Cholesky of PSD matrix."""
        P = tf.constant([[4.0, 2.0], [2.0, 2.0]], dtype=tf.float32)
        L = safe_cholesky(P)
        
        # L @ L^T should equal P
        reconstructed = tf.matmul(L, L, transpose_b=True)
        tf.debugging.assert_near(reconstructed, P, atol=1e-5)

    def test_lower_triangular_output(self):
        """Test that output is lower triangular."""
        P = tf.constant([[4.0, 2.0], [2.0, 2.0]], dtype=tf.float32)
        L = safe_cholesky(P)
        
        # Upper triangle (excluding diagonal) should be zero
        upper = tf.linalg.band_part(L, 0, -1) - tf.linalg.diag(tf.linalg.diag_part(L))
        tf.debugging.assert_near(upper, tf.zeros_like(upper), atol=1e-6)

    def test_nearly_singular_matrix(self):
        """Test handling of nearly singular matrix."""
        # Nearly singular matrix
        P = tf.constant([[1.0, 0.999999], [0.999999, 1.0]], dtype=tf.float32)
        L = safe_cholesky(P, eps=1e-4)
        
        # Should succeed without error
        self.assertEqual(L.shape, (2, 2))


class TestComputeParticleCovariance(unittest.TestCase):
    """Test cases for particle covariance computation."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_uniform_weights(self):
        """Test covariance with uniform weights (None)."""
        particles = tf.constant([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0]
        ], dtype=tf.float32)
        
        cov = compute_particle_covariance(particles)
        
        self.assertEqual(cov.shape, (2, 2))
        # Should be symmetric
        sym_error = tf.reduce_max(tf.abs(cov - tf.transpose(cov)))
        self.assertLess(float(sym_error), 1e-5)

    def test_with_weights(self):
        """Test covariance with explicit weights."""
        particles = tf.constant([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0]
        ], dtype=tf.float32)
        weights = tf.constant([0.4, 0.3, 0.2, 0.1], dtype=tf.float32)
        
        cov = compute_particle_covariance(particles, weights)
        
        self.assertEqual(cov.shape, (2, 2))
        # Should be positive definite
        eigvals = tf.linalg.eigvalsh(cov)
        self.assertTrue(tf.reduce_all(eigvals > 0))

    def test_single_particle(self):
        """Test with single particle (covariance should be regularized identity)."""
        particles = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        cov = compute_particle_covariance(particles)
        
        # Should be regularized (positive definite)
        eigvals = tf.linalg.eigvalsh(cov)
        self.assertTrue(tf.reduce_all(eigvals > 0))


if __name__ == '__main__':
    unittest.main()
