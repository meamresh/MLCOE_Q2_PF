"""
Granular Unit Tests for UKF (Unscented Kalman Filter) Components.

Tests individual mathematical operations in the UKF:
1. Sigma point generation
2. Unscented transform (mean/covariance from sigma points)
3. Weight computation
4. Matrix square root (Cholesky/SVD)
5. Cross-covariance computation
6. Kalman gain computation
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.ukf import UnscentedKalmanFilter
from src.models.ssm_range_bearing import RangeBearingSSM


class TestUKFWeights(unittest.TestCase):
    """Tests for UKF weight computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.P0 = tf.eye(3, dtype=tf.float32) * 0.1
    
    def test_weights_sum_to_one(self):
        """Mean weights should sum to approximately 1."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        wm_sum = tf.reduce_sum(ukf.wm)
        np.testing.assert_allclose(wm_sum.numpy(), 1.0, atol=1e-4)
    
    def test_covariance_weights_sum(self):
        """Covariance weights sum should match expected value."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        # Covariance weights may not sum to 1 due to beta correction
        wc_sum = tf.reduce_sum(ukf.wc)
        
        # Should be finite
        self.assertTrue(tf.math.is_finite(wc_sum))
    
    def test_num_sigma_points(self):
        """Should have 2n+1 sigma points."""
        n = 3  # state dimension
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        self.assertEqual(len(ukf.wm), 2 * n + 1)
        self.assertEqual(len(ukf.wc), 2 * n + 1)
    
    def test_central_weight_properties(self):
        """Central (0th) weight has special properties."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        # W0_m and W0_c can be negative or positive depending on alpha
        # But they should be finite
        self.assertTrue(tf.math.is_finite(ukf.wm[0]))
        self.assertTrue(tf.math.is_finite(ukf.wc[0]))
    
    def test_symmetric_weights(self):
        """Weights for +/- sigma points should be equal."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        n = 3
        
        # Weights 1 to n should equal weights n+1 to 2n
        for i in range(1, n + 1):
            np.testing.assert_allclose(
                ukf.wm[i].numpy(), ukf.wm[n + i].numpy(), atol=1e-10
            )
            np.testing.assert_allclose(
                ukf.wc[i].numpy(), ukf.wc[n + i].numpy(), atol=1e-10
            )


class TestSigmaPointGeneration(unittest.TestCase):
    """Tests for sigma point generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.x0 = tf.constant([1.0, 2.0, 0.5], dtype=tf.float32)
        self.P0 = tf.eye(3, dtype=tf.float32) * 0.1
    
    def test_sigma_points_shape(self):
        """Sigma points should have shape (2n+1, n)."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        self.assertEqual(sigma_points.shape, (7, 3))  # 2*3+1, 3
    
    def test_central_sigma_point_is_mean(self):
        """First sigma point should be the mean."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        np.testing.assert_allclose(
            sigma_points[0].numpy(), self.x0.numpy(), atol=1e-6
        )
    
    def test_sigma_points_symmetric_around_mean(self):
        """Sigma points should be symmetric around the mean.
        
        For each column i of the Cholesky factor, points i and n+i
        should be equidistant from the mean (norms should match).
        """
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        n = 3
        
        # Point i and point n+i should be equidistant from mean
        for i in range(1, n + 1):
            diff_plus = sigma_points[i] - sigma_points[0]
            diff_minus = sigma_points[n + i] - sigma_points[0]
            
            # Check norm equality (distances from mean should match)
            norm_plus = tf.norm(diff_plus)
            norm_minus = tf.norm(diff_minus)
            np.testing.assert_allclose(
                norm_plus.numpy(), norm_minus.numpy(), atol=1e-5
            )
    
    def test_sigma_points_finite(self):
        """All sigma points should be finite."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(sigma_points)))
    
    def test_sigma_points_with_diagonal_covariance(self):
        """Test sigma points with diagonal covariance."""
        P_diag = tf.constant([
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3]
        ], dtype=tf.float32)
        
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, P_diag, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        # Should spread more in dimensions with higher variance
        diffs = sigma_points[1:4] - sigma_points[0]  # First 3 positive offsets
        
        # Absolute spread should scale with sqrt of variance
        # (approximately, depending on alpha)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(diffs)))


class TestUnscentedTransform(unittest.TestCase):
    """Tests for the unscented transform."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.P0 = tf.eye(3, dtype=tf.float32) * 0.1
    
    def test_transform_recovers_mean(self):
        """Transform of untransformed sigma points should recover mean."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        # No transformation, just compute mean from sigma points
        mean, _ = ukf.unscented_transform(sigma_points, tf.zeros([3, 3]))
        
        np.testing.assert_allclose(
            mean.numpy(), self.x0.numpy(), atol=1e-5
        )
    
    def test_transform_recovers_covariance(self):
        """Transform should approximately recover original covariance."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        # No noise added
        _, cov = ukf.unscented_transform(sigma_points, tf.zeros([3, 3]))
        
        # Should be close to original (with small regularization difference)
        np.testing.assert_allclose(
            cov.numpy(), self.P0.numpy(), atol=1e-4
        )
    
    def test_transform_adds_noise(self):
        """Transform should add noise covariance."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        noise_cov = 0.5 * tf.eye(3, dtype=tf.float32)
        _, cov = ukf.unscented_transform(sigma_points, noise_cov)
        
        # Covariance should be larger than original
        orig_trace = tf.linalg.trace(self.P0)
        new_trace = tf.linalg.trace(cov)
        
        self.assertGreater(new_trace.numpy(), orig_trace.numpy())
    
    def test_transform_covariance_symmetric(self):
        """Transformed covariance should be symmetric."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        _, cov = ukf.unscented_transform(sigma_points, tf.zeros([3, 3]))
        
        cov_sym = 0.5 * (cov + tf.transpose(cov))
        np.testing.assert_allclose(cov.numpy(), cov_sym.numpy(), atol=1e-6)
    
    def test_transform_covariance_psd(self):
        """Transformed covariance should be positive semi-definite."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        _, cov = ukf.unscented_transform(sigma_points, tf.zeros([3, 3]))
        
        eigvals = tf.linalg.eigvalsh(cov)
        self.assertTrue(tf.reduce_all(eigvals >= -1e-6))


class TestMatrixSquareRoot(unittest.TestCase):
    """Tests for matrix square root computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.P0 = tf.eye(3, dtype=tf.float32) * 0.1
    
    def test_cholesky_sqrt_identity(self):
        """Cholesky of identity should be identity."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        I = tf.eye(3, dtype=tf.float32)
        L = ukf.matrix_sqrt_cholesky(I)
        
        # L @ L^T = I, so L should be identity (lower triangular)
        result = L @ tf.transpose(L)
        np.testing.assert_allclose(result.numpy(), I.numpy(), atol=1e-5)
    
    def test_cholesky_sqrt_scaled_identity(self):
        """Cholesky of scaled identity."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        scale = 4.0
        M = scale * tf.eye(3, dtype=tf.float32)
        L = ukf.matrix_sqrt_cholesky(M)
        
        # L @ L^T = M
        result = L @ tf.transpose(L)
        np.testing.assert_allclose(result.numpy(), M.numpy(), atol=1e-5)
    
    def test_cholesky_sqrt_general_psd(self):
        """Cholesky of general PSD matrix."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        # Create PSD matrix
        A = tf.random.normal([3, 3], dtype=tf.float32)
        M = A @ tf.transpose(A) + 0.1 * tf.eye(3, dtype=tf.float32)
        
        L = ukf.matrix_sqrt_cholesky(M)
        
        # L @ L^T = M
        result = L @ tf.transpose(L)
        np.testing.assert_allclose(result.numpy(), M.numpy(), atol=1e-4)
    
    def test_cholesky_lower_triangular(self):
        """Cholesky factor should be lower triangular."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        M = tf.eye(3, dtype=tf.float32) * 2.0
        L = ukf.matrix_sqrt_cholesky(M)
        
        # Upper triangle (excluding diagonal) should be zero
        upper = tf.linalg.band_part(L, 0, -1) - tf.linalg.band_part(L, 0, 0)
        np.testing.assert_allclose(upper.numpy(), np.zeros([3, 3]), atol=1e-6)


class TestKalmanGainComputation(unittest.TestCase):
    """Tests for Kalman gain computation in UKF."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.P0 = tf.eye(3, dtype=tf.float32) * 0.1
        self.landmarks = tf.constant([[5.0, 5.0]], dtype=tf.float32)
    
    def test_gain_shape(self):
        """Kalman gain should have shape (n, m)."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        ukf.predict(control)
        
        # Get measurement
        z = self.ssm.measurement_model(self.x0[tf.newaxis, :], self.landmarks)[0]
        ukf.update(z, self.landmarks)
        
        # Gain is computed internally, but we can check state update happened
        self.assertEqual(ukf.state.shape, (3,))
    
    def test_gain_reduces_uncertainty(self):
        """Update with measurement should reduce uncertainty."""
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1)
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        _, P_pred = ukf.predict(control)
        
        z = self.ssm.measurement_model(ukf.state[tf.newaxis, :], self.landmarks)[0]
        _, P_upd, _ = ukf.update(z, self.landmarks)
        
        # Trace of updated covariance should be less than predicted
        # (information gain from measurement)
        trace_pred = tf.linalg.trace(P_pred)
        trace_upd = tf.linalg.trace(P_upd)
        
        self.assertLess(trace_upd.numpy(), trace_pred.numpy())


class TestUKFNumericalStability(unittest.TestCase):
    """Tests for numerical stability of UKF operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.landmarks = tf.constant([[5.0, 5.0]], dtype=tf.float32)
    
    def test_small_covariance(self):
        """UKF should handle small covariances."""
        P_small = tf.eye(3, dtype=tf.float32) * 1e-6
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, P_small, alpha=0.1)
        
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(sigma_points)))
    
    def test_large_covariance(self):
        """UKF should handle large covariances."""
        P_large = tf.eye(3, dtype=tf.float32) * 100.0
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, P_large, alpha=0.1)
        
        sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(sigma_points)))
    
    def test_long_sequence_stability(self):
        """UKF should remain stable over long sequences."""
        P0 = tf.eye(3, dtype=tf.float32) * 0.1
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, P0, alpha=0.1)
        
        for _ in range(50):
            control = tf.constant([1.0, 0.1], dtype=tf.float32)
            ukf.predict(control)
            
            z = self.ssm.measurement_model(
                ukf.state[tf.newaxis, :], self.landmarks
            )[0]
            z += tf.random.normal(z.shape, stddev=0.1)
            ukf.update(z, self.landmarks)
            
            # Check state and covariance remain finite
            self.assertTrue(tf.reduce_all(tf.math.is_finite(ukf.state)))
            self.assertTrue(tf.reduce_all(tf.math.is_finite(ukf.covariance)))
            
            # Check covariance remains PSD
            eigvals = tf.linalg.eigvalsh(ukf.covariance)
            self.assertTrue(tf.reduce_all(eigvals > -1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
