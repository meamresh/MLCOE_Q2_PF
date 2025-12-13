"""
Unit tests for Kalman filter.
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.kalman import KalmanFilter


class TestKalmanFilter(unittest.TestCase):
    """Test cases for Kalman Filter."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        
        # Simple 1D model for testing
        self.A = tf.constant([[1.0]], dtype=tf.float32)
        self.C = tf.constant([[1.0]], dtype=tf.float32)
        self.x0 = tf.constant([[0.0]], dtype=tf.float32)
        self.P0 = tf.constant([[1.0]], dtype=tf.float32)
        self.Q = tf.constant([[0.1]], dtype=tf.float32)
        self.R = tf.constant([[0.5]], dtype=tf.float32)

    def test_filter_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter(
            A=self.A, C=self.C, x0=self.x0, P0=self.P0,
            Q=self.Q, R=self.R
        )
        
        self.assertIsNotNone(kf.A)
        self.assertIsNotNone(kf.C)
        self.assertIsNotNone(kf.x)
        self.assertIsNotNone(kf.P)

    def test_predict_step(self):
        """Test prediction step."""
        kf = KalmanFilter(
            A=self.A, C=self.C, x0=self.x0, P0=self.P0,
            Q=self.Q, R=self.R
        )
        
        x_pred, P_pred = kf.predict()
        
        # Check shapes
        self.assertEqual(tf.shape(x_pred)[0], 1)
        self.assertEqual(tf.shape(P_pred)[0], 1)
        self.assertEqual(tf.shape(P_pred)[1], 1)
        
        # For 1D case: x_pred = A @ x0 = x0 (since A=1)
        tf.debugging.assert_near(x_pred, self.x0, rtol=1e-6)
        
        # P_pred = A @ P0 @ A^T + Q = P0 + Q
        P_expected = self.P0 + self.Q
        tf.debugging.assert_near(P_pred, P_expected, rtol=1e-6)

    def test_update_step(self):
        """Test measurement update step."""
        kf = KalmanFilter(
            A=self.A, C=self.C, x0=self.x0, P0=self.P0,
            Q=self.Q, R=self.R
        )
        
        # First predict
        x_pred, P_pred = kf.predict()
        
        # Store predicted covariance before update
        P_pred_before = tf.identity(kf.P)
        
        # Then update with measurement
        z = tf.constant([[1.0]], dtype=tf.float32)
        x_filt, P_filt, K, y, S = kf.update(z)
        
        # Check shapes
        self.assertEqual(tf.shape(x_filt)[0], 1)
        self.assertEqual(tf.shape(P_filt)[0], 1)
        self.assertEqual(tf.shape(K)[0], 1)
        self.assertEqual(tf.shape(y)[0], 1)
        self.assertEqual(tf.shape(S)[0], 1)
        
        # Check that covariance decreased (information increased)
        # Filtered covariance should be less than predicted covariance
        self.assertLess(float(P_filt[0, 0]), float(P_pred_before[0, 0]))
        
        # Check that filtered covariance is positive
        self.assertGreater(float(P_filt[0, 0]), 0.0)

    def test_filter_sequence(self):
        """Test filtering over a sequence of measurements."""
        kf = KalmanFilter(
            A=self.A, C=self.C, x0=self.x0, P0=self.P0,
            Q=self.Q, R=self.R
        )
        
        # Create sequence of measurements
        Z = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)
        
        results = kf.filter(Z)
        
        # Check output keys
        self.assertIn("x_pred", results)
        self.assertIn("P_pred", results)
        self.assertIn("x_filt", results)
        self.assertIn("P_filt", results)
        
        # Check shapes
        self.assertEqual(tf.shape(results["x_pred"])[0], 3)
        self.assertEqual(tf.shape(results["x_filt"])[0], 3)

    def test_joseph_update(self):
        """Test Joseph-stabilized covariance update."""
        kf_riccati = KalmanFilter(
            A=self.A, C=self.C, x0=self.x0, P0=self.P0,
            Q=self.Q, R=self.R
        )
        
        kf_joseph = KalmanFilter(
            A=self.A, C=self.C, x0=self.x0, P0=self.P0,
            Q=self.Q, R=self.R
        )
        
        kf_riccati.predict()
        kf_joseph.predict()
        
        z = tf.constant([[1.0]], dtype=tf.float32)
        
        # Riccati update
        x_r, P_r, _, _, _ = kf_riccati.update(z, joseph=False)
        
        # Joseph update
        x_j, P_j, _, _, _ = kf_joseph.update(z, joseph=True)
        
        # Means should be identical
        tf.debugging.assert_near(x_r, x_j, rtol=1e-6)
        
        # Covariances should be very close (algebraically equivalent)
        tf.debugging.assert_near(P_r, P_j, rtol=1e-5)

    def test_2d_model(self):
        """Test Kalman filter with 2D state space."""
        A_2d = tf.constant([[1.0, 1.0], [0.0, 1.0]], dtype=tf.float32)
        C_2d = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        x0_2d = tf.constant([[0.0], [0.0]], dtype=tf.float32)
        P0_2d = tf.eye(2, dtype=tf.float32)
        Q_2d = tf.eye(2, dtype=tf.float32) * 0.1
        R_2d = tf.constant([[0.5]], dtype=tf.float32)
        
        kf = KalmanFilter(
            A=A_2d, C=C_2d, x0=x0_2d, P0=P0_2d,
            Q=Q_2d, R=R_2d
        )
        
        Z = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        results = kf.filter(Z)
        
        # Check shapes
        self.assertEqual(tf.shape(results["x_filt"])[0], 2)
        self.assertEqual(tf.shape(results["x_filt"])[1], 2)
        self.assertEqual(tf.shape(results["P_filt"])[0], 2)
        self.assertEqual(tf.shape(results["P_filt"])[1], 2)
        self.assertEqual(tf.shape(results["P_filt"])[2], 2)


if __name__ == '__main__':
    unittest.main()

