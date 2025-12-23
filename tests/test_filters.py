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
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.models.ssm_range_bearing import RangeBearingSSM


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


class TestNonlinearFilters(unittest.TestCase):
    """Test cases for EKF and UKF on the range-bearing model."""

    def setUp(self) -> None:
        """Set up nonlinear model and filters."""
        tf.random.set_seed(123)
        self.dt = 0.1

        # Process and measurement noise
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05

        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.initial_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.initial_cov = tf.eye(3, dtype=tf.float32) * 0.1

        self.landmarks = tf.constant(
            [[5.0, 5.0],
             [-5.0, 5.0]],
            dtype=tf.float32,
        )

        self.ekf = ExtendedKalmanFilter(self.ssm, self.initial_state,
                                        self.initial_cov)
        self.ukf = UnscentedKalmanFilter(self.ssm, self.initial_state,
                                         self.initial_cov,
                                         alpha=0.1, beta=1.0, kappa=0.0)

    def _one_step_measurement(self, state: tf.Tensor) -> tf.Tensor:
        """Generate a single noisy measurement from a given state."""
        true_meas = self.ssm.measurement_model(state, self.landmarks)[0]
        meas_std = tf.sqrt(tf.linalg.diag_part(self.ssm.R))
        noise = tf.random.normal([tf.shape(self.landmarks)[0], 2],
                                 mean=0.0, stddev=meas_std,
                                 dtype=tf.float32)
        return true_meas + noise

    def test_ekf_predict_and_update(self) -> None:
        """EKF predict/update should maintain PSD covariance and finite values."""
        control = tf.constant([1.0, 0.1], dtype=tf.float32)

        # Predict
        x_pred, P_pred = self.ekf.predict(control)
        self.assertEqual(x_pred.shape, (3,))
        self.assertEqual(P_pred.shape, (3, 3))

        # Update
        z = self._one_step_measurement(self.initial_state)
        x_upd, P_upd, residual = self.ekf.update(z, self.landmarks)

        self.assertEqual(x_upd.shape, (3,))
        self.assertEqual(P_upd.shape, (3, 3))
        self.assertEqual(residual.shape[0], 2 * tf.shape(self.landmarks)[0])

        # Covariance should be symmetric and PSD
        tf.debugging.assert_near(P_upd, 0.5 * (P_upd + tf.transpose(P_upd)),
                                 atol=1e-6)
        eigvals = tf.linalg.eigvalsh(P_upd)
        self.assertTrue(tf.reduce_all(eigvals > 0.0))

    def test_ukf_predict_and_update(self) -> None:
        """UKF predict/update should maintain PSD covariance and finite values."""
        control = tf.constant([1.0, 0.1], dtype=tf.float32)

        # Predict
        x_pred, P_pred = self.ukf.predict(control)
        self.assertEqual(x_pred.shape, (3,))
        self.assertEqual(P_pred.shape, (3, 3))

        # Update
        z = self._one_step_measurement(self.initial_state)
        x_upd, P_upd, residual = self.ukf.update(z, self.landmarks)

        self.assertEqual(x_upd.shape, (3,))
        self.assertEqual(P_upd.shape, (3, 3))
        self.assertEqual(residual.shape[0], 2 * tf.shape(self.landmarks)[0])

        # Covariance should be symmetric and PSD
        tf.debugging.assert_near(P_upd, 0.5 * (P_upd + tf.transpose(P_upd)),
                                 atol=1e-6)
        eigvals = tf.linalg.eigvalsh(P_upd)
        self.assertTrue(tf.reduce_all(eigvals > 0.0))

    def test_ekf_ukf_close_on_short_horizon(self) -> None:
        """On a short horizon, EKF and UKF estimates should be reasonably close."""
        num_steps = 10
        state = tf.identity(self.initial_state)

        ekf_states = []
        ukf_states = []

        for step in range(num_steps):
            t = tf.cast(step, tf.float32)
            v = 1.0 + 0.2 * tf.sin(t * 0.1)
            omega = 0.1 + 0.05 * tf.cos(t * 0.1)
            control = tf.stack([v, omega])

            # Propagate true state
            state = self.ssm.motion_model(state, control)[0]

            # Measurement
            z = self._one_step_measurement(state)

            # EKF
            self.ekf.predict(control)
            self.ekf.update(z, self.landmarks)
            ekf_states.append(tf.identity(self.ekf.state))

            # UKF
            self.ukf.predict(control)
            self.ukf.update(z, self.landmarks)
            ukf_states.append(tf.identity(self.ukf.state))

        ekf_states = tf.stack(ekf_states)
        ukf_states = tf.stack(ukf_states)

        # Position estimates should not diverge wildly
        pos_diff = tf.norm(ekf_states[:, :2] - ukf_states[:, :2], axis=1)
        mean_pos_diff = tf.reduce_mean(pos_diff)
        self.assertLess(float(mean_pos_diff), 5.0)


if __name__ == '__main__':
    unittest.main()

