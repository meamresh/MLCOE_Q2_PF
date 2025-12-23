"""
Unit tests for state-space models.
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ssm_lgssm import LGSSM
from src.models.ssm_range_bearing import RangeBearingSSM


class TestLGSSM(unittest.TestCase):
    """Test cases for Linear Gaussian State-Space Model."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        
        # Simple 2D constant velocity model
        self.nx, self.ny, self.nv, self.nw = 4, 2, 2, 2
        
        self.A = tf.constant([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=tf.float32)
        
        self.B = tf.constant([
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0]
        ], dtype=tf.float32) * 0.5
        
        self.C = tf.constant([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=tf.float32)
        
        self.D = tf.constant([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=tf.float32) * 5.0
        
        self.m0 = tf.zeros(self.nx, dtype=tf.float32)
        self.P0 = tf.eye(self.nx, dtype=tf.float32) * 10.0
        
        self.Q = tf.matmul(self.B, self.B, transpose_b=True)
        self.R = tf.matmul(self.D, self.D, transpose_b=True)

    def test_model_creation(self):
        """Test LGSSM model creation."""
        model = LGSSM(
            A=self.A, B=self.B, C=self.C, D=self.D,
            m0=self.m0, P0=self.P0, Q=self.Q, R=self.R,
            nx=self.nx, ny=self.ny, nv=self.nv, nw=self.nw
        )
        
        self.assertEqual(model.nx, self.nx)
        self.assertEqual(model.ny, self.ny)
        self.assertEqual(model.nv, self.nv)
        self.assertEqual(model.nw, self.nw)
        
        # Check tensor shapes
        self.assertEqual(tf.shape(model.A)[0], self.nx)
        self.assertEqual(tf.shape(model.C)[0], self.ny)

    def test_from_config(self):
        """Test LGSSM creation from config dictionary."""
        cfg = {
            "dimensions": {
                "nx": 4, "ny": 2, "nv": 2, "nw": 2
            },
            "A": [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
            "B_raw": [[0.5, 0.0], [1.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
            "C": [[1, 0, 0, 0], [0, 0, 1, 0]],
            "D": [[1.0, 0.0], [0.0, 1.0]],
            "params": {
                "sigma_a": 0.5,
                "sigma_z": 5.0,
                "Sigma0_diag": [10.0, 5.0, 10.0, 5.0]
            }
        }
        
        model = LGSSM.from_config(cfg)
        
        self.assertEqual(model.nx, 4)
        self.assertEqual(model.ny, 2)
        self.assertEqual(tf.shape(model.A)[0], 4)
        self.assertEqual(tf.shape(model.C)[0], 2)

    def test_sample_shapes(self):
        """Test that sample() returns correct shapes."""
        model = LGSSM(
            A=self.A, B=self.B, C=self.C, D=self.D,
            m0=self.m0, P0=self.P0, Q=self.Q, R=self.R,
            nx=self.nx, ny=self.ny, nv=self.nv, nw=self.nw
        )
        
        N = 100
        X, Y = model.sample(N=N, seed=42)
        
        # Check shapes
        self.assertEqual(tf.shape(X)[0], N)
        self.assertEqual(tf.shape(X)[1], self.nx)
        self.assertEqual(tf.shape(Y)[0], N)
        self.assertEqual(tf.shape(Y)[1], self.ny)

    def test_sample_deterministic_with_seed(self):
        """Test that sampling with same seed produces same results."""
        model = LGSSM(
            A=self.A, B=self.B, C=self.C, D=self.D,
            m0=self.m0, P0=self.P0, Q=self.Q, R=self.R,
            nx=self.nx, ny=self.ny, nv=self.nv, nw=self.nw
        )
        
        X1, Y1 = model.sample(N=10, seed=123)
        X2, Y2 = model.sample(N=10, seed=123)
        
        # Should be identical with same seed
        tf.debugging.assert_near(X1, X2, rtol=1e-6)
        tf.debugging.assert_near(Y1, Y2, rtol=1e-6)

    def test_initial_state_distribution(self):
        """Test that initial state follows N(m0, P0)."""
        model = LGSSM(
            A=self.A, B=self.B, C=self.C, D=self.D,
            m0=self.m0, P0=self.P0, Q=self.Q, R=self.R,
            nx=self.nx, ny=self.ny, nv=self.nv, nw=self.nw
        )
        
        # Sample many initial states
        X_samples = []
        for _ in range(1000):
            X, _ = model.sample(N=1, seed=None)
            X_samples.append(X[0])
        
        X_samples = tf.stack(X_samples)
        mean_est = tf.reduce_mean(X_samples, axis=0)
        cov_est = tf.linalg.matmul(
            X_samples - mean_est,
            X_samples - mean_est,
            transpose_a=True
        ) / 999.0
        
        # Check mean is close to m0
        tf.debugging.assert_near(mean_est, self.m0, atol=0.5)
        # Check covariance is close to P0 (within reasonable tolerance)
        tf.debugging.assert_near(cov_est, self.P0, atol=2.0)


class TestRangeBearingSSM(unittest.TestCase):
    """Test cases for Range-Bearing nonlinear state-space model."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        self.ssm = RangeBearingSSM(dt=self.dt)

        # Simple landmark configuration
        self.landmarks = tf.constant(
            [[5.0, 0.0],
             [0.0, 5.0]],
            dtype=tf.float32,
        )

    def test_default_parameters(self) -> None:
        """Test default initialization and parameter shapes."""
        self.assertEqual(self.ssm.state_dim, 3)
        self.assertEqual(self.ssm.meas_per_landmark, 2)

        self.assertEqual(self.ssm.Q.shape, (3, 3))
        self.assertEqual(self.ssm.R.shape, (2, 2))

        # Q and R should be symmetric positive-definite
        q_eigvals = tf.linalg.eigvalsh(self.ssm.Q)
        r_eigvals = tf.linalg.eigvalsh(self.ssm.R)
        self.assertTrue(tf.reduce_all(q_eigvals > 0.0))
        self.assertTrue(tf.reduce_all(r_eigvals > 0.0))

    def test_full_measurement_cov_shape(self) -> None:
        """Test construction of full measurement covariance."""
        m = 3
        R_full = self.ssm.full_measurement_cov(m)

        self.assertEqual(R_full.shape, (2 * m, 2 * m))

        # Diagonal blocks should equal R
        for i in range(m):
            start = 2 * i
            end = start + 2
            block = R_full[start:end, start:end]
            tf.debugging.assert_near(block, self.ssm.R, atol=1e-6)

    def test_motion_model_shapes_and_dynamics(self) -> None:
        """Test motion model output shape and basic dynamics."""
        state = tf.constant([1.0, 2.0, 0.0], dtype=tf.float32)
        control = tf.constant([1.0, 0.0], dtype=tf.float32)  # move along +x

        next_state = self.ssm.motion_model(state, control)
        self.assertEqual(next_state.shape, (1, 3))

        # With theta=0 and v=1, x should increase by dt, y unchanged
        self.assertAlmostEqual(float(next_state[0, 0]), 1.0 + self.dt, places=5)
        self.assertAlmostEqual(float(next_state[0, 1]), 2.0, places=5)

    def test_measurement_model_shapes_and_values(self) -> None:
        """Test measurement model shape and simple range/bearing values."""
        state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)

        meas = self.ssm.measurement_model(state, self.landmarks)
        # Shape: (batch=1, num_landmarks, 2)
        self.assertEqual(meas.shape, (1, 2, 2))

        meas_np = meas.numpy()[0]
        # Landmark at (5, 0): range ≈ 5, bearing ≈ 0
        self.assertAlmostEqual(meas_np[0, 0], 5.0, places=4)
        self.assertAlmostEqual(meas_np[0, 1], 0.0, places=4)

    def test_jacobian_shapes(self) -> None:
        """Test Jacobian shapes for motion and measurement models."""
        state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        control = tf.constant([1.0, 0.1], dtype=tf.float32)

        F = self.ssm.motion_jacobian(state, control)
        self.assertEqual(F.shape, (1, 3, 3))

        H = self.ssm.measurement_jacobian(state, self.landmarks)
        # For 2 landmarks, measurement dimension is 4 -> H shape (1, 4, 3)
        self.assertEqual(H.shape, (1, 4, 3))


if __name__ == '__main__':
    unittest.main()

