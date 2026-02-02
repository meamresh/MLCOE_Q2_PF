"""
Unit tests for advanced state-space models: Lorenz-96 and Multi-Target Acoustic.
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ssm_lorenz96 import Lorenz96Model
from src.models.ssm_multi_target_acoustic import MultiTargetAcousticSSM


class TestLorenz96Model(unittest.TestCase):
    """Test cases for Lorenz-96 chaotic dynamical system."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.model = Lorenz96Model(dim=40, F=8.0, dt=0.05)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.dim, 40)
        self.assertEqual(self.model.F, 8.0)
        self.assertEqual(self.model.dt, 0.05)

    def test_derivative_shape(self):
        """Test derivative output shape."""
        x = tf.random.normal([40], dtype=tf.float32)
        dx = self.model.derivative(x)
        
        self.assertEqual(dx.shape, (40,))

    def test_derivative_batch(self):
        """Test derivative with batched input."""
        x = tf.random.normal([10, 40], dtype=tf.float32)
        dx = self.model.derivative(x)
        
        self.assertEqual(dx.shape, (10, 40))

    def test_step_shape(self):
        """Test single step output shape."""
        x = tf.random.normal([40], dtype=tf.float32)
        x_next = self.model.step(x)
        
        self.assertEqual(x_next.shape, (40,))

    def test_step_finite(self):
        """Test that step produces finite values."""
        x = tf.random.normal([40], dtype=tf.float32) * 0.1
        x_next = self.model.step(x)
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(x_next)))

    def test_generate_trajectory_shape(self):
        """Test trajectory generation shape."""
        x0 = tf.random.normal([40], dtype=tf.float32) * 0.1
        n_steps = 100
        trajectory = self.model.generate_trajectory(x0, n_steps, spinup=10)
        
        # Should have n_steps + 1 states (including initial)
        self.assertEqual(trajectory.shape, (n_steps + 1, 40))

    def test_trajectory_finite(self):
        """Test that generated trajectory is finite."""
        x0 = tf.random.normal([40], dtype=tf.float32) * 0.1
        trajectory = self.model.generate_trajectory(x0, n_steps=50, spinup=10)
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(trajectory)))

    def test_chaotic_behavior(self):
        """Test that system exhibits chaotic behavior (sensitivity to initial conditions)."""
        x0_a = tf.ones(40, dtype=tf.float32)
        x0_b = x0_a + tf.random.normal([40], dtype=tf.float32) * 1e-6
        
        # Run both for some steps
        traj_a = self.model.generate_trajectory(x0_a, n_steps=100, spinup=0)
        traj_b = self.model.generate_trajectory(x0_b, n_steps=100, spinup=0)
        
        # Initially similar
        initial_diff = tf.reduce_mean(tf.abs(traj_a[0] - traj_b[0]))
        self.assertLess(float(initial_diff), 1e-4)
        
        # Should diverge (chaotic)
        final_diff = tf.reduce_mean(tf.abs(traj_a[-1] - traj_b[-1]))
        self.assertGreater(float(final_diff), float(initial_diff))

    def test_custom_parameters(self):
        """Test model with custom parameters."""
        model = Lorenz96Model(dim=20, F=4.0, dt=0.01)
        
        self.assertEqual(model.dim, 20)
        self.assertEqual(model.F, 4.0)
        self.assertEqual(model.dt, 0.01)
        
        x = tf.random.normal([20], dtype=tf.float32)
        x_next = model.step(x)
        self.assertEqual(x_next.shape, (20,))


class TestMultiTargetAcousticSSM(unittest.TestCase):
    """Test cases for Multi-Target Acoustic state-space model."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.ssm = MultiTargetAcousticSSM(
            num_targets=2,
            num_sensors=9,
            area_size=30.0,
            dt=1.0,
            psi=10.0,
            d0=0.1,
            sigma_w=0.1,
            sensor_grid_size=3
        )

    def test_initialization(self):
        """Test SSM initialization."""
        self.assertEqual(self.ssm.C, 2)  # num_targets
        self.assertEqual(self.ssm.N_s, 9)  # num_sensors
        self.assertEqual(self.ssm.state_dim, 8)  # 4 * num_targets

    def test_sensor_positions(self):
        """Test sensor grid creation."""
        self.assertEqual(self.ssm.sensor_positions.shape, (9, 2))
        
        # All sensors should be within area
        self.assertTrue(tf.reduce_all(self.ssm.sensor_positions >= 0))
        self.assertTrue(tf.reduce_all(self.ssm.sensor_positions <= 30.0))

    def test_motion_model_shape(self):
        """Test motion model output shape."""
        state = tf.random.normal([8], dtype=tf.float32)
        state_batch = state[tf.newaxis, :]
        
        next_state = self.ssm.motion_model(state_batch)
        
        self.assertEqual(next_state.shape, (1, 8))

    def test_motion_model_batch(self):
        """Test motion model with batched input."""
        states = tf.random.normal([10, 8], dtype=tf.float32)
        
        next_states = self.ssm.motion_model(states)
        
        self.assertEqual(next_states.shape, (10, 8))

    def test_measurement_model_shape(self):
        """Test measurement model output shape."""
        state = tf.random.normal([8], dtype=tf.float32) + 15.0  # Center of area
        state_batch = state[tf.newaxis, :]
        
        measurements = self.ssm.measurement_model(state_batch)
        
        self.assertEqual(measurements.shape, (1, 9))

    def test_measurement_model_positive(self):
        """Test that acoustic measurements are positive (amplitude)."""
        state = tf.constant([15.0, 15.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0], dtype=tf.float32)
        state_batch = state[tf.newaxis, :]
        
        measurements = self.ssm.measurement_model(state_batch)
        
        self.assertTrue(tf.reduce_all(measurements > 0))

    def test_measurement_jacobian_shape(self):
        """Test measurement Jacobian shape."""
        state = tf.constant([15.0, 15.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0], dtype=tf.float32)
        
        H = self.ssm.measurement_jacobian(state)
        
        # Should be (num_sensors, state_dim)
        self.assertEqual(H.shape, (9, 8))

    def test_motion_jacobian_shape(self):
        """Test motion Jacobian shape."""
        state = tf.random.normal([8], dtype=tf.float32)
        
        F = self.ssm.motion_jacobian(state)
        
        # Returns batched output (1, state_dim, state_dim) for 1D input
        self.assertEqual(F.shape, (1, 8, 8))

    def test_sample_initial_state(self):
        """Test initial state sampling."""
        samples = self.ssm.sample_initial_state(num_samples=10)
        
        self.assertEqual(samples.shape, (10, 8))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(samples)))

    def test_sample_process_noise(self):
        """Test process noise sampling."""
        noise = self.ssm.sample_process_noise(shape=10)
        
        # Returns shape (batch, num_targets, 4) -> need to flatten
        self.assertEqual(noise.shape[0], 10)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(noise)))

    def test_process_noise_covariance(self):
        """Test process noise covariance matrix."""
        Q = self.ssm.Q
        
        self.assertEqual(Q.shape, (8, 8))
        # Should be symmetric
        sym_error = tf.reduce_max(tf.abs(Q - tf.transpose(Q)))
        self.assertLess(float(sym_error), 1e-6)
        # Should be PSD
        eigvals = tf.linalg.eigvalsh(Q)
        self.assertTrue(tf.reduce_all(eigvals >= 0))

    def test_measurement_noise_covariance(self):
        """Test measurement noise covariance matrix."""
        R = self.ssm.R
        
        self.assertEqual(R.shape, (9, 9))
        # Should be symmetric
        sym_error = tf.reduce_max(tf.abs(R - tf.transpose(R)))
        self.assertLess(float(sym_error), 1e-6)
        # Should be PSD
        eigvals = tf.linalg.eigvalsh(R)
        self.assertTrue(tf.reduce_all(eigvals >= 0))


class TestMultiTargetAcousticSSMVariants(unittest.TestCase):
    """Test cases for different configurations of Multi-Target Acoustic SSM."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_single_target(self):
        """Test with single target."""
        ssm = MultiTargetAcousticSSM(num_targets=1, num_sensors=4, sensor_grid_size=2)
        
        self.assertEqual(ssm.state_dim, 4)
        
        state = tf.constant([15.0, 15.0, 1.0, 1.0], dtype=tf.float32)
        z = ssm.measurement_model(state[tf.newaxis, :])[0]
        self.assertEqual(z.shape, (4,))

    def test_many_targets(self):
        """Test with many targets."""
        ssm = MultiTargetAcousticSSM(num_targets=5, num_sensors=16, sensor_grid_size=4)
        
        self.assertEqual(ssm.state_dim, 20)
        
        state = tf.random.normal([20], dtype=tf.float32) + 15.0
        z = ssm.measurement_model(state[tf.newaxis, :])[0]
        self.assertEqual(z.shape, (16,))

    def test_different_noise_levels(self):
        """Test with different noise levels."""
        ssm_low = MultiTargetAcousticSSM(num_targets=2, sigma_w=0.01)
        ssm_high = MultiTargetAcousticSSM(num_targets=2, sigma_w=1.0)
        
        # High noise SSM should have larger R diagonal
        r_low = tf.linalg.diag_part(ssm_low.R)
        r_high = tf.linalg.diag_part(ssm_high.R)
        
        self.assertTrue(tf.reduce_all(r_high > r_low))

    def test_custom_area_size(self):
        """Test with custom area size."""
        ssm = MultiTargetAcousticSSM(num_targets=2, area_size=100.0, sensor_grid_size=3)
        
        self.assertEqual(ssm.area_size, 100.0)
        # Sensors should be spread over larger area
        self.assertTrue(tf.reduce_max(ssm.sensor_positions) <= 100.0)


class TestSSMIntegration(unittest.TestCase):
    """Integration tests for state-space models with filters."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_acoustic_ssm_with_ekf_interface(self):
        """Test that Acoustic SSM works with EKF-like interface."""
        ssm = MultiTargetAcousticSSM(num_targets=2, num_sensors=9, sensor_grid_size=3)
        
        # Simulate initial state
        x0 = ssm.sample_initial_state(num_samples=1)
        x0_flat = tf.reshape(x0, [-1])
        
        # Motion model
        x1 = ssm.motion_model(x0)
        self.assertEqual(x1.shape[1], 8)
        
        # Measurement model
        z = ssm.measurement_model(x1)
        self.assertEqual(z.shape[1], 9)
        
        # Jacobians - returns batched output
        F = ssm.motion_jacobian(x0_flat)
        H = ssm.measurement_jacobian(x0_flat)
        
        self.assertEqual(F.shape, (1, 8, 8))
        self.assertEqual(H.shape, (9, 8))

    def test_lorenz96_trajectory_integration(self):
        """Test Lorenz-96 trajectory generation for data assimilation."""
        model = Lorenz96Model(dim=40, F=8.0, dt=0.05)
        
        # Generate trajectory
        x0 = tf.random.normal([40], dtype=tf.float32)
        trajectory = model.generate_trajectory(x0, n_steps=100, spinup=500)
        
        # Should be stable after spinup
        self.assertTrue(tf.reduce_all(tf.math.is_finite(trajectory)))
        
        # Values should be bounded (typical Lorenz-96 values are O(1)-O(10))
        max_val = tf.reduce_max(tf.abs(trajectory))
        self.assertLess(float(max_val), 100.0)


if __name__ == '__main__':
    unittest.main()
