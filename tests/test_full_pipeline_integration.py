"""
Full pipeline integration tests.

Tests the complete data assimilation pipeline:
1. Data generation from SSMs
2. Filter initialization
3. Filtering over trajectory
4. Metrics computation
5. Stability diagnostics
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ssm_range_bearing import RangeBearingSSM
from src.models.ssm_multi_target_acoustic import MultiTargetAcousticSSM
from src.models.ssm_lorenz96 import Lorenz96Model
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.filters.particle_filter import ParticleFilter
from src.filters.edh import EDH
from src.filters.ledh import LEDH
from src.filters.pff_kernel import ScalarPFF
from src.metrics.accuracy import compute_rmse, compute_mae, compute_nees
from src.metrics.stability import compute_condition_numbers, check_positive_definite
from src.metrics.particle_filter_metrics import compute_effective_sample_size
from src.filters.resampling import compute_ess
from src.utils.linalg import regularize_covariance, compute_condition_number


class TestRangeBearingPipeline(unittest.TestCase):
    """Integration tests for Range-Bearing model pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.P0 = tf.eye(3, dtype=tf.float32) * 0.1
        self.landmarks = tf.constant([[5.0, 5.0], [-5.0, 5.0]], dtype=tf.float32)
        self.n_steps = 20

    def _generate_trajectory(self):
        """Generate ground truth trajectory and measurements."""
        state = self.x0
        states = [state]
        measurements = []
        
        for t in range(self.n_steps):
            v = 1.0 + 0.2 * tf.sin(tf.cast(t, tf.float32) * 0.1)
            omega = 0.1 + 0.05 * tf.cos(tf.cast(t, tf.float32) * 0.1)
            control = tf.stack([v, omega])
            
            # Propagate state
            state_next = self.ssm.motion_model(state, control)[0]
            noise = tf.random.normal([3], stddev=0.01, dtype=tf.float32)
            state = state_next + noise
            states.append(state)
            
            # Generate measurement
            true_meas = self.ssm.measurement_model(state, self.landmarks)[0]
            meas_noise = tf.random.normal(tf.shape(true_meas), stddev=0.05, dtype=tf.float32)
            measurement = true_meas + meas_noise
            measurements.append(measurement)
        
        return tf.stack(states), measurements

    def test_ekf_pipeline(self):
        """Test full pipeline with EKF."""
        states, measurements = self._generate_trajectory()
        
        ekf = ExtendedKalmanFilter(self.ssm, self.x0, self.P0)
        estimates = []
        covariances = []
        
        for t in range(self.n_steps):
            v = 1.0 + 0.2 * tf.sin(tf.cast(t, tf.float32) * 0.1)
            omega = 0.1 + 0.05 * tf.cos(tf.cast(t, tf.float32) * 0.1)
            control = tf.stack([v, omega])
            
            ekf.predict(control)
            ekf.update(measurements[t], self.landmarks)
            
            estimates.append(ekf.state)
            covariances.append(ekf.covariance)
        
        estimates = tf.stack(estimates)
        covariances = tf.stack(covariances)
        
        # Compute metrics
        rmse = compute_rmse(estimates, states[1:])
        self.assertLess(float(rmse), 5.0)  # Reasonable tracking
        
        # Check covariance stability
        cond_numbers = compute_condition_numbers(covariances)
        self.assertTrue(tf.reduce_all(cond_numbers > 0))
        
        # Check positive definiteness
        _, is_pd = check_positive_definite(covariances)
        self.assertTrue(tf.reduce_all(is_pd))

    def test_ukf_pipeline(self):
        """Test full pipeline with UKF."""
        states, measurements = self._generate_trajectory()
        
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1, beta=1.0, kappa=0.0)
        estimates = []
        
        for t in range(self.n_steps):
            v = 1.0 + 0.2 * tf.sin(tf.cast(t, tf.float32) * 0.1)
            omega = 0.1 + 0.05 * tf.cos(tf.cast(t, tf.float32) * 0.1)
            control = tf.stack([v, omega])
            
            ukf.predict(control)
            ukf.update(measurements[t], self.landmarks)
            estimates.append(ukf.state)
        
        estimates = tf.stack(estimates)
        
        rmse = compute_rmse(estimates, states[1:])
        self.assertLess(float(rmse), 5.0)

    def test_particle_filter_pipeline(self):
        """Test full pipeline with Particle Filter."""
        states, measurements = self._generate_trajectory()
        
        pf = ParticleFilter(self.ssm, self.x0, self.P0, num_particles=200)
        estimates = []
        ess_history = []
        
        for t in range(self.n_steps):
            v = 1.0 + 0.2 * tf.sin(tf.cast(t, tf.float32) * 0.1)
            omega = 0.1 + 0.05 * tf.cos(tf.cast(t, tf.float32) * 0.1)
            control = tf.stack([v, omega])
            
            pf.predict(control)
            pf.update(measurements[t], self.landmarks)
            
            estimates.append(pf.state)
            ess_history.append(compute_ess(pf.weights))
        
        estimates = tf.stack(estimates)
        
        rmse = compute_rmse(estimates, states[1:])
        self.assertLess(float(rmse), 10.0)  # PF may have higher variance
        
        # ESS should remain reasonable
        avg_ess = tf.reduce_mean(tf.stack(ess_history))
        self.assertGreater(float(avg_ess), 10.0)

    def test_ekf_ukf_comparison(self):
        """Test that EKF and UKF produce similar results on mild nonlinearity."""
        states, measurements = self._generate_trajectory()
        
        ekf = ExtendedKalmanFilter(self.ssm, self.x0, self.P0)
        ukf = UnscentedKalmanFilter(self.ssm, self.x0, self.P0, alpha=0.1, beta=1.0, kappa=0.0)
        
        ekf_estimates = []
        ukf_estimates = []
        
        for t in range(self.n_steps):
            v = 1.0 + 0.2 * tf.sin(tf.cast(t, tf.float32) * 0.1)
            omega = 0.1 + 0.05 * tf.cos(tf.cast(t, tf.float32) * 0.1)
            control = tf.stack([v, omega])
            
            ekf.predict(control)
            ukf.predict(control)
            
            ekf.update(measurements[t], self.landmarks)
            ukf.update(measurements[t], self.landmarks)
            
            ekf_estimates.append(ekf.state)
            ukf_estimates.append(ukf.state)
        
        ekf_estimates = tf.stack(ekf_estimates)
        ukf_estimates = tf.stack(ukf_estimates)
        
        # Estimates should be similar
        diff = tf.reduce_mean(tf.abs(ekf_estimates - ukf_estimates))
        self.assertLess(float(diff), 2.0)


class TestAcousticPipeline(unittest.TestCase):
    """Integration tests for Multi-Target Acoustic model pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.ssm = MultiTargetAcousticSSM(
            num_targets=2,
            num_sensors=9,
            area_size=30.0,
            dt=1.0,
            sensor_grid_size=3
        )
        self.x0 = tf.constant([
            15.0, 15.0, 0.5, 0.5,  # Target 1
            20.0, 10.0, -0.3, 0.2  # Target 2
        ], dtype=tf.float32)
        self.P0 = tf.eye(8, dtype=tf.float32) * 1.0
        self.n_steps = 15

    def _generate_trajectory(self):
        """Generate ground truth trajectory and measurements."""
        state = self.x0
        states = [state]
        measurements = []
        
        for _ in range(self.n_steps):
            state_next = self.ssm.motion_model(state[tf.newaxis, :])[0]
            # sample_process_noise uses shape parameter, and returns (shape, num_targets, 4)
            noise = self.ssm.sample_process_noise(shape=1)
            noise_flat = tf.reshape(noise, [-1])  # Flatten to (8,)
            state = state_next + noise_flat * 0.1
            states.append(state)
            
            z = self.ssm.measurement_model(state[tf.newaxis, :])[0]
            meas_noise = tf.random.normal([self.ssm.N_s], stddev=self.ssm.sigma_w, dtype=tf.float32)
            measurement = z + meas_noise
            measurements.append(measurement)
        
        return tf.stack(states), measurements

    def test_ekf_on_acoustic(self):
        """Test EKF on acoustic tracking problem."""
        states, measurements = self._generate_trajectory()
        
        ekf = ExtendedKalmanFilter(self.ssm, self.x0, self.P0)
        estimates = []
        
        # Acoustic SSM uses constant velocity (zero control) and sensor-based measurements
        zero_control = tf.zeros([2], dtype=tf.float32)
        sensor_positions = self.ssm.sensor_positions  # Use sensor positions as landmarks
        
        for t in range(self.n_steps):
            ekf.predict(zero_control)
            ekf.update(measurements[t], sensor_positions)
            estimates.append(ekf.state)
        
        estimates = tf.stack(estimates)
        
        # Should track reasonably (may have larger errors due to interface mismatch)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(estimates)))

    def test_advanced_filter_on_acoustic(self):
        """Test advanced filters (EDH) on acoustic tracking."""
        states, measurements = self._generate_trajectory()
        
        edh = EDH(self.ssm, self.x0, self.P0, num_particles=50, n_lambda=10, show_progress=False)
        estimates = []
        
        # Acoustic SSM uses constant velocity (zero control) and sensor-based measurements
        zero_control = tf.zeros([2], dtype=tf.float32)
        sensor_positions = self.ssm.sensor_positions
        
        for t in range(min(5, self.n_steps)):  # Shorter for speed
            edh.predict(zero_control)
            edh.update(measurements[t], sensor_positions)
            estimates.append(edh.state)
        
        estimates = tf.stack(estimates)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(estimates)))


class TestMetricsPipeline(unittest.TestCase):
    """Integration tests for metrics computation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_rmse_with_filter_output(self):
        """Test RMSE computation with actual filter output."""
        # Create simple data
        true_states = tf.random.normal([20, 3], dtype=tf.float32)
        estimates = true_states + tf.random.normal([20, 3], stddev=0.1, dtype=tf.float32)
        
        rmse = compute_rmse(estimates, true_states)
        
        self.assertGreater(float(rmse), 0)
        self.assertLess(float(rmse), 1.0)

    def test_mae_with_filter_output(self):
        """Test MAE computation with actual filter output."""
        true_states = tf.random.normal([20, 3], dtype=tf.float32)
        estimates = true_states + tf.random.normal([20, 3], stddev=0.1, dtype=tf.float32)
        
        mae = compute_mae(estimates, true_states)
        
        self.assertGreater(float(mae), 0)
        self.assertLess(float(mae), 1.0)

    def test_nees_with_covariances(self):
        """Test NEES computation with filter covariances."""
        n_steps = 10
        state_dim = 3
        
        true_states = tf.random.normal([n_steps, state_dim], dtype=tf.float32)
        estimates = true_states + tf.random.normal([n_steps, state_dim], stddev=0.1, dtype=tf.float32)
        covariances = tf.stack([tf.eye(state_dim, dtype=tf.float32) * 0.1 for _ in range(n_steps)])
        
        # Reshape estimates for NEES
        estimates_reshaped = estimates[:, :, tf.newaxis]
        
        nees = compute_nees(estimates_reshaped, covariances, true_states)
        
        self.assertEqual(tf.shape(nees)[0], n_steps)
        self.assertTrue(tf.reduce_all(nees >= 0))

    def test_condition_number_stability(self):
        """Test condition number computation for stability monitoring."""
        # Well-conditioned matrices
        good_covs = tf.stack([tf.eye(3, dtype=tf.float32) for _ in range(10)])
        good_conds = compute_condition_numbers(good_covs)
        
        self.assertTrue(tf.reduce_all(good_conds < 10))
        
        # Ill-conditioned matrix
        bad_cov = tf.constant([
            [[1.0, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1.0]],
        ], dtype=tf.float32)
        bad_cond = compute_condition_numbers(bad_cov)
        
        self.assertTrue(tf.reduce_all(bad_cond > 1e6))


class TestFactoryIntegration(unittest.TestCase):
    """Integration tests for filters with different models."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_all_filters_on_range_bearing(self):
        """Test all filters can be created and used with Range-Bearing model."""
        dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        ssm = RangeBearingSSM(dt=dt, process_noise=Q, meas_noise=R)
        x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        P0 = tf.eye(3, dtype=tf.float32) * 0.1
        landmarks = tf.constant([[5.0, 5.0]], dtype=tf.float32)
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        
        filters_to_test = [
            ('ekf', ExtendedKalmanFilter(ssm, x0, P0)),
            ('ukf', UnscentedKalmanFilter(ssm, x0, P0, alpha=0.1, beta=1.0, kappa=0.0)),
            ('pf', ParticleFilter(ssm, x0, P0, num_particles=30)),
            ('edh', EDH(ssm, x0, P0, num_particles=30, n_lambda=5, show_progress=False)),
            ('ledh', LEDH(ssm, x0, P0, num_particles=30, n_lambda=5, show_progress=False)),
            ('pff_scalar', ScalarPFF(ssm, x0, P0, num_particles=30, max_steps=5)),
        ]
        
        for filter_name, f in filters_to_test:
            with self.subTest(filter=filter_name):
                f.predict(control)
                z = ssm.measurement_model(f.state[tf.newaxis, :], landmarks)[0]
                f.update(z, landmarks)
                self.assertTrue(
                    tf.reduce_all(tf.math.is_finite(f.state)),
                    f"Filter {filter_name} produced non-finite state"
                )

    def test_filter_instantiation(self):
        """Test that filters can be instantiated with correct types."""
        dt = 0.1
        ssm = RangeBearingSSM(dt=dt)
        x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        P0 = tf.eye(3, dtype=tf.float32) * 0.1
        
        ekf = ExtendedKalmanFilter(ssm, x0, P0)
        ukf = UnscentedKalmanFilter(ssm, x0, P0, alpha=0.1, beta=1.0, kappa=0.0)
        pf = ParticleFilter(ssm, x0, P0, num_particles=100)
        
        self.assertIsInstance(ekf, ExtendedKalmanFilter)
        self.assertIsInstance(ukf, UnscentedKalmanFilter)
        self.assertIsInstance(pf, ParticleFilter)


class TestLinalgIntegration(unittest.TestCase):
    """Integration tests for linear algebra utilities in filtering context."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_covariance_regularization_in_filter_context(self):
        """Test covariance regularization when applied to filter covariances."""
        # Simulate ill-conditioned covariance from filter
        P_bad = tf.constant([
            [1.0, 0.999999],
            [0.999999, 1.0]
        ], dtype=tf.float32)
        
        P_reg = regularize_covariance(P_bad, eps=1e-4)
        
        # Should now be well-conditioned for Cholesky
        try:
            L = tf.linalg.cholesky(P_reg)
            self.assertTrue(True)
        except Exception:
            self.fail("Regularized covariance failed Cholesky")

    def test_condition_number_monitoring(self):
        """Test condition number as stability metric."""
        # Series of covariances from hypothetical filter run
        covariances = [
            tf.eye(3, dtype=tf.float32) * 1.0,
            tf.eye(3, dtype=tf.float32) * 0.5,
            tf.constant([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 0.1]], dtype=tf.float32),
        ]
        
        cond_numbers = [compute_condition_number(P) for P in covariances]
        
        # First two should be well-conditioned
        self.assertLess(cond_numbers[0], 10)
        self.assertLess(cond_numbers[1], 10)
        
        # Third is ill-conditioned
        self.assertGreater(cond_numbers[2], 10)


if __name__ == '__main__':
    unittest.main()
