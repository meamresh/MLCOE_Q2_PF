"""
Unit tests for Particle Filter.
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.particle_filter import ParticleFilter
from src.models.ssm_range_bearing import RangeBearingSSM
from src.metrics.particle_filter_metrics import (
    compute_effective_sample_size,
    compute_weight_entropy,
    compute_weight_variance
)


class TestParticleFilter(unittest.TestCase):
    """Test cases for Particle Filter."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.initial_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.initial_cov = tf.eye(3, dtype=tf.float32) * 0.1
        self.num_particles = 100
        
        self.landmarks = tf.constant(
            [[5.0, 5.0], [-5.0, 5.0]],
            dtype=tf.float32
        )

    def test_filter_initialization(self):
        """Test Particle Filter initialization."""
        pf = ParticleFilter(
            self.ssm, self.initial_state, self.initial_cov,
            num_particles=self.num_particles
        )
        
        # Check shapes
        self.assertEqual(pf.particles.shape, (self.num_particles, 3))
        self.assertEqual(pf.weights.shape, (self.num_particles,))
        self.assertEqual(pf.state.shape, (3,))
        self.assertEqual(pf.covariance.shape, (3, 3))
        
        # Check weights sum to 1
        weights_sum = tf.reduce_sum(pf.weights)
        tf.debugging.assert_near(weights_sum, 1.0, atol=1e-5)
        
        # Check weights are non-negative
        self.assertTrue(tf.reduce_all(pf.weights >= 0.0))
        
        # Check covariance is symmetric and PSD
        cov_sym = 0.5 * (pf.covariance + tf.transpose(pf.covariance))
        tf.debugging.assert_near(pf.covariance, cov_sym, atol=1e-6)
        eigvals = tf.linalg.eigvalsh(pf.covariance)
        self.assertTrue(tf.reduce_all(eigvals > 0.0))

    def test_predict_step(self):
        """Test prediction step."""
        pf = ParticleFilter(
            self.ssm, self.initial_state, self.initial_cov,
            num_particles=self.num_particles
        )
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        state_before = tf.identity(pf.state)
        particles_before = tf.identity(pf.particles)
        
        state_pred, cov_pred = pf.predict(control)
        
        # Check shapes
        self.assertEqual(state_pred.shape, (3,))
        self.assertEqual(cov_pred.shape, (3, 3))
        
        # State should have changed (motion)
        self.assertFalse(tf.reduce_all(tf.abs(state_pred - state_before) < 1e-6))
        
        # Particles should have changed
        self.assertFalse(tf.reduce_all(tf.abs(pf.particles - particles_before) < 1e-6))
        
        # Covariance should be symmetric and PSD
        cov_sym = 0.5 * (cov_pred + tf.transpose(cov_pred))
        tf.debugging.assert_near(cov_pred, cov_sym, atol=1e-6)
        eigvals = tf.linalg.eigvalsh(cov_pred)
        self.assertTrue(tf.reduce_all(eigvals > 0.0))

    def test_update_step(self):
        """Test update step."""
        pf = ParticleFilter(
            self.ssm, self.initial_state, self.initial_cov,
            num_particles=self.num_particles
        )
        
        # Generate measurement
        true_meas = self.ssm.measurement_model(
            self.initial_state[tf.newaxis, :], self.landmarks
        )[0]
        meas_std = tf.sqrt(tf.linalg.diag_part(self.ssm.R))
        noise = tf.random.normal([tf.shape(self.landmarks)[0], 2],
                                 mean=0.0, stddev=meas_std, dtype=tf.float32)
        measurement = true_meas + noise
        
        weights_before = tf.identity(pf.weights)
        
        state_upd, cov_upd, residual, did_resample = pf.update(measurement, self.landmarks)
        
        # Check shapes
        self.assertEqual(state_upd.shape, (3,))
        self.assertEqual(cov_upd.shape, (3, 3))
        self.assertEqual(residual.shape[0], 2 * tf.shape(self.landmarks)[0])
        self.assertIsInstance(did_resample, bool)
        
        # Weights should still sum to 1
        weights_sum = tf.reduce_sum(pf.weights)
        tf.debugging.assert_near(weights_sum, 1.0, atol=1e-5)
        
        # Weights should be non-negative
        self.assertTrue(tf.reduce_all(pf.weights >= 0.0))
        
        # Covariance should be symmetric and PSD
        cov_sym = 0.5 * (cov_upd + tf.transpose(cov_upd))
        tf.debugging.assert_near(cov_upd, cov_sym, atol=1e-6)
        eigvals = tf.linalg.eigvalsh(cov_upd)
        self.assertTrue(tf.reduce_all(eigvals > 0.0))

    def test_resampling_trigger(self):
        """Test that resampling is triggered when ESS is low."""
        pf = ParticleFilter(
            self.ssm, self.initial_state, self.initial_cov,
            num_particles=self.num_particles,
            resample_threshold=0.5
        )
        
        # Create degenerate weights (one particle has most weight)
        degenerate_weights = tf.ones(self.num_particles, dtype=tf.float32) * 1e-6
        # Set first particle to have most weight
        indices = tf.constant([[0]], dtype=tf.int32)
        updates = tf.constant([1.0 - (self.num_particles - 1) * 1e-6], dtype=tf.float32)
        degenerate_weights = tf.tensor_scatter_nd_update(degenerate_weights, indices, updates)
        pf.weights = degenerate_weights
        
        # Check ESS is low
        ess = compute_effective_sample_size(pf.weights)
        self.assertLess(float(ess.numpy()), self.num_particles * 0.5)
        
        # Generate measurement
        true_meas = self.ssm.measurement_model(
            self.initial_state[tf.newaxis, :], self.landmarks
        )[0]
        measurement = true_meas
        
        # Update should trigger resampling
        _, _, _, did_resample = pf.update(measurement, self.landmarks)
        
        # Resampling should have occurred
        self.assertTrue(did_resample)
        
        # After resampling, weights should be uniform
        expected_weight = 1.0 / self.num_particles
        tf.debugging.assert_near(
            pf.weights,
            tf.ones(self.num_particles, dtype=tf.float32) * expected_weight,
            atol=1e-4
        )

    def test_predict_update_sequence(self):
        """Test a sequence of predict and update steps."""
        pf = ParticleFilter(
            self.ssm, self.initial_state, self.initial_cov,
            num_particles=self.num_particles
        )
        
        states = []
        covariances = []
        
        for step in range(5):
            control = tf.constant([1.0, 0.1], dtype=tf.float32)
            pf.predict(control)
            
            # Generate measurement
            true_meas = self.ssm.measurement_model(
                pf.state[tf.newaxis, :], self.landmarks
            )[0]
            measurement = true_meas + tf.random.normal(
                [tf.shape(self.landmarks)[0], 2],
                stddev=0.1, dtype=tf.float32
            )
            
            state, cov, _, _ = pf.update(measurement, self.landmarks)
            states.append(state)
            covariances.append(cov)
            
            # Check weights sum to 1 at each step
            weights_sum = tf.reduce_sum(pf.weights)
            tf.debugging.assert_near(weights_sum, 1.0, atol=1e-5)
        
        # All states should be finite
        for state in states:
            self.assertTrue(tf.reduce_all(tf.math.is_finite(state)))
        
        # All covariances should be PSD
        for cov in covariances:
            eigvals = tf.linalg.eigvalsh(cov)
            self.assertTrue(tf.reduce_all(eigvals > 0.0))

    def test_state_estimate_computation(self):
        """Test that state estimate is weighted mean of particles."""
        pf = ParticleFilter(
            self.ssm, self.initial_state, self.initial_cov,
            num_particles=self.num_particles
        )
        
        # Manually compute weighted mean
        weights_expanded = tf.expand_dims(pf.weights, axis=1)
        expected_state = tf.reduce_sum(weights_expanded * pf.particles, axis=0)
        
        # Should match computed state
        tf.debugging.assert_near(pf.state, expected_state, atol=1e-6)

    def test_covariance_estimate_computation(self):
        """Test that covariance estimate is weighted covariance."""
        pf = ParticleFilter(
            self.ssm, self.initial_state, self.initial_cov,
            num_particles=self.num_particles
        )
        
        # Manually compute weighted covariance
        state_mean = pf.state
        diff = pf.particles - state_mean
        weights_expanded = tf.expand_dims(pf.weights, axis=1)
        weighted_diff = weights_expanded * diff
        expected_cov = tf.matmul(weighted_diff, diff, transpose_a=True)
        expected_cov = 0.5 * (expected_cov + tf.transpose(expected_cov))
        
        # Should be close (allowing for regularization)
        cov_diff = tf.linalg.norm(pf.covariance - expected_cov)
        self.assertLess(float(cov_diff.numpy()), 1e-4)


class TestParticleFilterMetrics(unittest.TestCase):
    """Test cases for particle filter metrics."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_compute_effective_sample_size(self):
        """Test ESS computation."""
        # Uniform weights (best case)
        uniform_weights = tf.ones(10, dtype=tf.float32) / 10.0
        ess_uniform = compute_effective_sample_size(uniform_weights)
        tf.debugging.assert_near(ess_uniform, 10.0, atol=1e-5)
        
        # Degenerate weights (worst case - one particle has all weight)
        degenerate_weights = tf.zeros(10, dtype=tf.float32)
        indices = tf.constant([[0]], dtype=tf.int32)
        updates = tf.constant([1.0], dtype=tf.float32)
        degenerate_weights = tf.tensor_scatter_nd_update(degenerate_weights, indices, updates)
        ess_degenerate = compute_effective_sample_size(degenerate_weights)
        tf.debugging.assert_near(ess_degenerate, 1.0, atol=1e-5)
        
        # ESS should be between 1 and num_particles
        self.assertGreaterEqual(float(ess_uniform.numpy()), 1.0)
        self.assertLessEqual(float(ess_uniform.numpy()), 10.0)

    def test_compute_weight_entropy(self):
        """Test weight entropy computation."""
        # Uniform weights (maximum entropy)
        uniform_weights = tf.ones(10, dtype=tf.float32) / 10.0
        entropy_uniform = compute_weight_entropy(uniform_weights, normalize=True)
        tf.debugging.assert_near(entropy_uniform, 1.0, atol=1e-5)
        
        # Degenerate weights (minimum entropy)
        degenerate_weights = tf.zeros(10, dtype=tf.float32)
        indices = tf.constant([[0]], dtype=tf.int32)
        updates = tf.constant([1.0], dtype=tf.float32)
        degenerate_weights = tf.tensor_scatter_nd_update(degenerate_weights, indices, updates)
        entropy_degenerate = compute_weight_entropy(degenerate_weights, normalize=True)
        self.assertLess(float(entropy_degenerate.numpy()), 0.1)
        
        # Normalized entropy should be in [0, 1]
        self.assertGreaterEqual(float(entropy_uniform.numpy()), 0.0)
        self.assertLessEqual(float(entropy_uniform.numpy()), 1.0)

    def test_compute_weight_entropy_not_normalized(self):
        """Test weight entropy without normalization."""
        uniform_weights = tf.ones(10, dtype=tf.float32) / 10.0
        entropy = compute_weight_entropy(uniform_weights, normalize=False)
        
        # Should be log(10) ≈ 2.303
        expected = tf.math.log(10.0)
        tf.debugging.assert_near(entropy, expected, atol=1e-5)

    def test_compute_weight_variance(self):
        """Test weight variance computation."""
        # Uniform weights (zero variance)
        uniform_weights = tf.ones(10, dtype=tf.float32) / 10.0
        var_uniform = compute_weight_variance(uniform_weights)
        tf.debugging.assert_near(var_uniform, 0.0, atol=1e-6)
        
        # Non-uniform weights (positive variance)
        non_uniform = tf.constant([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 dtype=tf.float32)
        var_non_uniform = compute_weight_variance(non_uniform)
        self.assertGreater(float(var_non_uniform.numpy()), 0.0)
        
        # Variance should be non-negative
        self.assertGreaterEqual(float(var_uniform.numpy()), 0.0)


if __name__ == '__main__':
    unittest.main()

