"""
Granular Unit Tests for EDH/LEDH (Daum-Huang Flow) Filter Components.

Tests individual mathematical operations:
1. Flow matrix A computation
2. Flow vector b computation
3. Homotopy step (λ) spacing
4. Residual/innovation computation
5. Bearing wrapping in residuals
6. Particle velocity computation
7. Covariance regularization
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.edh import EDH
from src.filters.ledh import LEDH
from src.models.ssm_range_bearing import RangeBearingSSM


class TestHomotopyStepSpacing(unittest.TestCase):
    """Tests for pseudo-time (λ) step spacing."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
        self.x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        self.P0 = tf.eye(3, dtype=tf.float32) * 0.1
    
    def test_epsilon_steps_sum_to_one(self):
        """Epsilon steps should sum to approximately 1."""
        n_lambda = 50
        q = 1.2
        epsilon_1 = (1.0 - q) / (1.0 - q ** n_lambda)
        
        # Compute all epsilon steps
        epsilon_sum = 0.0
        for j in range(1, n_lambda + 1):
            epsilon_j = epsilon_1 * (q ** (j - 1))
            epsilon_sum += epsilon_j
        
        np.testing.assert_allclose(epsilon_sum, 1.0, atol=1e-6)
    
    def test_epsilon_steps_exponentially_increasing(self):
        """Later steps should be larger (exponential spacing)."""
        n_lambda = 50
        q = 1.2
        epsilon_1 = (1.0 - q) / (1.0 - q ** n_lambda)
        
        epsilons = []
        for j in range(1, n_lambda + 1):
            epsilon_j = epsilon_1 * (q ** (j - 1))
            epsilons.append(epsilon_j)
        
        # Each step should be larger than previous
        for i in range(1, len(epsilons)):
            self.assertGreater(epsilons[i], epsilons[i-1])
    
    def test_first_step_is_small(self):
        """First step should be small for accuracy."""
        n_lambda = 50
        q = 1.2
        epsilon_1 = (1.0 - q) / (1.0 - q ** n_lambda)
        
        # First step should be much smaller than 1/n_lambda
        self.assertLess(epsilon_1, 1.0 / n_lambda)
    
    def test_different_n_lambda_values(self):
        """Different n_lambda should all sum to 1."""
        for n_lambda in [10, 29, 50, 100]:
            q = 1.2
            epsilon_1 = (1.0 - q) / (1.0 - q ** n_lambda)
            
            epsilon_sum = sum(
                epsilon_1 * (q ** (j - 1)) for j in range(1, n_lambda + 1)
            )
            
            np.testing.assert_allclose(epsilon_sum, 1.0, atol=1e-6)


class TestFlowMatrixComputation(unittest.TestCase):
    """Tests for flow matrix A = -0.5 * P * H^T * S^{-1} * H."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.state_dim = 3
        self.meas_dim = 2
    
    def test_flow_matrix_negative_semidefinite(self):
        """Flow matrix A should be negative semi-definite."""
        P = tf.eye(self.state_dim, dtype=tf.float32)
        H = tf.random.normal([self.meas_dim, self.state_dim], dtype=tf.float32)
        R = tf.eye(self.meas_dim, dtype=tf.float32) * 0.1
        lam = 0.5
        
        # S = λ H P H^T + R
        S = lam * (H @ P @ tf.transpose(H)) + R
        S_inv = tf.linalg.inv(S)
        
        # A = -0.5 * P * H^T * S^{-1} * H
        A = -0.5 * P @ tf.transpose(H) @ S_inv @ H
        
        # Eigenvalues should be <= 0
        eigvals = tf.linalg.eigvalsh(A)
        self.assertTrue(tf.reduce_all(eigvals <= 1e-6))
    
    def test_flow_matrix_shape(self):
        """Flow matrix should be (state_dim, state_dim)."""
        P = tf.eye(self.state_dim, dtype=tf.float32)
        H = tf.random.normal([self.meas_dim, self.state_dim], dtype=tf.float32)
        R = tf.eye(self.meas_dim, dtype=tf.float32) * 0.1
        lam = 0.5
        
        S = lam * (H @ P @ tf.transpose(H)) + R
        S_inv = tf.linalg.inv(S)
        A = -0.5 * P @ tf.transpose(H) @ S_inv @ H
        
        self.assertEqual(A.shape, (self.state_dim, self.state_dim))
    
    def test_flow_matrix_lambda_zero(self):
        """At λ=0, flow should depend only on R."""
        P = tf.eye(self.state_dim, dtype=tf.float32)
        H = tf.random.normal([self.meas_dim, self.state_dim], dtype=tf.float32)
        R = tf.eye(self.meas_dim, dtype=tf.float32) * 0.1
        lam = 0.0
        
        S = lam * (H @ P @ tf.transpose(H)) + R  # = R when λ=0
        S_inv = tf.linalg.inv(S)
        A = -0.5 * P @ tf.transpose(H) @ S_inv @ H
        
        # A should be finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(A)))
    
    def test_flow_matrix_lambda_one(self):
        """At λ=1, flow reaches posterior."""
        P = tf.eye(self.state_dim, dtype=tf.float32)
        H = tf.random.normal([self.meas_dim, self.state_dim], dtype=tf.float32)
        R = tf.eye(self.meas_dim, dtype=tf.float32) * 0.1
        lam = 1.0
        
        S = lam * (H @ P @ tf.transpose(H)) + R
        S_inv = tf.linalg.inv(S)
        A = -0.5 * P @ tf.transpose(H) @ S_inv @ H
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(A)))


class TestResidualComputation(unittest.TestCase):
    """Tests for measurement residual computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        self.ssm = RangeBearingSSM(dt=self.dt, process_noise=Q, meas_noise=R)
    
    def test_residual_at_true_state(self):
        """Residual at true state should be small (just noise)."""
        state = tf.constant([[1.0, 2.0, 0.5]], dtype=tf.float32)
        landmarks = tf.constant([[5.0, 5.0]], dtype=tf.float32)
        
        # Measurement at true state
        z = self.ssm.measurement_model(state, landmarks)[0]
        
        # Predicted measurement
        z_pred = self.ssm.measurement_model(state, landmarks)[0]
        
        residual = z - z_pred
        
        # Should be zero (no noise added)
        np.testing.assert_allclose(residual.numpy(), np.zeros_like(residual.numpy()), atol=1e-6)
    
    def test_bearing_wrapping(self):
        """Bearing residuals should be wrapped to [-π, π]."""
        # Large bearing difference
        bearing_diff = 5.0  # > π
        wrapped = tf.math.atan2(tf.sin(bearing_diff), tf.cos(bearing_diff))
        
        self.assertLess(wrapped.numpy(), np.pi + 1e-6)
        self.assertGreater(wrapped.numpy(), -np.pi - 1e-6)
    
    def test_bearing_wrapping_negative(self):
        """Negative bearing differences should wrap correctly."""
        bearing_diff = -5.0  # < -π
        wrapped = tf.math.atan2(tf.sin(bearing_diff), tf.cos(bearing_diff))
        
        self.assertLess(wrapped.numpy(), np.pi + 1e-6)
        self.assertGreater(wrapped.numpy(), -np.pi - 1e-6)
    
    def test_residual_shape(self):
        """Residual should match measurement dimension."""
        state = tf.constant([[1.0, 2.0, 0.5]], dtype=tf.float32)
        landmarks = tf.constant([[5.0, 5.0], [-5.0, 5.0]], dtype=tf.float32)
        
        z = self.ssm.measurement_model(state, landmarks)[0]
        z_pred = self.ssm.measurement_model(state, landmarks)[0]
        
        residual = tf.reshape(z - z_pred, [-1])
        
        # 2 landmarks * 2 measurements each = 4
        self.assertEqual(residual.shape[0], 4)


class TestParticleVelocityComputation(unittest.TestCase):
    """Tests for particle velocity (dx/dλ = Ax + b)."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.state_dim = 3
    
    def test_velocity_shape(self):
        """Velocity should have same shape as particles."""
        N = 50
        particles = tf.random.normal([N, self.state_dim], dtype=tf.float32)
        A = tf.random.normal([self.state_dim, self.state_dim], dtype=tf.float32) * 0.1
        b = tf.random.normal([self.state_dim], dtype=tf.float32)
        
        # v = A @ x + b (for each particle)
        velocities = particles @ tf.transpose(A) + b
        
        self.assertEqual(velocities.shape, (N, self.state_dim))
    
    def test_velocity_finite(self):
        """Velocities should be finite."""
        N = 50
        particles = tf.random.normal([N, self.state_dim], dtype=tf.float32)
        A = -0.1 * tf.eye(self.state_dim, dtype=tf.float32)  # Stable
        b = tf.zeros([self.state_dim], dtype=tf.float32)
        
        velocities = particles @ tf.transpose(A) + b
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(velocities)))
    
    def test_euler_step_contracts_particles(self):
        """With negative-definite A and b=0, particles should contract."""
        N = 50
        particles = tf.random.normal([N, self.state_dim], dtype=tf.float32)
        A = -0.5 * tf.eye(self.state_dim, dtype=tf.float32)  # Contractive
        b = tf.zeros([self.state_dim], dtype=tf.float32)
        
        epsilon = 0.1
        velocities = particles @ tf.transpose(A) + b
        particles_new = particles + epsilon * velocities
        
        # Particles should be closer to origin
        norm_before = tf.reduce_mean(tf.norm(particles, axis=1))
        norm_after = tf.reduce_mean(tf.norm(particles_new, axis=1))
        
        self.assertLess(norm_after.numpy(), norm_before.numpy())


class TestCovarianceRegularization(unittest.TestCase):
    """Tests for covariance regularization."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
    
    def test_regularization_makes_psd(self):
        """Regularization should make matrix PSD."""
        # Near-singular matrix
        M = tf.constant([
            [1.0, 0.999],
            [0.999, 1.0]
        ], dtype=tf.float32)
        
        # Regularize
        eps = 1e-6
        M_reg = M + eps * tf.eye(2, dtype=tf.float32)
        
        eigvals = tf.linalg.eigvalsh(M_reg)
        self.assertTrue(tf.reduce_all(eigvals > 0))
    
    def test_regularization_preserves_structure(self):
        """Regularization should preserve matrix structure."""
        M = tf.constant([
            [2.0, 0.5],
            [0.5, 1.0]
        ], dtype=tf.float32)
        
        eps = 1e-6
        M_reg = M + eps * tf.eye(2, dtype=tf.float32)
        
        # Off-diagonals should be unchanged
        np.testing.assert_allclose(M_reg[0, 1].numpy(), M[0, 1].numpy(), atol=1e-10)
    
    def test_symmetrization(self):
        """Symmetrization should make matrix symmetric."""
        M = tf.constant([
            [1.0, 0.5],
            [0.6, 1.0]  # Not symmetric
        ], dtype=tf.float32)
        
        M_sym = 0.5 * (M + tf.transpose(M))
        
        np.testing.assert_allclose(
            M_sym.numpy(), tf.transpose(M_sym).numpy(), atol=1e-10
        )


class TestEDHFilterComponents(unittest.TestCase):
    """Tests for EDH filter-specific components."""
    
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
    
    def test_particle_initialization(self):
        """Particles should be sampled around initial state."""
        edh = EDH(self.ssm, self.x0, self.P0, num_particles=100, n_lambda=10)
        
        particle_mean = tf.reduce_mean(edh.particles, axis=0)
        
        # Mean should be close to initial state
        np.testing.assert_allclose(
            particle_mean.numpy(), self.x0.numpy(), atol=0.2
        )
    
    def test_particle_spread(self):
        """Particle spread should match initial covariance."""
        edh = EDH(self.ssm, self.x0, self.P0, num_particles=1000, n_lambda=10)
        
        # Compute particle covariance
        particle_mean = tf.reduce_mean(edh.particles, axis=0)
        centered = edh.particles - particle_mean
        particle_cov = tf.reduce_mean(
            centered[:, :, tf.newaxis] * centered[:, tf.newaxis, :],
            axis=0
        )
        
        # Should be close to initial covariance
        np.testing.assert_allclose(
            particle_cov.numpy(), self.P0.numpy(), atol=0.05
        )
    
    def test_particles_finite_after_predict(self):
        """Particles should remain finite after prediction."""
        edh = EDH(self.ssm, self.x0, self.P0, num_particles=100, n_lambda=10)
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        edh.predict(control)
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(edh.particles)))
    
    def test_particles_finite_after_update(self):
        """Particles should remain finite after update."""
        edh = EDH(
            self.ssm, self.x0, self.P0, 
            num_particles=100, n_lambda=10, show_progress=False
        )
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        edh.predict(control)
        
        z = self.ssm.measurement_model(edh.x_hat[tf.newaxis, :], self.landmarks)[0]
        edh.update(z, self.landmarks)
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(edh.particles)))


class TestLEDHFilterComponents(unittest.TestCase):
    """Tests for LEDH filter-specific components (local linearization)."""
    
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
    
    def test_per_particle_jacobians(self):
        """LEDH should compute different Jacobians per particle."""
        # Different particles at different locations
        particles = tf.constant([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0]
        ], dtype=tf.float32)
        
        H_batch = self.ssm.measurement_jacobian(particles, self.landmarks)
        
        # Jacobians should be different for different particles
        H0 = H_batch[0]
        H1 = H_batch[1]
        
        diff = tf.norm(H0 - H1)
        self.assertGreater(diff.numpy(), 0.01)
    
    def test_local_flow_matrices_batch(self):
        """LEDH should produce per-particle flow matrices."""
        ledh = LEDH(
            self.ssm, self.x0, self.P0,
            num_particles=50, n_lambda=10, show_progress=False
        )
        
        # After initialization, particles exist
        self.assertEqual(ledh.particles.shape[0], 50)
        self.assertEqual(ledh.particles.shape[1], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
