"""
Unit tests for advanced filters: EDH, LEDH, PFF, PFPF.
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.edh import EDH
from src.filters.ledh import LEDH
from src.filters.pff_kernel import ScalarPFF, MatrixPFF
from src.filters.pfpf_filter import PFPFEDHFilter, PFPFLEDHFilter
from src.models.ssm_range_bearing import RangeBearingSSM


class TestEDH(unittest.TestCase):
    """Test cases for Exact Daum-Huang filter."""

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

    def test_initialization(self):
        """Test EDH initialization."""
        edh = EDH(
            self.ssm, self.x0, self.P0,
            num_particles=50, n_lambda=10,
            show_progress=False
        )
        
        self.assertIsNotNone(edh)
        self.assertEqual(edh.particles.shape[0], 50)
        self.assertEqual(edh.particles.shape[1], 3)

    def test_predict_step(self):
        """Test EDH prediction step."""
        edh = EDH(
            self.ssm, self.x0, self.P0,
            num_particles=50, n_lambda=10,
            show_progress=False
        )
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        state_pred, cov_pred = edh.predict(control)
        
        self.assertEqual(state_pred.shape, (3,))
        self.assertEqual(cov_pred.shape, (3, 3))

    def test_update_step(self):
        """Test EDH update step."""
        edh = EDH(
            self.ssm, self.x0, self.P0,
            num_particles=50, n_lambda=10,
            show_progress=False
        )
        
        # Generate measurement
        z = self.ssm.measurement_model(self.x0[tf.newaxis, :], self.landmarks)[0]
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        edh.predict(control)
        result = edh.update(z, self.landmarks)  # Returns (state, cov) or updates in place
        
        self.assertEqual(edh.state.shape, (3,))
        self.assertEqual(edh.P.shape, (3, 3))

    def test_predict_update_sequence(self):
        """Test EDH over multiple steps."""
        edh = EDH(
            self.ssm, self.x0, self.P0,
            num_particles=50, n_lambda=10,
            show_progress=False
        )
        
        for _ in range(3):
            control = tf.constant([1.0, 0.1], dtype=tf.float32)
            edh.predict(control)
            
            z = self.ssm.measurement_model(edh.state[tf.newaxis, :], self.landmarks)[0]
            edh.update(z, self.landmarks)
        
        # State should be finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(edh.state)))


class TestLEDH(unittest.TestCase):
    """Test cases for Local Exact Daum-Huang filter."""

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

    def test_initialization(self):
        """Test LEDH initialization."""
        ledh = LEDH(
            self.ssm, self.x0, self.P0,
            num_particles=50, n_lambda=10,
            show_progress=False
        )
        
        self.assertIsNotNone(ledh)
        self.assertEqual(ledh.particles.shape[0], 50)
        self.assertEqual(ledh.particles.shape[1], 3)

    def test_predict_step(self):
        """Test LEDH prediction step."""
        ledh = LEDH(
            self.ssm, self.x0, self.P0,
            num_particles=50, n_lambda=10,
            show_progress=False
        )
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        state_pred, cov_pred = ledh.predict(control)
        
        self.assertEqual(state_pred.shape, (3,))
        self.assertEqual(cov_pred.shape, (3, 3))

    def test_update_step(self):
        """Test LEDH update step."""
        ledh = LEDH(
            self.ssm, self.x0, self.P0,
            num_particles=50, n_lambda=10,
            show_progress=False
        )
        
        z = self.ssm.measurement_model(self.x0[tf.newaxis, :], self.landmarks)[0]
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        ledh.predict(control)
        result = ledh.update(z, self.landmarks)  # Returns (state, cov) or updates in place
        
        self.assertEqual(ledh.state.shape, (3,))
        self.assertEqual(ledh.P.shape, (3, 3))


class TestScalarPFF(unittest.TestCase):
    """Test cases for Scalar Kernel Particle Flow Filter."""

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

    def test_initialization(self):
        """Test ScalarPFF initialization."""
        pff = ScalarPFF(
            self.ssm, self.x0, self.P0,
            num_particles=50,
            step_size=0.05,
            max_steps=10
        )
        
        self.assertIsNotNone(pff)
        self.assertEqual(pff.particles.shape[0], 50)

    def test_predict_step(self):
        """Test ScalarPFF prediction step."""
        pff = ScalarPFF(
            self.ssm, self.x0, self.P0,
            num_particles=50,
            step_size=0.05,
            max_steps=10
        )
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pff.predict(control)  # Updates state in place
        
        self.assertEqual(pff.state.shape, (3,))
        self.assertEqual(pff.P.shape, (3, 3))

    def test_update_step(self):
        """Test ScalarPFF update step."""
        pff = ScalarPFF(
            self.ssm, self.x0, self.P0,
            num_particles=50,
            step_size=0.05,
            max_steps=10
        )
        
        z = self.ssm.measurement_model(self.x0[tf.newaxis, :], self.landmarks)[0]
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pff.predict(control)
        pff.update(z, self.landmarks)  # Updates state in place
        
        self.assertEqual(pff.state.shape, (3,))
        self.assertEqual(pff.P.shape, (3, 3))

    def test_particles_finite(self):
        """Test that particles remain finite after filtering."""
        pff = ScalarPFF(
            self.ssm, self.x0, self.P0,
            num_particles=50,
            step_size=0.05,
            max_steps=10
        )
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pff.predict(control)
        
        z = self.ssm.measurement_model(pff.state[tf.newaxis, :], self.landmarks)[0]
        pff.update(z, self.landmarks)
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pff.particles)))


class TestMatrixPFF(unittest.TestCase):
    """Test cases for Matrix Kernel Particle Flow Filter."""

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

    def test_initialization(self):
        """Test MatrixPFF initialization."""
        pff = MatrixPFF(
            self.ssm, self.x0, self.P0,
            num_particles=50,
            step_size=0.05,
            max_steps=10
        )
        
        self.assertIsNotNone(pff)
        self.assertEqual(pff.particles.shape[0], 50)

    def test_predict_step(self):
        """Test MatrixPFF prediction step."""
        pff = MatrixPFF(
            self.ssm, self.x0, self.P0,
            num_particles=50,
            step_size=0.05,
            max_steps=10
        )
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pff.predict(control)  # Updates state in place
        
        self.assertEqual(pff.state.shape, (3,))
        self.assertEqual(pff.P.shape, (3, 3))

    def test_update_step(self):
        """Test MatrixPFF update step."""
        pff = MatrixPFF(
            self.ssm, self.x0, self.P0,
            num_particles=50,
            step_size=0.05,
            max_steps=10
        )
        
        z = self.ssm.measurement_model(self.x0[tf.newaxis, :], self.landmarks)[0]
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pff.predict(control)
        pff.update(z, self.landmarks)  # Updates state in place
        
        self.assertEqual(pff.state.shape, (3,))
        self.assertEqual(pff.P.shape, (3, 3))


class TestPFPFEDH(unittest.TestCase):
    """Test cases for Particle Flow Particle Filter (EDH variant)."""

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
        self.num_particles = 50

    def _make_filter(self, n_lambda=10):
        return PFPFEDHFilter(
            self.ssm, self.x0, self.P0,
            num_particles=self.num_particles, n_lambda=n_lambda,
            show_progress=False,
        )

    def test_initialization(self):
        """Test PFPF-EDH initialization."""
        pfpf = self._make_filter()
        self.assertIsNotNone(pfpf)

    def test_predict_step(self):
        """Test PFPF-EDH prediction step."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        self.assertEqual(pfpf.state.shape, (3,))
        self.assertEqual(pfpf.P.shape, (3, 3))

    def test_update_step(self):
        """Test PFPF-EDH update step."""
        pfpf = self._make_filter()
        z = self.ssm.measurement_model(self.x0[tf.newaxis, :], self.landmarks)[0]
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        pfpf.update(z, self.landmarks)
        self.assertEqual(pfpf.state.shape, (3,))
        self.assertEqual(pfpf.P.shape, (3, 3))

    def test_particles_finite_after_update(self):
        """Particles should remain finite after a predict-update cycle."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        z = self.ssm.measurement_model(pfpf.state[tf.newaxis, :], self.landmarks)[0]
        pfpf.update(z, self.landmarks)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pfpf.particles)))

    def test_state_finite_after_update(self):
        """State estimate should remain finite after update."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        z = self.ssm.measurement_model(pfpf.state[tf.newaxis, :], self.landmarks)[0]
        pfpf.update(z, self.landmarks)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pfpf.state)))

    def test_weights_sum_to_one(self):
        """Weights should sum to ~1 after normalization."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        z = self.ssm.measurement_model(pfpf.state[tf.newaxis, :], self.landmarks)[0]
        pfpf.update(z, self.landmarks)
        weight_sum = float(tf.reduce_sum(pfpf.weights).numpy())
        self.assertAlmostEqual(weight_sum, 1.0, places=4)

    def test_covariance_psd(self):
        """Global covariance P should remain PSD after predict."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        eigvals = tf.linalg.eigvalsh(pfpf.P)
        self.assertTrue(tf.reduce_all(eigvals >= -1e-5))

    def test_predict_changes_particles(self):
        """Predict should move particles from their initial positions."""
        pfpf = self._make_filter()
        particles_before = pfpf.particles.numpy().copy()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        self.assertFalse(
            np.allclose(pfpf.particles.numpy(), particles_before, atol=1e-6))

    def test_predict_update_sequence(self):
        """Multi-step predict-update should remain finite and stable."""
        pfpf = self._make_filter(n_lambda=5)
        for _ in range(3):
            control = tf.constant([1.0, 0.1], dtype=tf.float32)
            pfpf.predict(control)
            z = self.ssm.measurement_model(
                pfpf.state[tf.newaxis, :], self.landmarks)[0]
            pfpf.update(z, self.landmarks)

        self.assertTrue(tf.reduce_all(tf.math.is_finite(pfpf.state)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pfpf.particles)))
        weight_sum = float(tf.reduce_sum(pfpf.weights).numpy())
        self.assertAlmostEqual(weight_sum, 1.0, places=4)


class TestPFPFLEDH(unittest.TestCase):
    """Test cases for Particle Flow Particle Filter (LEDH variant)."""

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
        self.num_particles = 50

    def _make_filter(self, n_lambda=10):
        return PFPFLEDHFilter(
            self.ssm, self.x0, self.P0,
            num_particles=self.num_particles, n_lambda=n_lambda,
            show_progress=False,
        )

    def test_initialization(self):
        """Test PFPF-LEDH initialization."""
        pfpf = self._make_filter()
        self.assertIsNotNone(pfpf)

    def test_predict_step(self):
        """Test PFPF-LEDH prediction step."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        self.assertEqual(pfpf.state.shape, (3,))
        self.assertEqual(pfpf.particle_covs.shape, (self.num_particles, 3, 3))

    def test_update_step(self):
        """Test PFPF-LEDH update step."""
        pfpf = self._make_filter()
        z = self.ssm.measurement_model(self.x0[tf.newaxis, :], self.landmarks)[0]
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        pfpf.update(z, self.landmarks)
        self.assertEqual(pfpf.state.shape, (3,))
        self.assertEqual(pfpf.particle_covs.shape, (self.num_particles, 3, 3))

    def test_particles_finite_after_update(self):
        """Particles should remain finite after a predict-update cycle."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        z = self.ssm.measurement_model(pfpf.state[tf.newaxis, :], self.landmarks)[0]
        pfpf.update(z, self.landmarks)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pfpf.particles)))

    def test_state_finite_after_update(self):
        """State estimate should remain finite after update."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        z = self.ssm.measurement_model(pfpf.state[tf.newaxis, :], self.landmarks)[0]
        pfpf.update(z, self.landmarks)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pfpf.state)))

    def test_weights_sum_to_one(self):
        """Weights should sum to ~1 after normalization."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        z = self.ssm.measurement_model(pfpf.state[tf.newaxis, :], self.landmarks)[0]
        pfpf.update(z, self.landmarks)
        weight_sum = float(tf.reduce_sum(pfpf.weights).numpy())
        self.assertAlmostEqual(weight_sum, 1.0, places=4)

    def test_weights_nonnegative(self):
        """All weights should be non-negative."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        z = self.ssm.measurement_model(pfpf.state[tf.newaxis, :], self.landmarks)[0]
        pfpf.update(z, self.landmarks)
        self.assertTrue(tf.reduce_all(pfpf.weights >= 0.0))

    def test_particle_covs_psd(self):
        """Per-particle covariances should be positive semi-definite."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        eigvals = tf.linalg.eigvalsh(pfpf.particle_covs)
        self.assertTrue(tf.reduce_all(eigvals >= -1e-5))

    def test_predict_changes_particles(self):
        """Predict should move particles from their initial positions."""
        pfpf = self._make_filter()
        particles_before = pfpf.particles.numpy().copy()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        self.assertFalse(
            np.allclose(pfpf.particles.numpy(), particles_before, atol=1e-6))

    def test_predict_update_sequence(self):
        """Multi-step predict-update should remain finite and stable."""
        pfpf = self._make_filter(n_lambda=5)
        for _ in range(3):
            control = tf.constant([1.0, 0.1], dtype=tf.float32)
            pfpf.predict(control)
            z = self.ssm.measurement_model(
                pfpf.state[tf.newaxis, :], self.landmarks)[0]
            pfpf.update(z, self.landmarks)

        self.assertTrue(tf.reduce_all(tf.math.is_finite(pfpf.state)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pfpf.particles)))
        weight_sum = float(tf.reduce_sum(pfpf.weights).numpy())
        self.assertAlmostEqual(weight_sum, 1.0, places=4)

    def test_ess_tracking(self):
        """ESS should be tracked and be in valid range."""
        pfpf = self._make_filter()
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        pfpf.predict(control)
        z = self.ssm.measurement_model(pfpf.state[tf.newaxis, :], self.landmarks)[0]
        pfpf.update(z, self.landmarks)
        ess = float(pfpf.ess_before_resample.numpy())
        self.assertGreater(ess, 0.0)
        self.assertLessEqual(ess, float(self.num_particles))


class TestFilterComparison(unittest.TestCase):
    """Integration tests comparing different advanced filters."""

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

    def test_all_filters_produce_finite_estimates(self):
        """Test that all advanced filters produce finite state estimates."""
        filters = [
            EDH(self.ssm, self.x0, self.P0, num_particles=30, n_lambda=5, show_progress=False),
            LEDH(self.ssm, self.x0, self.P0, num_particles=30, n_lambda=5, show_progress=False),
            ScalarPFF(self.ssm, self.x0, self.P0, num_particles=30, max_steps=5),
            MatrixPFF(self.ssm, self.x0, self.P0, num_particles=30, max_steps=5),
            PFPFEDHFilter(self.ssm, self.x0, self.P0, num_particles=30, n_lambda=5, show_progress=False),
            PFPFLEDHFilter(self.ssm, self.x0, self.P0, num_particles=30, n_lambda=5, show_progress=False),
        ]
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        
        for f in filters:
            f.predict(control)
            z = self.ssm.measurement_model(f.state[tf.newaxis, :], self.landmarks)[0]
            f.update(z, self.landmarks)
            
            self.assertTrue(
                tf.reduce_all(tf.math.is_finite(f.state)),
                f"Filter {type(f).__name__} produced non-finite state"
            )

    def test_covariance_positive_definite(self):
        """Test that all filters maintain PSD covariance."""
        filters = [
            EDH(self.ssm, self.x0, self.P0, num_particles=30, n_lambda=5, show_progress=False),
            LEDH(self.ssm, self.x0, self.P0, num_particles=30, n_lambda=5, show_progress=False),
            ScalarPFF(self.ssm, self.x0, self.P0, num_particles=30, max_steps=5),
            MatrixPFF(self.ssm, self.x0, self.P0, num_particles=30, max_steps=5),
        ]
        
        control = tf.constant([1.0, 0.1], dtype=tf.float32)
        
        for f in filters:
            f.predict(control)
            z = self.ssm.measurement_model(f.state[tf.newaxis, :], self.landmarks)[0]
            f.update(z, self.landmarks)
            
            # Get covariance (different filters use different attribute names)
            cov = getattr(f, 'covariance', None) or getattr(f, 'P', None)
            if cov is not None:
                # Check covariance is PSD (all eigenvalues >= 0)
                eigvals = tf.linalg.eigvalsh(cov)
                self.assertTrue(
                    tf.reduce_all(eigvals >= -1e-6),
                    f"Filter {type(f).__name__} has non-PSD covariance"
                )


if __name__ == '__main__':
    unittest.main()
