"""
Unit tests for ``src/filters/bonus/differentiable_pfpf_ledh.py``.

Covers (granularly):
  Pure helpers (XLA-compatible utilities):
    - _as_2d_observations
    - _regularize_cov
    - _initial_moments
    - _xla_log_abs_det_batched   (d=1, 2, 3, and slogdet fallback)
  KitagawaPFPFLEDHLogLikelihood (the production HMC path):
    - construction + geometric pseudo-time schedule
    - __call__ returns finite scalar
    - differentiability wrt (sigma_v_sq, sigma_w_sq)
    - resampler helper _maybe_ot_resample_1d
  DifferentiablePFPFLEDHLogLikelihood (generic SSM path):
    - construction
    - __call__ on a scalar Kitagawa SSM (finite scalar)
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.differentiable_pfpf_ledh import (
    _as_2d_observations,
    _regularize_cov,
    _initial_moments,
    _xla_log_abs_det_batched,
    DifferentiablePFPFLEDHLogLikelihood,
    KitagawaPFPFLEDHLogLikelihood,
    _LOG_FLOOR,
)
from src.models.ssm_katigawa import PMCMCNonlinearSSM


class TestAs2dObservations(unittest.TestCase):

    def test_1d_to_column(self):
        y = tf.constant([1.0, 2.0, 3.0])
        out = _as_2d_observations(y)
        self.assertEqual(out.shape, (3, 1))

    def test_2d_passthrough(self):
        y = tf.random.normal([5, 2])
        out = _as_2d_observations(y)
        self.assertEqual(out.shape, (5, 2))

    def test_casts_to_float32(self):
        out = _as_2d_observations(tf.constant([1, 2, 3], dtype=tf.int32))
        self.assertEqual(out.dtype, tf.float32)


class TestRegularizeCov(unittest.TestCase):

    def test_symmetrises(self):
        cov = tf.constant([[1.0, 2.0], [0.0, 1.0]])
        out = _regularize_cov(cov)
        np.testing.assert_allclose(out.numpy(), tf.transpose(out).numpy(), atol=1e-6)

    def test_adds_ridge(self):
        cov = tf.zeros([2, 2])
        out = _regularize_cov(cov, eps=1e-3)
        np.testing.assert_allclose(np.diag(out.numpy()), [1e-3, 1e-3], atol=1e-7)

    def test_batched(self):
        cov = tf.zeros([4, 3, 3])
        out = _regularize_cov(cov, eps=1e-2)
        self.assertEqual(out.shape, (4, 3, 3))
        np.testing.assert_allclose(np.diag(out.numpy()[0]), [1e-2] * 3, atol=1e-7)


class TestInitialMoments(unittest.TestCase):

    def test_defaults_from_Q(self):
        ssm = PMCMCNonlinearSSM(sigma_v_sq=4.0, sigma_w_sq=1.0)
        mean, cov = _initial_moments(ssm, state_dim=1)
        self.assertEqual(mean.shape, (1,))
        self.assertEqual(cov.shape, (1, 1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(cov)))
        # Covariance must be positive after regularisation.
        self.assertGreater(float(cov[0, 0]), 0.0)


class TestXLALogAbsDetBatched(unittest.TestCase):

    def test_scalar_dim1(self):
        m = tf.reshape(tf.constant([2.0, 3.0, 0.5]), [3, 1, 1])
        out = _xla_log_abs_det_batched(m, state_dim=1)
        np.testing.assert_allclose(
            out.numpy(), np.log([2.0, 3.0, 0.5]), atol=1e-5
        )

    def test_dim2_matches_numpy(self):
        m = tf.constant([[[1.0, 2.0], [3.0, 5.0]]])  # det = -1
        out = _xla_log_abs_det_batched(m, state_dim=2)
        np.testing.assert_allclose(out.numpy(), [np.log(1.0)], atol=1e-5)

    def test_dim3_matches_numpy(self):
        rng = np.random.default_rng(0)
        mats = rng.normal(size=(2, 3, 3)).astype(np.float32) + 3 * np.eye(3)
        out = _xla_log_abs_det_batched(tf.constant(mats), state_dim=3)
        expected = np.log(np.abs(np.linalg.det(mats)))
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-4)

    def test_dim4_fallback_slogdet(self):
        rng = np.random.default_rng(1)
        mats = rng.normal(size=(2, 4, 4)).astype(np.float32) + 5 * np.eye(4)
        out = _xla_log_abs_det_batched(tf.constant(mats), state_dim=4)
        expected = np.log(np.abs(np.linalg.det(mats)))
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-3)

    def test_floors_zero_determinant(self):
        m = tf.zeros([1, 1, 1])
        out = _xla_log_abs_det_batched(m, state_dim=1)
        self.assertTrue(tf.math.is_finite(out[0]))


class TestKitagawaPFPFLEDHConstruction(unittest.TestCase):

    def test_attributes(self):
        ll = KitagawaPFPFLEDHLogLikelihood(
            num_particles=30, n_lambda=5, initial_var=3.0
        )
        self.assertEqual(ll.num_particles, 30)
        self.assertEqual(ll.n_lambda, 5)
        self.assertEqual(ll.initial_var, 3.0)

    def test_pseudo_time_sums_to_one(self):
        ll = KitagawaPFPFLEDHLogLikelihood(num_particles=10, n_lambda=7)
        self.assertEqual(len(ll.epsilons), 7)
        np.testing.assert_allclose(sum(ll.epsilons), 1.0, atol=1e-5)
        self.assertTrue(all(e > 0 for e in ll.epsilons))


class TestKitagawaPFPFLEDHCall(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(7)
        self.ll = KitagawaPFPFLEDHLogLikelihood(
            num_particles=20, n_lambda=4, initial_var=5.0, resample_threshold=0.5
        )
        # Tiny synthetic Kitagawa observation series.
        self.y = tf.constant([0.3, -0.1, 0.8, 1.2, -0.5], tf.float32)

    def test_returns_finite_scalar(self):
        ll_val = self.ll(
            sigma_v_sq=tf.constant(10.0),
            sigma_w_sq=tf.constant(1.0),
            y_obs=self.y,
        )
        self.assertEqual(ll_val.shape, ())
        self.assertEqual(ll_val.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(ll_val))
        self.assertGreater(float(ll_val), _LOG_FLOOR)

    def test_accepts_2d_observation_column(self):
        ll_val = self.ll(
            sigma_v_sq=tf.constant(8.0),
            sigma_w_sq=tf.constant(1.0),
            y_obs=tf.reshape(self.y, [-1, 1]),
        )
        self.assertEqual(ll_val.shape, ())
        self.assertTrue(tf.math.is_finite(ll_val))

    def test_differentiable_wrt_params(self):
        sv = tf.Variable(10.0)
        sw = tf.Variable(1.0)
        with tf.GradientTape() as tape:
            ll_val = self.ll(sv, sw, self.y)
        grads = tape.gradient(ll_val, [sv, sw])
        self.assertEqual(len(grads), 2)
        for g in grads:
            self.assertIsNotNone(g)
            self.assertTrue(tf.math.is_finite(g))

    def test_clamps_nonpositive_variance(self):
        """Negative variance must be floored, not produce NaN."""
        ll_val = self.ll(
            sigma_v_sq=tf.constant(-1.0),
            sigma_w_sq=tf.constant(-2.0),
            y_obs=self.y,
        )
        self.assertTrue(tf.math.is_finite(ll_val))

    def test_resample_helper_shapes(self):
        N = self.ll.num_particles
        particles = tf.random.normal([N])
        log_w = tf.fill([N], -tf.math.log(float(N)))
        particle_vars = tf.fill([N], 5.0)
        p, lw, pv = self.ll._maybe_ot_resample_1d(particles, log_w, particle_vars)
        self.assertEqual(p.shape, (N,))
        self.assertEqual(lw.shape, (N,))
        self.assertEqual(pv.shape, (N,))


class TestDifferentiablePFPFLEDHGeneric(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(13)
        self.ssm = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0)
        self.ll = DifferentiablePFPFLEDHLogLikelihood(
            num_particles=20,
            n_lambda=4,
            sinkhorn_epsilon=0.5,
            sinkhorn_iters=10,
            resample_threshold=0.7,
        )

    def test_construction_pseudo_time(self):
        self.assertEqual(len(self.ll.epsilons), 4)
        np.testing.assert_allclose(sum(self.ll.epsilons), 1.0, atol=1e-5)

    def test_call_finite_scalar(self):
        y = tf.random.normal([6])
        ll_val = self.ll(self.ssm, y)
        self.assertEqual(ll_val.shape, ())
        self.assertEqual(ll_val.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(ll_val))

    def test_call_with_2d_observations(self):
        y = tf.random.normal([5, 1])
        ll_val = self.ll(self.ssm, y)
        self.assertEqual(ll_val.shape, ())
        self.assertTrue(tf.math.is_finite(ll_val))

    def test_forward_variable_T_finite_scalar(self):
        """The tf.while_loop variant must return a finite scalar too."""
        y = tf.random.normal([5])
        ll_val = self.ll.forward_variable_T(self.ssm, y)
        self.assertEqual(ll_val.shape, ())
        self.assertEqual(ll_val.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(ll_val))

    def test_forward_variable_T_handles_changing_length(self):
        """Same graph must run for two different T without error."""
        a = self.ll.forward_variable_T(self.ssm, tf.random.normal([4]))
        b = self.ll.forward_variable_T(self.ssm, tf.random.normal([7]))
        self.assertTrue(tf.math.is_finite(a))
        self.assertTrue(tf.math.is_finite(b))


if __name__ == "__main__":
    unittest.main()
