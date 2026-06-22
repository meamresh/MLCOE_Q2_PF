"""
Unit tests for the SVSSM differentiable LEDH log-likelihood filters:

  - src/filters/bonus/extra_bonus/differentiable_ledh_svssm.py
      DifferentiableLEDHLogLikelihoodSVSSM (univariate)
      module helpers: _safe_scalar, _as_real_log_scalar, LOG_CHI2 constants
  - src/filters/bonus/extra_bonus/differentiable_ledh_svssm_outerjit.py
      DifferentiableLEDHLogLikelihoodSVSSMOuterJIT
  - src/filters/bonus/extra_bonus/differentiable_ledh_svssm_multivariate.py
      expm_2x2_batch, expm_pade_batch, pade_scaling_for_dim, _safe_nd
      DifferentiableLEDHLogLikelihoodSVSSMmulti (diagonal + full-Phi paths)

All filter calls use tiny N / n_lambda / T to keep runtime low.
"""

import math
import unittest
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))


def _expm_ref(A: np.ndarray) -> np.ndarray:
    """Reference matrix exponential via eigendecomposition.

    Independent of the module under test and scipy-free (CI installs only
    requirements.txt, which deliberately omits scipy). Valid for the
    generically-diagonalisable small random matrices used in these tests.
    """
    w, V = np.linalg.eig(A)
    return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    DifferentiableLEDHLogLikelihoodSVSSM,
    _safe_scalar,
    _as_real_log_scalar,
    LOG_CHI2_MEAN,
    LOG_CHI2_VAR,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_outerjit import (
    DifferentiableLEDHLogLikelihoodSVSSMOuterJIT,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
    expm_2x2_batch,
    expm_pade_batch,
    pade_scaling_for_dim,
    _safe_nd,
)


def _svssm_obs(T=6, mu=0.0, phi=0.9, sigma_eta=0.5, seed=0):
    tf.random.set_seed(seed)
    h = tf.constant(mu, tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


# ---------------------------------------------------------------------------
# Module-level scalar helpers
# ---------------------------------------------------------------------------

class TestScalarHelpers(unittest.TestCase):

    def test_log_chi2_constants(self):
        self.assertAlmostEqual(LOG_CHI2_MEAN, -1.270362845461478, places=10)
        self.assertAlmostEqual(LOG_CHI2_VAR, math.pi ** 2 / 2.0, places=10)

    def test_safe_scalar_replaces_nonfinite(self):
        x = tf.constant([float("nan"), float("inf"), 1.0, -1.0])
        out = _safe_scalar(x)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))
        self.assertAlmostEqual(float(out[2]), 1.0)

    def test_safe_scalar_clamps(self):
        out = _safe_scalar(tf.constant([1e9, -1e9]))
        self.assertLessEqual(float(out[0]), 1e4 + 1)
        self.assertGreaterEqual(float(out[1]), -1e4 - 1)

    def test_as_real_log_scalar_finite(self):
        out = _as_real_log_scalar(tf.constant(3.5))
        self.assertEqual(out.dtype, tf.float32)
        self.assertAlmostEqual(float(out), 3.5, places=5)

    def test_as_real_log_scalar_floors_nan(self):
        out = _as_real_log_scalar(tf.constant(float("nan")))
        self.assertAlmostEqual(float(out), -1e6)


# ---------------------------------------------------------------------------
# Univariate filter
# ---------------------------------------------------------------------------

class TestSVSSMUnivariate(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.y = _svssm_obs(T=6)

    def _make(self, **kw):
        params = dict(num_particles=24, n_lambda=4, jit_compile=False)
        params.update(kw)
        return DifferentiableLEDHLogLikelihoodSVSSM(**params)

    def test_construction_pseudo_time_sums_to_one(self):
        ll = self._make(n_lambda=8)
        self.assertEqual(len(ll.epsilons), 8)
        np.testing.assert_allclose(sum(ll.epsilons), 1.0, atol=1e-5)

    def test_invalid_integrator_raises(self):
        with self.assertRaises(ValueError):
            self._make(integrator="midpoint")

    def test_invalid_init_type_raises(self):
        with self.assertRaises(ValueError):
            self._make(init_type="weird")

    def test_call_finite_scalar_eager(self):
        ll = self._make(jit_compile=False)
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(0.25), self.y)
        self.assertEqual(out.shape, ())
        self.assertEqual(out.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(out))

    def test_call_finite_scalar_jit(self):
        ll = self._make(jit_compile=True)
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(0.25), self.y)
        self.assertTrue(tf.math.is_finite(out))

    def test_euler_integrator_runs(self):
        ll = self._make(integrator="euler", jit_compile=False)
        out = ll(tf.constant(0.0), tf.constant(0.85), tf.constant(0.3), self.y)
        self.assertTrue(tf.math.is_finite(out))

    def test_init_type_variants_run(self):
        for it in ("stationary", "fixed_mu", "diffuse"):
            ll = self._make(init_type=it, jit_compile=False)
            out = ll(tf.constant(0.1), tf.constant(0.8), tf.constant(0.2), self.y)
            self.assertTrue(tf.math.is_finite(out), msg=it)

    def test_differentiable_wrt_theta(self):
        ll = self._make(jit_compile=False)
        mu = tf.Variable(0.0)
        phi = tf.Variable(0.9)
        sig = tf.Variable(0.25)
        with tf.GradientTape() as tape:
            out = ll(mu, phi, sig, self.y)
        grads = tape.gradient(out, [mu, phi, sig])
        # At least one parameter must carry a finite, non-None gradient.
        finite = [g for g in grads if g is not None and bool(tf.math.is_finite(g))]
        self.assertGreater(len(finite), 0)

    def test_y_obs_2d_column_accepted(self):
        ll = self._make(jit_compile=False)
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(0.25),
                 tf.reshape(self.y, [-1, 1]))
        self.assertTrue(tf.math.is_finite(out))

    def test_negative_variance_floored(self):
        ll = self._make(jit_compile=False)
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(-5.0), self.y)
        self.assertTrue(tf.math.is_finite(out))


# ---------------------------------------------------------------------------
# Outer-JIT univariate filter
# ---------------------------------------------------------------------------

class TestSVSSMOuterJIT(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.y = _svssm_obs(T=6)

    def test_invalid_integrator_raises(self):
        with self.assertRaises(ValueError):
            DifferentiableLEDHLogLikelihoodSVSSMOuterJIT(integrator="nope")

    def test_per_step_jit_call_finite(self):
        ll = DifferentiableLEDHLogLikelihoodSVSSMOuterJIT(
            num_particles=24, n_lambda=4, jit_compile=True, outer_jit=False
        )
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(0.25), self.y)
        self.assertTrue(tf.math.is_finite(out))

    def test_outer_jit_call_finite(self):
        ll = DifferentiableLEDHLogLikelihoodSVSSMOuterJIT(
            num_particles=24, n_lambda=4, jit_compile=True, outer_jit=True
        )
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(0.25), self.y)
        self.assertTrue(tf.math.is_finite(out))


# ---------------------------------------------------------------------------
# Multivariate batched-expm helpers
# ---------------------------------------------------------------------------

class TestExpm2x2Batch(unittest.TestCase):

    def test_zero_matrix_is_identity(self):
        A = tf.zeros([3, 2, 2])
        out = expm_2x2_batch(A)
        np.testing.assert_allclose(
            out.numpy(), np.tile(np.eye(2), (3, 1, 1)), atol=1e-5
        )

    def test_matches_reference_random(self):
        rng = np.random.default_rng(0)
        mats = rng.normal(scale=0.5, size=(5, 2, 2)).astype(np.float32)
        out = expm_2x2_batch(tf.constant(mats)).numpy()
        for k in range(5):
            np.testing.assert_allclose(out[k], _expm_ref(mats[k]), atol=1e-4)

    def test_diagonal_matrix(self):
        A = tf.constant([[[0.3, 0.0], [0.0, -0.7]]])
        out = expm_2x2_batch(A).numpy()[0]
        np.testing.assert_allclose(np.diag(out), [math.exp(0.3), math.exp(-0.7)], atol=1e-5)


class TestExpmPadeBatch(unittest.TestCase):

    def test_matches_reference_d3(self):
        rng = np.random.default_rng(2)
        mats = rng.normal(scale=0.2, size=(4, 3, 3)).astype(np.float32)
        s = pade_scaling_for_dim(3)
        out = expm_pade_batch(tf.constant(mats), s).numpy()
        for k in range(4):
            np.testing.assert_allclose(out[k], _expm_ref(mats[k]), atol=1e-3)

    def test_zero_is_identity(self):
        A = tf.zeros([2, 4, 4])
        out = expm_pade_batch(A, pade_scaling_for_dim(4)).numpy()
        np.testing.assert_allclose(out, np.tile(np.eye(4), (2, 1, 1)), atol=1e-5)


class TestPadeScalingForDim(unittest.TestCase):

    def test_nonnegative(self):
        for d in (1, 2, 3, 5, 8):
            self.assertGreaterEqual(pade_scaling_for_dim(d), 0)

    def test_monotone_increasing(self):
        self.assertLessEqual(pade_scaling_for_dim(2), pade_scaling_for_dim(8))


class TestSafeNd(unittest.TestCase):

    def test_replaces_and_clamps(self):
        x = tf.constant([[float("nan"), 1e9], [-1e9, 2.0]])
        out = _safe_nd(x)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))
        self.assertLessEqual(float(tf.reduce_max(out)), 1e4 + 1)


# ---------------------------------------------------------------------------
# Multivariate filter (diagonal Phi + full-Phi entry points)
# ---------------------------------------------------------------------------

class TestSVSSMMultivariate(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.d = 2
        self.T = 5
        self.y = tf.random.normal([self.T, self.d]) * 0.5

    def _make(self, **kw):
        params = dict(state_dim=self.d, num_particles=16, n_lambda=3)
        params.update(kw)
        return DifferentiableLEDHLogLikelihoodSVSSMmulti(**params)

    def test_invalid_init_type_raises(self):
        with self.assertRaises(ValueError):
            self._make(init_type="bad")

    def test_diagonal_call_finite_scalar(self):
        ll = self._make()
        mu = tf.zeros([self.d])
        phi_diag = tf.constant([0.9, 0.85])
        sig_diag_sq = tf.constant([0.25, 0.2])
        out = ll(mu, phi_diag, sig_diag_sq, self.y)
        self.assertEqual(out.shape, ())
        self.assertTrue(tf.math.is_finite(out))

    def test_full_phi_call_finite_scalar(self):
        ll = self._make()
        mu = tf.zeros([self.d])
        Phi = tf.constant([[0.9, 0.05], [0.0, 0.85]])
        L_eta = tf.constant([[0.5, 0.0], [0.1, 0.45]])
        out = ll.call_mat_phi(mu, Phi, L_eta, self.y)
        self.assertEqual(out.shape, ())
        self.assertTrue(tf.math.is_finite(out))

    def test_init_type_variants(self):
        mu = tf.zeros([self.d])
        phi_diag = tf.constant([0.9, 0.85])
        sig_diag_sq = tf.constant([0.25, 0.2])
        for it in ("stationary", "fixed_mu", "diffuse"):
            ll = self._make(init_type=it)
            out = ll(mu, phi_diag, sig_diag_sq, self.y)
            self.assertTrue(tf.math.is_finite(out), msg=it)


if __name__ == "__main__":
    unittest.main()
