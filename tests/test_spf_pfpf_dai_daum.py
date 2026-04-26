"""
Unit tests for SPF Dai-Daum and PFPF Dai-Daum filters.

Covers:
  - spf_dai_daum: compute_M, cond_number, DaiDaumStochasticParticleFlow
  - pfpf_dai_daum: compute_M, PFPFDaiDaumFilter
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ssm_dai_daum_bearing_only import DaiDaumBearingSSM, DTYPE
from src.filters.spf_dai_daum import (
    DaiDaumStochasticParticleFlow,
    SPFConfig,
)
from src.filters.pfpf_dai_daum import (
    compute_M as pfpf_compute_M,
    PFPFDaiDaumFilter,
    SPFConfig as PFPFSPFConfig,
)


class TestSPFDaiDaumInit(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssm = DaiDaumBearingSSM()
        self.config = SPFConfig(
            n_particles=20,
            lambda_steps=10,
            bvp_mesh_points=5,
            bvp_max_iter=10,
            verbose=False,
        )

    def _make_spf(self, homotopy="linear"):
        return DaiDaumStochasticParticleFlow(
            ssm=self.ssm,
            initial_state=self.ssm.prior_mean,
            initial_covariance=self.ssm.prior_cov,
            config=self.config,
            homotopy_mode=homotopy,
        )

    def test_instantiation(self):
        spf = self._make_spf()
        self.assertIsNotNone(spf)

    def test_predict_does_not_crash(self):
        spf = self._make_spf()
        spf.predict()

    def test_state_finite_after_predict(self):
        """State should remain finite after prediction."""
        spf = self._make_spf()
        state, cov = spf.predict()
        self.assertTrue(tf.reduce_all(tf.math.is_finite(state)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(cov)))

    def test_covariance_psd(self):
        """Initial covariance should be PSD."""
        spf = self._make_spf()
        eigvals = tf.linalg.eigvalsh(spf.covariance)
        self.assertTrue(tf.reduce_all(eigvals >= -1e-6))


class TestPFPFDaiDaumComputeM(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssm = DaiDaumBearingSSM()

    def test_shape(self):
        x_ref = tf.constant([4.0, 4.0], dtype=DTYPE)
        z = self.ssm._h_vec(x_ref)
        beta = tf.constant(0.5, dtype=DTYPE)
        M = pfpf_compute_M(self.ssm, x_ref, beta, z)
        self.assertEqual(M.shape, (2, 2))

    def test_symmetric(self):
        x_ref = tf.constant([4.0, 4.0], dtype=DTYPE)
        z = self.ssm._h_vec(x_ref)
        beta = tf.constant(0.5, dtype=DTYPE)
        M = pfpf_compute_M(self.ssm, x_ref, beta, z)
        np.testing.assert_allclose(M.numpy(), M.numpy().T, atol=1e-8)

    def test_positive_semidefinite(self):
        x_ref = tf.constant([4.0, 4.0], dtype=DTYPE)
        z = self.ssm._h_vec(x_ref)
        beta = tf.constant(0.5, dtype=DTYPE)
        M = pfpf_compute_M(self.ssm, x_ref, beta, z)
        eigvals = tf.linalg.eigvalsh(M)
        self.assertTrue(tf.reduce_all(eigvals >= -1e-6))


class TestPFPFDaiDaumFilter(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssm = DaiDaumBearingSSM()
        self.config = PFPFSPFConfig(
            n_particles=15,
            lambda_steps=5,
            bvp_mesh_points=5,
            bvp_max_iter=5,
            verbose=False,
        )

    def _make_filter(self):
        return PFPFDaiDaumFilter(
            ssm=self.ssm,
            initial_state=self.ssm.prior_mean,
            initial_covariance=self.ssm.prior_cov,
            config=self.config,
        )

    def test_instantiation(self):
        pf = self._make_filter()
        self.assertIsNotNone(pf)

    def test_particles_initialized_finite(self):
        pf = self._make_filter()
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pf.particles)))

    def test_particle_count(self):
        pf = self._make_filter()
        self.assertEqual(pf.particles.shape[0], pf.num_particles)

    def test_particles_have_spread(self):
        """Initialized particles should not be collapsed to a single point."""
        pf = self._make_filter()
        std = tf.math.reduce_std(pf.particles, axis=0)
        self.assertTrue(tf.reduce_all(std > 1e-6))


if __name__ == "__main__":
    unittest.main()
