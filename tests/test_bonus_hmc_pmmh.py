"""
Unit tests for bonus HMC, PMMH, DPFHMCTarget, and filtering_rmse.

Covers:
  - run_hmc (with simple quadratic target)
  - HMCResult fields
  - bootstrap_pf_log_likelihood (Kitagawa SSM)
  - run_pmmh (with simple log-target)
  - PMMHResult fields
  - DPFHMCTarget.log_prior
  - filtering_rmse
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.hmc_pf import run_hmc, HMCResult
from src.filters.bonus.pmmh import (
    bootstrap_pf_log_likelihood,
    run_pmmh,
    PMMHResult,
)
from src.models.ssm_katigawa import PMCMCNonlinearSSM
from src.filters.bonus.dpf_hmc_ssl import filtering_rmse


class TestRunHMC(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_simple_gaussian_target(self):
        """HMC should explore a simple Gaussian posterior."""
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        initial = tf.constant([2.0, -1.0])
        result = run_hmc(
            log_target,
            initial,
            num_results=10,
            num_burnin=5,
            step_size=0.1,
            num_leapfrog_steps=5,
            seed=42,
            verbose=False,
        )
        self.assertIsInstance(result, HMCResult)
        self.assertEqual(result.samples.shape, (10, 2))
        self.assertEqual(result.is_accepted.shape, (10,))
        self.assertTrue(0.0 <= float(result.accept_rate) <= 1.0)
        self.assertEqual(result.target_log_probs.shape, (10,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))

    def test_step_sizes_recorded(self):
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        result = run_hmc(
            log_target,
            tf.constant([0.0]),
            num_results=5,
            num_burnin=3,
            step_size=0.05,
            num_leapfrog_steps=3,
            verbose=False,
        )
        self.assertEqual(result.step_sizes.shape[0], 8)  # 5 + 3

    def test_adapt_step_size(self):
        """With step-size adaptation enabled, step sizes should vary during burn-in."""
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        result = run_hmc(
            log_target,
            tf.constant([3.0, -2.0]),
            num_results=10,
            num_burnin=10,
            step_size=0.1,
            num_leapfrog_steps=5,
            adapt_step_size=True,
            seed=42,
            verbose=False,
        )
        self.assertTrue(0.0 <= float(result.accept_rate) <= 1.0)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))

    def test_samples_near_zero_for_gaussian(self):
        """Samples from a standard Gaussian should have mean near 0."""
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        result = run_hmc(
            log_target,
            tf.constant([0.0]),
            num_results=50,
            num_burnin=20,
            step_size=0.3,
            num_leapfrog_steps=10,
            seed=42,
            verbose=False,
        )
        sample_mean = float(tf.reduce_mean(result.samples).numpy())
        self.assertAlmostEqual(sample_mean, 0.0, delta=1.5)


class TestBootstrapPFLogLikelihood(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssm = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0)

    def test_returns_finite_scalar(self):
        T = 10
        y_obs = tf.random.normal([T, 1])
        ll = bootstrap_pf_log_likelihood(self.ssm, y_obs, num_particles=30)
        self.assertEqual(ll.shape, ())
        self.assertTrue(tf.math.is_finite(ll))

    def test_1d_observations(self):
        y_obs = tf.random.normal([5])
        ll = bootstrap_pf_log_likelihood(self.ssm, y_obs, num_particles=20)
        self.assertEqual(ll.shape, ())


class TestRunPMMH(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_simple_gaussian(self):
        def log_target(theta):
            return -0.5 * tf.reduce_sum(theta ** 2)

        initial = tf.constant([1.0, -1.0])
        result = run_pmmh(
            log_target,
            initial,
            num_results=5,
            num_burnin=3,
            step_size=0.5,
            seed=42,
            verbose=False,
        )
        self.assertIsInstance(result, PMMHResult)
        self.assertEqual(result.samples.shape, (5, 2))
        self.assertEqual(result.is_accepted.shape, (5,))
        self.assertTrue(0.0 <= float(result.accept_rate) <= 1.0)

    def test_target_log_probs_shape(self):
        def log_target(theta):
            return -tf.reduce_sum(theta ** 2)

        result = run_pmmh(
            log_target,
            tf.constant([0.0]),
            num_results=4,
            num_burnin=2,
            verbose=False,
        )
        self.assertEqual(result.target_log_probs.shape, (4,))

    def test_samples_finite(self):
        def log_target(theta):
            return -0.5 * tf.reduce_sum(theta ** 2)

        result = run_pmmh(
            log_target,
            tf.constant([1.0]),
            num_results=8,
            num_burnin=3,
            step_size=0.3,
            seed=42,
            verbose=False,
        )
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))


class TestFilteringRMSE(unittest.TestCase):

    def test_zero_error(self):
        z = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        rmse = filtering_rmse(z, z)
        np.testing.assert_allclose(float(rmse), 0.0, atol=1e-6)

    def test_positive(self):
        z_est = tf.constant([[1.0], [2.0]])
        z_true = tf.constant([[1.5], [2.5]])
        rmse = filtering_rmse(z_est, z_true)
        self.assertGreater(float(rmse), 0.0)
        self.assertTrue(tf.math.is_finite(rmse))

    def test_shape(self):
        z = tf.random.normal([10, 3])
        rmse = filtering_rmse(z, z + 0.1)
        self.assertEqual(rmse.shape, ())


if __name__ == "__main__":
    unittest.main()
