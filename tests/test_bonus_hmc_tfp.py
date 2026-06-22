"""
Unit tests for the TensorFlow-Probability HMC wrappers in
``src/filters/bonus/hmc_tfp.py``.

Covers (granularly):
  - _finite_log_prob               (NaN/Inf sanitisation)
  - _with_fixed_crn                (CRN determinism + finitisation)
  - _kernel_step_size / _kernel_is_accepted / _kernel_target_log_prob
    (best-effort field extraction against a real bootstrapped kernel)
  - _trace_fn                      (tuple shape/content)
  - run_hmc_tfp                    (fused sample_chain path AND the verbose
                                    one_step progress-loop path)
  - run_hmc_tfp_multi_chain        (stacking + shared/independent CRN)
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.hmc_tfp import (
    _finite_log_prob,
    _with_fixed_crn,
    _kernel_step_size,
    _kernel_is_accepted,
    _kernel_target_log_prob,
    _trace_fn,
    run_hmc_tfp,
    run_hmc_tfp_multi_chain,
)
from src.filters.bonus.hmc_pf import HMCResult, MultiChainHMCResult

tfm = tfp.mcmc


def _gaussian_log_target(q):
    return -0.5 * tf.reduce_sum(q ** 2)


class TestFiniteLogProb(unittest.TestCase):

    def test_passes_finite_values(self):
        v = tf.constant([1.0, -2.5, 3.0])
        out = _finite_log_prob(v)
        np.testing.assert_allclose(out.numpy(), v.numpy())

    def test_replaces_nan(self):
        v = tf.constant([float("nan"), 1.0])
        out = _finite_log_prob(v)
        self.assertAlmostEqual(float(out[0]), -1e6)
        self.assertAlmostEqual(float(out[1]), 1.0)

    def test_replaces_inf(self):
        v = tf.constant([float("inf"), float("-inf")])
        out = _finite_log_prob(v)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))
        self.assertAlmostEqual(float(out[0]), -1e6)
        self.assertAlmostEqual(float(out[1]), -1e6)

    def test_casts_to_float32(self):
        out = _finite_log_prob(tf.constant(1.0, tf.float64))
        self.assertEqual(out.dtype, tf.float32)


class TestWithFixedCRN(unittest.TestCase):

    def test_determinism_across_calls(self):
        """A stochastic target wrapped with a fixed CRN must be deterministic."""
        def stochastic_target(q):
            return tf.reduce_sum(q) + tf.random.normal([])

        wrapped = _with_fixed_crn(stochastic_target, crn_seed=123)
        q = tf.constant([0.5, -0.5])
        a = float(wrapped(q))
        b = float(wrapped(q))
        self.assertAlmostEqual(a, b, places=5)

    def test_none_crn_is_passthrough_finite(self):
        wrapped = _with_fixed_crn(_gaussian_log_target, crn_seed=None)
        out = wrapped(tf.constant([1.0, 1.0]))
        self.assertTrue(tf.math.is_finite(out))
        self.assertEqual(out.shape, ())

    def test_finitises_target(self):
        wrapped = _with_fixed_crn(lambda q: tf.constant(float("nan")), crn_seed=7)
        out = wrapped(tf.constant([0.0]))
        self.assertAlmostEqual(float(out), -1e6)


class TestKernelFieldExtractors(unittest.TestCase):
    """Drive the best-effort extractors with a real (bootstrapped) kernel."""

    def setUp(self):
        tf.random.set_seed(0)
        inner = tfm.HamiltonianMonteCarlo(
            target_log_prob_fn=_gaussian_log_target,
            step_size=0.1,
            num_leapfrog_steps=3,
        )
        self.kernel = tfm.DualAveragingStepSizeAdaptation(
            inner_kernel=inner, num_adaptation_steps=5, target_accept_prob=0.65,
        )
        self.state = tf.constant([0.5, -0.5])
        self.kr = self.kernel.bootstrap_results(self.state)
        self.new_state, self.new_kr = self.kernel.one_step(
            self.state, self.kr, seed=1
        )

    def test_step_size_finite_positive(self):
        eps = _kernel_step_size(self.new_kr)
        self.assertEqual(eps.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(eps))
        self.assertGreater(float(eps), 0.0)

    def test_is_accepted_bool(self):
        acc = _kernel_is_accepted(self.new_kr)
        self.assertEqual(acc.dtype, tf.bool)

    def test_target_log_prob_finite(self):
        lp = _kernel_target_log_prob(self.new_kr)
        self.assertEqual(lp.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(lp))

    def test_trace_fn_tuple(self):
        acc, lp, eps = _trace_fn(self.new_state, self.new_kr)
        self.assertEqual(acc.dtype, tf.bool)
        self.assertEqual(lp.dtype, tf.float32)
        self.assertEqual(eps.dtype, tf.float32)


class TestRunHMCTFPFusedPath(unittest.TestCase):
    """verbose=False  =>  uses the fused tfm.sample_chain code path."""

    def setUp(self):
        tf.random.set_seed(42)

    def test_result_type_and_shapes(self):
        result = run_hmc_tfp(
            _gaussian_log_target,
            initial_state=tf.constant([2.0, -1.0]),
            num_results=8,
            num_burnin=4,
            step_size=0.2,
            num_leapfrog_steps=4,
            seed=42,
            verbose=False,
        )
        self.assertIsInstance(result, HMCResult)
        self.assertEqual(result.samples.shape, (8, 2))
        self.assertEqual(result.is_accepted.shape, (8,))
        self.assertEqual(result.target_log_probs.shape, (8,))
        self.assertEqual(result.step_sizes.shape, (8,))

    def test_accept_rate_in_unit_interval(self):
        result = run_hmc_tfp(
            _gaussian_log_target,
            initial_state=tf.constant([0.0]),
            num_results=6,
            num_burnin=3,
            step_size=0.1,
            num_leapfrog_steps=3,
            seed=1,
            verbose=False,
        )
        self.assertTrue(0.0 <= float(result.accept_rate) <= 1.0)

    def test_samples_finite(self):
        result = run_hmc_tfp(
            _gaussian_log_target,
            initial_state=tf.constant([1.0, 1.0]),
            num_results=10,
            num_burnin=5,
            step_size=0.15,
            num_leapfrog_steps=4,
            seed=7,
            verbose=False,
        )
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.step_sizes)))

    def test_no_adaptation_keeps_step_size_constant(self):
        result = run_hmc_tfp(
            _gaussian_log_target,
            initial_state=tf.constant([0.0]),
            num_results=5,
            num_burnin=4,
            step_size=0.123,
            num_leapfrog_steps=3,
            seed=3,
            adapt_step_size=False,
            verbose=False,
        )
        np.testing.assert_allclose(
            result.step_sizes.numpy(), 0.123, atol=1e-5
        )

    def test_dtypes(self):
        result = run_hmc_tfp(
            _gaussian_log_target,
            initial_state=tf.constant([0.0]),
            num_results=4,
            num_burnin=2,
            verbose=False,
        )
        self.assertEqual(result.samples.dtype, tf.float32)
        self.assertEqual(result.is_accepted.dtype, tf.bool)
        self.assertEqual(result.target_log_probs.dtype, tf.float32)


class TestRunHMCTFPStepLoopPath(unittest.TestCase):
    """verbose=True + progress_every>0  =>  uses the explicit one_step loop."""

    def setUp(self):
        tf.random.set_seed(42)

    def test_step_loop_shapes(self):
        result = run_hmc_tfp(
            _gaussian_log_target,
            initial_state=tf.constant([1.0, -1.0]),
            num_results=6,
            num_burnin=4,
            step_size=0.2,
            num_leapfrog_steps=3,
            seed=42,
            verbose=True,
            progress_every=2,
        )
        self.assertIsInstance(result, HMCResult)
        # Post-burn-in samples only -> num_results rows.
        self.assertEqual(result.samples.shape, (6, 2))
        # The step loop records traces for EVERY step (burn + sample).
        self.assertEqual(result.is_accepted.shape, (10,))
        self.assertEqual(result.target_log_probs.shape, (10,))
        self.assertEqual(result.step_sizes.shape, (10,))

    def test_step_loop_accept_rate_unit_interval(self):
        result = run_hmc_tfp(
            _gaussian_log_target,
            initial_state=tf.constant([0.0]),
            num_results=5,
            num_burnin=2,
            step_size=0.15,
            num_leapfrog_steps=3,
            seed=11,
            verbose=True,
            progress_every=1,
        )
        self.assertTrue(0.0 <= float(result.accept_rate) <= 1.0)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))

    def test_progress_every_zero_falls_back_to_fused(self):
        """verbose=True but progress_every=0 -> fused path (num_results traces)."""
        result = run_hmc_tfp(
            _gaussian_log_target,
            initial_state=tf.constant([0.0]),
            num_results=5,
            num_burnin=3,
            step_size=0.1,
            num_leapfrog_steps=3,
            seed=5,
            verbose=True,
            progress_every=0,
        )
        self.assertEqual(result.is_accepted.shape, (5,))


class TestRunHMCTFPMultiChain(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_multi_chain_shapes(self):
        inits = tf.constant([[2.0, -1.0], [-2.0, 1.0], [0.0, 0.0]])
        result = run_hmc_tfp_multi_chain(
            _gaussian_log_target,
            initial_states=inits,
            num_results=5,
            num_burnin=3,
            step_size=0.2,
            num_leapfrog_steps=3,
            seed=42,
            verbose=False,
        )
        self.assertIsInstance(result, MultiChainHMCResult)
        self.assertEqual(result.samples.shape, (3, 5, 2))
        self.assertEqual(result.is_accepted.shape, (3, 5))
        self.assertEqual(result.accept_rate.shape, (3,))
        self.assertEqual(result.target_log_probs.shape, (3, 5))
        self.assertEqual(result.step_sizes.shape, (3, 5))

    def test_multi_chain_accept_rates_unit_interval(self):
        inits = tf.constant([[1.0], [-1.0]])
        result = run_hmc_tfp_multi_chain(
            _gaussian_log_target,
            initial_states=inits,
            num_results=4,
            num_burnin=2,
            step_size=0.1,
            num_leapfrog_steps=3,
            seed=1,
            verbose=False,
        )
        for r in result.accept_rate.numpy():
            self.assertTrue(0.0 <= float(r) <= 1.0)

    def test_shared_vs_independent_crn_both_run(self):
        inits = tf.constant([[0.5], [-0.5]])
        for share in (True, False):
            result = run_hmc_tfp_multi_chain(
                _gaussian_log_target,
                initial_states=inits,
                num_results=3,
                num_burnin=2,
                step_size=0.1,
                num_leapfrog_steps=2,
                seed=9,
                share_crn_across_chains=share,
                verbose=False,
            )
            self.assertEqual(result.samples.shape, (2, 3, 1))
            self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))


if __name__ == "__main__":
    unittest.main()
