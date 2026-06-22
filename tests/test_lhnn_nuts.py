"""
Unit tests for ``src/filters/bonus/lhnn_nuts.py`` — L-HNN-accelerated NUTS
with online error monitoring + real-target MH correction.

Covers (granularly):
  Dataclasses:
    - NUTSResult / MultiChainNUTSResult (construction + field access)
  Leapfrog + tree primitives:
    - _real_target_value           (finite + non-finite floor)
    - _real_hamiltonian            (= -logpi + 0.5||p||^2)
    - _no_u_turn                   (straight / turned trajectories)
    - _leapfrog_step_lhnn          (shape + reversibility-ish)
    - _leapfrog_step_real          (exact-gradient leapfrog on a Gaussian)
    - _SamplerState                (counter init)
  Samplers:
    - run_lhnn_nuts                (tiny run; result shapes + invariants)
    - run_lhnn_nuts_multi_chain    (stacking)
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.lhnn_nuts import (
    NUTSResult,
    MultiChainNUTSResult,
    _real_target_value,
    _real_hamiltonian,
    _no_u_turn,
    _leapfrog_step_lhnn,
    _leapfrog_step_real,
    _SamplerState,
    run_lhnn_nuts,
    run_lhnn_nuts_multi_chain,
)
from src.filters.bonus.lhnn_hmc_pf import LatentHNN


def _gaussian_log_target(q):
    return -0.5 * tf.reduce_sum(q ** 2)


def _make_lhnn(d=1):
    tf.random.set_seed(0)
    lhnn = LatentHNN(d=d, hidden_units=8, num_hidden=2)
    _ = lhnn(tf.zeros([1, 2 * d]))
    return lhnn


class TestDataclasses(unittest.TestCase):

    def test_nuts_result_fields(self):
        r = NUTSResult(
            samples=tf.zeros([3, 2]),
            is_accepted=tf.ones([3], tf.bool),
            accept_rate=tf.constant(1.0),
            target_log_probs=tf.zeros([3]),
            step_sizes=tf.ones([5]),
            avg_tree_depth=tf.constant(2.0),
            tree_depths=tf.ones([5]),
            real_grad_evals_per_iter=tf.zeros([5]),
            total_real_grad_evals=0,
            total_lhnn_leapfrog_evals=10,
            total_error_triggers=0,
        )
        self.assertEqual(r.samples.shape, (3, 2))
        self.assertEqual(r.total_lhnn_leapfrog_evals, 10)

    def test_multichain_result_fields(self):
        r = MultiChainNUTSResult(
            samples=tf.zeros([2, 3, 1]),
            is_accepted=tf.ones([2, 3], tf.bool),
            accept_rate=tf.ones([2]),
            target_log_probs=tf.zeros([2, 3]),
            step_sizes=tf.ones([2, 5]),
            tree_depths=tf.ones([2, 5]),
            per_chain_diagnostics=[],
        )
        self.assertEqual(r.samples.shape, (2, 3, 1))
        self.assertEqual(r.per_chain_diagnostics, [])


class TestRealTargetHelpers(unittest.TestCase):

    def test_real_target_value_matches(self):
        q = tf.constant([1.0, 2.0])
        val = _real_target_value(_gaussian_log_target, q, crn_seed=0)
        self.assertAlmostEqual(val, -0.5 * 5.0, places=4)

    def test_real_target_value_floors_nonfinite(self):
        val = _real_target_value(lambda q: tf.constant(float("nan")),
                                 tf.constant([0.0]), crn_seed=0)
        self.assertEqual(val, -1e9)

    def test_real_hamiltonian(self):
        q = tf.constant([1.0])      # -logpi = 0.5
        p = tf.constant([2.0])      # K = 0.5*4 = 2.0
        H = _real_hamiltonian(_gaussian_log_target, q, p, crn_seed=0)
        self.assertAlmostEqual(H, 0.5 + 2.0, places=4)


class TestNoUTurn(unittest.TestCase):

    def test_straight_trajectory_continues(self):
        q_minus = tf.constant([0.0])
        q_plus = tf.constant([1.0])
        p = tf.constant([1.0])
        self.assertTrue(_no_u_turn(q_minus, q_plus, p, p))

    def test_turned_trajectory_stops(self):
        q_minus = tf.constant([0.0])
        q_plus = tf.constant([1.0])
        p_minus = tf.constant([-1.0])  # momentum opposes the displacement
        p_plus = tf.constant([-1.0])
        self.assertFalse(_no_u_turn(q_minus, q_plus, p_minus, p_plus))


class TestLeapfrogSteps(unittest.TestCase):

    def test_lhnn_leapfrog_shapes(self):
        lhnn = _make_lhnn(d=2)
        q = tf.constant([0.5, -0.5])
        p = tf.constant([1.0, 0.0])
        q_new, p_new = _leapfrog_step_lhnn(lhnn, q, p, eps=0.05)
        self.assertEqual(q_new.shape, (2,))
        self.assertEqual(p_new.shape, (2,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(q_new)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(p_new)))

    def test_real_leapfrog_on_gaussian(self):
        """For -logpi=0.5||q||^2, grad=-q; a small step moves q by ~eps*p."""
        q = tf.constant([0.0])
        p = tf.constant([1.0])
        eps = 0.01
        q_new, p_new = _leapfrog_step_real(
            _gaussian_log_target, q, p, eps=eps, crn_seed=0
        )
        # q_new = q + eps*p + (eps^2/2)*grad(q),  grad(0) = 0 -> q_new ~ eps
        self.assertAlmostEqual(float(q_new[0]), eps, places=4)
        self.assertTrue(tf.math.is_finite(p_new[0]))


class TestSamplerState(unittest.TestCase):

    def test_init(self):
        rng = np.random.default_rng(0)
        st = _SamplerState(fallback_remaining=3, cooldown_steps=10, rng=rng)
        self.assertEqual(st.fallback_remaining, 3)
        self.assertEqual(st.cooldown_steps, 10)
        self.assertEqual(st.real_grad_count, 0)
        self.assertEqual(st.lhnn_grad_count, 0)
        self.assertFalse(st.error_triggered_this_iter)


class TestRunLHNNNUTS(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.lhnn = _make_lhnn(d=1)

    def test_result_shapes(self):
        result = run_lhnn_nuts(
            _gaussian_log_target,
            initial_state=tf.constant([0.5]),
            lhnn=self.lhnn,
            num_results=5,
            num_burnin=3,
            step_size=0.05,
            max_treedepth=4,
            seed=42,
            verbose=False,
        )
        self.assertIsInstance(result, NUTSResult)
        self.assertEqual(result.samples.shape, (5, 1))
        self.assertEqual(result.is_accepted.shape, (5,))
        self.assertEqual(result.target_log_probs.shape, (5,))
        total = 5 + 3
        self.assertEqual(result.step_sizes.shape, (total,))
        self.assertEqual(result.tree_depths.shape, (total,))

    def test_accept_rate_unit_interval_and_finite(self):
        result = run_lhnn_nuts(
            _gaussian_log_target,
            initial_state=tf.constant([0.0]),
            lhnn=self.lhnn,
            num_results=6,
            num_burnin=2,
            step_size=0.05,
            max_treedepth=4,
            seed=1,
            verbose=False,
        )
        self.assertTrue(0.0 <= float(result.accept_rate) <= 1.0)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))
        self.assertGreaterEqual(result.total_lhnn_leapfrog_evals, 0)
        self.assertGreaterEqual(result.total_error_triggers, 0)

    def test_kinetic_mh_runs(self):
        result = run_lhnn_nuts(
            _gaussian_log_target,
            initial_state=tf.constant([0.0]),
            lhnn=self.lhnn,
            num_results=4,
            num_burnin=2,
            step_size=0.05,
            max_treedepth=3,
            seed=2,
            verbose=False,
            kinetic_mh=True,
        )
        self.assertEqual(result.samples.shape, (4, 1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))


class TestRunLHNNNUTSMultiChain(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.lhnn = _make_lhnn(d=1)

    def test_multi_chain_shapes(self):
        inits = tf.constant([[0.5], [-0.5]])
        result = run_lhnn_nuts_multi_chain(
            _gaussian_log_target,
            initial_states=inits,
            lhnn=self.lhnn,
            num_results=4,
            num_burnin=2,
            step_size=0.05,
            max_treedepth=3,
            seed=42,
            verbose=False,
        )
        self.assertIsInstance(result, MultiChainNUTSResult)
        self.assertEqual(result.samples.shape, (2, 4, 1))
        self.assertEqual(result.is_accepted.shape, (2, 4))
        self.assertEqual(result.accept_rate.shape, (2,))
        self.assertEqual(len(result.per_chain_diagnostics), 2)


if __name__ == "__main__":
    unittest.main()
