"""
Unit tests for experiment utilities.

Covers:
  - FilterResult dataclass
  - generate_generic_trajectory
  - run_filter_on_trajectory
  - aggregate_results
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.experiment import (
    FilterResult,
    run_filter_on_trajectory,
    generate_generic_trajectory,
    aggregate_results,
)
from src.models.ssm_katigawa import PMCMCNonlinearSSM


class TestFilterResult(unittest.TestCase):

    def test_creation(self):
        r = FilterResult(
            filter_name="TestFilter",
            estimates=tf.zeros([5, 1]),
            rmse=0.5,
            execution_time=0.1,
        )
        self.assertEqual(r.filter_name, "TestFilter")
        self.assertAlmostEqual(r.rmse, 0.5)
        self.assertTrue(r.success)
        self.assertEqual(r.error_message, "")

    def test_with_failure(self):
        r = FilterResult(
            filter_name="Bad",
            estimates=tf.zeros([5, 1]),
            rmse=float('inf'),
            execution_time=0.01,
            success=False,
            error_message="diverged",
        )
        self.assertFalse(r.success)
        self.assertEqual(r.error_message, "diverged")

    def test_extra_dict(self):
        r = FilterResult(
            filter_name="X",
            estimates=tf.zeros([1, 1]),
            rmse=0.1,
            execution_time=0.0,
            extra={"ess": 50.0},
        )
        self.assertEqual(r.extra["ess"], 50.0)


class TestGenerateGenericTrajectory(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssm = PMCMCNonlinearSSM()

    def test_output_shapes(self):
        states, measurements, controls, obs_avail = generate_generic_trajectory(
            self.ssm, n_steps=10, seed=42,
        )
        self.assertEqual(states.shape[0], 10)
        self.assertEqual(measurements.shape[0], 10)
        self.assertEqual(controls.shape[0], 10)
        self.assertEqual(obs_avail.shape[0], 10)

    def test_finite_outputs(self):
        states, measurements, _, _ = generate_generic_trajectory(
            self.ssm, n_steps=8, seed=99,
        )
        self.assertTrue(tf.reduce_all(tf.math.is_finite(states)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(measurements)))

    def test_custom_initial_state(self):
        x0 = tf.constant([5.0])
        states, _, _, _ = generate_generic_trajectory(
            self.ssm, n_steps=5, x0=x0, seed=1,
        )
        np.testing.assert_allclose(states[0].numpy(), [5.0], atol=1e-5)


class TestAggregateResults(unittest.TestCase):

    def test_basic_aggregation(self):
        results = [
            FilterResult("A", tf.zeros([1, 1]), rmse=1.0, execution_time=0.1),
            FilterResult("A", tf.zeros([1, 1]), rmse=2.0, execution_time=0.2),
            FilterResult("B", tf.zeros([1, 1]), rmse=0.5, execution_time=0.05),
        ]
        agg = aggregate_results(results)
        self.assertIn("A", agg)
        self.assertIn("B", agg)
        self.assertAlmostEqual(agg["A"]["mean_rmse"], 1.5)
        self.assertEqual(agg["A"]["n_runs"], 2)
        self.assertEqual(agg["B"]["n_runs"], 1)

    def test_success_rate(self):
        results = [
            FilterResult("C", tf.zeros([1, 1]), rmse=1.0, execution_time=0.1, success=True),
            FilterResult("C", tf.zeros([1, 1]), rmse=float('inf'), execution_time=0.1, success=False),
        ]
        agg = aggregate_results(results)
        self.assertAlmostEqual(agg["C"]["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
