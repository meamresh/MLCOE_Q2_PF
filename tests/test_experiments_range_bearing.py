"""
Tests for range-bearing EKF/UKF experiments.
"""

from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path

import tensorflow as tf
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.exp_range_bearing_ekf_ukf import (  # noqa: E402
    run_experiment,
    run_parameter_tuning,
)


class TestRangeBearingExperiments(unittest.TestCase):
    """Integration tests for range-bearing EKF/UKF experiments."""

    def setUp(self) -> None:
        """Create a temporary output directory."""
        self.tmp_dir = Path("reports/test_range_bearing_tmp")
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up temporary output directory."""
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_run_experiment_creates_outputs(self) -> None:
        """run_experiment should create plot and metrics files."""
        out_dir = self.tmp_dir / "experiments"

        results = run_experiment(
            scenario="moderate",
            num_steps=50,
            alpha=0.1,
            beta=1.0,
            kappa=0.0,
            output_dir=out_dir,
            seed=123,
        )

        # Basic sanity checks on results
        self.assertIn("true_states", results)
        self.assertIn("ekf_states", results)
        self.assertIn("ukf_states", results)

        # Files should exist
        scenario = results["scenario"]
        png_path = out_dir / f"ekf_ukf_comparison_{scenario}.png"
        csv_path = out_dir / f"ekf_ukf_metrics_{scenario}.csv"

        self.assertTrue(png_path.exists(), f"Missing {png_path}")
        self.assertTrue(csv_path.exists(), f"Missing {csv_path}")

    def test_run_parameter_tuning_creates_outputs(self) -> None:
        """run_parameter_tuning should perform grid search and save results."""
        out_dir = self.tmp_dir / "tuning"

        grid_results = run_parameter_tuning(
            scenario="strong",
            num_steps=30,
            alpha_values=[0.01, 0.1],
            beta_values=[1.0],
            kappa_values=[0.0],
            output_dir=out_dir,
        )

        # Check grid search results structure
        self.assertIn("best_params", grid_results)
        self.assertIn("best_rmse", grid_results)
        self.assertIn("all_results", grid_results)

        best_rmse = grid_results["best_rmse"]
        self.assertTrue(tf.math.is_finite(best_rmse))

        # Files should exist
        scenario = grid_results["scenario"]
        png_path = out_dir / f"grid_search_{scenario}.png"
        csv_path = out_dir / f"grid_search_results_{scenario}.csv"

        self.assertTrue(png_path.exists(), f"Missing {png_path}")
        self.assertTrue(csv_path.exists(), f"Missing {csv_path}")


if __name__ == "__main__":
    unittest.main()


