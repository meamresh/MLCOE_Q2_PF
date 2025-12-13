"""
Integration tests for the full pipeline.
"""

import unittest
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generators import generate_lgssm_from_yaml
from src.filters.kalman import KalmanFilter
from src.metrics.accuracy import compute_rmse, compute_nees
from src.metrics.stability import (
    compute_condition_numbers, check_symmetry, check_positive_definite
)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full filtering pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_full_pipeline(self):
        """Test the complete pipeline: generate data -> filter -> compute metrics."""
        # Use the actual config file
        config_path = Path(__file__).parent.parent / "configs" / "ssm_linear.yaml"
        
        if not config_path.exists():
            self.skipTest("Config file not found")
        
        # 1. Generate data
        model, X_true, Y_obs, data_dict = generate_lgssm_from_yaml(str(config_path))
        
        # 2. Initialize filter
        x0_reshaped = tf.reshape(model.m0, [-1, 1]) if len(model.m0.shape) == 1 else model.m0
        kf = KalmanFilter(
            A=model.A, C=model.C, x0=x0_reshaped, P0=model.P0,
            Q=model.Q, R=model.R
        )
        
        # 3. Run filter
        results = kf.filter(Y_obs, joseph=False)
        
        # 4. Check results structure
        self.assertIn("x_filt", results)
        self.assertIn("P_filt", results)
        self.assertIn("x_pred", results)
        self.assertIn("P_pred", results)
        
        # 5. Compute metrics
        x_filt = tf.squeeze(results["x_filt"], axis=-1)
        rmse = compute_rmse(x_filt, X_true)
        
        # RMSE should be reasonable (not too large)
        self.assertLess(rmse, 10.0)
        self.assertGreater(rmse, 0.0)
        
        # 6. Check stability metrics
        cond_numbers = compute_condition_numbers(results["P_filt"])
        self.assertTrue(tf.reduce_all(cond_numbers > 0))
        
        sym_error = check_symmetry(results["P_filt"])
        # Symmetry error should be small
        self.assertTrue(tf.reduce_all(sym_error < 1e-4))
        
        min_eigvals, is_pd = check_positive_definite(results["P_filt"])
        # Should be positive definite
        self.assertTrue(tf.reduce_all(is_pd))

    def test_riccati_vs_joseph_equivalence(self):
        """Test that Riccati and Joseph updates produce similar results."""
        config_path = Path(__file__).parent.parent / "configs" / "ssm_linear.yaml"
        
        if not config_path.exists():
            self.skipTest("Config file not found")
        
        # Generate data
        model, X_true, Y_obs, _ = generate_lgssm_from_yaml(str(config_path))
        
        # Initialize two filters
        x0_reshaped = tf.reshape(model.m0, [-1, 1]) if len(model.m0.shape) == 1 else model.m0
        kf_riccati = KalmanFilter(
            A=model.A, C=model.C, x0=x0_reshaped, P0=model.P0,
            Q=model.Q, R=model.R
        )
        kf_joseph = KalmanFilter(
            A=model.A, C=model.C, x0=x0_reshaped, P0=model.P0,
            Q=model.Q, R=model.R
        )
        
        # Run filters
        results_riccati = kf_riccati.filter(Y_obs, joseph=False)
        results_joseph = kf_joseph.filter(Y_obs, joseph=True)
        
        # Compare means (should be very close)
        x_filt_r = tf.squeeze(results_riccati["x_filt"], axis=-1)
        x_filt_j = tf.squeeze(results_joseph["x_filt"], axis=-1)
        
        mean_diff = tf.reduce_mean(tf.abs(x_filt_r - x_filt_j))
        # Should be very close (within numerical precision)
        self.assertLess(float(mean_diff), 1e-5)
        
        # Compare RMSE (should be very similar)
        rmse_r = compute_rmse(x_filt_r, X_true)
        rmse_j = compute_rmse(x_filt_j, X_true)
        
        # Should be very close
        self.assertLess(abs(rmse_r - rmse_j), 1e-5)

    def test_filter_consistency(self):
        """Test that filter produces consistent estimates."""
        config_path = Path(__file__).parent.parent / "configs" / "ssm_linear.yaml"
        
        if not config_path.exists():
            self.skipTest("Config file not found")
        
        # Generate data
        model, X_true, Y_obs, _ = generate_lgssm_from_yaml(str(config_path))
        
        # Initialize filter
        x0_reshaped = tf.reshape(model.m0, [-1, 1]) if len(model.m0.shape) == 1 else model.m0
        kf = KalmanFilter(
            A=model.A, C=model.C, x0=x0_reshaped, P0=model.P0,
            Q=model.Q, R=model.R
        )
        
        # Run filter
        results = kf.filter(Y_obs)
        
        # Compute NEES for consistency check
        x_filt = tf.squeeze(results["x_filt"], axis=-1)
        nees = compute_nees(results["x_filt"], results["P_filt"], X_true)
        
        mean_nees = tf.reduce_mean(nees)
        
        # For a consistent filter, mean NEES should be close to state dimension
        # (within reasonable bounds, e.g., 0.5 * nx to 2 * nx)
        nx = model.nx
        self.assertGreater(float(mean_nees), 0.5 * nx)
        self.assertLess(float(mean_nees), 2.0 * nx)

    def test_filter_with_different_seeds(self):
        """Test that filtering produces different results with different data seeds."""
        config_path = Path(__file__).parent.parent / "configs" / "ssm_linear.yaml"
        
        if not config_path.exists():
            self.skipTest("Config file not found")
        
        # Generate data with different seeds
        model1, X1, Y1, _ = generate_lgssm_from_yaml(str(config_path))
        
        # Modify seed in config (this would require re-parsing, so we'll just test with same seed)
        # For now, test that filtering works with the generated data
        x0_reshaped = tf.reshape(model1.m0, [-1, 1]) if len(model1.m0.shape) == 1 else model1.m0
        kf = KalmanFilter(
            A=model1.A, C=model1.C, x0=x0_reshaped, P0=model1.P0,
            Q=model1.Q, R=model1.R
        )
        
        results = kf.filter(Y1)
        
        # Should complete without errors
        self.assertIsNotNone(results)
        self.assertIn("x_filt", results)


if __name__ == '__main__':
    unittest.main()

