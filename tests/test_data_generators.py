"""
Unit tests for data generators.
"""

import unittest
import tensorflow as tf
import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generators import generate_lgssm_from_yaml, _parse_yaml_simple


class TestDataGenerators(unittest.TestCase):
    """Test cases for data generators."""

    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)

    def test_parse_yaml_simple(self):
        """Test simple YAML parser."""
        yaml_content = """
name: test_model
seed: 42
N: 100

dimensions:
  nx: 4
  ny: 2
  nv: 2
  nw: 2

params:
  sigma_a: 0.5
  sigma_z: 5.0
  Sigma0_diag: [10.0, 5.0, 10.0, 5.0]

A:
  - [1, 1, 0, 0]
  - [0, 1, 0, 0]
  - [0, 0, 1, 1]
  - [0, 0, 0, 1]

B_raw:
  - [0.5, 0.0]
  - [1.0, 0.0]
  - [0.0, 0.5]
  - [0.0, 1.0]

C:
  - [1, 0, 0, 0]
  - [0, 0, 1, 0]

D:
  - [1.0, 0.0]
  - [0.0, 1.0]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            cfg = _parse_yaml_simple(temp_path)
            
            # Check parsed values
            self.assertEqual(cfg["name"], "test_model")
            self.assertEqual(cfg["seed"], 42)
            self.assertEqual(cfg["N"], 100)
            self.assertEqual(cfg["dimensions"]["nx"], 4)
            self.assertEqual(cfg["dimensions"]["ny"], 2)
            self.assertEqual(cfg["params"]["sigma_a"], 0.5)
            self.assertEqual(cfg["params"]["sigma_z"], 5.0)
            self.assertEqual(len(cfg["A"]), 4)
            self.assertEqual(len(cfg["C"]), 2)
        finally:
            os.unlink(temp_path)

    def test_generate_lgssm_from_yaml(self):
        """Test LGSSM generation from YAML file."""
        # Use the actual config file
        config_path = Path(__file__).parent.parent / "configs" / "ssm_linear.yaml"
        
        if config_path.exists():
            model, X, Y, data_dict = generate_lgssm_from_yaml(str(config_path))
            
            # Check model
            self.assertIsNotNone(model)
            self.assertEqual(model.nx, 4)
            self.assertEqual(model.ny, 2)
            
            # Check data shapes
            N = tf.shape(X)[0]
            self.assertEqual(tf.shape(X)[1], 4)
            self.assertEqual(tf.shape(Y)[1], 2)
            
            # Check data_dict keys
            self.assertIn("t", data_dict)
            self.assertIn("x_true", data_dict)
            self.assertIn("y_true", data_dict)
            self.assertIn("x_obs", data_dict)
            self.assertIn("y_obs", data_dict)
            
            # Check data_dict values are tensors
            self.assertIsInstance(data_dict["t"], tf.Tensor)
            self.assertIsInstance(data_dict["x_true"], tf.Tensor)


if __name__ == '__main__':
    unittest.main()

