"""
Unit tests for ``src/filters/bonus/extra_bonus/svssm_neural_ot_training.py``.

Covers (granularly):
  - gen_svssm_observations            (shape, finiteness, determinism)
  - _safe_scalar
  - SVSSMTrainingDataset              (__len__, split_train_val sizes + disjoint)
  - generate_svssm_training_data      (M = grid*seeds*T; array shapes)
  - SVSSMNeuralOTTrainer
      * loss_mode validation
      * supervised training run -> TrainingHistory with monotone-length curves
      * best-weights restoration / checkpoint save
"""

import unittest
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.extra_bonus.svssm_neural_ot_training import (
    gen_svssm_observations,
    _safe_scalar,
    SVSSMTrainingDataset,
    generate_svssm_training_data,
    SVSSMNeuralOTTrainer,
    TrainingHistory,
)


def _tiny_dataset(M=12, N=16):
    rng = np.random.default_rng(0)
    return SVSSMTrainingDataset(
        particles_norm=rng.normal(size=(M, N)).astype(np.float32),
        weights=np.full((M, N), 1.0 / N, np.float32),
        ctx=rng.normal(size=(M, 7)).astype(np.float32),
        targets_norm=rng.normal(size=(M, N)).astype(np.float32),
        theta_idx=np.zeros(M, np.int32),
        t_idx=np.arange(M, dtype=np.int32),
    )


def _tiny_model(N=16):
    tf.random.set_seed(0)
    net = ConditionalMGradNet(
        state_dim=1, n_ridges=4, d_set=8, d_scalar=8, n_scalar_ctx=7
    )
    _ = net(tf.zeros([2, N]), tf.fill([2, N], 1.0 / N), tf.zeros([2, 7]))
    return net


class TestGenObservations(unittest.TestCase):

    def test_shape_and_finite(self):
        y = gen_svssm_observations(T=8, mu=0.0, phi=0.9, sigma_eta=0.5, seed=1)
        self.assertEqual(y.shape, (8,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y)))

    def test_deterministic_for_seed(self):
        a = gen_svssm_observations(5, 0.0, 0.9, 0.5, seed=42)
        b = gen_svssm_observations(5, 0.0, 0.9, 0.5, seed=42)
        np.testing.assert_allclose(a.numpy(), b.numpy())


class TestSafeScalar(unittest.TestCase):

    def test_nonfinite_to_zero_and_clamp(self):
        out = _safe_scalar(tf.constant([float("nan"), 1e9, -1e9, 1.0]))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertLessEqual(float(out[1]), 1e4 + 1)


class TestDataset(unittest.TestCase):

    def test_len(self):
        ds = _tiny_dataset(M=10)
        self.assertEqual(len(ds), 10)

    def test_split_sizes(self):
        ds = _tiny_dataset(M=20)
        train, val = ds.split_train_val(val_frac=0.25, seed=0)
        self.assertEqual(len(val), 5)
        self.assertEqual(len(train), 15)

    def test_split_disjoint_and_complete(self):
        ds = _tiny_dataset(M=20)
        train, val = ds.split_train_val(val_frac=0.25, seed=1)
        train_ts = set(train.t_idx.tolist())
        val_ts = set(val.t_idx.tolist())
        self.assertEqual(train_ts & val_ts, set())
        self.assertEqual(train_ts | val_ts, set(range(20)))

    def test_split_preserves_widths(self):
        ds = _tiny_dataset(M=16, N=16)
        train, _ = ds.split_train_val(val_frac=0.5)
        self.assertEqual(train.particles_norm.shape[1], 16)
        self.assertEqual(train.ctx.shape[1], 7)


class TestGenerateTrainingData(unittest.TestCase):

    def test_shapes_and_count(self):
        grid = [(0.0, 0.9, 0.5), (0.1, 0.85, 0.4)]
        T, N = 4, 16
        ds = generate_svssm_training_data(
            grid, T=T, N=N, n_lambda=3, sinkhorn_epsilon=1.0,
            sinkhorn_iters=5, seeds_per_theta=1, base_seed=42, verbose=False,
        )
        M = len(grid) * 1 * T
        self.assertEqual(len(ds), M)
        self.assertEqual(ds.particles_norm.shape, (M, N))
        self.assertEqual(ds.weights.shape, (M, N))
        self.assertEqual(ds.ctx.shape, (M, 7))
        self.assertEqual(ds.targets_norm.shape, (M, N))
        self.assertEqual(ds.theta_idx.shape, (M,))
        self.assertTrue(np.all(np.isfinite(ds.particles_norm)))
        self.assertTrue(np.all(np.isfinite(ds.targets_norm)))

    def test_seeds_per_theta_multiplies(self):
        grid = [(0.0, 0.9, 0.5)]
        T = 3
        ds = generate_svssm_training_data(
            grid, T=T, N=12, n_lambda=2, seeds_per_theta=2,
            sinkhorn_iters=5, verbose=False,
        )
        self.assertEqual(len(ds), 1 * 2 * T)


class TestTrainer(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.model = _tiny_model(N=16)

    def test_invalid_loss_mode_raises(self):
        with self.assertRaises(ValueError):
            SVSSMNeuralOTTrainer(self.model, loss_mode="bogus")

    def test_construction_defaults(self):
        tr = SVSSMNeuralOTTrainer(self.model, learning_rate=1e-3, batch_size=8)
        self.assertEqual(tr.batch_size, 8)
        self.assertEqual(tr.loss_mode, "supervised")

    def test_supervised_training_returns_history(self):
        ds = _tiny_dataset(M=24, N=16)
        train, val = ds.split_train_val(val_frac=0.25, seed=0)
        tr = SVSSMNeuralOTTrainer(
            self.model, learning_rate=1e-3, batch_size=8, max_epochs=3,
            patience=5, loss_mode="supervised",
        )
        hist = tr.train(train, val, verbose=False)
        self.assertIsInstance(hist, TrainingHistory)
        self.assertGreaterEqual(len(hist.train_loss), 1)
        self.assertEqual(len(hist.train_loss), len(hist.val_loss))
        self.assertTrue(np.isfinite(hist.best_val_loss))
        self.assertGreaterEqual(hist.best_epoch, 0)
        self.assertEqual(hist.loss_mode, "supervised")

    def test_checkpoint_saved(self):
        ds = _tiny_dataset(M=16, N=16)
        train, val = ds.split_train_val(val_frac=0.25, seed=0)
        tr = SVSSMNeuralOTTrainer(
            self.model, batch_size=8, max_epochs=2, patience=5,
        )
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "model.weights.h5"
            tr.train(train, val, checkpoint_path=ckpt, verbose=False)
            # Keras may append a suffix; assert the directory got a file.
            self.assertTrue(any(Path(d).iterdir()))


if __name__ == "__main__":
    unittest.main()
