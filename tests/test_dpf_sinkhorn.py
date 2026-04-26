"""
Unit tests for DPF Sinkhorn module and differentiable resampling.

Covers:
  - _cost_matrix
  - sinkhorn_potentials
  - entropy_regularized_transport
  - det_resample
  - soft_resample
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.dpf.sinkhorn import (
    _cost_matrix,
    sinkhorn_potentials,
    entropy_regularized_transport,
)
from src.filters.dpf.resampling import det_resample, soft_resample


class TestCostMatrix(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_shape(self):
        x = tf.random.normal([10, 3])
        y = tf.random.normal([10, 3])
        C = _cost_matrix(x, y)
        self.assertEqual(C.shape, (10, 10))

    def test_non_negative(self):
        x = tf.random.normal([8, 2])
        y = tf.random.normal([8, 2])
        C = _cost_matrix(x, y)
        self.assertTrue(tf.reduce_all(C >= 0))

    def test_symmetric_same_input(self):
        x = tf.random.normal([6, 2])
        C = _cost_matrix(x, x)
        np.testing.assert_allclose(C.numpy(), C.numpy().T, atol=1e-5)

    def test_diagonal_zero_same_input(self):
        x = tf.random.normal([5, 2])
        C = _cost_matrix(x, x)
        np.testing.assert_allclose(tf.linalg.diag_part(C).numpy(), 0.0, atol=1e-5)


class TestSinkhornPotentials(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.N = 10
        self.a = tf.fill([self.N], 1.0 / self.N)
        self.b = tf.fill([self.N], 1.0 / self.N)
        self.x = tf.random.normal([self.N, 2])
        self.y = tf.random.normal([self.N, 2])

    def test_output_shapes(self):
        f, g = sinkhorn_potentials(self.a, self.b, self.x, self.y, n_iters=20)
        self.assertEqual(f.shape, (self.N,))
        self.assertEqual(g.shape, (self.N,))

    def test_finite_outputs(self):
        f, g = sinkhorn_potentials(self.a, self.b, self.x, self.y, n_iters=30)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(f)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(g)))


class TestEntropyRegularizedTransport(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.N = 10
        self.a = tf.fill([self.N], 1.0 / self.N)
        self.b = tf.fill([self.N], 1.0 / self.N)
        self.x = tf.random.normal([self.N, 2])

    def test_output_shape(self):
        P = entropy_regularized_transport(
            self.a, self.b, self.x, self.x, epsilon=0.5, n_iters=30
        )
        self.assertEqual(P.shape, (self.N, self.N))

    def test_row_sums(self):
        P = entropy_regularized_transport(
            self.a, self.b, self.x, self.x, epsilon=0.5, n_iters=50
        )
        row_sums = tf.reduce_sum(P, axis=1).numpy()
        np.testing.assert_allclose(row_sums, self.a.numpy(), atol=0.05)

    def test_col_sums(self):
        P = entropy_regularized_transport(
            self.a, self.b, self.x, self.x, epsilon=0.5, n_iters=50
        )
        col_sums = tf.reduce_sum(P, axis=0).numpy()
        np.testing.assert_allclose(col_sums, self.b.numpy(), atol=0.05)

    def test_gradient_flows(self):
        x = tf.Variable(tf.random.normal([self.N, 2]))
        with tf.GradientTape() as tape:
            P = entropy_regularized_transport(
                self.a, self.b, x, x, epsilon=1.0, n_iters=20
            )
            loss = tf.reduce_sum(P)
        grads = tape.gradient(loss, x)
        self.assertIsNotNone(grads)


class TestDetResample(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_output_shapes(self):
        N, d = 10, 2
        x = tf.random.normal([N, d])
        log_w = tf.random.normal([N])
        x_new, w_new = det_resample(x, log_w, epsilon=1.0, n_iters=20)
        self.assertEqual(x_new.shape, (N, d))
        self.assertEqual(w_new.shape, (N,))

    def test_uniform_weights_output(self):
        N = 8
        x = tf.random.normal([N, 2])
        log_w = tf.zeros([N])
        _, w_new = det_resample(x, log_w, epsilon=1.0, n_iters=20)
        np.testing.assert_allclose(w_new.numpy(), 1.0 / N, atol=1e-5)

    def test_high_weight_concentration(self):
        N = 10
        x = tf.random.normal([N, 1])
        log_w = tf.constant([-100.0] * (N - 1) + [0.0])
        x_new, _ = det_resample(x, log_w, epsilon=0.1, n_iters=50)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(x_new)))


class TestSoftResample(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_output_shapes(self):
        N, d = 12, 3
        x = tf.random.normal([N, d])
        log_w = tf.random.normal([N])
        x_r, log_w_r = soft_resample(x, log_w, alpha=0.5)
        self.assertEqual(x_r.shape, (N, d))
        self.assertEqual(log_w_r.shape, (N,))

    def test_finite_outputs(self):
        N = 10
        x = tf.random.normal([N, 2])
        log_w = tf.random.normal([N])
        x_r, log_w_r = soft_resample(x, log_w, alpha=0.5)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(x_r)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_w_r)))

    def test_alpha_one_uniform(self):
        """alpha=1 → pure uniform sampling, weights should approximately cancel."""
        N = 20
        x = tf.random.normal([N, 2])
        log_w = tf.zeros([N])
        x_r, log_w_r = soft_resample(x, log_w, alpha=1.0)
        self.assertEqual(x_r.shape, (N, 2))


if __name__ == "__main__":
    unittest.main()
