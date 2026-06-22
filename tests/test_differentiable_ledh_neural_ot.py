"""
Unit tests for ``src/filters/bonus/differentiable_ledh_neural_ot.py`` — the
Kitagawa-space differentiable LEDH filter with neural-OT resampling.

Covers (granularly):
  - construction (attributes + geometric pseudo-time schedule)
  - __call__ on a scalar Kitagawa SSM with explicit theta
  - __call__ with inferred theta (theta=None -> from ssm.Q / ssm.R)
  - 1-D vs 2-D observation handling
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.differentiable_ledh_neural_ot import DifferentiableLEDHNeuralOT
from src.models.ssm_katigawa import PMCMCNonlinearSSM


def _tiny_mgradnet(N=16):
    tf.random.set_seed(0)
    net = ConditionalMGradNet(
        state_dim=1, n_ridges=4, d_set=8, d_scalar=8, n_scalar_ctx=6
    )
    x = tf.random.normal([N])
    w = tf.nn.softmax(tf.random.normal([N]))
    ctx = tf.zeros([6])
    _ = net(x, w, ctx)
    return net


class TestConstruction(unittest.TestCase):

    def test_attributes(self):
        model = _tiny_mgradnet()
        ll = DifferentiableLEDHNeuralOT(
            model, num_particles=16, n_lambda=4, jit_compile=False
        )
        self.assertEqual(ll.num_particles, 16)
        self.assertEqual(ll.n_lambda, 4)
        self.assertIs(ll.neural_ot_model, model)

    def test_pseudo_time_sums_to_one(self):
        model = _tiny_mgradnet()
        ll = DifferentiableLEDHNeuralOT(model, num_particles=16, n_lambda=6)
        self.assertEqual(len(ll.epsilons), 6)
        np.testing.assert_allclose(sum(ll.epsilons), 1.0, atol=1e-5)


class TestCall(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(7)
        self.model = _tiny_mgradnet(N=16)
        self.ssm = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0)
        self.ll = DifferentiableLEDHNeuralOT(
            self.model, num_particles=16, n_lambda=3, jit_compile=False
        )

    def test_explicit_theta_finite_scalar(self):
        y = tf.random.normal([5])
        theta = tf.constant([np.log(10.0), np.log(1.0)], tf.float32)
        out = self.ll(self.ssm, y, theta=theta)
        self.assertEqual(out.shape, ())
        self.assertEqual(out.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(out))

    def test_inferred_theta_finite_scalar(self):
        y = tf.random.normal([5])
        out = self.ll(self.ssm, y)  # theta=None -> inferred from Q/R
        self.assertEqual(out.shape, ())
        self.assertTrue(tf.math.is_finite(out))

    def test_2d_observations(self):
        y = tf.random.normal([4, 1])
        out = self.ll(self.ssm, y)
        self.assertEqual(out.shape, ())
        self.assertTrue(tf.math.is_finite(out))


if __name__ == "__main__":
    unittest.main()
