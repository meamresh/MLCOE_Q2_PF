"""
Unit tests for Neural OT networks and utilities.

Covers:
  - mask_context_columns
  - build_context_scalars
  - _compute_ess
  - ConditionalMGradNet (forward pass)
  - DeepONetMonotoneOT (forward pass)
  - HyperDeepONetMonotoneOT (forward pass)
  - DifferentiableLEDHLogLikelihood (instantiation, log_likelihood)
  - DifferentiableLEDHNeuralOT (instantiation)
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.neural_ot_resampling import (
    mask_context_columns,
    build_context_scalars,
    _compute_ess,
)
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet, DeepSetEncoder
from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.hyper_deeponet_ot import HyperDeepONetMonotoneOT
from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood


class TestMaskContextColumns(unittest.TestCase):

    def test_no_masking(self):
        ctx = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = mask_context_columns(ctx, None)
        np.testing.assert_allclose(out.numpy(), ctx.numpy())

    def test_empty_list(self):
        ctx = tf.constant([[1.0, 2.0]])
        out = mask_context_columns(ctx, [])
        np.testing.assert_allclose(out.numpy(), ctx.numpy())

    def test_mask_single_column(self):
        ctx = tf.constant([[1.0, 2.0, 3.0]])
        out = mask_context_columns(ctx, [1])
        expected = [[1.0, 0.0, 3.0]]
        np.testing.assert_allclose(out.numpy(), expected)

    def test_mask_preserves_shape(self):
        ctx = tf.random.normal([5, 6])
        out = mask_context_columns(ctx, [0, 3])
        self.assertEqual(out.shape, (5, 6))


class TestBuildContextScalars(unittest.TestCase):

    def test_output_shape(self):
        theta = tf.constant([0.5, -0.3])
        ctx = build_context_scalars(theta, t=5.0, y_t=1.0, ess=20.0, epsilon=2.0)
        self.assertEqual(ctx.shape, (6,))

    def test_finite(self):
        theta = tf.constant([1.0, 2.0])
        ctx = build_context_scalars(theta, t=10.0, y_t=0.5, ess=50.0, epsilon=1.0)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ctx)))


class TestComputeESS(unittest.TestCase):

    def test_uniform_weights(self):
        N = 20
        w = tf.fill([N], 1.0 / N)
        ess = _compute_ess(w)
        np.testing.assert_allclose(float(ess), N, atol=0.1)

    def test_degenerate_weights(self):
        w = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0])
        ess = _compute_ess(w)
        np.testing.assert_allclose(float(ess), 1.0, atol=0.01)


class TestConditionalMGradNet(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.net = ConditionalMGradNet(
            state_dim=1, n_ridges=4, d_set=8, d_scalar=8, n_scalar_ctx=6,
        )

    def test_forward_shape(self):
        N = 10
        x = tf.random.normal([N, 1])
        w = tf.nn.softmax(tf.random.normal([N]))
        ctx = tf.random.normal([6])
        out = self.net(x, w, ctx)
        self.assertEqual(out.shape, (N, 1))

    def test_finite_output(self):
        N = 8
        x = tf.random.normal([N, 1])
        w = tf.nn.softmax(tf.random.normal([N]))
        ctx = tf.random.normal([6])
        out = self.net(x, w, ctx)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))


class TestDeepSetEncoder(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.enc = DeepSetEncoder(state_dim=1, d_hidden=8, d_output=8)

    def test_forward_shape(self):
        N = 10
        x = tf.random.normal([N, 1])
        w = tf.nn.softmax(tf.random.normal([N]))
        out = self.enc(x, w)
        self.assertEqual(out.shape, (8,))


class TestDeepONetMonotoneOT(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.net = DeepONetMonotoneOT(
            state_dim=1, n_basis=4, d_branch=8, d_trunk=8, n_scalar_ctx=6,
        )

    def test_forward_shape(self):
        N = 10
        x = tf.random.normal([N, 1])
        w = tf.nn.softmax(tf.random.normal([N]))
        ctx = tf.random.normal([6])
        out = self.net(x, w, ctx)
        self.assertEqual(out.shape, (N, 1))

    def test_finite(self):
        x = tf.random.normal([5, 1])
        w = tf.nn.softmax(tf.random.normal([5]))
        ctx = tf.zeros([6])
        out = self.net(x, w, ctx)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))


class TestHyperDeepONetMonotoneOT(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.net = HyperDeepONetMonotoneOT(
            state_dim=1, n_basis=4, d_branch=8, d_trunk=8, n_scalar_ctx=6,
        )

    def test_forward_shape(self):
        N = 8
        x = tf.random.normal([N, 1])
        w = tf.nn.softmax(tf.random.normal([N]))
        ctx = tf.random.normal([6])
        out = self.net(x, w, ctx)
        self.assertEqual(out.shape, (N, 1))


class TestDifferentiableLEDHLogLikelihood(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.filt = DifferentiableLEDHLogLikelihood(
            num_particles=10,
            n_lambda=3,
            sinkhorn_epsilon=2.0,
            sinkhorn_iters=10,
            jit_compile=False,
        )

    def test_instantiation(self):
        self.assertEqual(self.filt.num_particles, 10)
        self.assertEqual(self.filt.n_lambda, 3)

    def test_log_likelihood_with_ssl_adapter(self):
        from src.models.gaussian_ssl import GaussianSSL, GaussianSSLasSSM

        ssl = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=4)
        _ = ssl.transition_params(ssl.get_initial_lstm_state(1), tf.zeros([1, 2]))
        z_traj = tf.random.normal([5, 2])
        adapter = GaussianSSLasSSM(ssl, z_traj)

        x_obs = tf.random.normal([5, 2])
        ll = self.filt(adapter, x_obs)
        self.assertEqual(ll.shape, ())
        self.assertTrue(tf.math.is_finite(ll))

    def test_ledh_returns_finite_with_ssl_adapter(self):
        """Verify LEDH filter returns finite log-likelihood with DifferentiableSSLAdapter."""
        from src.models.gaussian_ssl import GaussianSSL, GaussianSSLasSSM
        from src.filters.bonus.dpf_hmc_ssl import DifferentiableSSLAdapter

        ssl = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=4)
        _ = ssl.transition_params(ssl.get_initial_lstm_state(1), tf.zeros([1, 2]))

        z_traj = tf.random.normal([5, 2])
        adapter = GaussianSSLasSSM(ssl, z_traj)

        log_R = tf.zeros([2])
        diff_adapter = DifferentiableSSLAdapter(
            mus=adapter._mus,
            sigmas=tf.ones_like(adapter._mus) * 0.3,
            C=ssl.C,
            b=ssl.b,
            log_R_diag=log_R,
            state_dim=2,
            obs_dim=2,
        )

        x_obs = tf.random.normal([5, 2])
        ll = self.filt(diff_adapter, x_obs)
        self.assertTrue(tf.math.is_finite(ll))


if __name__ == "__main__":
    unittest.main()
