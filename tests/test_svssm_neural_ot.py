"""
Unit tests for the SVSSM neural-OT filters and their context builders:

  - differentiable_ledh_neural_ot_svssm.py
      build_svssm_context_scalars (7-D)
      _compute_ess
      DifferentiableLEDHNeuralOTSVSSM (eager + JIT-block call)
  - differentiable_ledh_neural_ot_svssm_jit.py
      DifferentiableLEDHNeuralOTSVSSMJIT (graph_mode variants)
  - differentiable_ledh_neural_ot_svssm_multivariate.py
      svssm_multi_ctx_dim, build_svssm_multi_context_scalars
      DifferentiableLEDHNeuralOTSVSSMmulti (construction + inheritance)

Uses a tiny ConditionalMGradNet so neural-OT resampling actually runs.
"""

import math
import unittest
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm import (
    build_svssm_context_scalars,
    _compute_ess,
    DifferentiableLEDHNeuralOTSVSSM,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_jit import (
    DifferentiableLEDHNeuralOTSVSSMJIT,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_multivariate import (
    svssm_multi_ctx_dim,
    build_svssm_multi_context_scalars,
    DifferentiableLEDHNeuralOTSVSSMmulti,
    SVSSM_MULTI_CTX_DIM_D1,
    SVSSM_MULTI_CTX_DIM_D2,
)


def _svssm_obs(T=5, mu=0.0, phi=0.9, sigma_eta=0.5, seed=0):
    tf.random.set_seed(seed)
    h = tf.constant(mu, tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def _tiny_mgradnet(n_scalar_ctx=7, N=16):
    tf.random.set_seed(0)
    net = ConditionalMGradNet(
        state_dim=1, n_ridges=4, d_set=8, d_scalar=8, n_scalar_ctx=n_scalar_ctx
    )
    # Build the weights with one forward pass.
    x = tf.random.normal([N])
    w = tf.nn.softmax(tf.random.normal([N]))
    ctx = tf.zeros([n_scalar_ctx])
    _ = net(x, w, ctx)
    return net


# ---------------------------------------------------------------------------
# Univariate context builder
# ---------------------------------------------------------------------------

class TestBuildSVSSMContext(unittest.TestCase):

    def test_shape(self):
        ctx = build_svssm_context_scalars(
            mu=0.0, phi=0.9, sigma_eta_sq=0.25, t=3.0, z_t=-1.0,
            ess=20.0, epsilon=1.0, T_max=50.0,
        )
        self.assertEqual(ctx.shape, (7,))

    def test_content(self):
        ctx = build_svssm_context_scalars(
            mu=0.5, phi=0.9, sigma_eta_sq=math.e, t=10.0, z_t=-2.0,
            ess=30.0, epsilon=2.0, T_max=100.0,
        ).numpy()
        self.assertAlmostEqual(ctx[0], 0.5, places=5)              # mu
        self.assertAlmostEqual(ctx[1], math.atanh(0.9), places=4)  # phi_raw
        self.assertAlmostEqual(ctx[2], 1.0, places=4)              # log(e)
        self.assertAlmostEqual(ctx[3], 0.1, places=5)              # t/T_max
        self.assertAlmostEqual(ctx[4], -2.0, places=5)             # z_t
        self.assertAlmostEqual(ctx[5], 30.0, places=4)             # ess
        self.assertAlmostEqual(ctx[6], 2.0, places=5)              # epsilon

    def test_phi_clipped_for_extreme(self):
        ctx = build_svssm_context_scalars(
            mu=0.0, phi=1.5, sigma_eta_sq=0.25, t=1.0, z_t=0.0,
            ess=10.0, epsilon=1.0,
        )
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ctx)))


class TestComputeESS(unittest.TestCase):

    def test_uniform(self):
        N = 20
        w = tf.fill([N], 1.0 / N)
        np.testing.assert_allclose(float(_compute_ess(w)), N, atol=0.1)

    def test_degenerate(self):
        w = tf.constant([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(float(_compute_ess(w)), 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Univariate neural-OT filter
# ---------------------------------------------------------------------------

class TestNeuralOTSVSSM(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.y = _svssm_obs(T=5)
        self.model = _tiny_mgradnet(n_scalar_ctx=7, N=16)

    def _make(self, **kw):
        params = dict(neural_ot_model=self.model, num_particles=16, n_lambda=3,
                      jit_compile=False)
        params.update(kw)
        return DifferentiableLEDHNeuralOTSVSSM(**params)

    def test_invalid_integrator_raises(self):
        with self.assertRaises(ValueError):
            self._make(integrator="rk4")

    def test_invalid_init_type_raises(self):
        with self.assertRaises(ValueError):
            self._make(init_type="nope")

    def test_call_finite_scalar(self):
        ll = self._make()
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(0.25), self.y)
        self.assertEqual(out.shape, ())
        self.assertEqual(out.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(out))

    def test_call_finite_jit_block(self):
        ll = self._make(jit_compile=True)
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(0.25), self.y)
        self.assertTrue(tf.math.is_finite(out))

    def test_model_stored(self):
        ll = self._make()
        self.assertIs(ll.neural_ot_model, self.model)


# ---------------------------------------------------------------------------
# Univariate JIT neural-OT filter
# ---------------------------------------------------------------------------

class TestNeuralOTSVSSMJIT(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.y = _svssm_obs(T=5)
        self.model = _tiny_mgradnet(n_scalar_ctx=7, N=16)

    def test_invalid_graph_mode_raises(self):
        with self.assertRaises(ValueError):
            DifferentiableLEDHNeuralOTSVSSMJIT(
                neural_ot_model=self.model, graph_mode="turbo"
            )

    def test_invalid_integrator_raises(self):
        with self.assertRaises(ValueError):
            DifferentiableLEDHNeuralOTSVSSMJIT(
                neural_ot_model=self.model, integrator="rk4"
            )

    def test_eager_mode_call_finite(self):
        ll = DifferentiableLEDHNeuralOTSVSSMJIT(
            neural_ot_model=self.model, num_particles=16, n_lambda=3,
            graph_mode="eager",
        )
        out = ll(tf.constant(0.0), tf.constant(0.9), tf.constant(0.25), self.y)
        self.assertEqual(out.shape, ())
        self.assertTrue(tf.math.is_finite(out))


# ---------------------------------------------------------------------------
# Multivariate context dim + builder
# ---------------------------------------------------------------------------

class TestSVSSMMultiCtxDim(unittest.TestCase):

    def test_d1_is_7(self):
        self.assertEqual(svssm_multi_ctx_dim(1), 7)
        self.assertEqual(SVSSM_MULTI_CTX_DIM_D1, 7)

    def test_d2_is_12(self):
        self.assertEqual(svssm_multi_ctx_dim(2), 12)
        self.assertEqual(SVSSM_MULTI_CTX_DIM_D2, 12)

    def test_d3_formula(self):
        # 3d + d(d-1)/2 + 3 + d  = 9 + 3 + 3 + 3 = 18
        self.assertEqual(svssm_multi_ctx_dim(3), 18)


class TestBuildSVSSMMultiContext(unittest.TestCase):

    def test_d2_shape(self):
        d = 2
        mu = tf.constant([0.0, 0.1])
        Phi = tf.constant([[0.9, 0.05], [0.0, 0.85]])
        sig_sq = tf.constant([0.25, 0.2])
        z_t = tf.constant([-1.0, -0.5])
        ctx = build_svssm_multi_context_scalars(
            mu=mu, Phi=Phi, sigma_eta_sq_diag=sig_sq, t=3.0, z_t=z_t,
            ess=20.0, epsilon=1.0, T_max=50.0, d=d,
        )
        self.assertEqual(ctx.shape, (svssm_multi_ctx_dim(d),))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ctx)))

    def test_d1_shape(self):
        d = 1
        ctx = build_svssm_multi_context_scalars(
            mu=tf.constant([0.0]), Phi=tf.constant([[0.9]]),
            sigma_eta_sq_diag=tf.constant([0.25]), t=1.0,
            z_t=tf.constant([-1.0]), ess=10.0, epsilon=1.0, T_max=20.0, d=d,
        )
        self.assertEqual(ctx.shape, (7,))

    def test_diag_atanh_entry(self):
        d = 2
        mu = tf.zeros([d])
        Phi = tf.constant([[0.9, 0.0], [0.0, 0.5]])
        sig_sq = tf.constant([1.0, 1.0])
        z_t = tf.zeros([d])
        ctx = build_svssm_multi_context_scalars(
            mu=mu, Phi=Phi, sigma_eta_sq_diag=sig_sq, t=0.0, z_t=z_t,
            ess=1.0, epsilon=1.0, T_max=1.0, d=d,
        ).numpy()
        # entries[d : 2d] are atanh(phi_diag)
        self.assertAlmostEqual(ctx[d + 0], math.atanh(0.9), places=4)
        self.assertAlmostEqual(ctx[d + 1], math.atanh(0.5), places=4)


class TestNeuralOTSVSSMMultiConstruction(unittest.TestCase):

    def test_stores_model_and_inherits(self):
        dummy = object()
        ll = DifferentiableLEDHNeuralOTSVSSMmulti(
            neural_ot_model=dummy, state_dim=2, num_particles=16, n_lambda=3,
        )
        self.assertIs(ll.neural_ot_model, dummy)
        self.assertEqual(ll.state_dim, 2)
        self.assertIsInstance(ll, DifferentiableLEDHNeuralOTSVSSMmulti)


class TestNeuralOTSVSSMMultiCall(unittest.TestCase):
    """End-to-end full-Phi forward pass with a tiny batched ConditionalMGradNet."""

    def setUp(self):
        tf.random.set_seed(0)
        self.d = 2
        self.T = 4
        self.N = 16
        ctx_dim = svssm_multi_ctx_dim(self.d)
        # Batched signature: model((B,N,d), (B,N), (B,ctx)) -> (B,N,d).
        self.model = ConditionalMGradNet(
            state_dim=self.d, n_ridges=4, d_set=8, d_scalar=8,
            n_scalar_ctx=ctx_dim,
        )
        _ = self.model(
            tf.zeros([1, self.N, self.d]),
            tf.fill([1, self.N], 1.0 / self.N),
            tf.zeros([1, ctx_dim]),
        )
        self.y = tf.random.normal([self.T, self.d]) * 0.5

    def test_call_mat_phi_finite_scalar(self):
        ll = DifferentiableLEDHNeuralOTSVSSMmulti(
            neural_ot_model=self.model, state_dim=self.d,
            num_particles=self.N, n_lambda=3,
        )
        mu = tf.zeros([self.d])
        Phi = tf.constant([[0.9, 0.05], [0.0, 0.85]])
        L_eta = tf.constant([[0.5, 0.0], [0.1, 0.45]])
        out = ll.call_mat_phi(mu, Phi, L_eta, self.y)
        self.assertEqual(out.shape, ())
        self.assertEqual(out.dtype, tf.float32)
        self.assertTrue(tf.math.is_finite(out))


if __name__ == "__main__":
    unittest.main()
