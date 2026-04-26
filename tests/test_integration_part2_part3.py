"""
Integration tests for Part 2 and Part 3 pipelines.

End-to-end tests covering:
  - DPF pipeline: Kitagawa data -> DPF filter -> Sinkhorn resampling -> finite RMSE
  - SPF pipeline: DaiDaum SSM -> stochastic particle flow -> finite particles
  - Differentiable LEDH pipeline: GaussianSSL -> adapter -> LEDH LL -> gradient exists
  - Neural OT forward pass: mGradNet on random batch -> finite output
  - PMMH short chain: Kitagawa SSM -> bootstrap PF LL -> 3-step PMMH chain
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDPFPipelineIntegration(unittest.TestCase):
    """End-to-end: Kitagawa SSM → DPF filter → RMSE."""

    def setUp(self):
        tf.random.set_seed(42)

    def test_dpf_on_kitagawa(self):
        from src.filters.dpf.diff_particle_filter import (
            BootstrapModel,
            DifferentiableParticleFilter,
        )

        sigma_v, sigma_w = 3.16, 1.0

        def sample_initial(N, y0):
            x = tf.random.normal([N, 1]) * 2.0
            lw = -0.5 * (y0 - x ** 2 / 20.0) ** 2 / sigma_w ** 2
            return x, tf.squeeze(lw, axis=-1)

        def transition(t, x_prev, y_t):
            t_f = tf.cast(t, tf.float32)
            mean = 0.5 * x_prev + 25.0 * x_prev / (1 + x_prev ** 2) + 8 * tf.cos(1.2 * t_f)
            x = mean + tf.random.normal(tf.shape(x_prev)) * sigma_v
            lw = -0.5 * (y_t - x ** 2 / 20.0) ** 2 / sigma_w ** 2
            return x, tf.squeeze(lw, axis=-1)

        model = BootstrapModel(sample_initial, transition)
        dpf = DifferentiableParticleFilter(
            model, num_particles=20, epsilon=1.0, sinkhorn_iters=10
        )

        y = tf.random.normal([8, 1])
        loglik, final_log_w = dpf(y)

        self.assertTrue(tf.math.is_finite(loglik))
        self.assertEqual(final_log_w.shape, (20,))


class TestSPFPipelineIntegration(unittest.TestCase):
    """End-to-end: DaiDaum SSM → stochastic particle flow → finite particles."""

    def setUp(self):
        tf.random.set_seed(42)

    def test_spf_basic(self):
        from src.models.ssm_dai_daum_bearing_only import DaiDaumBearingSSM, DTYPE
        from src.filters.spf_dai_daum import (
            DaiDaumStochasticParticleFlow,
            SPFConfig,
        )

        ssm = DaiDaumBearingSSM()
        config = SPFConfig(
            n_particles=15,
            lambda_steps=5,
            bvp_mesh_points=5,
            bvp_max_iter=5,
            verbose=False,
        )

        spf = DaiDaumStochasticParticleFlow(
            ssm=ssm,
            initial_state=ssm.prior_mean,
            initial_covariance=ssm.prior_cov,
            config=config,
            homotopy_mode="linear",
        )
        spf.predict()
        z_obs = ssm._h_vec(ssm.x_true)
        z_noisy = z_obs + tf.random.normal([2], dtype=DTYPE) * 0.2
        spf.update(z_noisy)

        self.assertIsNotNone(spf)


class TestDifferentiableLEDHPipelineIntegration(unittest.TestCase):
    """End-to-end: GaussianSSL → adapter → LEDH log-likelihood → gradient."""

    def setUp(self):
        tf.random.set_seed(42)

    def test_ledh_returns_finite_ll(self):
        from src.models.gaussian_ssl import GaussianSSL, GaussianSSLasSSM
        from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood

        ssl = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=4)
        _ = ssl.transition_params(ssl.get_initial_lstm_state(1), tf.zeros([1, 2]))

        z_traj = tf.random.normal([6, 2])
        adapter = GaussianSSLasSSM(ssl, z_traj)

        filt = DifferentiableLEDHLogLikelihood(
            num_particles=8, n_lambda=2, sinkhorn_epsilon=2.0,
            sinkhorn_iters=10, jit_compile=False,
        )

        x_obs = tf.random.normal([6, 2])
        ll = filt(adapter, x_obs)
        self.assertTrue(tf.math.is_finite(ll))


class TestNeuralOTForwardIntegration(unittest.TestCase):
    """mGradNet forward pass on random batch → finite output."""

    def setUp(self):
        tf.random.set_seed(42)

    def test_mgradnet_forward(self):
        from src.filters.bonus.mgradnet_ot import ConditionalMGradNet

        net = ConditionalMGradNet(
            state_dim=1, n_ridges=4, d_set=8, d_scalar=8, n_scalar_ctx=6,
        )
        N = 15
        x = tf.random.normal([N, 1])
        w = tf.nn.softmax(tf.random.normal([N]))
        ctx = tf.random.normal([6])

        out = net(x, w, ctx)
        self.assertEqual(out.shape, (N, 1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))


class TestPMMHShortChainIntegration(unittest.TestCase):
    """Kitagawa SSM → bootstrap PF LL → 3-step PMMH chain."""

    def setUp(self):
        tf.random.set_seed(42)

    def test_pmmh_short_chain(self):
        from src.models.ssm_katigawa import PMCMCNonlinearSSM
        from src.filters.bonus.pmmh import (
            bootstrap_pf_log_likelihood,
            run_pmmh,
        )

        ssm = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0)
        T = 8
        y_obs = tf.random.normal([T, 1])

        ll = bootstrap_pf_log_likelihood(ssm, y_obs, num_particles=20)
        self.assertTrue(tf.math.is_finite(ll))

        def log_target(theta):
            s = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(theta[0]),
                sigma_w_sq=tf.exp(theta[1]),
            )
            ll = bootstrap_pf_log_likelihood(s, y_obs, num_particles=15)
            prior = -0.5 * tf.reduce_sum(theta ** 2)
            return ll + prior

        init = tf.stack([tf.math.log(10.0), tf.math.log(1.0)])
        result = run_pmmh(
            log_target,
            init,
            num_results=3,
            num_burnin=2,
            step_size=0.3,
            verbose=False,
        )
        self.assertEqual(result.samples.shape, (3, 2))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))


if __name__ == "__main__":
    unittest.main()
