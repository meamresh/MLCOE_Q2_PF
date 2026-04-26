"""
Unit tests for uncovered state-space models:
  - PMCMCNonlinearSSM  (Kitagawa / Andrieu 2010)
  - DaiDaumBearingSSM  (Dai & Daum 2021)
  - GaussianSSL, GaussianSSLasSSM, generate_ssl_data
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ssm_katigawa import PMCMCNonlinearSSM
from src.models.ssm_dai_daum_bearing_only import (
    DaiDaumBearingSSM, h_vec, DTYPE,
)
from src.models.gaussian_ssl import GaussianSSL, GaussianSSLasSSM, generate_ssl_data


# =========================================================================
# PMCMCNonlinearSSM
# =========================================================================

class TestPMCMCNonlinearSSM(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssm = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0)

    def test_initialization(self):
        self.assertEqual(self.ssm.state_dim, 1)
        self.assertEqual(self.ssm.Q.shape, (1, 1))
        self.assertEqual(self.ssm.R.shape, (1, 1))
        np.testing.assert_allclose(self.ssm.Q.numpy(), [[10.0]])
        np.testing.assert_allclose(self.ssm.R.numpy(), [[1.0]])

    def test_motion_model_shape(self):
        state = tf.constant([[2.0]], dtype=tf.float32)
        control = tf.constant([[1.0]], dtype=tf.float32)
        out = self.ssm.motion_model(state, control)
        self.assertEqual(out.shape, (1, 1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))

    def test_motion_model_1d_input(self):
        state = tf.constant([2.0], dtype=tf.float32)
        control = tf.constant([1.0], dtype=tf.float32)
        out = self.ssm.motion_model(state, control)
        self.assertEqual(out.shape, (1, 1))

    def test_motion_model_batch(self):
        state = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)
        control = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)
        out = self.ssm.motion_model(state, control)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))

    def test_measurement_model_shape(self):
        state = tf.constant([[3.0]], dtype=tf.float32)
        out = self.ssm.measurement_model(state)
        self.assertEqual(out.shape, (1, 1))
        np.testing.assert_allclose(out.numpy(), [[9.0 / 20.0]], atol=1e-5)

    def test_motion_jacobian_shape(self):
        state = tf.constant([[2.0]], dtype=tf.float32)
        control = tf.constant([[1.0]], dtype=tf.float32)
        jac = self.ssm.motion_jacobian(state, control)
        self.assertEqual(jac.shape, (1, 1, 1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(jac)))

    def test_measurement_jacobian_shape(self):
        state = tf.constant([[5.0]], dtype=tf.float32)
        jac = self.ssm.measurement_jacobian(state)
        self.assertEqual(jac.shape, (1, 1, 1))
        np.testing.assert_allclose(jac.numpy(), [[[0.5]]], atol=1e-5)

    def test_full_measurement_cov(self):
        R = self.ssm.full_measurement_cov(1)
        self.assertEqual(R.shape, (1, 1))

    def test_custom_parameters(self):
        ssm2 = PMCMCNonlinearSSM(sigma_v_sq=5.0, sigma_w_sq=2.0, initial_var=3.0)
        np.testing.assert_allclose(ssm2.Q.numpy(), [[5.0]])
        np.testing.assert_allclose(ssm2.R.numpy(), [[2.0]])
        self.assertAlmostEqual(ssm2.initial_var, 3.0)


# =========================================================================
# DaiDaumBearingSSM
# =========================================================================

class TestDaiDaumBearingSSM(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssm = DaiDaumBearingSSM()

    def test_initialization(self):
        self.assertEqual(self.ssm.state_dim, 2)
        self.assertEqual(self.ssm.meas_dim, 2)
        self.assertEqual(self.ssm.Q.shape, (2, 2))
        self.assertEqual(self.ssm.R.shape, (2, 2))

    def test_motion_model_identity(self):
        state = tf.constant([4.0, 4.0], dtype=DTYPE)
        next_state, jac = self.ssm.motion_model(state)
        np.testing.assert_allclose(next_state.numpy(), state.numpy(), atol=1e-10)
        np.testing.assert_allclose(jac.numpy(), np.eye(2), atol=1e-10)

    def test_measurement_model_shape(self):
        state = tf.constant([4.0, 4.0], dtype=DTYPE)
        h, jac = self.ssm.measurement_model(state)
        self.assertEqual(h.shape, (2,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(h)))

    def test_log_prior_finite(self):
        x = tf.constant([3.0, 5.0], dtype=DTYPE)
        lp = self.ssm.log_prior(x)
        self.assertTrue(tf.math.is_finite(lp))

    def test_log_prior_max_at_mean(self):
        lp_mean = self.ssm.log_prior(self.ssm.prior_mean)
        lp_off = self.ssm.log_prior(self.ssm.prior_mean + tf.constant([1.0, 1.0], dtype=DTYPE))
        self.assertGreater(float(lp_mean), float(lp_off))

    def test_gradient_log_prior_shape(self):
        x = tf.constant([4.0, 4.0], dtype=DTYPE)
        grad = self.ssm.gradient_log_prior(x)
        self.assertEqual(grad.shape, (2,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(grad)))

    def test_hessian_log_prior_shape(self):
        x = tf.constant([4.0, 4.0], dtype=DTYPE)
        hess = self.ssm.hessian_log_prior(x)
        self.assertEqual(hess.shape, (2, 2))

    def test_log_likelihood_finite(self):
        x = tf.constant([4.0, 4.0], dtype=DTYPE)
        z = self.ssm._h_vec(x)
        ll = self.ssm.log_likelihood(x, z)
        self.assertTrue(tf.math.is_finite(ll))

    def test_gradient_log_likelihood_shape(self):
        x = tf.constant([4.0, 4.0], dtype=DTYPE)
        z = self.ssm._h_vec(x)
        grad = self.ssm.gradient_log_likelihood(x, z)
        self.assertEqual(grad.shape, (2,))

    def test_hessian_log_likelihood_shape(self):
        x = tf.constant([4.0, 4.0], dtype=DTYPE)
        z = self.ssm._h_vec(x)
        hess = self.ssm.hessian_log_likelihood(x, z)
        self.assertEqual(hess.shape, (2, 2))


class TestHVecStandalone(unittest.TestCase):

    def test_h_vec_shape(self):
        x = tf.constant([4.0, 4.0], dtype=DTYPE)
        h = h_vec(x)
        self.assertEqual(h.shape, (2,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(h)))


# =========================================================================
# GaussianSSL
# =========================================================================

class TestGaussianSSL(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssl = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=8)
        _ = self.ssl.transition_params(
            self.ssl.get_initial_lstm_state(1), tf.zeros([1, 2])
        )

    def test_construction(self):
        self.assertEqual(self.ssl.state_dim_val, 2)
        self.assertEqual(self.ssl.obs_dim_val, 2)
        self.assertEqual(self.ssl.lstm_units, 8)

    def test_R_matrix_shape(self):
        R = self.ssl.R_matrix
        self.assertEqual(R.shape, (2, 2))
        eigvals = tf.linalg.eigvalsh(R)
        self.assertTrue(tf.reduce_all(eigvals > 0))

    def test_log_R_diag_trainable(self):
        self.assertTrue(self.ssl.log_R_diag.trainable)

    def test_initial_lstm_state(self):
        h, c = self.ssl.get_initial_lstm_state(batch_size=3)
        self.assertEqual(h.shape, (3, 8))
        self.assertEqual(c.shape, (3, 8))

    def test_transition_params_shapes(self):
        lstm_state = self.ssl.get_initial_lstm_state(batch_size=2)
        z_prev = tf.zeros([2, 2])
        mu, sigma, new_state = self.ssl.transition_params(lstm_state, z_prev)
        self.assertEqual(mu.shape, (2, 2))
        self.assertEqual(sigma.shape, (2, 2))
        self.assertTrue(tf.reduce_all(sigma > 0))

    def test_emission_log_prob(self):
        z = tf.constant([[1.0, 0.5]], dtype=tf.float32)
        x = tf.constant([[1.0, 0.5]], dtype=tf.float32)
        lp = self.ssl.emission_log_prob(z, x)
        self.assertEqual(lp.shape, (1,))
        self.assertTrue(tf.math.is_finite(lp[0]))

    def test_forward_messages_shapes(self):
        lstm_state = self.ssl.get_initial_lstm_state(batch_size=5)
        z_prev = tf.zeros([5, 2])
        x_t = tf.ones([5, 2])
        alpha, gamma_mu, gamma_var, new_state = self.ssl.forward_messages(
            lstm_state, z_prev, x_t
        )
        self.assertEqual(alpha.shape, (5,))
        self.assertEqual(gamma_mu.shape, (5, 2))
        self.assertEqual(gamma_var.shape, (5, 2, 2))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(alpha)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(gamma_mu)))


# =========================================================================
# GaussianSSLasSSM adapter
# =========================================================================

class TestGaussianSSLasSSM(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssl = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=8)
        _ = self.ssl.transition_params(
            self.ssl.get_initial_lstm_state(1), tf.zeros([1, 2])
        )
        z_traj = tf.random.normal([10, 2])
        self.adapter = GaussianSSLasSSM(self.ssl, z_traj)

    def test_attributes(self):
        self.assertEqual(self.adapter.state_dim, 2)
        self.assertEqual(self.adapter.Q.shape, (2, 2))
        self.assertEqual(self.adapter.R.shape, (2, 2))

    def test_motion_model_shape(self):
        state = tf.constant([1.0, 0.5], dtype=tf.float32)
        control = tf.constant([0.0], dtype=tf.float32)
        out = self.adapter.motion_model(state, control)
        self.assertEqual(out.shape[1], 2)

    def test_measurement_model_shape(self):
        state = tf.constant([[1.0, 0.5]], dtype=tf.float32)
        out = self.adapter.measurement_model(state)
        self.assertEqual(out.shape, (1, 2))

    def test_motion_jacobian_shape(self):
        state = tf.constant([1.0, 0.5], dtype=tf.float32)
        control = tf.constant([0.0], dtype=tf.float32)
        jac = self.adapter.motion_jacobian(state, control)
        self.assertEqual(jac.shape[1], 2)
        self.assertEqual(jac.shape[2], 2)

    def test_measurement_jacobian_shape(self):
        state = tf.constant([1.0, 0.5], dtype=tf.float32)
        jac = self.adapter.measurement_jacobian(state)
        self.assertEqual(jac.shape[1], 2)
        self.assertEqual(jac.shape[2], 2)


# =========================================================================
# generate_ssl_data
# =========================================================================

class TestGenerateSSLData(unittest.TestCase):

    def test_sine_dynamics(self):
        z, x = generate_ssl_data(T=20, state_dim=2, obs_dim=2, dynamics="sine", seed=0)
        self.assertEqual(z.shape, (20, 2))
        self.assertEqual(x.shape, (20, 2))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(z)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(x)))

    def test_circle_dynamics(self):
        z, x = generate_ssl_data(T=15, dynamics="circle", seed=1)
        self.assertEqual(z.shape, (15, 2))

    def test_line_dynamics(self):
        z, x = generate_ssl_data(T=10, dynamics="line", seed=2)
        self.assertEqual(z.shape, (10, 2))

    def test_swiss_roll_dynamics(self):
        z, x = generate_ssl_data(T=10, dynamics="swiss_roll", seed=3)
        self.assertEqual(z.shape, (10, 2))

    def test_unknown_dynamics_raises(self):
        with self.assertRaises(ValueError):
            generate_ssl_data(dynamics="unknown")

    def test_reproducibility(self):
        z1, x1 = generate_ssl_data(T=10, seed=99)
        z2, x2 = generate_ssl_data(T=10, seed=99)
        np.testing.assert_allclose(z1.numpy(), z2.numpy(), atol=1e-6)
        np.testing.assert_allclose(x1.numpy(), x2.numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
