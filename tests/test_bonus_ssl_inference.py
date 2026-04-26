"""
Unit tests for SSL inference routines (Particle Gibbs + DPF-HMC warm-up/filtering).

Covers:
  - PGResult NamedTuple
  - particle_gibbs_ssl (tiny run)
  - warmup_em
  - compute_filtered_trajectory
  - pmmh_ssl_inference (tiny run)
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gaussian_ssl import GaussianSSL, generate_ssl_data
from src.filters.bonus.ssl_particle_gibbs import PGResult, particle_gibbs_ssl
from src.filters.bonus.dpf_hmc_ssl import (
    warmup_em,
    compute_filtered_trajectory,
)


class TestPGResult(unittest.TestCase):

    def test_fields(self):
        result = PGResult(
            z_samples=tf.zeros([2, 5, 2]),
            log_marginal=tf.zeros([2]),
            ssl_losses=[1.0, 0.5],
        )
        self.assertEqual(result.z_samples.shape, (2, 5, 2))
        self.assertEqual(result.log_marginal.shape, (2,))
        self.assertEqual(len(result.ssl_losses), 2)


class TestParticleGibbsSSL(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssl = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=4)
        _ = self.ssl.transition_params(
            self.ssl.get_initial_lstm_state(1), tf.zeros([1, 2])
        )
        _, self.x_obs = generate_ssl_data(T=8, state_dim=2, obs_dim=2, dynamics="sine", seed=0)

    def test_runs_and_returns_pg_result(self):
        result = particle_gibbs_ssl(
            self.ssl,
            self.x_obs,
            n_particles=5,
            n_iterations=2,
            n_m_steps=1,
            verbose=False,
        )
        self.assertIsInstance(result, PGResult)
        self.assertEqual(result.z_samples.shape, (2, 8, 2))
        self.assertEqual(result.log_marginal.shape, (2,))
        self.assertEqual(len(result.ssl_losses), 2)

    def test_z_samples_finite(self):
        result = particle_gibbs_ssl(
            self.ssl,
            self.x_obs,
            n_particles=5,
            n_iterations=2,
            n_m_steps=1,
            verbose=False,
        )
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.z_samples)))


class TestWarmupEM(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssl = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=4)
        _ = self.ssl.transition_params(
            self.ssl.get_initial_lstm_state(1), tf.zeros([1, 2])
        )
        _, self.x_obs = generate_ssl_data(T=8, state_dim=2, obs_dim=2, dynamics="sine", seed=1)

    def test_returns_correct_shape(self):
        z_ref, warmup_lls = warmup_em(
            self.ssl,
            self.x_obs,
            n_particles=5,
            n_outer=2,
            n_m_steps=1,
            m_step_lr=1e-3,
            verbose=False,
        )
        self.assertEqual(z_ref.shape, (8, 2))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(z_ref)))
        self.assertEqual(len(warmup_lls), 2)

    def test_lstm_weights_updated(self):
        w_before = [v.numpy().copy() for v in self.ssl.trainable_variables]
        warmup_em(
            self.ssl,
            self.x_obs,
            n_particles=5,
            n_outer=3,
            n_m_steps=2,
            verbose=False,
        )
        w_after = [v.numpy() for v in self.ssl.trainable_variables]
        changed = any(
            not np.allclose(b, a, atol=1e-8)
            for b, a in zip(w_before, w_after)
        )
        self.assertTrue(changed, "Warm-up should update at least some SSL weights")


class TestComputeFilteredTrajectory(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.ssl = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=4)
        _ = self.ssl.transition_params(
            self.ssl.get_initial_lstm_state(1), tf.zeros([1, 2])
        )
        _, self.x_obs = generate_ssl_data(T=8, state_dim=2, obs_dim=2, dynamics="sine", seed=2)
        self.z_ref = tf.random.normal([8, 2]) * 0.1

    def test_output_shape(self):
        z_est = compute_filtered_trajectory(
            self.ssl, self.x_obs, self.z_ref, n_particles=5
        )
        self.assertEqual(z_est.shape, (8, 2))

    def test_finite(self):
        z_est = compute_filtered_trajectory(
            self.ssl, self.x_obs, self.z_ref, n_particles=5
        )
        self.assertTrue(tf.reduce_all(tf.math.is_finite(z_est)))


if __name__ == "__main__":
    unittest.main()
