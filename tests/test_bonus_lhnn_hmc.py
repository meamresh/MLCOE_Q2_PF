"""
Unit tests for Latent Hamiltonian Neural Network (L-HNN) accelerated HMC.

Covers:
  - LHNNConfig dataclass
  - LHNNHMCDiagnostics dataclass + properties
  - LatentHNN model (forward, potential_grad_and_hamiltonian, hamiltonian)
  - generate_training_data
  - train_lhnn (tiny run)
  - ess_per_gradient
  - run_lhnn_hmc (tiny run with pre-trained L-HNN)
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.bonus.lhnn_hmc_pf import (
    LHNNConfig,
    LHNNHMCDiagnostics,
    LatentHNN,
    generate_training_data,
    train_lhnn,
    run_lhnn_hmc,
    ess_per_gradient,
)
from src.filters.bonus.hmc_pf import HMCResult


class TestLHNNConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = LHNNConfig()
        self.assertEqual(cfg.hidden_units, 256)
        self.assertEqual(cfg.num_hidden, 3)
        self.assertEqual(cfg.epochs, 3000)
        self.assertGreater(cfg.error_threshold, 0)

    def test_custom(self):
        cfg = LHNNConfig(hidden_units=32, epochs=10)
        self.assertEqual(cfg.hidden_units, 32)
        self.assertEqual(cfg.epochs, 10)


class TestLHNNHMCDiagnostics(unittest.TestCase):

    def test_properties(self):
        diag = LHNNHMCDiagnostics(
            training_gradient_evals=100,
            sampling_real_gradient_evals=20,
            total_mcmc_iterations=50,
            leapfrog_steps_per_iter=10,
        )
        self.assertEqual(diag.total_leapfrog_steps, 500)
        self.assertEqual(diag.total_real_gradient_evals, 120)
        self.assertAlmostEqual(diag.sampling_fallback_intensity, 20 / 500)


class TestLatentHNN(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.d = 2
        self.lhnn = LatentHNN(d=self.d, hidden_units=8, num_hidden=2)
        _ = self.lhnn(tf.zeros([1, 2 * self.d]))

    def test_forward_shape(self):
        z = tf.random.normal([5, 2 * self.d])
        out = self.lhnn(z)
        self.assertEqual(out.shape, (5, self.d))

    def test_finite_output(self):
        z = tf.random.normal([3, 2 * self.d])
        out = self.lhnn(z)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out)))

    def test_potential_grad_and_hamiltonian(self):
        q = tf.random.normal([self.d])
        p = tf.random.normal([self.d])
        dH_dq, H_val = self.lhnn.potential_grad_and_hamiltonian(q, p)
        self.assertEqual(dH_dq.shape, (self.d,))
        self.assertEqual(H_val.shape, ())
        self.assertTrue(tf.math.is_finite(H_val))

    def test_hamiltonian_scalar(self):
        q = tf.random.normal([self.d])
        p = tf.random.normal([self.d])
        H = self.lhnn.hamiltonian(q, p)
        self.assertEqual(H.shape, ())
        self.assertTrue(tf.math.is_finite(H))


class TestGenerateTrainingData(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_output_shapes_and_count(self):
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        q, p, dq, dp, n_evals = generate_training_data(
            log_target,
            initial_state=tf.constant([0.0, 0.0]),
            num_trajectories=2,
            steps_per_trajectory=5,
            step_size=0.1,
            seed=42,
            verbose=False,
        )
        expected_points = 2 * 5
        self.assertEqual(q.shape, (expected_points, 2))
        self.assertEqual(p.shape, (expected_points, 2))
        self.assertEqual(dq.shape, (expected_points, 2))
        self.assertEqual(dp.shape, (expected_points, 2))
        self.assertGreater(n_evals, 0)

    def test_finite(self):
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        q, p, dq, dp, _ = generate_training_data(
            log_target,
            initial_state=tf.constant([1.0]),
            num_trajectories=2,
            steps_per_trajectory=3,
            step_size=0.05,
            seed=99,
            verbose=False,
        )
        self.assertTrue(tf.reduce_all(tf.math.is_finite(q)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(dp)))

    def test_gradient_eval_count(self):
        """Gradient evals = num_traj * (steps_per_traj + 1)."""
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        n_traj, steps = 3, 7
        _, _, _, _, n_evals = generate_training_data(
            log_target,
            initial_state=tf.constant([0.0, 0.0]),
            num_trajectories=n_traj,
            steps_per_trajectory=steps,
            step_size=0.1,
            seed=42,
            verbose=False,
        )
        self.assertEqual(n_evals, n_traj * (steps + 1))

    def test_dq_equals_p(self):
        """dq/dt should equal p (exact kinetic energy derivative)."""
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        _, p, dq, _, _ = generate_training_data(
            log_target,
            initial_state=tf.constant([0.5]),
            num_trajectories=2,
            steps_per_trajectory=4,
            step_size=0.05,
            seed=42,
            verbose=False,
        )
        np.testing.assert_allclose(dq.numpy(), p.numpy(), atol=1e-6)


class TestTrainLHNN(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_training_updates_weights(self):
        d = 1
        lhnn = LatentHNN(d=d, hidden_units=8, num_hidden=2)
        _ = lhnn(tf.zeros([1, 2 * d]))

        N = 20
        q_data = tf.random.normal([N, d])
        p_data = tf.random.normal([N, d])
        dq_dt = p_data
        dp_dt = -q_data

        w_before = [v.numpy().copy() for v in lhnn.trainable_variables]

        train_lhnn(
            lhnn, q_data, p_data, dq_dt, dp_dt,
            epochs=5, lr=1e-2, batch_size=10, verbose=False,
        )

        w_after = [v.numpy() for v in lhnn.trainable_variables]
        changed = any(not np.allclose(b, a, atol=1e-8) for b, a in zip(w_before, w_after))
        self.assertTrue(changed)

    def test_training_reduces_loss(self):
        """Training should reduce the Hamilton residual loss."""
        d = 1
        lhnn = LatentHNN(d=d, hidden_units=16, num_hidden=2)
        _ = lhnn(tf.zeros([1, 2 * d]))

        N = 40
        q_data = tf.random.normal([N, d])
        p_data = tf.random.normal([N, d])
        dq_dt = p_data
        dp_dt = -q_data

        def compute_loss():
            z = tf.concat([q_data, p_data], axis=1)
            with tf.GradientTape() as tape:
                tape.watch(z)
                lam = lhnn(z)
                H = tf.reduce_sum(lam, axis=1)
            dH_dz = tape.gradient(H, z)
            loss_k = tf.reduce_mean(tf.abs(dH_dz[:, d:] - dq_dt))
            loss_p = tf.reduce_mean(tf.abs(-dH_dz[:, :d] - dp_dt))
            return float((loss_k + loss_p).numpy())

        loss_before = compute_loss()
        train_lhnn(
            lhnn, q_data, p_data, dq_dt, dp_dt,
            epochs=50, lr=1e-2, batch_size=20, verbose=False,
        )
        loss_after = compute_loss()
        self.assertLess(loss_after, loss_before)


class TestESSPerGradient(unittest.TestCase):

    def test_returns_positive_scalar(self):
        samples = tf.random.normal([50, 2])
        result = ess_per_gradient(samples, total_gradient_evals=100)
        self.assertEqual(result.shape, ())
        self.assertGreater(float(result), 0)

    def test_handles_zero_evals(self):
        samples = tf.random.normal([20, 1])
        result = ess_per_gradient(samples, total_gradient_evals=0)
        self.assertTrue(tf.math.is_finite(result))


class TestRunLHNNHMC(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_with_pretrained_lhnn(self):
        """Run L-HNN HMC with a pre-trained (untrained) L-HNN on a simple target."""
        d = 1

        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        lhnn = LatentHNN(d=d, hidden_units=8, num_hidden=2)
        _ = lhnn(tf.zeros([1, 2 * d]))

        result, lhnn_out, diagnostics = run_lhnn_hmc(
            log_target,
            initial_state=tf.constant([1.0]),
            num_results=5,
            num_burnin=3,
            step_size=0.1,
            num_leapfrog_steps=3,
            seed=42,
            verbose=False,
            pretrained_lhnn=lhnn,
        )
        self.assertIsInstance(result, HMCResult)
        self.assertEqual(result.samples.shape, (5, 1))
        self.assertTrue(0.0 <= float(result.accept_rate) <= 1.0)
        self.assertIsInstance(diagnostics, LHNNHMCDiagnostics)
        self.assertEqual(diagnostics.training_gradient_evals, 0)

    def test_samples_finite(self):
        """All samples should be finite."""
        d = 2

        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        lhnn = LatentHNN(d=d, hidden_units=8, num_hidden=2)
        _ = lhnn(tf.zeros([1, 2 * d]))

        result, _, _ = run_lhnn_hmc(
            log_target,
            initial_state=tf.constant([0.5, -0.5]),
            num_results=8,
            num_burnin=3,
            step_size=0.05,
            num_leapfrog_steps=3,
            seed=42,
            verbose=False,
            pretrained_lhnn=lhnn,
        )
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result.samples)))

    def test_diagnostics_consistency(self):
        """Diagnostics total_leapfrog_steps should equal iters * steps."""
        d = 1

        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        lhnn = LatentHNN(d=d, hidden_units=8, num_hidden=2)
        _ = lhnn(tf.zeros([1, 2 * d]))

        n_results, n_burnin, L = 6, 4, 5
        _, _, diag = run_lhnn_hmc(
            log_target,
            initial_state=tf.constant([0.0]),
            num_results=n_results,
            num_burnin=n_burnin,
            step_size=0.05,
            num_leapfrog_steps=L,
            seed=42,
            verbose=False,
            pretrained_lhnn=lhnn,
        )
        self.assertEqual(diag.total_mcmc_iterations, n_results + n_burnin)
        self.assertEqual(diag.leapfrog_steps_per_iter, L)
        self.assertEqual(
            diag.total_leapfrog_steps, (n_results + n_burnin) * L)


if __name__ == "__main__":
    unittest.main()
