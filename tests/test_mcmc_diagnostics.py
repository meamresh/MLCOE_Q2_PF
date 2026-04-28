"""
Unit tests for src.utils.mcmc_diagnostics + the multi-chain / mass-matrix
machinery in src.filters.bonus.hmc_pf.

The diagnostic suite covers:

- well-mixed i.i.d. Gaussian chains            -> rank_rhat ~ 1.0, ESS ~ N
- non-overlapping chains (means at +/- 5)      -> rank_rhat >> 1.01
- credible-interval coverage on N(0,1)         -> ~95% over Monte-Carlo replicates
- convergence verdict thresholds
- format_diagnostics_table renders without crashing
- mass-matrix adaptation produces a well-formed run on an anisotropic
  Gaussian target (smoke test; we assert finite samples + reasonable accept).
- shared CRN across chains: aligned target seeds, distinct sample paths.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.mcmc_diagnostics import (
    bulk_ess,
    tail_ess,
    split_rhat,
    rank_rhat,
    credible_interval,
    coverage,
    convergence_verdict,
    diagnostics_summary,
    format_diagnostics_table,
)
from src.filters.bonus import hmc_pf as hmc_pf_module
from src.filters.bonus.hmc_pf import (
    run_hmc,
    run_hmc_multi_chain,
    disperse_initial_states,
)


tfd = tfp.distributions


class TestRankRhatWellMixed(unittest.TestCase):
    """Two i.i.d. Gaussian chains -> rank_rhat ~ 1.0, bulk-ESS near nominal."""

    def setUp(self):
        tf.random.set_seed(0)
        np.random.seed(0)
        self.C = 4
        self.S = 1000
        self.D = 2

    def test_iid_gaussian_chains_pass(self):
        x = tf.random.normal([self.C, self.S, self.D], seed=0)
        rh = rank_rhat(x).numpy()
        sh = split_rhat(x).numpy()
        b_ess = bulk_ess(x).numpy()
        t_ess = tail_ess(x).numpy()

        self.assertTrue(np.all(rh < 1.05), f"rank_rhat too large: {rh}")
        self.assertTrue(np.all(sh < 1.05), f"split_rhat too large: {sh}")
        # i.i.d. samples -> bulk-ESS should be a substantial fraction of N.
        self.assertTrue(np.all(b_ess > 0.5 * self.C * self.S),
                        f"bulk_ess too small: {b_ess}")
        self.assertTrue(np.all(t_ess > 0.3 * self.C * self.S),
                        f"tail_ess too small: {t_ess}")

    def test_verdict_passes_for_well_mixed(self):
        x = tf.random.normal([self.C, self.S, self.D], seed=1)
        s = diagnostics_summary(x, ["a", "b"])
        self.assertTrue(all(s.converged), f"verdict failed: {s}")


class TestRankRhatBroken(unittest.TestCase):
    """Two non-overlapping chains -> rank_rhat >> 1.01."""

    def test_disjoint_chains_fail(self):
        tf.random.set_seed(1)
        # chain 0 around +5, chain 1 around -5, chain 2 around +5, chain 3 around -5
        means = tf.constant([[5.0], [-5.0], [5.0], [-5.0]])  # (4, 1)
        eps = tf.random.normal([4, 1000, 1], stddev=0.1)
        x = means[:, None, :] + eps  # (4, 1000, 1)

        rh = rank_rhat(x).numpy()
        sh = split_rhat(x).numpy()

        self.assertTrue(np.all(rh > 1.5),
                        f"rank_rhat should be huge for disjoint chains: {rh}")
        self.assertTrue(np.all(sh > 1.5),
                        f"split_rhat should be huge for disjoint chains: {sh}")

    def test_verdict_fails_for_disjoint(self):
        tf.random.set_seed(2)
        means = tf.constant([[3.0], [-3.0], [3.0], [-3.0]])
        eps = tf.random.normal([4, 800, 1], stddev=0.05)
        x = means[:, None, :] + eps
        s = diagnostics_summary(x, ["theta"])
        self.assertFalse(s.converged[0])


class TestCredibleIntervalCoverage(unittest.TestCase):
    """Frequentist coverage check: 95% CIs from N(0,1) cover 0 ~95% of the time."""

    def test_coverage_close_to_nominal(self):
        np.random.seed(7)
        n_replicates = 200
        N = 2000
        hits = 0
        for r in range(n_replicates):
            tf.random.set_seed(100 + r)
            x = tf.random.normal([1, N, 1])
            ci = credible_interval(x, level=0.95)  # (1, 2)
            cov = coverage(ci, tf.constant([0.0])).numpy()[0]
            if cov:
                hits += 1
        emp = hits / n_replicates
        # 95% empirical coverage with 200 replicates: 95% binomial CI ~ [0.91, 0.98]
        self.assertGreaterEqual(emp, 0.88)
        self.assertLessEqual(emp, 1.0)

    def test_coverage_truth_outside_interval(self):
        # Truth far from posterior mean -> coverage False
        x = tf.random.normal([2, 500, 1], seed=11)
        ci = credible_interval(x, level=0.95)
        cov = coverage(ci, tf.constant([10.0])).numpy()
        self.assertFalse(bool(cov[0]))


class TestConvergenceVerdict(unittest.TestCase):

    def test_thresholds(self):
        # rank_rhat below threshold + ESS above 400 -> PASS
        v = convergence_verdict(
            tf.constant([1.005, 1.02]),
            tf.constant([500.0, 200.0]),
            tf.constant([500.0, 1500.0]),
        ).numpy()
        # param 0: PASS (R=1.005<1.01, bulk=500>400, tail=500>400)
        # param 1: FAIL (R=1.02>=1.01)
        self.assertTrue(bool(v[0]))
        self.assertFalse(bool(v[1]))

    def test_low_tail_fails(self):
        v = convergence_verdict(
            tf.constant([1.005]),
            tf.constant([800.0]),
            tf.constant([350.0]),  # below threshold
        ).numpy()
        self.assertFalse(bool(v[0]))


class TestFormatTable(unittest.TestCase):

    def test_table_renders(self):
        x = tf.random.normal([4, 600, 2], seed=21)
        s = diagnostics_summary(x, ["alpha", "beta"], truth=[0.0, 0.0])
        out = format_diagnostics_table("Test method", s)
        self.assertIn("alpha", out)
        self.assertIn("beta", out)
        self.assertIn("rankR^", out)
        self.assertIn("bulk-ESS", out)
        self.assertIn("Overall verdict", out)


class TestMassMatrixSmoke(unittest.TestCase):
    """Smoke test: HMC with adapt_mass_matrix=True runs and produces samples."""

    def test_anisotropic_gaussian(self):
        # Anisotropic target: variances (1.0, 9.0). Identity-M HMC will under-mix
        # the slow direction unless the step size is shrunk; adaptive M should
        # rescale automatically.
        sigmas = tf.constant([1.0, 3.0])

        def log_target(q):
            return -0.5 * tf.reduce_sum((q / sigmas) ** 2)

        initial = tf.constant([0.0, 0.0])
        res = run_hmc(
            log_target,
            initial,
            num_results=200,
            num_burnin=400,
            step_size=0.5,
            num_leapfrog_steps=8,
            target_accept_prob=0.65,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            seed=99,
            verbose=False,
        )
        self.assertEqual(res.samples.shape, (200, 2))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(res.samples)))
        # With adaptation we expect a non-trivial accept rate
        self.assertGreater(float(res.accept_rate), 0.2)
        # And the empirical posterior std on each axis should be in the right ballpark
        emp_std = tf.math.reduce_std(res.samples, axis=0).numpy()
        self.assertGreater(float(emp_std[1]), float(emp_std[0]))


class TestMultiChainRunner(unittest.TestCase):
    """run_hmc_multi_chain spawns the right shape and per-chain seeds differ."""

    def test_shape_and_distinct_chains(self):
        def log_target(q):
            return -0.5 * tf.reduce_sum(q ** 2)

        init = tf.constant([0.0, 0.0])
        inits = disperse_initial_states(init, num_chains=3, scale=0.5, seed=7)
        self.assertEqual(inits.shape, (3, 2))
        res = run_hmc_multi_chain(
            log_target,
            initial_states=inits,
            num_results=50,
            num_burnin=20,
            step_size=0.3,
            num_leapfrog_steps=5,
            target_accept_prob=0.65,
            seed=5,
            adapt_step_size=False,
            adapt_mass_matrix=False,
            verbose=False,
        )
        self.assertEqual(res.samples.shape, (3, 50, 2))
        self.assertEqual(res.is_accepted.shape, (3, 50))
        # Distinct seeds -> chains shouldn't be identical
        s0 = res.samples[0].numpy()
        s1 = res.samples[1].numpy()
        self.assertFalse(np.allclose(s0, s1))


class TestEnsure3D(unittest.TestCase):
    """Single-chain (S, D) inputs should be auto-promoted."""

    def test_single_chain_promote(self):
        x = tf.random.normal([100, 2], seed=33)
        # Should not raise; bulk_ess returns shape (D,)
        b = bulk_ess(x)
        self.assertEqual(b.shape, (2,))
        # split_rhat with one chain returns NaN or similar; tolerant assertion.
        # We just check it runs.
        _ = split_rhat(x)


class TestSharedCRN(unittest.TestCase):
    """Shared CRN across chains: aligned target seeds, distinct sample paths.

    Rationale (cross-link with Report_II_Addendum_Diagnostics §4.5).
    For stochastic targets (LEDH / particle-filter likelihood), each chain
    must see the *same* CRN realisation per iteration; otherwise R-hat
    diagnoses CRN drift between chains rather than MCMC non-convergence.
    """

    @staticmethod
    def _stochastic_target(q):
        # Deterministic mean + tiny stochastic perturbation re-seeded by
        # the global TF seed inside _eval_target_and_grad (via tf.random.set_seed
        # immediately before this call).  Same seed -> same lp value.
        noise = tf.reduce_sum(tf.random.normal([2], stddev=0.01))
        return -0.5 * tf.reduce_sum(q * q) + noise

    def test_shared_crn_aligns_target_seeds_across_chains(self):
        # Capture the crn_seed argument every time _eval_target_and_grad is
        # called (this is the symbol used by run_hmc inside hmc_pf.py).
        original = hmc_pf_module._eval_target_and_grad
        recorded: list[tuple[int, int]] = []  # (chain_index, crn_seed)
        chain_idx_box = {"c": -1}

        def spy(fn, q, crn_seed):
            recorded.append((chain_idx_box["c"], int(crn_seed)))
            return original(fn, q, crn_seed)

        # Wrap run_hmc to bump chain_idx_box around each chain.
        original_run_hmc = hmc_pf_module.run_hmc

        def run_hmc_spy(*args, **kwargs):
            chain_idx_box["c"] += 1
            return original_run_hmc(*args, **kwargs)

        init = tf.constant([0.0, 0.0])
        inits = disperse_initial_states(init, num_chains=3, scale=0.5, seed=7)

        with mock.patch.object(hmc_pf_module, "_eval_target_and_grad", spy), \
             mock.patch.object(hmc_pf_module, "run_hmc", run_hmc_spy):
            res = run_hmc_multi_chain(
                self._stochastic_target,
                initial_states=inits,
                num_results=10,
                num_burnin=5,
                step_size=0.2,
                num_leapfrog_steps=3,
                target_accept_prob=0.65,
                seed=300,
                adapt_step_size=False,
                adapt_mass_matrix=False,
                share_crn_across_chains=True,
                verbose=False,
            )

        # Per-chain seed traces must be (a) non-empty and (b) bit-identical.
        per_chain_seeds: dict[int, list[int]] = {}
        for c_idx, s in recorded:
            per_chain_seeds.setdefault(c_idx, []).append(s)
        self.assertEqual(len(per_chain_seeds), 3)
        chain_seeds = list(per_chain_seeds.values())
        self.assertGreater(len(chain_seeds[0]), 0)
        for cs in chain_seeds[1:]:
            self.assertEqual(cs, chain_seeds[0],
                             "Shared CRN: per-iteration target seeds must "
                             "match across chains.")

        # Sample paths still differ across chains because momentum draws
        # and MH-accept randomness use chain-private base_seed.
        self.assertEqual(res.samples.shape, (3, 10, 2))
        s0 = res.samples[0].numpy()
        s1 = res.samples[1].numpy()
        s2 = res.samples[2].numpy()
        self.assertFalse(np.allclose(s0, s1),
                         "Chains 0 and 1 should not be identical.")
        self.assertFalse(np.allclose(s0, s2),
                         "Chains 0 and 2 should not be identical.")

    def test_unshared_crn_yields_distinct_target_seeds(self):
        # Sanity check: the legacy (per-chain CRN) policy gives different
        # target seeds across chains.  This is the failure mode we are
        # fixing in the multi-chain default.
        original = hmc_pf_module._eval_target_and_grad
        recorded: list[tuple[int, int]] = []
        chain_idx_box = {"c": -1}

        def spy(fn, q, crn_seed):
            recorded.append((chain_idx_box["c"], int(crn_seed)))
            return original(fn, q, crn_seed)

        original_run_hmc = hmc_pf_module.run_hmc

        def run_hmc_spy(*args, **kwargs):
            chain_idx_box["c"] += 1
            return original_run_hmc(*args, **kwargs)

        init = tf.constant([0.0, 0.0])
        inits = disperse_initial_states(init, num_chains=2, scale=0.5, seed=7)

        with mock.patch.object(hmc_pf_module, "_eval_target_and_grad", spy), \
             mock.patch.object(hmc_pf_module, "run_hmc", run_hmc_spy):
            run_hmc_multi_chain(
                self._stochastic_target,
                initial_states=inits,
                num_results=8,
                num_burnin=3,
                step_size=0.2,
                num_leapfrog_steps=3,
                target_accept_prob=0.65,
                seed=300,
                adapt_step_size=False,
                adapt_mass_matrix=False,
                share_crn_across_chains=False,
                verbose=False,
            )

        per_chain_seeds: dict[int, list[int]] = {}
        for c_idx, s in recorded:
            per_chain_seeds.setdefault(c_idx, []).append(s)
        self.assertEqual(len(per_chain_seeds), 2)
        # The first iteration (i=0) of each chain should produce a
        # different crn_seed when CRN is per-chain.
        self.assertNotEqual(
            per_chain_seeds[0][0], per_chain_seeds[1][0],
            "Per-chain CRN: iteration-0 target seeds must differ across chains."
        )


if __name__ == "__main__":
    unittest.main()
