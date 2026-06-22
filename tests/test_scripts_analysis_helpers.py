"""
Unit tests for the pure analysis/diagnostics helpers in the experiment
scripts (these are importable, side-effect-free numeric functions; the
surrounding CLI ``main()`` drivers and plotting are out of scope).

Modules under test (loaded via importlib since ``scripts/`` is not a package):
  - scripts/exp/analyze_v2_mv_vehtari.py
      _split_chains, gelman_rubin, autocorr_via_fft, ess_geyer,
      split_rhat_bulk_ess, tail_ess, diagnostics_for
  - scripts/exp/analyze_phase19.py
      summarize_marginals, shift_table
"""

import importlib.util
import unittest
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent


def _load(module_name: str, rel_path: str):
    path = _REPO / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


vehtari = _load("_analyze_v2_mv_vehtari", "scripts/exp/analyze_v2_mv_vehtari.py")
phase19 = _load("_analyze_phase19", "scripts/exp/analyze_phase19.py")


# ---------------------------------------------------------------------------
# analyze_v2_mv_vehtari helpers
# ---------------------------------------------------------------------------

class TestSplitChains(unittest.TestCase):

    def test_even_length(self):
        chains = np.arange(2 * 10).reshape(2, 10)
        out = vehtari._split_chains(chains)
        self.assertEqual(out.shape, (4, 5))

    def test_odd_length_truncates(self):
        chains = np.arange(2 * 11).reshape(2, 11)
        out = vehtari._split_chains(chains)
        self.assertEqual(out.shape, (4, 5))  # 11//2 = 5

    def test_first_half_preserved(self):
        chains = np.array([[0, 1, 2, 3]])
        out = vehtari._split_chains(chains)
        np.testing.assert_array_equal(out[0], [0, 1])
        np.testing.assert_array_equal(out[1], [2, 3])


class TestGelmanRubin(unittest.TestCase):

    def test_well_mixed_near_one(self):
        rng = np.random.default_rng(0)
        chains = rng.normal(size=(4, 2000))
        rhat = vehtari.gelman_rubin(chains)
        self.assertTrue(0.98 <= rhat <= 1.05)

    def test_separated_chains_large(self):
        rng = np.random.default_rng(1)
        # Two clusters of chains with very different means -> poor mixing.
        c = np.concatenate([
            rng.normal(-5.0, 0.1, size=(2, 1000)),
            rng.normal(+5.0, 0.1, size=(2, 1000)),
        ], axis=0)
        rhat = vehtari.gelman_rubin(c)
        self.assertGreater(rhat, 1.2)

    def test_returns_float(self):
        self.assertIsInstance(
            vehtari.gelman_rubin(np.random.default_rng(2).normal(size=(2, 100))),
            float,
        )


class TestAutocorrViaFFT(unittest.TestCase):

    def test_lag0_is_one(self):
        rng = np.random.default_rng(0)
        acf = vehtari.autocorr_via_fft(rng.normal(size=512))
        self.assertAlmostEqual(acf[0], 1.0, places=6)

    def test_white_noise_decorrelated(self):
        rng = np.random.default_rng(3)
        acf = vehtari.autocorr_via_fft(rng.normal(size=4096))
        # Lags >= 1 should be small for white noise.
        self.assertLess(np.max(np.abs(acf[1:50])), 0.15)

    def test_length(self):
        x = np.zeros(64)
        x[0] = 1.0
        acf = vehtari.autocorr_via_fft(x + np.arange(64) * 1e-9)
        self.assertEqual(acf.shape, (64,))


class TestESSGeyer(unittest.TestCase):

    def test_iid_ess_near_total(self):
        rng = np.random.default_rng(0)
        chains = rng.normal(size=(4, 2000))
        ess = vehtari.ess_geyer(chains)
        total = 4 * 2000
        # iid -> ESS should be a large fraction of the sample count.
        self.assertGreater(ess, 0.5 * total)
        self.assertLessEqual(ess, total * 1.2)

    def test_positive(self):
        rng = np.random.default_rng(5)
        self.assertGreater(vehtari.ess_geyer(rng.normal(size=(2, 500))), 0.0)


class TestSplitRhatBulkESS(unittest.TestCase):

    def test_tuple_shape(self):
        rng = np.random.default_rng(0)
        rhat, bulk = vehtari.split_rhat_bulk_ess(rng.normal(size=(4, 1000)))
        self.assertTrue(np.isfinite(rhat))
        self.assertGreater(bulk, 0.0)


class TestTailESS(unittest.TestCase):

    def test_positive_finite(self):
        rng = np.random.default_rng(0)
        t = vehtari.tail_ess(rng.normal(size=(4, 1000)))
        self.assertTrue(np.isfinite(t))
        self.assertGreater(t, 0.0)


class TestDiagnosticsFor(unittest.TestCase):

    def test_rows_and_keys(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=(4, 500, 2))
        rows = vehtari.diagnostics_for(samples, ["a", "b"])
        self.assertEqual(len(rows), 2)
        for r in rows:
            for k in ("param", "median", "sd", "rhat", "bulk_ess",
                      "tail_ess", "rhat_ok", "bulk_ok", "tail_ok"):
                self.assertIn(k, r)
        self.assertEqual(rows[0]["param"], "a")
        self.assertEqual(rows[1]["param"], "b")


# ---------------------------------------------------------------------------
# analyze_phase19 helpers
# ---------------------------------------------------------------------------

class TestSummarizeMarginals(unittest.TestCase):

    def test_keys_sigma_eta(self):
        rng = np.random.default_rng(0)
        samples = np.abs(rng.normal(size=(2, 200, 3))) + 0.1
        out = phase19.summarize_marginals(samples, sigma_eta_sq=False)
        self.assertEqual(set(out.keys()), {"mu", "phi", "sigma_eta"})
        for stats in out.values():
            for k in ("median", "mean", "sd", "q05", "q95"):
                self.assertIn(k, stats)

    def test_keys_sigma_eta_sq(self):
        rng = np.random.default_rng(0)
        samples = np.abs(rng.normal(size=(2, 200, 3))) + 0.1
        out = phase19.summarize_marginals(samples, sigma_eta_sq=True)
        self.assertIn("sigma_eta_sq", out)
        self.assertNotIn("sigma_eta", out)

    def test_sqrt_conversion(self):
        """With sigma_eta_sq=False the third column is sqrt'd before summary."""
        samples = np.empty((1, 4, 3), np.float64)
        samples[..., 0] = 0.0
        samples[..., 1] = 0.5
        samples[..., 2] = 4.0          # variance
        out = phase19.summarize_marginals(samples, sigma_eta_sq=False)
        self.assertAlmostEqual(out["sigma_eta"]["median"], 2.0, places=6)

    def test_no_sqrt_when_sq(self):
        samples = np.empty((1, 4, 3), np.float64)
        samples[..., 0] = 0.0
        samples[..., 1] = 0.5
        samples[..., 2] = 4.0
        out = phase19.summarize_marginals(samples, sigma_eta_sq=True)
        self.assertAlmostEqual(out["sigma_eta_sq"]["median"], 4.0, places=6)


class TestShiftTable(unittest.TestCase):

    def test_shift_and_ci_shrink(self):
        no_data = {
            "mu": {"median": 0.0, "sd": 1.0, "q05": -2.0, "q95": 2.0},
        }
        with_data = {
            "mu": {"median": 1.0, "sd": 0.5, "q05": 0.0, "q95": 1.0},
        }
        rows = phase19.shift_table(no_data, with_data, params=("mu",))
        self.assertEqual(len(rows), 1)
        p, nd_med, wd_med, shift, ci_nd, ci_wd, ci_shrink = rows[0]
        self.assertEqual(p, "mu")
        self.assertAlmostEqual(nd_med, 0.0)
        self.assertAlmostEqual(wd_med, 1.0)
        self.assertAlmostEqual(shift, 1.0, places=6)        # (1-0)/1
        self.assertAlmostEqual(ci_nd, 4.0, places=6)        # 2-(-2)
        self.assertAlmostEqual(ci_wd, 1.0, places=6)        # 1-0
        self.assertAlmostEqual(ci_shrink, 75.0, places=4)   # (4-1)/4*100

    def test_zero_sd_guarded(self):
        no_data = {"phi": {"median": 0.0, "sd": 0.0, "q05": 0.0, "q95": 0.0}}
        with_data = {"phi": {"median": 0.5, "sd": 0.1, "q05": 0.4, "q95": 0.6}}
        rows = phase19.shift_table(no_data, with_data, params=("phi",))
        self.assertTrue(np.isfinite(rows[0][3]))   # shift uses max(sd, 1e-9)


if __name__ == "__main__":
    unittest.main()
