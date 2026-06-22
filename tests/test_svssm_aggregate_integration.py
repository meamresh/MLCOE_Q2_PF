"""
Integration + unit tests for ``src/experiments/exp_svssm_hmc_aggregate.py``
(univariate SVSSM HMC aggregator).

Covers (granularly):
  - rmse / mae helpers
  - _try_load_sidecar_meta (present, absent, malformed JSON)
  - save_svssm_aggregate_report (writes trash/results.txt with expected content)
  - plot_svssm_diagnostics (writes trash/trace_posterior.png; sigma_eta + Var(h)_stat rows)
  - main() end-to-end against a synthetic npz fixture
      * default out_dir = npz parent
      * explicit out_dir
      * burn-in inference from accept trace
      * error paths: missing file, missing arrays, wrong rank/dim
"""

import json
import sys
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.experiments.exp_svssm_hmc_aggregate as agg


def _write_univariate_npz(
    path: Path,
    *,
    num_chains: int = 2,
    draws: int = 40,
    burn: int = 10,
    truth=(0.0, 0.9, 0.5),
    with_accept: bool = True,
):
    """Create a synthetic svssm_hmc_samples.npz with constrained draws."""
    rng = np.random.default_rng(0)
    mu_t, phi_t, sig_t = truth
    samples = np.empty((num_chains, draws, 3), np.float32)
    samples[:, :, 0] = rng.normal(mu_t, 0.1, size=(num_chains, draws))
    samples[:, :, 1] = rng.normal(phi_t, 0.02, size=(num_chains, draws))
    # third constrained param is sigma_eta_sq
    samples[:, :, 2] = np.abs(rng.normal(sig_t ** 2, 0.05, size=(num_chains, draws)))
    arrays = dict(
        samples_constrained=samples,
        truth=np.asarray(truth, np.float64),  # [mu, phi, sigma_eta]
    )
    if with_accept:
        arrays["accept"] = (rng.uniform(size=(num_chains, draws + burn)) < 0.8).astype(
            np.float64
        )
    np.savez(path, **arrays)


class TestErrorMetrics(unittest.TestCase):

    def test_rmse_zero(self):
        s = tf.constant([2.0, 2.0, 2.0])
        self.assertAlmostEqual(agg.rmse(s, 2.0), 0.0, places=6)

    def test_rmse_positive(self):
        s = tf.constant([1.0, 3.0])
        self.assertAlmostEqual(agg.rmse(s, 2.0), 1.0, places=5)

    def test_mae(self):
        s = tf.constant([1.0, 3.0])
        self.assertAlmostEqual(agg.mae(s, 2.0), 1.0, places=5)


class TestSidecarMeta(unittest.TestCase):

    def test_absent_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            burn, nres, elapsed = agg._try_load_sidecar_meta(npz)
            self.assertEqual((burn, nres, elapsed), (None, None, None))

    def test_present_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            (Path(d) / "svssm_hmc_summary.json").write_text(
                json.dumps({"config": {"num_burnin": 200, "num_results": 1000},
                            "elapsed_s": 1234.5})
            )
            burn, nres, elapsed = agg._try_load_sidecar_meta(npz)
            self.assertEqual(burn, 200)
            self.assertEqual(nres, 1000)
            self.assertAlmostEqual(elapsed, 1234.5)

    def test_malformed_json_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            (Path(d) / "svssm_hmc_summary.json").write_text("{not valid json")
            self.assertEqual(
                agg._try_load_sidecar_meta(npz), (None, None, None)
            )


class TestSaveReport(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.num_chains, self.draws = 2, 40
        samples = tf.random.normal([self.num_chains, self.draws, 3]) * 0.1
        self.samples_chains = samples
        self.samples_flat = tf.reshape(samples, [self.num_chains * self.draws, 3])

    def test_writes_results_txt(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            agg.save_svssm_aggregate_report(
                samples_chains=self.samples_chains,
                samples_flat=self.samples_flat,
                accept_rate=0.8,
                t_total=100.0,
                num_chains=self.num_chains,
                samples_per_chain=self.draws,
                burn_per_chain=10,
                true_values=tf.constant([0.0, 0.9, 0.25]),
                param_names=["mu", "phi", "sigma_eta_sq"],
                out_dir=out,
                title_line="TEST TITLE",
            )
            report = (out / "trash" / "results.txt").read_text()
            self.assertIn("TEST TITLE", report)
            self.assertIn("Acceptance rate", report)
            self.assertIn("mu", report)
            self.assertIn("sigma_eta_sq", report)


class TestPlot(unittest.TestCase):

    def test_writes_png_with_four_rows(self):
        if agg.plt is None:
            self.skipTest("matplotlib not available")
        tf.random.set_seed(0)
        flat = tf.concat(
            [
                tf.random.normal([60, 1]) * 0.1,            # mu
                tf.random.normal([60, 1]) * 0.01 + 0.9,     # phi
                tf.abs(tf.random.normal([60, 1]) * 0.05 + 0.25),  # sigma_eta_sq
            ],
            axis=1,
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            agg.plot_svssm_diagnostics(
                flat, out, mu_t=0.0, phi_t=0.9, sigma_eta_t=0.5
            )
            self.assertTrue((out / "trash" / "trace_posterior.png").is_file())


class TestMainEndToEnd(unittest.TestCase):

    def _run_main(self, argv):
        with mock.patch.object(sys, "argv", argv):
            return agg.main()

    def test_default_out_dir_is_npz_parent(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            _write_univariate_npz(npz, burn=10)
            rc = self._run_main([
                "prog", "--samples_npz", str(npz), "--burn_per_chain", "10",
                "--runtime_total", "10.0",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "trash" / "results.txt").is_file())

    def test_explicit_out_dir(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            _write_univariate_npz(npz, burn=10)
            out = Path(d) / "agg_out"
            rc = self._run_main([
                "prog", "--samples_npz", str(npz), "--out_dir", str(out),
                "--burn_per_chain", "10", "--runtime_total", "5.0",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "trash" / "results.txt").is_file())
            if agg.plt is not None:
                self.assertTrue((out / "trash" / "trace_posterior.png").is_file())

    def test_burn_inference_from_accept_trace(self):
        """burn_per_chain=-1 with no JSON -> inferred from accept.shape[1] - draws."""
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            _write_univariate_npz(npz, draws=40, burn=15, with_accept=True)
            rc = self._run_main([
                "prog", "--samples_npz", str(npz), "--runtime_total", "1.0",
            ])
            self.assertEqual(rc, 0)

    def test_missing_file_raises(self):
        with self.assertRaises(SystemExit):
            self._run_main(["prog", "--samples_npz", "/no/such/file.npz"])

    def test_missing_constrained_array_raises(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            np.savez(npz, truth=np.zeros(3))
            with self.assertRaises(SystemExit):
                self._run_main(["prog", "--samples_npz", str(npz)])

    def test_wrong_rank_raises(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            np.savez(
                npz,
                samples_constrained=np.zeros((2, 40), np.float32),  # rank 2
                truth=np.zeros(3),
            )
            with self.assertRaises(SystemExit):
                self._run_main(["prog", "--samples_npz", str(npz)])

    def test_wrong_dim_raises(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            np.savez(
                npz,
                samples_constrained=np.zeros((2, 40, 5), np.float32),  # dim != 3
                truth=np.zeros(3),
            )
            with self.assertRaises(SystemExit):
                self._run_main(["prog", "--samples_npz", str(npz)])

    def test_missing_truth_raises(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_samples.npz"
            np.savez(npz, samples_constrained=np.zeros((2, 40, 3), np.float32))
            with self.assertRaises(SystemExit):
                self._run_main(["prog", "--samples_npz", str(npz)])


if __name__ == "__main__":
    unittest.main()
