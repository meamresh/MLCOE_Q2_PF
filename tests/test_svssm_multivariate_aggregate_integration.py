"""
Integration + unit tests for
``src/experiments/exp_svssm_hmc_multivariate_aggregate.py``.

Covers (granularly):
  - rmse / mae helpers
  - _build_param_names ordering (mu_*, phi_*, sigma_eta_sq_*)
  - _try_load_sidecar_meta (multi sidecar, univariate fallback, malformed)
  - save_multi_aggregate_report (writes trash/results.txt; flattened 3d names)
  - plot_multi_diagnostics (writes trace_posterior_multivariate.png; 4d rows)
  - main() end-to-end against a synthetic rank-4 npz fixture
      * default + explicit out_dir
      * error paths: missing file/array, wrong rank, wrong truth shape
"""

from __future__ import annotations

import json
import sys
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.experiments.exp_svssm_hmc_multivariate_aggregate as magg


def _write_multivariate_npz(
    path: Path,
    *,
    d: int = 2,
    num_chains: int = 2,
    draws: int = 40,
    burn: int = 10,
    with_accept: bool = True,
    sidecar: str | None = None,
):
    """Synthetic svssm_hmc_multi_samples.npz: samples (chains, draws, 3, d)."""
    rng = np.random.default_rng(1)
    mu = np.linspace(-0.2, 0.2, d)
    phi = np.linspace(0.85, 0.92, d)
    sig_sq = np.linspace(0.2, 0.3, d)
    truth = np.stack([mu, phi, sig_sq], axis=0).astype(np.float32)  # (3, d)

    samples = np.empty((num_chains, draws, 3, d), np.float32)
    for c in range(d):
        samples[:, :, 0, c] = rng.normal(mu[c], 0.05, (num_chains, draws))
        samples[:, :, 1, c] = rng.normal(phi[c], 0.01, (num_chains, draws))
        samples[:, :, 2, c] = np.abs(rng.normal(sig_sq[c], 0.03, (num_chains, draws)))

    arrays = dict(samples_constrained=samples, truth=truth)
    if with_accept:
        arrays["accept"] = (
            rng.uniform(size=(num_chains, draws + burn)) < 0.8
        ).astype(np.float64)
    np.savez(path, **arrays)

    if sidecar is not None:
        (path.with_name(sidecar)).write_text(
            json.dumps({"config": {"num_burnin": burn, "num_results": draws},
                        "elapsed_s": 99.0})
        )


class TestErrorMetrics(unittest.TestCase):

    def test_rmse(self):
        self.assertAlmostEqual(magg.rmse(tf.constant([1.0, 3.0]), 2.0), 1.0, places=5)

    def test_mae(self):
        self.assertAlmostEqual(magg.mae(tf.constant([0.0, 4.0]), 2.0), 2.0, places=5)


class TestBuildParamNames(unittest.TestCase):

    def test_d1(self):
        self.assertEqual(
            magg._build_param_names(1), ["mu_0", "phi_0", "sigma_eta_sq_0"]
        )

    def test_d2_ordering(self):
        names = magg._build_param_names(2)
        self.assertEqual(
            names,
            ["mu_0", "mu_1", "phi_0", "phi_1", "sigma_eta_sq_0", "sigma_eta_sq_1"],
        )

    def test_length_is_3d(self):
        self.assertEqual(len(magg._build_param_names(3)), 9)


class TestSidecarMeta(unittest.TestCase):

    def test_multi_sidecar_preferred(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_multi_samples.npz"
            (npz.with_name("svssm_hmc_multi_summary.json")).write_text(
                json.dumps({"config": {"num_burnin": 50, "num_results": 500},
                            "elapsed_s": 12.0})
            )
            burn, nres, elapsed = magg._try_load_sidecar_meta(npz)
            self.assertEqual((burn, nres), (50, 500))
            self.assertAlmostEqual(elapsed, 12.0)

    def test_univariate_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_multi_samples.npz"
            (npz.with_name("svssm_hmc_summary.json")).write_text(
                json.dumps({"config": {"num_burnin": 33}, "elapsed_s": 7.0})
            )
            burn, _, elapsed = magg._try_load_sidecar_meta(npz)
            self.assertEqual(burn, 33)
            self.assertAlmostEqual(elapsed, 7.0)

    def test_absent(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_multi_samples.npz"
            self.assertEqual(magg._try_load_sidecar_meta(npz), (None, None, None))


class TestSaveReport(unittest.TestCase):

    def test_writes_results_txt(self):
        tf.random.set_seed(0)
        d, num_chains, draws = 2, 2, 40
        flat3d = 3 * d
        samples = tf.random.normal([num_chains, draws, flat3d]) * 0.1
        flat = tf.reshape(samples, [num_chains * draws, flat3d])
        names = magg._build_param_names(d)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            magg.save_multi_aggregate_report(
                samples_chains=samples,
                samples_flat=flat,
                accept_rate=0.8,
                t_total=10.0,
                num_chains=num_chains,
                samples_per_chain=draws,
                burn_per_chain=10,
                true_values=tf.zeros([flat3d]),
                param_names=names,
                d=d,
                out_dir=out,
                title_line="MULTI TITLE",
            )
            report = (out / "trash" / "results.txt").read_text()
            self.assertIn("MULTI TITLE", report)
            self.assertIn("d=2", report)
            self.assertIn("phi_1", report)


class TestPlot(unittest.TestCase):

    def test_writes_png(self):
        if magg.plt is None:
            self.skipTest("matplotlib not available")
        tf.random.set_seed(0)
        d = 2
        flat3d = 3 * d
        cols = []
        for _ in range(d):  # mu
            cols.append(tf.random.normal([50, 1]) * 0.05)
        for _ in range(d):  # phi
            cols.append(tf.random.normal([50, 1]) * 0.01 + 0.9)
        for _ in range(d):  # sigma_eta_sq
            cols.append(tf.abs(tf.random.normal([50, 1]) * 0.03 + 0.25))
        flat = tf.concat(cols, axis=1)
        truth_np = np.array([[0.0, 0.1], [0.9, 0.88], [0.25, 0.2]], np.float32)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            magg.plot_multi_diagnostics(flat, out, d=d, truth_np=truth_np)
            self.assertTrue(
                (out / "trash" / "trace_posterior_multivariate.png").is_file()
            )


class TestMainEndToEnd(unittest.TestCase):

    def _run_main(self, argv):
        with mock.patch.object(sys, "argv", argv):
            return magg.main()

    def test_default_out_dir(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_multi_samples.npz"
            _write_multivariate_npz(npz, d=2, burn=10)
            rc = self._run_main([
                "prog", "--samples_npz", str(npz),
                "--burn_per_chain", "10", "--runtime_total", "10.0",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "trash" / "results.txt").is_file())

    def test_explicit_out_dir_and_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_multi_samples.npz"
            _write_multivariate_npz(
                npz, d=2, burn=10, sidecar="svssm_hmc_multi_summary.json"
            )
            out = Path(d) / "out"
            rc = self._run_main([
                "prog", "--samples_npz", str(npz), "--out_dir", str(out),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "trash" / "results.txt").is_file())
            if magg.plt is not None:
                self.assertTrue(
                    (out / "trash" / "trace_posterior_multivariate.png").is_file()
                )

    def test_d3_runs(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_multi_samples.npz"
            _write_multivariate_npz(npz, d=3, burn=10)
            rc = self._run_main([
                "prog", "--samples_npz", str(npz),
                "--burn_per_chain", "10", "--runtime_total", "1.0",
            ])
            self.assertEqual(rc, 0)

    def test_missing_file_raises(self):
        with self.assertRaises(SystemExit):
            self._run_main(["prog", "--samples_npz", "/no/such.npz"])

    def test_rank3_rejected(self):
        """Univariate-shaped npz must be rejected by the multivariate aggregator."""
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_multi_samples.npz"
            np.savez(
                npz,
                samples_constrained=np.zeros((2, 40, 3), np.float32),  # rank 3
                truth=np.zeros((3, 2), np.float32),
            )
            with self.assertRaises(SystemExit):
                self._run_main(["prog", "--samples_npz", str(npz)])

    def test_wrong_truth_shape_raises(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "svssm_hmc_multi_samples.npz"
            np.savez(
                npz,
                samples_constrained=np.zeros((2, 40, 3, 2), np.float32),
                truth=np.zeros((3, 3), np.float32),  # mismatched d
            )
            with self.assertRaises(SystemExit):
                self._run_main(["prog", "--samples_npz", str(npz)])


if __name__ == "__main__":
    unittest.main()
