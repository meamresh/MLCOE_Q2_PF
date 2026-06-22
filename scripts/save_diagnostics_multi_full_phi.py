"""Compute Vehtari rank-Rhat + bulk-ESS + tail-ESS for a full-Phi
multivariate run, save as both JSON and human-readable .txt.

Operates on the saved samples.npz, so it works for already-finished
runs without rerunning HMC. Computes both:
- Raw parameters: mu, phi_diag, phi_off, sigma_eta_sq
- Derived stationary covariance: Sigma_h diagonals, off-diagonal, rho_h
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "exp"))
from compare_svssm_hmc_methods import rank_rhat, bulk_ess, tail_ess

# Reuse the Smith-doubling implementation from the stationary plot script.
sys.path.insert(0, os.path.dirname(__file__))
from plot_trace_stationary_cov import smith_doubling


def diagnostics(x: np.ndarray, truth: float) -> dict:
    """x: (chains, draws); truth: scalar."""
    flat = x.reshape(-1)
    finite_mask = np.isfinite(x)
    finite_frac = float(finite_mask.mean())
    flat_finite = flat[np.isfinite(flat)]
    med = float(np.nanmedian(flat))
    q025 = float(np.nanpercentile(flat, 2.5))
    q975 = float(np.nanpercentile(flat, 97.5))
    # Replace any NaN with median for diagnostic stats (rank-Rhat etc.
    # need finite values).
    x_clean = np.where(np.isfinite(x), x, med)
    return {
        "truth": float(truth),
        "median": med,
        "q025": q025, "q975": q975,
        "rank_rhat": float(rank_rhat(x_clean)),
        "bulk_ess": float(bulk_ess(x_clean)),
        "tail_ess": float(tail_ess(x_clean)),
        "covered_95ci": bool(q025 <= truth <= q975),
        "finite_frac": finite_frac,
    }


def fmt_row(name: str, r: dict) -> str:
    cov = "Y" if r["covered_95ci"] else "N"
    return (f"{name:<24}{r['truth']:>+10.4f}{r['median']:>+11.4f}"
            f"{r['q025']:>+11.4f}{r['q975']:>+11.4f}"
            f"{r['rank_rhat']:>11.4f}{r['bulk_ess']:>10.0f}"
            f"{r['tail_ess']:>10.0f}{cov:>5}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--rhat_threshold", type=float, default=1.05,
                   help="Relaxed Vehtari rank-Rhat threshold for PASS.")
    p.add_argument("--ess_threshold", type=float, default=400.0,
                   help="Stan-style bulk/tail-ESS floor for strict PASS.")
    args = p.parse_args()

    # Load the run config (truth + priors + HMC settings) from the
    # summary.json so the diagnostics file is self-documenting -- you
    # never have to dig into another file to know what produced it.
    cfg = None
    summ_path = os.path.join(args.out_dir,
                               "svssm_hmc_multi_full_phi_summary.json")
    if os.path.isfile(summ_path):
        try:
            sj = json.load(open(summ_path))
            # Stitched summaries nest the per-chain config under per_chain_summaries;
            # both the single-run and stitched forms expose a top-level "config".
            cfg = sj.get("config")
            if cfg is None and "per_chain_summaries" in sj and sj["per_chain_summaries"]:
                cfg = sj["per_chain_summaries"][0]["summary"]["config"]
        except Exception:
            cfg = None

    npz = np.load(os.path.join(args.out_dir,
                                  "svssm_hmc_multi_full_phi_samples.npz"))
    mu = npz["mu"]                  # (chains, draws, d)
    phi_d = npz["phi_diag"]         # (chains, draws, d)
    phi_o = npz["phi_off"]          # (chains, draws, d(d-1)/2)
    sig2 = npz["sigma_eta_sq"]      # (chains, draws, d)
    mu_truth = npz["mu_truth"]
    phi_d_truth = npz["phi_diag_truth"]
    phi_o_truth = npz["phi_off_truth"]
    sig2_truth = npz["sigma_eta_sq_truth"]
    Phi_truth = npz["Phi_truth"]

    chains, draws, d = mu.shape
    # Upper-triangular off-diagonal index order (row-major), matching
    # build_phi_matrix and the operator context layout.
    off_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]

    # --- Raw params (d-generic) ---
    raw_rows = []
    for i in range(d):
        raw_rows.append((f"mu_{i}", diagnostics(mu[..., i], mu_truth[i])))
    for i in range(d):
        raw_rows.append((f"phi_diag_{i}",
                         diagnostics(phi_d[..., i], phi_d_truth[i])))
    for k, (i, j) in enumerate(off_idx):
        raw_rows.append((f"phi_off_{i}{j}",
                         diagnostics(phi_o[..., k], phi_o_truth[k])))
    for i in range(d):
        raw_rows.append((f"sigma_eta_sq_{i}",
                         diagnostics(sig2[..., i], sig2_truth[i])))

    # --- Derived stationary covariance Sigma_h (d-generic) ---
    Phi = np.zeros((chains, draws, d, d), dtype=np.float64)
    for i in range(d):
        Phi[..., i, i] = phi_d[..., i]
    for k, (i, j) in enumerate(off_idx):
        Phi[..., i, j] = phi_o[..., k]
    Sigma_eta = np.zeros((chains, draws, d, d), dtype=np.float64)
    for i in range(d):
        Sigma_eta[..., i, i] = sig2[..., i]
    eig = np.linalg.eigvals(Phi)
    stat_mask = np.max(np.abs(eig), axis=-1) < 1.0
    Sigma_h = smith_doubling(Phi, Sigma_eta, n_doublings=15)
    Sigma_h[~stat_mask] = np.nan

    Sigma_eta_truth = np.diag(np.asarray(sig2_truth, dtype=np.float64))
    Sigma_h_truth = smith_doubling(Phi_truth.astype(np.float64),
                                     Sigma_eta_truth, n_doublings=15)

    derived_rows = []
    # diagonal stationary variances
    for i in range(d):
        derived_rows.append((f"Sigma_h_{i}{i}",
                             diagnostics(Sigma_h[..., i, i],
                                          float(Sigma_h_truth[i, i]))))
    # off-diagonal stationary covariances + correlations
    for (i, j) in off_idx:
        sij = Sigma_h[..., i, j]
        derived_rows.append((f"Sigma_h_{i}{j}",
                             diagnostics(sij, float(Sigma_h_truth[i, j]))))
        rho = sij / np.sqrt(np.maximum(
            Sigma_h[..., i, i] * Sigma_h[..., j, j], 1e-12))
        rho_truth = float(Sigma_h_truth[i, j] / np.sqrt(
            Sigma_h_truth[i, i] * Sigma_h_truth[j, j]))
        derived_rows.append((f"rho_h_{i}{j}", diagnostics(rho, rho_truth)))

    # --- Verdicts ---
    def pass_relaxed(r):
        return (r["rank_rhat"] <= args.rhat_threshold and r["covered_95ci"])

    def pass_strict(r):
        return (r["rank_rhat"] <= 1.01 and
                r["bulk_ess"] >= args.ess_threshold and
                r["tail_ess"] >= args.ess_threshold and
                r["covered_95ci"])

    # --- Format header ---
    header = (f"{'quantity':<24}{'truth':>10}{'median':>11}"
              f"{'2.5%':>11}{'97.5%':>11}{'rank-Rhat':>11}"
              f"{'bulk-ESS':>10}{'tail-ESS':>10}{'cov':>5}")
    sep = "-" * len(header)

    lines = []
    lines.append("Vehtari diagnostics — full-Phi V1 multivariate HMC")
    lines.append(f"Run: {args.out_dir}")
    lines.append(f"chains={chains}  draws={draws}  d={d}")
    lines.append(f"Thresholds: relaxed rank-Rhat <= {args.rhat_threshold}; "
                 f"strict rank-Rhat <= 1.01 AND bulk/tail-ESS >= "
                 f"{args.ess_threshold:.0f}")
    lines.append("")
    # ---- Self-documenting RUN CONFIGURATION block (truth + priors) ----
    lines.append("RUN CONFIGURATION")
    lines.append(sep)
    # Truth comes straight from the npz (always present), priors from cfg.
    lines.append(f"  truth mu          : {np.asarray(mu_truth).tolist()}")
    lines.append(f"  truth phi_diag    : {np.asarray(phi_d_truth).tolist()}")
    lines.append(f"  truth phi_off     : {np.asarray(phi_o_truth).tolist()}")
    lines.append(f"  truth sigma_eta_sq: {np.asarray(sig2_truth).tolist()}")
    if cfg is not None:
        lines.append(f"  T={cfg.get('T')}  N={cfg.get('N')}  L={cfg.get('L')}  "
                     f"n_lambda={cfg.get('n_lambda')}  step_size={cfg.get('step_size')}  "
                     f"num_burnin={cfg.get('num_burnin')}  num_results={cfg.get('num_results')}")
        lines.append(f"  prior mu       ~ N({cfg.get('prior_mu_loc')}, "
                     f"{cfg.get('prior_mu_scale')}^2)")
        lines.append(f"  prior phi_raw  ~ N({cfg.get('prior_phi_raw_loc')}, "
                     f"{cfg.get('prior_phi_raw_scale')}^2)  [phi_diag = tanh(phi_raw)]")
        lines.append(f"  prior phi_off  ~ N(0, {cfg.get('prior_phi_off_scale')}^2)")
        lines.append(f"  prior log_sig2 ~ N({cfg.get('prior_log_sigma_eta_sq_loc')}, "
                     f"{cfg.get('prior_log_sigma_eta_sq_scale')}^2)")
        lines.append(f"  data_seed={cfg.get('data_seed')}  base_seed={cfg.get('base_seed')}  "
                     f"dense_mass={not cfg.get('diagonal_mass', False)}")
    else:
        lines.append("  (priors unavailable: summary.json with config not found "
                     "in this dir; truth above is from the samples npz)")
    lines.append("")
    lines.append("RAW PARAMETERS")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for name, r in raw_rows:
        lines.append(fmt_row(name, r))
    lines.append(sep)
    lines.append("")
    lines.append("DERIVED STATIONARY COVARIANCE Sigma_h "
                 "= Phi Sigma_h Phi^T + Sigma_eta")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for name, r in derived_rows:
        lines.append(fmt_row(name, r))
    lines.append(sep)
    lines.append("")
    raw_relaxed = sum(pass_relaxed(r) for _, r in raw_rows)
    raw_strict = sum(pass_strict(r) for _, r in raw_rows)
    der_relaxed = sum(pass_relaxed(r) for _, r in derived_rows)
    der_strict = sum(pass_strict(r) for _, r in derived_rows)
    lines.append(f"Raw params:     {raw_relaxed}/{len(raw_rows)} pass relaxed, "
                 f"{raw_strict}/{len(raw_rows)} pass strict")
    lines.append(f"Derived (Sigma_h): {der_relaxed}/{len(derived_rows)} pass "
                 f"relaxed, {der_strict}/{len(derived_rows)} pass strict")

    txt = "\n".join(lines) + "\n"
    txt_path = os.path.join(args.out_dir,
                              "svssm_hmc_multi_full_phi_diagnostics.txt")
    with open(txt_path, "w") as f:
        f.write(txt)

    json_path = os.path.join(args.out_dir,
                                "svssm_hmc_multi_full_phi_diagnostics.json")
    out = {
        "chains": chains, "draws": draws, "d": d,
        "truth": {
            "mu": np.asarray(mu_truth).tolist(),
            "phi_diag": np.asarray(phi_d_truth).tolist(),
            "phi_off": np.asarray(phi_o_truth).tolist(),
            "sigma_eta_sq": np.asarray(sig2_truth).tolist(),
        },
        "config": cfg,   # full run config incl. priors + HMC settings (None if not found)
        "thresholds": {"rhat_relaxed": args.rhat_threshold,
                        "rhat_strict": 1.01, "ess": args.ess_threshold},
        "raw": {name: r for name, r in raw_rows},
        "derived_stationary": {name: r for name, r in derived_rows},
        "summary": {
            "raw_pass_relaxed": raw_relaxed, "raw_pass_strict": raw_strict,
            "derived_pass_relaxed": der_relaxed,
            "derived_pass_strict": der_strict,
            "raw_total": len(raw_rows), "derived_total": len(derived_rows),
        },
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(txt)
    print(f"saved {txt_path}")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
