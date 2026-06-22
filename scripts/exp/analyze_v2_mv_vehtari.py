"""
Post-hoc Vehtari diagnostics for the multivariate V2 demo.

Loads `v2_mv_samples.npz` from a completed run of
`exp_v2_multivariate_demo.py` and computes per-marginal:
  - split-Rhat (rank-normalized, Vehtari et al. 2021)
  - bulk-ESS
  - tail-ESS

Vehtari thresholds:
  - rank-Rhat <= 1.01
  - bulk-ESS  >= 400  (for percentile estimation)
  - tail-ESS  >= 400  (for tail quantiles)

Also reproduces the ridge-collapse summary table with the larger
budget and prints any marginal that fails any Vehtari threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """Split each chain in half. (M, N) -> (2M, N//2)."""
    M, N = chains.shape
    half = N // 2
    return chains[:, :2 * half].reshape(M, 2, half).reshape(2 * M, half)


def gelman_rubin(chains: np.ndarray) -> float:
    """Split-Rhat (Vehtari et al. 2021 rank-Rhat variant uses the same
    base formula on rank-normalized samples; here we use plain split-Rhat).
    chains: (M, N) where M is num chains and N is num draws.
    Returns Rhat (>= 1).
    """
    split = _split_chains(chains)  # (2M, N//2)
    M2, N2 = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    grand_mean = chain_means.mean()
    B = N2 * chain_means.var(ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        return float("nan")
    var_hat = ((N2 - 1) / N2) * W + B / N2
    return float(np.sqrt(var_hat / W))


def autocorr_via_fft(x: np.ndarray) -> np.ndarray:
    """Autocorrelation function of 1D series x via FFT. Returns rho_0..rho_{N-1}."""
    x = x - x.mean()
    N = len(x)
    f = np.fft.fft(x, n=2 * N)
    acf = np.fft.ifft(f * np.conj(f)).real[:N]
    acf /= acf[0]
    return acf


def ess_geyer(chains: np.ndarray) -> float:
    """ESS via Geyer's initial monotone sequence estimator (the Stan/Vehtari
    standard).
    chains: (M, N). Returns scalar ESS.
    """
    M, N = chains.shape
    # Compute autocorrelation per chain, average over chains
    acfs = np.stack([autocorr_via_fft(chains[m]) for m in range(M)], axis=0)
    rho = acfs.mean(axis=0)  # (N,)
    # Sum of paired autocorrelations rho_{2k} + rho_{2k+1}, truncating at first
    # non-positive pair (Geyer 1992).
    K = (N - 1) // 2
    paired = rho[1:2 * K + 1].reshape(K, 2).sum(axis=1)
    # Find first non-positive paired sum
    neg_idx = np.where(paired <= 0)[0]
    cutoff = neg_idx[0] if len(neg_idx) > 0 else K
    if cutoff == 0:
        tau = 1.0
    else:
        tau = 1.0 + 2.0 * paired[:cutoff].sum()
    if tau <= 0:
        return float(M * N)
    return float(M * N / tau)


def split_rhat_bulk_ess(chains: np.ndarray):
    """chains: (num_chains, num_draws) — single marginal."""
    rhat = gelman_rubin(chains)
    bulk = ess_geyer(_split_chains(chains))  # bulk-ESS uses split chains
    return rhat, bulk


def tail_ess(chains: np.ndarray, lower_q: float = 0.05,
             upper_q: float = 0.95):
    """Tail-ESS = min(ESS of I{x<=q_5%}, ESS of I{x>=q_95%}).
    Vehtari et al. (2021) recommendation."""
    flat = chains.reshape(-1)
    q_lo = np.quantile(flat, lower_q)
    q_hi = np.quantile(flat, upper_q)
    ind_lo = (chains <= q_lo).astype(np.float32)
    ind_hi = (chains >= q_hi).astype(np.float32)
    ess_lo = ess_geyer(_split_chains(ind_lo))
    ess_hi = ess_geyer(_split_chains(ind_hi))
    return min(ess_lo, ess_hi)


def diagnostics_for(samples_const: np.ndarray, names: list[str]):
    """samples_const: (chains, draws, dim). Returns list of dicts."""
    rows = []
    for j, n in enumerate(names):
        marg = samples_const[..., j]  # (chains, draws)
        rhat, bulk = split_rhat_bulk_ess(marg)
        t_ess = tail_ess(marg)
        rows.append({
            "param": n,
            "median": float(np.median(marg)),
            "sd": float(np.std(marg)),
            "rhat": rhat,
            "bulk_ess": bulk,
            "tail_ess": t_ess,
            "rhat_ok": rhat <= 1.01,
            "bulk_ok": bulk >= 400,
            "tail_ok": t_ess >= 400,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "v2_multivariate_demo_long")
    args = p.parse_args()
    in_dir = Path(args.in_dir)
    npz_path = in_dir / "v2_mv_samples.npz"
    if not npz_path.exists():
        print(f"[error] samples not found at {npz_path}")
        return 1

    data = np.load(npz_path)
    free_const = data["free_const"]    # (chains, draws, 12)
    fixed_const = data["fixed_const"]  # (chains, draws, 8)

    free_names = ["mu_1", "mu_2", "phi_1", "phi_2",
                  "sigma_eta_1_sq", "sigma_eta_2_sq",
                  "A_11", "A_12", "A_21", "A_22",
                  "sigma_eps_1_sq", "sigma_eps_2_sq"]
    fixed_names = ["mu_1", "mu_2", "phi_1", "phi_2",
                   "sigma_eta_1_sq", "sigma_eta_2_sq",
                   "sigma_eps_1_sq", "sigma_eps_2_sq"]

    print(f"FREE: shape={free_const.shape}, "
          f"FIXED: shape={fixed_const.shape}\n")

    print("=" * 95)
    print("FREE A (12-param) — Vehtari diagnostics")
    print("=" * 95)
    print(f"{'param':>20}  {'median':>10}  {'sd':>10}  "
          f"{'rhat':>8}  {'bulk_ess':>10}  {'tail_ess':>10}  flags")
    print("-" * 95)
    free_rows = diagnostics_for(free_const, free_names)
    for r in free_rows:
        flags = []
        if not r["rhat_ok"]: flags.append("rhat")
        if not r["bulk_ok"]: flags.append("bulk")
        if not r["tail_ok"]: flags.append("tail")
        flag_str = ",".join(flags) if flags else "OK"
        print(f"{r['param']:>20}  {r['median']:10.4f}  {r['sd']:10.4f}  "
              f"{r['rhat']:8.4f}  {r['bulk_ess']:10.1f}  "
              f"{r['tail_ess']:10.1f}  {flag_str}")

    print()
    print("=" * 95)
    print("FIXED A=I (8-param) — Vehtari diagnostics")
    print("=" * 95)
    print(f"{'param':>20}  {'median':>10}  {'sd':>10}  "
          f"{'rhat':>8}  {'bulk_ess':>10}  {'tail_ess':>10}  flags")
    print("-" * 95)
    fixed_rows = diagnostics_for(fixed_const, fixed_names)
    for r in fixed_rows:
        flags = []
        if not r["rhat_ok"]: flags.append("rhat")
        if not r["bulk_ok"]: flags.append("bulk")
        if not r["tail_ok"]: flags.append("tail")
        flag_str = ",".join(flags) if flags else "OK"
        print(f"{r['param']:>20}  {r['median']:10.4f}  {r['sd']:10.4f}  "
              f"{r['rhat']:8.4f}  {r['bulk_ess']:10.1f}  "
              f"{r['tail_ess']:10.1f}  {flag_str}")

    # ---- Ridge collapse summary (sd FREE / sd FIXED) ----
    print()
    print("=" * 60)
    print("Ridge collapse: sd(FREE) / sd(FIXED)")
    print("=" * 60)
    for n in fixed_names:
        fr = next(r for r in free_rows if r["param"] == n)
        fx = next(r for r in fixed_rows if r["param"] == n)
        ratio = fr["sd"] / fx["sd"] if fx["sd"] > 0 else float("nan")
        print(f"  {n:>20}: free_sd={fr['sd']:8.4f}  fixed_sd={fx['sd']:8.4f}  "
              f"ratio={ratio:6.2f}x")
    # ---- A entries (FREE only) ----
    print()
    print("A matrix entries (FREE only — ridge fingerprint):")
    for n in ["A_11", "A_12", "A_21", "A_22"]:
        fr = next(r for r in free_rows if r["param"] == n)
        print(f"  {n:>5}: median={fr['median']:8.4f}  sd={fr['sd']:8.4f}")

    # Save
    out = {
        "free_diagnostics": free_rows,
        "fixed_diagnostics": fixed_rows,
    }
    with open(in_dir / "v2_mv_vehtari.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {in_dir}/v2_mv_vehtari.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
