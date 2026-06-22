"""Trace plot of the *derived* stationary covariance Sigma_h
for the full-Phi V1 multivariate run.

For each posterior sample (Phi, Sigma_eta) we solve
    Sigma_h = Phi Sigma_h Phi^T + Sigma_eta
via Smith doubling (matches what the filter uses for h_0). The
diagonal entries are the marginal stationary variances of each
latent dimension --- the quantity the data identifies sharpest
in stochastic-volatility-style models (in univariate SVSSM the
(phi, sigma_eta) ridge collapses to a clean estimate of
sigma_h^2 = sigma_eta^2 / (1 - phi^2)). For upper-triangular Phi
at d=2:

  Sigma_h[1,1] = sigma_eta,1^2 / (1 - phi_11^2)
                 (second row has no off-diagonal --> univariate)
  Sigma_h[0,0] = coupled function of all 5 params via spillover
  rho_h        = Sigma_h[0,1] / sqrt(Sigma_h[0,0] Sigma_h[1,1])
                 = cross-asset stationary correlation
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "exp"))
from compare_svssm_hmc_methods import rank_rhat, bulk_ess, tail_ess


def smith_doubling(Phi: np.ndarray, Sigma_eta: np.ndarray,
                    n_doublings: int = 15) -> np.ndarray:
    """Solve Sigma = Phi Sigma Phi^T + Sigma_eta for Sigma (d, d).

    Same algorithm as the filter's _discrete_lyapunov_solve, ported to
    numpy. Works on a single matrix or vectorized over a leading dim.
    """
    X = Sigma_eta.copy()
    A = Phi.copy()
    for _ in range(n_doublings):
        X = X + A @ X @ A.swapaxes(-1, -2)
        A = A @ A
    return 0.5 * (X + X.swapaxes(-1, -2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    npz = np.load(os.path.join(args.out_dir,
                                  "svssm_hmc_multi_full_phi_samples.npz"))
    phi_d = npz["phi_diag"]    # (chains, draws, d)
    phi_o = npz["phi_off"]     # (chains, draws, d(d-1)/2)
    sig2 = npz["sigma_eta_sq"] # (chains, draws, d)

    chains, draws, d = phi_d.shape
    off_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]

    # Build per-sample upper-triangular Phi and diagonal Sigma_eta (d-generic).
    Phi = np.zeros((chains, draws, d, d), dtype=np.float64)
    for i in range(d):
        Phi[..., i, i] = phi_d[..., i]
    for k, (i, j) in enumerate(off_idx):
        Phi[..., i, j] = phi_o[..., k]
    Sigma_eta = np.zeros((chains, draws, d, d), dtype=np.float64)
    for i in range(d):
        Sigma_eta[..., i, i] = sig2[..., i]

    # Smith doubling (vectorized) — discard samples with spec_rad >= 1
    # to avoid Lyapunov divergence on transient HMC excursions.
    eig = np.linalg.eigvals(Phi)
    stat_mask = np.max(np.abs(eig), axis=-1) < 1.0
    Sigma_h = smith_doubling(Phi, Sigma_eta, n_doublings=15)
    Sigma_h[~stat_mask] = np.nan

    # Truth
    Phi_truth = npz["Phi_truth"]
    sig2_truth = npz["sigma_eta_sq_truth"]
    Sigma_eta_truth = np.diag(np.asarray(sig2_truth, dtype=np.float64))
    Sigma_h_truth = smith_doubling(Phi_truth.astype(np.float64),
                                     Sigma_eta_truth, n_doublings=15)

    # d-generic derived quantities: diagonal variances, off-diagonal
    # covariances, and cross-asset correlations.
    panels = []   # (label, array (chains,draws), truth)
    for i in range(d):
        panels.append((f"$\\Sigma_{{h,{i}{i}}}$", Sigma_h[..., i, i],
                       float(Sigma_h_truth[i, i])))
    for (i, j) in off_idx:
        sij = Sigma_h[..., i, j]
        panels.append((f"$\\Sigma_{{h,{i}{j}}}$", sij,
                       float(Sigma_h_truth[i, j])))
        rho = sij / np.sqrt(np.maximum(
            Sigma_h[..., i, i] * Sigma_h[..., j, j], 1e-12))
        rho_t = float(Sigma_h_truth[i, j] / np.sqrt(
            Sigma_h_truth[i, i] * Sigma_h_truth[j, j]))
        panels.append((f"$\\rho_{{h,{i}{j}}}$", rho, rho_t))

    # Vehtari diagnostics on the derived quantities.
    def vehtari_row(name: str, x: np.ndarray, t: float) -> dict:
        # Drop chains with too many NaN (transient non-stationary excursions)
        finite_frac = np.mean(np.isfinite(x))
        x_fin = np.where(np.isfinite(x), x, np.nan)
        med = float(np.nanmedian(x_fin))
        q025 = float(np.nanpercentile(x_fin, 2.5))
        q975 = float(np.nanpercentile(x_fin, 97.5))
        x_clean = np.nan_to_num(x_fin, nan=med)
        rh = rank_rhat(x_clean)
        be = bulk_ess(x_clean)
        te = tail_ess(x_clean)
        cov = q025 <= t <= q975
        return dict(name=name, truth=t, median=med, q025=q025, q975=q975,
                     rhat=rh, bulk_ess=be, tail_ess=te, covered=cov,
                     finite_frac=finite_frac)

    rows = [vehtari_row(name, x, t) for (name, x, t) in panels]

    print(f'{"quantity":<22}{"truth":>9}{"median":>10}{"2.5%":>10}{"97.5%":>10}'
          f'{"rank-Rhat":>11}{"bulk-ESS":>10}{"tail-ESS":>10}{"finite":>9}{"cov":>5}')
    print('-' * 105)
    for r in rows:
        cov_s = "Y" if r["covered"] else "N"
        print(f'{r["name"]:<22}{r["truth"]:>+9.3f}{r["median"]:>+10.3f}'
              f'{r["q025"]:>+10.3f}{r["q975"]:>+10.3f}'
              f'{r["rhat"]:>11.4f}{r["bulk_ess"]:>10.0f}{r["tail_ess"]:>10.0f}'
              f'{r["finite_frac"]*100:>8.1f}%{cov_s:>5}')

    # Plot (reuses the same d-generic `panels` list)
    n_chains = chains
    colors = plt.cm.tab10(np.arange(n_chains))

    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 2.0 * len(panels)),
                              gridspec_kw={"width_ratios": [3, 1]})
    if len(panels) == 1:
        axes = axes.reshape(1, 2)

    for r, (name, x, t) in enumerate(panels):
        ax_t = axes[r, 0]
        ax_h = axes[r, 1]
        for c in range(n_chains):
            ax_t.plot(x[c], color=colors[c], alpha=0.6, linewidth=0.6,
                       label=f"chain {c+1}" if r == 0 else None)
        ax_t.axhline(t, color="red", linestyle="--", linewidth=1.2,
                      label="truth" if r == 0 else None)
        ax_t.set_ylabel(name, fontsize=11)
        ax_t.grid(True, alpha=0.3)
        if r == 0:
            ax_t.legend(loc="upper right", fontsize=8, ncol=n_chains + 1)
        if r == len(panels) - 1:
            ax_t.set_xlabel("draw")

        flat = x[np.isfinite(x)]
        ax_h.hist(flat, bins=50, color="seagreen",
                   orientation="horizontal", alpha=0.8)
        ax_h.axhline(t, color="red", linestyle="--", linewidth=1.2)
        ax_h.set_yticks([])
        ax_h.grid(True, alpha=0.3)

    fig.suptitle(
        "Derived stationary covariance traces: full-$\\Phi$ V1 multivariate "
        f"($d{{=}}2$), {n_chains} chains $\\times$ {draws} draws",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_path = os.path.join(args.out_dir,
                              "svssm_hmc_multi_full_phi_stationary_trace.png")
    fig.savefig(out_path, dpi=140)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
