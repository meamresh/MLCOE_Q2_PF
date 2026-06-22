"""Plot the multivariate V2 posterior story for §11.9.

Four plots, written to the same output directory as v2_mv_samples.npz:

  1. v2_mv_marginals_compare.png — 2x4 grid of shared param histograms,
     FREE (blue) vs FIXED (green), truth-line in red, ridge prediction
     in dashed red.

  2. v2_mv_A_ridge.png — 2x2 grid of A entry FREE histograms with truth
     and ±3 sd shaded. The wide blue cloud IS the ridge.

  3. v2_mv_A_traces.png — 2x2 grid of A_11, A_12, A_21, A_22 traces by
     chain, showing each chain stuck on a different ridge segment.
     Direct visual evidence of Vehtari failure.

  4. v2_mv_vehtari_ess.png — horizontal bar chart of bulk-ESS per
     parameter, FREE (blue) vs FIXED (green), 400 threshold line.
     Visually dramatic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(in_dir: Path):
    samples = np.load(in_dir / "v2_mv_samples.npz")
    free = samples["free_const"]      # (chains, draws, 12)
    fixed = samples["fixed_const"]    # (chains, draws, 8)
    free_raw = samples["free_raw"]
    fixed_raw = samples["fixed_raw"]
    with open(in_dir / "v2_mv_vehtari.json") as f:
        veht = json.load(f)
    return free, fixed, free_raw, fixed_raw, veht


def plot_marginals(out_path, free, fixed, truth_dict, ridge_dict):
    """2x4 grid of shared-param histograms, FREE vs FIXED + truth + ridge."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    # Shared params indices: [mu1, mu2, phi1, phi2, s2_1, s2_2, se2_1, se2_2]
    # FREE cols (after constraining): 0,1,2,3,4,5,10,11
    # FIXED cols: 0..7
    shared = [
        ("mu_1",                 0,  0,  truth_dict["mu_1"],            ridge_dict["mu_1"]),
        ("mu_2",                 1,  1,  truth_dict["mu_2"],            ridge_dict["mu_2"]),
        ("phi_1",                2,  2,  truth_dict["phi_1"],           truth_dict["phi_1"]),
        ("phi_2",                3,  3,  truth_dict["phi_2"],           truth_dict["phi_2"]),
        ("sigma_eta_1_sq",       4,  4,  truth_dict["sigma_eta_1_sq"],  ridge_dict["sigma_eta_1_sq"]),
        ("sigma_eta_2_sq",       5,  5,  truth_dict["sigma_eta_2_sq"],  ridge_dict["sigma_eta_2_sq"]),
        ("sigma_eps_1_sq",      10,  6,  truth_dict["sigma_eps_1_sq"],  truth_dict["sigma_eps_1_sq"]),
        ("sigma_eps_2_sq",      11,  7,  truth_dict["sigma_eps_2_sq"],  truth_dict["sigma_eps_2_sq"]),
    ]
    for ax, (name, fi, fxi, tval, ridge_v) in zip(axes.flatten(), shared):
        fc = free[..., fi].ravel()
        xc = fixed[..., fxi].ravel()
        lo = float(min(np.percentile(fc, 1), np.percentile(xc, 1)))
        hi = float(max(np.percentile(fc, 99), np.percentile(xc, 99)))
        # Clip very-wide tails for readability
        bins = np.linspace(lo, hi, 60)
        ax.hist(fc, bins=bins, density=True, alpha=0.5, color="C0",
                label=f"FREE A (sd={fc.std():.3f})")
        ax.hist(xc, bins=bins, density=True, alpha=0.5, color="C2",
                label=f"A=I (sd={xc.std():.3f})")
        ax.axvline(tval, color="red", ls="--", lw=1.5,
                   label=f"V1 truth = {tval:.2f}")
        if abs(ridge_v - tval) > 1e-6:
            ax.axvline(ridge_v, color="orange", ls=":", lw=1.5,
                       label=f"A=I ridge = {ridge_v:.2f}")
        ax.set_title(name)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("Multivariate V2: marginal collapse under A=I restriction",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_A_ridge(out_path, free, truth_A):
    """2x2 grid of A entry FREE histograms with truth marked."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # FREE cols 6=A_11, 7=A_12, 8=A_21, 9=A_22
    A_specs = [
        (axes[0, 0], "A_11", 6, truth_A[0, 0]),
        (axes[0, 1], "A_12", 7, truth_A[0, 1]),
        (axes[1, 0], "A_21", 8, truth_A[1, 0]),
        (axes[1, 1], "A_22", 9, truth_A[1, 1]),
    ]
    for ax, name, idx, tval in A_specs:
        col = free[..., idx].ravel()
        ax.hist(col, bins=60, density=True, alpha=0.65, color="C0")
        ax.axvline(tval, color="red", ls="--", lw=2,
                   label=f"truth = {tval:.1f}")
        ax.axvline(col.mean(), color="black", ls="-", lw=1,
                   label=f"posterior mean = {col.mean():.2f}")
        ax.set_title(f"{name}  (sd={col.std():.2f}, "
                     f"95% CI = [{np.percentile(col, 2.5):.1f}, "
                     f"{np.percentile(col, 97.5):.1f}])")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("FREE A: each entry's posterior is essentially the prior "
                 r"$\mathcal{N}(0, 25)$ — the ridge swallows the data signal",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_A_traces(out_path, free, truth_A):
    """Trace plots of A entries by chain, showing chain non-convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    n_chains = free.shape[0]
    A_specs = [
        (axes[0, 0], "A_11", 6, truth_A[0, 0]),
        (axes[0, 1], "A_12", 7, truth_A[0, 1]),
        (axes[1, 0], "A_21", 8, truth_A[1, 0]),
        (axes[1, 1], "A_22", 9, truth_A[1, 1]),
    ]
    colors = ["C0", "C1", "C2", "C3"]
    for ax, name, idx, tval in A_specs:
        for c in range(n_chains):
            trace = free[c, :, idx]
            ax.plot(trace, color=colors[c], alpha=0.7, lw=0.6,
                    label=f"chain {c+1} (mean {trace.mean():.2f})")
        ax.axhline(tval, color="red", ls="--", lw=1.5,
                   label=f"truth = {tval:.1f}")
        ax.set_title(f"{name} trace (each chain on a different ridge segment)")
        ax.set_xlabel("sample index (post-burnin)")
        ax.set_ylabel(name)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("FREE A trace plots: chains do NOT converge to a "
                 "common posterior (each stuck on a different ridge segment)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_vehtari_ess(out_path, veht):
    """Horizontal bar chart of bulk-ESS per parameter, FREE vs FIXED."""
    free_rows = veht["free_diagnostics"]
    fixed_rows = veht["fixed_diagnostics"]

    free_names = [r["param"] for r in free_rows]
    free_ess = [r["bulk_ess"] for r in free_rows]
    fixed_names = [r["param"] for r in fixed_rows]
    fixed_ess = [r["bulk_ess"] for r in fixed_rows]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    y = np.arange(len(free_names))
    bars = ax.barh(y, free_ess, color="C0", alpha=0.7)
    # Color A entries red, sigma_eps green
    for i, n in enumerate(free_names):
        if n.startswith("A_"):
            bars[i].set_color("crimson")
            bars[i].set_alpha(0.85)
        elif n.startswith("sigma_eps"):
            bars[i].set_color("forestgreen")
            bars[i].set_alpha(0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(free_names)
    ax.axvline(400, color="red", ls="--", lw=2, label="Vehtari threshold = 400")
    ax.set_xlabel("bulk-ESS (out of 12000 samples)")
    ax.set_title("FREE A (12-param): A entries are stuck;\n"
                 r"$\sigma_\varepsilon^2$ mixes properly because it's off-ridge")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    y = np.arange(len(fixed_names))
    ax.barh(y, fixed_ess, color="forestgreen", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(fixed_names)
    ax.axvline(400, color="red", ls="--", lw=2, label="Vehtari threshold = 400")
    ax.set_xlabel("bulk-ESS (out of 12000 samples)")
    ax.set_title("FIXED A=I (8-param): every parameter passes Vehtari")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Vehtari bulk-ESS per marginal — the identifiability "
                 "signature is the ESS gap",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "v2_multivariate_demo_long")
    args = p.parse_args()
    in_dir = Path(args.in_dir)

    free, fixed, _, _, veht = load_data(in_dir)
    print(f"FREE samples shape: {free.shape}, FIXED: {fixed.shape}")

    truth = {
        "mu_1": 1.0, "mu_2": 1.0,
        "phi_1": 0.9, "phi_2": 0.85,
        "sigma_eta_1_sq": 0.25, "sigma_eta_2_sq": 0.36,
        "sigma_eps_1_sq": 0.09, "sigma_eps_2_sq": 0.09,
    }
    # Under A=I ridge (c_i = A_true_ii = 2): mu_i -> 2*1 = 2, sigma_eta_i^2 -> 4*sigma_eta_true_i^2
    ridge = {
        "mu_1": 2.0, "mu_2": 2.0,
        "phi_1": 0.9, "phi_2": 0.85,
        "sigma_eta_1_sq": 1.0, "sigma_eta_2_sq": 1.44,
        "sigma_eps_1_sq": 0.09, "sigma_eps_2_sq": 0.09,
    }
    A_true = np.array([[2.0, 0.0], [0.0, 2.0]])

    plot_marginals(in_dir / "v2_mv_marginals_compare.png",
                   free, fixed, truth, ridge)
    print(f"wrote {in_dir}/v2_mv_marginals_compare.png")

    plot_A_ridge(in_dir / "v2_mv_A_ridge.png", free, A_true)
    print(f"wrote {in_dir}/v2_mv_A_ridge.png")

    plot_A_traces(in_dir / "v2_mv_A_traces.png", free, A_true)
    print(f"wrote {in_dir}/v2_mv_A_traces.png")

    plot_vehtari_ess(in_dir / "v2_mv_vehtari_ess.png", veht)
    print(f"wrote {in_dir}/v2_mv_vehtari_ess.png")


if __name__ == "__main__":
    raise SystemExit(main())
