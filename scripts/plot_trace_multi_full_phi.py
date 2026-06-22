"""Per-parameter trace plot for the 4-chain full-Phi production run.

Reads the samples.npz at <out_dir>, emits a (7 rows x 2 cols) trace +
running-mean plot. Each chain a different colour; truth as a dashed
horizontal. Saved as <out_dir>/svssm_hmc_multi_full_phi_trace.png.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True,
                   help="Run directory containing *_samples.npz")
    p.add_argument("--burn_in_warning", type=int, default=0,
                   help="If >0, draw a vertical line at this draw index "
                        "to mark a post-hoc burn-in cutoff for diagnostics.")
    args = p.parse_args()

    npz_path = os.path.join(args.out_dir, "svssm_hmc_multi_full_phi_samples.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    s = np.load(npz_path)

    mu = s["mu"]                # (chains, draws, d)
    phi_d = s["phi_diag"]       # (chains, draws, d)
    phi_o = s["phi_off"]        # (chains, draws, d(d-1)/2)
    sig2 = s["sigma_eta_sq"]    # (chains, draws, d)

    mu_truth = s["mu_truth"]
    phi_d_truth = s["phi_diag_truth"]
    phi_o_truth = s["phi_off_truth"]
    sig2_truth = s["sigma_eta_sq_truth"]

    d = mu.shape[2]
    off_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]

    # d-generic panel list: mu_i, phi_diag_ii, phi_off_ij (cross-asset),
    # sigma_eta_sq_i.
    rows = []
    for i in range(d):
        rows.append((f"$\\mu_{{{i}}}$", mu[:, :, i], float(mu_truth[i])))
    for i in range(d):
        rows.append((f"$\\phi_{{{i}{i}}}$", phi_d[:, :, i],
                     float(phi_d_truth[i])))
    for k, (i, j) in enumerate(off_idx):
        rows.append((f"$\\phi_{{{i}{j}}}$ (x-asset)", phi_o[:, :, k],
                     float(phi_o_truth[k])))
    for i in range(d):
        rows.append((f"$\\sigma^2_{{\\eta,{i}}}$", sig2[:, :, i],
                     float(sig2_truth[i])))

    n_chains = mu.shape[0]
    n_draws = mu.shape[1]
    colors = plt.cm.tab10(np.arange(n_chains))

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 2.0 * len(rows)),
                              gridspec_kw={"width_ratios": [3, 1]})

    for r, (name, x, truth) in enumerate(rows):
        ax_t = axes[r, 0]
        ax_h = axes[r, 1]
        for c in range(n_chains):
            ax_t.plot(x[c], color=colors[c], alpha=0.6, linewidth=0.6,
                       label=f"chain {c+1}" if r == 0 else None)
        ax_t.axhline(truth, color="red", linestyle="--", linewidth=1.2,
                      label="truth" if r == 0 else None)
        if args.burn_in_warning > 0:
            ax_t.axvline(args.burn_in_warning, color="gray",
                          linestyle=":", linewidth=1.0)
        ax_t.set_ylabel(name, fontsize=11)
        ax_t.grid(True, alpha=0.3)
        if r == 0:
            ax_t.legend(loc="upper right", fontsize=8, ncol=n_chains + 1)
        if r == len(rows) - 1:
            ax_t.set_xlabel("draw")

        # Right-side: marginal histogram pooled across chains
        flat = x.reshape(-1)
        ax_h.hist(flat, bins=50, color="steelblue",
                   orientation="horizontal", alpha=0.8)
        ax_h.axhline(truth, color="red", linestyle="--", linewidth=1.2)
        ax_h.set_yticks([])
        ax_h.grid(True, alpha=0.3)

    fig.suptitle(
        f"Trace + marginal: full-$\\Phi$ V1 multivariate ($d{{=}}{d}$), "
        f"{n_chains} chains $\\times$ {n_draws} draws",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_path = os.path.join(args.out_dir, "svssm_hmc_multi_full_phi_trace.png")
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
