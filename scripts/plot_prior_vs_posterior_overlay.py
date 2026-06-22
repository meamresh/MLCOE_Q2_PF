"""Overlay prior-only marginals and with-likelihood marginals on the
same axes for visual prior-dominance assessment.

If the two histograms are indistinguishable on a parameter, the data
is not informing that parameter at this T.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compare_prior_vs_posterior import extract_scalars, load_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior_dir", type=str, required=True)
    p.add_argument("--posterior_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.posterior_dir

    prior_npz = load_run(args.prior_dir)
    post_npz = load_run(args.posterior_dir)
    prior_s = extract_scalars(prior_npz)
    post_s = extract_scalars(post_npz)

    # Build truths
    Phi_t = post_npz["Phi_truth"].astype(np.float64)
    Seta_t = np.diag(post_npz["sigma_eta_sq_truth"].astype(np.float64))
    from plot_trace_stationary_cov import smith_doubling
    Sh_t = smith_doubling(Phi_t, Seta_t, n_doublings=15)
    rho_t = Sh_t[0, 1] / np.sqrt(Sh_t[0, 0] * Sh_t[1, 1])

    truths = {
        "mu_0": float(post_npz["mu_truth"][0]),
        "mu_1": float(post_npz["mu_truth"][1]),
        "phi_diag_0": float(post_npz["phi_diag_truth"][0]),
        "phi_diag_1": float(post_npz["phi_diag_truth"][1]),
        "phi_off_01": float(post_npz["phi_off_truth"][0]),
        "sigma_eta_sq_0": float(post_npz["sigma_eta_sq_truth"][0]),
        "sigma_eta_sq_1": float(post_npz["sigma_eta_sq_truth"][1]),
        "rho_h": float(rho_t),
    }

    panels = [
        ("$\\mu_0$",          "mu_0"),
        ("$\\mu_1$",          "mu_1"),
        ("$\\phi_{00}$",      "phi_diag_0"),
        ("$\\phi_{11}$",      "phi_diag_1"),
        ("$\\phi_{01}$",      "phi_off_01"),
        ("$\\sigma^2_{\\eta,0}$", "sigma_eta_sq_0"),
        ("$\\sigma^2_{\\eta,1}$", "sigma_eta_sq_1"),
        ("$\\rho_h$",          "rho_h"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    for i, (label, key) in enumerate(panels):
        ax = axes[i]
        pr = prior_s[key].ravel()
        po = post_s[key].ravel()
        pr = pr[np.isfinite(pr)]
        po = po[np.isfinite(po)]

        # Symmetric range covering both
        lo = float(np.percentile(np.concatenate([pr, po]), 1.0))
        hi = float(np.percentile(np.concatenate([pr, po]), 99.0))
        bins = np.linspace(lo, hi, 60)

        ax.hist(pr, bins=bins, alpha=0.45, color="steelblue",
                 density=True, label="prior-only HMC")
        ax.hist(po, bins=bins, alpha=0.45, color="darkorange",
                 density=True, label="posterior")
        ax.axvline(truths[key], color="red", linestyle="--",
                    linewidth=1.4, label="truth")
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Prior-only vs with-likelihood marginals: where the data informs",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(args.out_dir,
                              "svssm_hmc_multi_full_phi_prior_vs_posterior.png")
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
