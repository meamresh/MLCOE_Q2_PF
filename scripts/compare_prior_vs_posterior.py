"""Side-by-side comparison: prior-only HMC vs with-likelihood HMC.

Gold-standard Phase-19-style prior-dominance diagnostic. Both runs use
the same priors and the same HMC machinery; only the likelihood term
is dropped in the prior-only run. If the two posteriors are
indistinguishable on a parameter, the data is not informing that param.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_trace_stationary_cov import smith_doubling


def load_run(out_dir: str):
    npz = np.load(os.path.join(out_dir,
                                  "svssm_hmc_multi_full_phi_samples.npz"))
    return npz


def extract_scalars(npz):
    mu = npz["mu"]; phi_d = npz["phi_diag"]
    phi_o = npz["phi_off"]; sig2 = npz["sigma_eta_sq"]
    out = {
        "mu_0": mu[..., 0],
        "mu_1": mu[..., 1],
        "phi_diag_0": phi_d[..., 0],
        "phi_diag_1": phi_d[..., 1],
        "phi_off_01": phi_o[..., 0],
        "sigma_eta_sq_0": sig2[..., 0],
        "sigma_eta_sq_1": sig2[..., 1],
    }
    chains, draws, _ = phi_d.shape
    Phi = np.zeros((chains, draws, 2, 2))
    Phi[..., 0, 0] = phi_d[..., 0]
    Phi[..., 1, 1] = phi_d[..., 1]
    Phi[..., 0, 1] = phi_o[..., 0]
    Seta = np.zeros((chains, draws, 2, 2))
    Seta[..., 0, 0] = sig2[..., 0]
    Seta[..., 1, 1] = sig2[..., 1]
    eig = np.linalg.eigvals(Phi)
    Sh = smith_doubling(Phi, Seta, n_doublings=15)
    Sh[np.max(np.abs(eig), axis=-1) >= 1.0] = np.nan
    s00 = Sh[..., 0, 0]; s11 = Sh[..., 1, 1]; s01 = Sh[..., 0, 1]
    rho = s01 / np.sqrt(np.maximum(s00 * s11, 1e-12))
    out["Sigma_h_00"] = s00
    out["Sigma_h_11"] = s11
    out["Sigma_h_01"] = s01
    out["rho_h"] = rho
    return out


def stats(x: np.ndarray) -> tuple:
    flat = x[np.isfinite(x)].ravel()
    if flat.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.median(flat)),
            float(np.std(flat)),
            float(np.percentile(flat, 2.5)),
            float(np.percentile(flat, 97.5)))


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
    prior_scalars = extract_scalars(prior_npz)
    post_scalars = extract_scalars(post_npz)

    # Truths from the posterior run (same data + truth in both).
    truths = {
        "mu_0": float(post_npz["mu_truth"][0]),
        "mu_1": float(post_npz["mu_truth"][1]),
        "phi_diag_0": float(post_npz["phi_diag_truth"][0]),
        "phi_diag_1": float(post_npz["phi_diag_truth"][1]),
        "phi_off_01": float(post_npz["phi_off_truth"][0]),
        "sigma_eta_sq_0": float(post_npz["sigma_eta_sq_truth"][0]),
        "sigma_eta_sq_1": float(post_npz["sigma_eta_sq_truth"][1]),
    }
    # Truth on derived
    Phi_t = post_npz["Phi_truth"].astype(np.float64)
    Seta_t = np.diag(post_npz["sigma_eta_sq_truth"].astype(np.float64))
    Sh_t = smith_doubling(Phi_t, Seta_t, n_doublings=15)
    truths["Sigma_h_00"] = float(Sh_t[0, 0])
    truths["Sigma_h_11"] = float(Sh_t[1, 1])
    truths["Sigma_h_01"] = float(Sh_t[0, 1])
    truths["rho_h"] = float(Sh_t[0, 1] /
                              np.sqrt(Sh_t[0, 0] * Sh_t[1, 1]))

    header = (f"{'param':<18}{'prior med':>11}{'prior SD':>10}"
              f"{'post med':>10}{'post SD':>10}"
              f"{'SD ratio':>10}{'med shift':>11}"
              f"{'verdict':>22}{'truth':>10}")
    sep = "-" * len(header)
    lines = [
        "Prior-only HMC vs with-likelihood HMC — full-Phi V1 multivariate",
        f"Prior-only run:    {args.prior_dir}",
        f"With-likelihood:   {args.posterior_dir}",
        f"Prior  chains × draws: "
        f"{prior_npz['mu'].shape[0]} × {prior_npz['mu'].shape[1]}",
        f"Posterior chains × draws: "
        f"{post_npz['mu'].shape[0]} × {post_npz['mu'].shape[1]}",
        "",
        "SD ratio  = post_SD / prior_SD       (1.0=no data info; <1=data narrows)",
        "med shift = (post_med - prior_med) / prior_SD   (data pulls posterior off prior center)",
        "",
        "RAW + DERIVED",
        header, sep,
    ]
    panels = ["mu_0", "mu_1", "phi_diag_0", "phi_diag_1", "phi_off_01",
              "sigma_eta_sq_0", "sigma_eta_sq_1",
              "Sigma_h_00", "Sigma_h_11", "Sigma_h_01", "rho_h"]
    rows = []
    for name in panels:
        p_med, p_sd, _, _ = stats(prior_scalars[name])
        po_med, po_sd, _, _ = stats(post_scalars[name])
        ratio = po_sd / max(p_sd, 1e-12)
        shift = abs(po_med - p_med) / max(p_sd, 1e-12)
        # Verdict: combine ratio + shift
        if ratio < 0.5 and shift > 0.3:
            verdict = "data informative"
        elif ratio < 0.7 or shift > 0.2:
            verdict = "moderate"
        else:
            verdict = "prior-dominated"
        truth = truths[name]
        lines.append(f"{name:<18}{p_med:>+11.3f}{p_sd:>10.3f}"
                     f"{po_med:>+10.3f}{po_sd:>10.3f}"
                     f"{ratio:>10.3f}{shift:>11.3f}"
                     f"{verdict:>22}{truth:>+10.3f}")
        rows.append({"param": name,
                      "prior_median": p_med, "prior_sd": p_sd,
                      "posterior_median": po_med, "posterior_sd": po_sd,
                      "sd_ratio": ratio, "med_shift": shift,
                      "verdict": verdict, "truth": truth})
    lines.append(sep)
    lines.append("")
    lines.append("Verdict rules:")
    lines.append("  data informative : SD ratio < 0.5  AND median shift > 0.3 sigma")
    lines.append("  moderate         : SD ratio < 0.7  OR  median shift > 0.2 sigma")
    lines.append("  prior-dominated  : neither (posterior ≈ prior)")

    txt = "\n".join(lines) + "\n"
    print(txt)

    out_txt = os.path.join(args.out_dir,
                              "svssm_hmc_multi_full_phi_prior_vs_posterior.txt")
    with open(out_txt, "w") as f:
        f.write(txt)
    out_json = os.path.join(args.out_dir,
                               "svssm_hmc_multi_full_phi_prior_vs_posterior.json")
    with open(out_json, "w") as f:
        json.dump({"rows": rows, "truths": truths,
                    "prior_dir": args.prior_dir,
                    "posterior_dir": args.posterior_dir}, f, indent=2)
    print(f"saved {out_txt}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
