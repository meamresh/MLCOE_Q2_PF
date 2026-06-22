"""Prior-vs-posterior shrinkage diagnostic for the full-Phi V1 run.

For each constrained parameter we compute:
  - prior SD (via Monte Carlo through the constraining transform)
  - posterior SD (sample SD on the saved npz)
  - shrinkage  = 1 - posterior_SD / prior_SD     in (-∞, 1]
                 (1.0 = perfect data-only collapse;
                  0.0 = posterior SD == prior SD == no learning;
                  negative = posterior wider than prior, usually a sign
                  of computational issue or model mismatch.)
  - posterior mean shift  = (posterior_mean - prior_mean) / prior_SD
                            (z-score of how far posterior mean has moved
                             from prior centroid; |z| >> 0.5 = data is
                             pulling the posterior away from prior)

Convention (Phase 19-style):
   shrinkage > 0.5  : data informative (good)
   shrinkage 0.2-0.5: moderately informative
   shrinkage < 0.2  : prior-dominated
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_trace_stationary_cov import smith_doubling


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_prior", type=int, default=200_000,
                   help="MC sample size for implied prior moments.")
    args = p.parse_args()

    s = np.load(os.path.join(args.out_dir,
                                "svssm_hmc_multi_full_phi_samples.npz"))
    summary = json.load(open(os.path.join(
        args.out_dir, "svssm_hmc_multi_full_phi_summary.json")))
    cfg = summary["config"]

    # Prior hyperparameters from the run config
    prior_mu_loc = cfg["prior_mu_loc"]
    prior_mu_scale = cfg["prior_mu_scale"]
    prior_phi_raw_loc = cfg["prior_phi_raw_loc"]
    prior_phi_raw_scale = cfg["prior_phi_raw_scale"]
    prior_log_sig_loc = cfg["prior_log_sigma_eta_sq_loc"]
    prior_log_sig_scale = cfg["prior_log_sigma_eta_sq_scale"]
    prior_phi_off_scale = cfg["prior_phi_off_scale"]

    rng = np.random.default_rng(0)
    N = args.n_prior

    # Implied priors on constrained params via MC
    prior_mu = rng.normal(prior_mu_loc, prior_mu_scale, N)
    raw_phi = rng.normal(prior_phi_raw_loc, prior_phi_raw_scale, N)
    prior_phi_diag = np.tanh(raw_phi)
    prior_phi_off = rng.normal(0.0, prior_phi_off_scale, N)
    log_sig = rng.normal(prior_log_sig_loc, prior_log_sig_scale, N)
    prior_sig2 = np.exp(log_sig)

    panels = [
        ("mu_0",          s["mu"][..., 0].ravel(),          prior_mu,
         float(s["mu_truth"][0])),
        ("mu_1",          s["mu"][..., 1].ravel(),          prior_mu,
         float(s["mu_truth"][1])),
        ("phi_diag_0",    s["phi_diag"][..., 0].ravel(),    prior_phi_diag,
         float(s["phi_diag_truth"][0])),
        ("phi_diag_1",    s["phi_diag"][..., 1].ravel(),    prior_phi_diag,
         float(s["phi_diag_truth"][1])),
        ("phi_off_01",    s["phi_off"][..., 0].ravel(),     prior_phi_off,
         float(s["phi_off_truth"][0])),
        ("sigma_eta_sq_0",s["sigma_eta_sq"][..., 0].ravel(),prior_sig2,
         float(s["sigma_eta_sq_truth"][0])),
        ("sigma_eta_sq_1",s["sigma_eta_sq"][..., 1].ravel(),prior_sig2,
         float(s["sigma_eta_sq_truth"][1])),
    ]

    header = (f"{'param':<18}{'prior med':>11}{'prior SD':>10}"
              f"{'post med':>10}{'post SD':>10}{'shrink':>9}"
              f"{'|shift|':>9}{'verdict':>22}{'truth':>10}")
    sep = "-" * len(header)
    lines = [
        "Prior-vs-posterior shrinkage check — full-Phi V1 multivariate",
        f"Run: {args.out_dir}",
        f"chains={s['mu'].shape[0]}  draws={s['mu'].shape[1]}",
        "Shrinkage = 1 - post_SD/prior_SD  (1.0=full collapse, 0=no learning)",
        "|shift|   = |post_mean - prior_mean| / prior_SD  (z-units)",
        "",
        header, sep,
    ]
    rows = []
    for name, post, prior, truth in panels:
        prior_mean = float(np.mean(prior))
        prior_sd = float(np.std(prior))
        prior_med = float(np.median(prior))
        post_mean = float(np.mean(post))
        post_sd = float(np.std(post))
        post_med = float(np.median(post))
        shrink = 1.0 - post_sd / max(prior_sd, 1e-12)
        shift = abs(post_mean - prior_mean) / max(prior_sd, 1e-12)
        if shrink > 0.5:
            verdict = "data informative"
        elif shrink > 0.2:
            verdict = "moderate"
        else:
            verdict = "prior-dominated"
        lines.append(f"{name:<18}{prior_med:>+11.3f}{prior_sd:>10.3f}"
                     f"{post_med:>+10.3f}{post_sd:>10.3f}"
                     f"{shrink:>+9.3f}{shift:>9.3f}"
                     f"{verdict:>22}{truth:>+10.3f}")
        rows.append({"param": name, "prior_median": prior_med,
                      "prior_sd": prior_sd, "posterior_median": post_med,
                      "posterior_sd": post_sd, "shrinkage": shrink,
                      "mean_shift_in_prior_sd": shift, "verdict": verdict,
                      "truth": truth})

    lines.append(sep)
    lines.append("")

    # ---- Derived stationary covariance: induced prior via Smith doubling ----
    # Sample N upper-tri Phi and diagonal Sigma_eta from the prior, solve
    # the Lyapunov equation, discard non-stationary draws.
    N_d = min(N, 50_000)  # 50k is plenty; full Lyapunov solve per sample
    raw_phi_a = rng.normal(prior_phi_raw_loc, prior_phi_raw_scale, N_d)
    raw_phi_b = rng.normal(prior_phi_raw_loc, prior_phi_raw_scale, N_d)
    phi_d_a = np.tanh(raw_phi_a)
    phi_d_b = np.tanh(raw_phi_b)
    phi_off_p = rng.normal(0.0, prior_phi_off_scale, N_d)
    log_sa = rng.normal(prior_log_sig_loc, prior_log_sig_scale, N_d)
    log_sb = rng.normal(prior_log_sig_loc, prior_log_sig_scale, N_d)
    s2_a = np.exp(log_sa); s2_b = np.exp(log_sb)
    Phi_p = np.zeros((N_d, 2, 2))
    Phi_p[:, 0, 0] = phi_d_a; Phi_p[:, 1, 1] = phi_d_b
    Phi_p[:, 0, 1] = phi_off_p
    Seta_p = np.zeros((N_d, 2, 2))
    Seta_p[:, 0, 0] = s2_a; Seta_p[:, 1, 1] = s2_b
    eig = np.linalg.eigvals(Phi_p)
    stat_mask = np.max(np.abs(eig), axis=-1) < 1.0
    Sh_p = smith_doubling(Phi_p, Seta_p, n_doublings=15)
    Sh_p[~stat_mask] = np.nan
    s00p = Sh_p[:, 0, 0]; s11p = Sh_p[:, 1, 1]; s01p = Sh_p[:, 0, 1]
    rho_p = s01p / np.sqrt(np.maximum(s00p * s11p, 1e-12))

    # Posterior side: same Smith doubling on posterior (Phi, Sigma_eta)
    mu_arr = s["mu"]; phi_d_arr = s["phi_diag"]
    phi_o_arr = s["phi_off"]; sig2_arr = s["sigma_eta_sq"]
    chains, draws, _ = phi_d_arr.shape
    Phi_post = np.zeros((chains, draws, 2, 2))
    Phi_post[..., 0, 0] = phi_d_arr[..., 0]
    Phi_post[..., 1, 1] = phi_d_arr[..., 1]
    Phi_post[..., 0, 1] = phi_o_arr[..., 0]
    Seta_post = np.zeros((chains, draws, 2, 2))
    Seta_post[..., 0, 0] = sig2_arr[..., 0]
    Seta_post[..., 1, 1] = sig2_arr[..., 1]
    Sh_post = smith_doubling(Phi_post, Seta_post, n_doublings=15)
    eig_post = np.linalg.eigvals(Phi_post)
    Sh_post[np.max(np.abs(eig_post), axis=-1) >= 1.0] = np.nan
    s00_post = Sh_post[..., 0, 0].ravel()
    s11_post = Sh_post[..., 1, 1].ravel()
    s01_post = Sh_post[..., 0, 1].ravel()
    rho_post = s01_post / np.sqrt(np.maximum(s00_post * s11_post, 1e-12))

    # Truth on derived
    Phi_truth = s["Phi_truth"].astype(np.float64)
    Seta_truth = np.diag(s["sigma_eta_sq_truth"].astype(np.float64))
    Sh_truth = smith_doubling(Phi_truth, Seta_truth, n_doublings=15)
    truth_rho = float(Sh_truth[0,1] / np.sqrt(Sh_truth[0,0]*Sh_truth[1,1]))

    derived = [
        ("Sigma_h_00", s00_post, s00p, float(Sh_truth[0,0])),
        ("Sigma_h_11", s11_post, s11p, float(Sh_truth[1,1])),
        ("Sigma_h_01", s01_post, s01p, float(Sh_truth[0,1])),
        ("rho_h",      rho_post, rho_p, truth_rho),
    ]

    lines.append("DERIVED STATIONARY COVARIANCE")
    lines.append(header); lines.append(sep)
    for name, post, prior, truth in derived:
        post_f = post[np.isfinite(post)]
        prior_f = prior[np.isfinite(prior)]
        prior_mean = float(np.mean(prior_f))
        prior_sd = float(np.std(prior_f))
        prior_med = float(np.median(prior_f))
        post_mean = float(np.mean(post_f))
        post_sd = float(np.std(post_f))
        post_med = float(np.median(post_f))
        shrink = 1.0 - post_sd / max(prior_sd, 1e-12)
        shift = abs(post_mean - prior_mean) / max(prior_sd, 1e-12)
        if shrink > 0.5: verdict = "data informative"
        elif shrink > 0.2: verdict = "moderate"
        else: verdict = "prior-dominated"
        lines.append(f"{name:<18}{prior_med:>+11.3f}{prior_sd:>10.3f}"
                     f"{post_med:>+10.3f}{post_sd:>10.3f}"
                     f"{shrink:>+9.3f}{shift:>9.3f}"
                     f"{verdict:>22}{truth:>+10.3f}")
        rows.append({"param": name, "prior_median": prior_med,
                      "prior_sd": prior_sd, "posterior_median": post_med,
                      "posterior_sd": post_sd, "shrinkage": shrink,
                      "mean_shift_in_prior_sd": shift, "verdict": verdict,
                      "truth": truth})
    lines.append(sep)
    lines.append("")

    lines.append("Notes on shrinkage thresholds:")
    lines.append(" >0.5  : data has clearly overridden the prior")
    lines.append(" 0.2-0.5: moderate update; data + prior both contribute")
    lines.append(" <0.2  : posterior ≈ prior  — defer claims about this param")
    txt = "\n".join(lines) + "\n"
    print(txt)

    out_txt = os.path.join(args.out_dir,
                              "svssm_hmc_multi_full_phi_prior_dominance.txt")
    with open(out_txt, "w") as f:
        f.write(txt)
    out_json = os.path.join(args.out_dir,
                               "svssm_hmc_multi_full_phi_prior_dominance.json")
    with open(out_json, "w") as f:
        json.dump({"thresholds": {"data_informative": 0.5,
                                     "moderate": 0.2}, "rows": rows},
                  f, indent=2)
    print(f"saved {out_txt}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
