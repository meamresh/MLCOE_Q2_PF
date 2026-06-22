"""
V2 three-way scale-fix comparison (section3 §11.7-extension empirical).

The 1D scale invariance of V2 (mu, sigma_eta, A) -> (c*mu, c*sigma_eta, A/c)
predicts that *any one* of the three parameters can be fixed to break the
invariance and identify the rest. §11.3 of section3 listed three options:

    Restriction      c-resolution            ridge point we land on
    ----------------------------------------------------------------------
    A = 1            c = A_true              (A_true * mu, A_true*sigma_eta, 1)
    sigma_eta = 1    c = 1/sigma_eta_true    (mu/sigma_eta_true, 1, A_true*sigma_eta_true)
    mu = 1           c = 1/mu_true           (1, sigma_eta_true/mu_true, A_true*mu_true)

This script runs all three on the SAME y_{1:T} and verifies:

  (a) every restriction collapses the ridge -> 4-dim identifiable posterior
  (b) phi and sigma_eps marginals are invariant across the three (the prediction)
  (c) the remaining marginals each land at their predicted ridge point.

Uses the exact Kalman likelihood from exp_v2_identifiability_demo.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from scripts.exp.exp_v2_identifiability_demo import (
    gen_v2_data,
    kalman_v2_loglik,
    run_hmc,
)

tfd = tfp.distributions
tfm = tfp.mcmc


# ---------------------------------------------------------------------------
# Target factories with each of the three scale restrictions
# ---------------------------------------------------------------------------

def make_target_fixed_A(y_obs, A_value):
    """A fixed at A_value, estimate (mu, phi, sigma_eta, sigma_eps)."""
    p_mu      = tfd.Normal(0.0, 5.0)
    p_phi_raw = tfd.Normal(0.0, 2.0)
    p_log_s2  = tfd.Normal(-2.0, 2.0)
    p_log_se2 = tfd.Normal(-2.0, 2.0)
    A_t = tf.constant(A_value, tf.float32)

    @tf.function
    def target(theta_raw):
        mu = theta_raw[0]
        phi_raw = theta_raw[1]
        log_s2 = theta_raw[2]
        log_se2 = theta_raw[3]
        phi = tf.tanh(phi_raw)
        sigma_eta_sq = tf.exp(log_s2)
        sigma_eps_sq = tf.exp(log_se2)
        lp = (p_mu.log_prob(mu)
              + p_phi_raw.log_prob(phi_raw)
              + p_log_s2.log_prob(log_s2)
              + p_log_se2.log_prob(log_se2))
        ll = kalman_v2_loglik(mu, phi, sigma_eta_sq, A_t, sigma_eps_sq, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float32))
        return lp + ll
    return target


def make_target_fixed_sigma_eta(y_obs, sigma_eta_value):
    """sigma_eta fixed at sigma_eta_value, estimate (mu, phi, A, sigma_eps)."""
    p_mu      = tfd.Normal(0.0, 5.0)
    p_phi_raw = tfd.Normal(0.0, 2.0)
    p_log_A   = tfd.Normal(0.0, 5.0)
    p_log_se2 = tfd.Normal(-2.0, 2.0)
    sigma_eta_sq_t = tf.constant(sigma_eta_value ** 2, tf.float32)

    @tf.function
    def target(theta_raw):
        mu = theta_raw[0]
        phi_raw = theta_raw[1]
        log_A = theta_raw[2]
        log_se2 = theta_raw[3]
        phi = tf.tanh(phi_raw)
        A = tf.exp(log_A)
        sigma_eps_sq = tf.exp(log_se2)
        lp = (p_mu.log_prob(mu)
              + p_phi_raw.log_prob(phi_raw)
              + p_log_A.log_prob(log_A)
              + p_log_se2.log_prob(log_se2))
        ll = kalman_v2_loglik(mu, phi, sigma_eta_sq_t, A, sigma_eps_sq, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float32))
        return lp + ll
    return target


def make_target_fixed_mu(y_obs, mu_value):
    """mu fixed at mu_value, estimate (phi, sigma_eta, A, sigma_eps)."""
    p_phi_raw = tfd.Normal(0.0, 2.0)
    p_log_s2  = tfd.Normal(-2.0, 2.0)
    p_log_A   = tfd.Normal(0.0, 5.0)
    p_log_se2 = tfd.Normal(-2.0, 2.0)
    mu_t = tf.constant(mu_value, tf.float32)

    @tf.function
    def target(theta_raw):
        phi_raw = theta_raw[0]
        log_s2 = theta_raw[1]
        log_A = theta_raw[2]
        log_se2 = theta_raw[3]
        phi = tf.tanh(phi_raw)
        sigma_eta_sq = tf.exp(log_s2)
        A = tf.exp(log_A)
        sigma_eps_sq = tf.exp(log_se2)
        lp = (p_phi_raw.log_prob(phi_raw)
              + p_log_s2.log_prob(log_s2)
              + p_log_A.log_prob(log_A)
              + p_log_se2.log_prob(log_se2))
        ll = kalman_v2_loglik(mu_t, phi, sigma_eta_sq, A, sigma_eps_sq, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float32))
        return lp + ll
    return target


# ---------------------------------------------------------------------------
# Constrain helpers (4-param, different parameter slot in each fix)
# ---------------------------------------------------------------------------

def summarize(samples, names, truth_dict=None):
    """samples: (chains, draws, dim); names: list of param names matching last dim."""
    flat = samples.reshape(-1, samples.shape[-1])
    rows = []
    for j, n in enumerate(names):
        col = flat[:, j]
        row = {
            "param": n,
            "median": float(np.median(col)),
            "mean": float(np.mean(col)),
            "sd": float(np.std(col)),
            "q025": float(np.percentile(col, 2.5)),
            "q975": float(np.percentile(col, 97.5)),
        }
        if truth_dict is not None and n in truth_dict:
            row["truth"] = float(truth_dict[n])
            row["covered"] = bool(row["q025"] <= truth_dict[n] <= row["q975"])
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=2.0)
    p.add_argument("--phi", type=float, default=0.9)
    p.add_argument("--sigma_eta", type=float, default=0.4)
    p.add_argument("--A", type=float, default=3.0)
    p.add_argument("--sigma_eps", type=float, default=0.3)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--num_chains", type=int, default=2)
    p.add_argument("--num_burnin", type=int, default=400)
    p.add_argument("--num_results", type=int, default=1000)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--step_size", type=float, default=0.05)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--dispersion", type=float, default=0.2)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "v2_three_way_fixes")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = dict(mu=args.mu, phi=args.phi, sigma_eta=args.sigma_eta,
                 A=args.A, sigma_eps=args.sigma_eps)

    # ---- predicted ridge points for each fix ----
    # Scale group: (mu, sigma_eta, A) -> (c*mu, c*sigma_eta, A/c)
    #   - Fix A=1     -> c = A_true              -> mu* = A_true*mu_true,         s*  = A_true*sigma_eta_true, A*=1
    #   - Fix sig_eta=1 -> c = 1/sigma_eta_true  -> mu* = mu_true/sigma_eta_true, s*  = 1,                      A*=A_true*sigma_eta_true
    #   - Fix mu=1    -> c = 1/mu_true           -> mu* = 1,                      s*  = sigma_eta_true/mu_true, A*=A_true*mu_true
    c_for_A1 = args.A
    c_for_sig1 = 1.0 / args.sigma_eta
    c_for_mu1 = 1.0 / args.mu
    ridge_A1 = dict(mu=c_for_A1 * args.mu,
                    sigma_eta=c_for_A1 * args.sigma_eta,
                    A=1.0,
                    sigma_eps=args.sigma_eps)
    ridge_sig1 = dict(mu=c_for_sig1 * args.mu,
                      sigma_eta=1.0,
                      A=args.A / c_for_sig1,
                      sigma_eps=args.sigma_eps)
    ridge_mu1 = dict(mu=1.0,
                     sigma_eta=c_for_mu1 * args.sigma_eta,
                     A=args.A / c_for_mu1,
                     sigma_eps=args.sigma_eps)

    print(f"[v2-3way] truth: {truth}")
    print(f"  predicted ridge points:")
    print(f"    A=1 fix         (c={c_for_A1:.3f}): {ridge_A1}")
    print(f"    sigma_eta=1 fix (c={c_for_sig1:.3f}): {ridge_sig1}")
    print(f"    mu=1 fix        (c={c_for_mu1:.3f}): {ridge_mu1}")

    # ---- data ----
    y_obs, _ = gen_v2_data(args.T, args.mu, args.phi, args.sigma_eta,
                            args.A, args.sigma_eps, seed=args.data_seed)
    print(f"  y_obs range: [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")

    rng = np.random.default_rng(args.base_seed)

    def run_with_target(target_fn, init_raw, names, label, truth_dict):
        nonlocal rng
        samples_all = []
        accs_all = []
        t0 = time.perf_counter()
        for c in range(args.num_chains):
            seed = args.base_seed + 1009 * (c + 1) + hash(label) % 997
            s, a = run_hmc(target_fn, init_raw + args.dispersion *
                            rng.standard_normal(len(init_raw)).astype(np.float32),
                            args.num_burnin, args.num_results,
                            args.step_size, args.L, seed)
            print(f"  [{label} chain {c+1}/{args.num_chains}] "
                  f"accept={a.mean():.3f} elapsed={time.perf_counter()-t0:.1f}s")
            samples_all.append(s)
            accs_all.append(a)
        samples_all = np.stack(samples_all, axis=0)
        return samples_all, accs_all

    # ============ Run 1: A=1 fix ============
    print("\n========== FIXED A=1 ==========")
    target_A1 = make_target_fixed_A(y_obs, A_value=1.0)
    # init raw: [mu, phi_raw, log_sigma_eta_sq, log_sigma_eps_sq]
    init_A1 = np.asarray([
        ridge_A1["mu"],
        math.atanh(min(max(args.phi, -0.999), 0.999)),
        math.log(max(ridge_A1["sigma_eta"] ** 2, 1e-8)),
        math.log(max(args.sigma_eps ** 2, 1e-8)),
    ], dtype=np.float32)
    samples_A1, accs_A1 = run_with_target(
        target_A1, init_A1,
        names=["mu", "phi_raw", "log_s2", "log_se2"],
        label="A=1", truth_dict=None,
    )
    # constrained: [mu, phi, sigma_eta_sq, sigma_eps_sq]
    A1_const = np.stack([
        samples_A1[..., 0],
        np.tanh(samples_A1[..., 1]),
        np.exp(samples_A1[..., 2]),
        np.exp(samples_A1[..., 3]),
    ], axis=-1)
    # Predicted ridge truth in (mu, phi, sigma_eta_sq, sigma_eps_sq):
    A1_truth = dict(mu=ridge_A1["mu"], phi=args.phi,
                    sigma_eta_sq=ridge_A1["sigma_eta"] ** 2,
                    sigma_eps_sq=ridge_A1["sigma_eps"] ** 2)
    A1_summary = summarize(A1_const,
                           names=["mu", "phi", "sigma_eta_sq", "sigma_eps_sq"],
                           truth_dict=A1_truth)

    # ============ Run 2: sigma_eta=1 fix ============
    print("\n========== FIXED sigma_eta=1 ==========")
    target_sig1 = make_target_fixed_sigma_eta(y_obs, sigma_eta_value=1.0)
    init_sig1 = np.asarray([
        ridge_sig1["mu"],
        math.atanh(min(max(args.phi, -0.999), 0.999)),
        math.log(max(ridge_sig1["A"], 1e-8)),
        math.log(max(args.sigma_eps ** 2, 1e-8)),
    ], dtype=np.float32)
    samples_sig1, accs_sig1 = run_with_target(
        target_sig1, init_sig1,
        names=["mu", "phi_raw", "log_A", "log_se2"],
        label="sig=1", truth_dict=None,
    )
    # constrained: [mu, phi, A, sigma_eps_sq]
    sig1_const = np.stack([
        samples_sig1[..., 0],
        np.tanh(samples_sig1[..., 1]),
        np.exp(samples_sig1[..., 2]),
        np.exp(samples_sig1[..., 3]),
    ], axis=-1)
    sig1_truth = dict(mu=ridge_sig1["mu"], phi=args.phi,
                      A=ridge_sig1["A"],
                      sigma_eps_sq=ridge_sig1["sigma_eps"] ** 2)
    sig1_summary = summarize(sig1_const,
                             names=["mu", "phi", "A", "sigma_eps_sq"],
                             truth_dict=sig1_truth)

    # ============ Run 3: mu=1 fix ============
    print("\n========== FIXED mu=1 ==========")
    target_mu1 = make_target_fixed_mu(y_obs, mu_value=1.0)
    init_mu1 = np.asarray([
        math.atanh(min(max(args.phi, -0.999), 0.999)),
        math.log(max(ridge_mu1["sigma_eta"] ** 2, 1e-8)),
        math.log(max(ridge_mu1["A"], 1e-8)),
        math.log(max(args.sigma_eps ** 2, 1e-8)),
    ], dtype=np.float32)
    samples_mu1, accs_mu1 = run_with_target(
        target_mu1, init_mu1,
        names=["phi_raw", "log_s2", "log_A", "log_se2"],
        label="mu=1", truth_dict=None,
    )
    # constrained: [phi, sigma_eta_sq, A, sigma_eps_sq]
    mu1_const = np.stack([
        np.tanh(samples_mu1[..., 0]),
        np.exp(samples_mu1[..., 1]),
        np.exp(samples_mu1[..., 2]),
        np.exp(samples_mu1[..., 3]),
    ], axis=-1)
    mu1_truth = dict(phi=args.phi,
                     sigma_eta_sq=ridge_mu1["sigma_eta"] ** 2,
                     A=ridge_mu1["A"],
                     sigma_eps_sq=ridge_mu1["sigma_eps"] ** 2)
    mu1_summary = summarize(mu1_const,
                            names=["phi", "sigma_eta_sq", "A", "sigma_eps_sq"],
                            truth_dict=mu1_truth)

    # ============ Print results ============
    def print_summary(label, rows):
        print(f"\n=== {label}: posterior summary ===")
        for r in rows:
            tval = r.get("truth", float("nan"))
            ck = "OK" if r.get("covered", False) else "NOT-cov"
            print(f"  {r['param']:>15}  truth={tval:>8.4f}  "
                  f"med={r['median']:>8.4f}  sd={r['sd']:>8.4f}  "
                  f"CI=[{r['q025']:>8.4f}, {r['q975']:>8.4f}]  {ck}")
    print_summary("A=1 fix", A1_summary)
    print_summary("sigma_eta=1 fix", sig1_summary)
    print_summary("mu=1 fix", mu1_summary)

    # ============ Invariance check on phi and sigma_eps ============
    def extract_phi_sigma_eps(const, names):
        idx_phi = names.index("phi")
        idx_se = names.index("sigma_eps_sq")
        return const[..., idx_phi].ravel(), const[..., idx_se].ravel()

    phi_A1, se_A1 = extract_phi_sigma_eps(
        A1_const, ["mu", "phi", "sigma_eta_sq", "sigma_eps_sq"])
    phi_sig1, se_sig1 = extract_phi_sigma_eps(
        sig1_const, ["mu", "phi", "A", "sigma_eps_sq"])
    phi_mu1, se_mu1 = extract_phi_sigma_eps(
        mu1_const, ["phi", "sigma_eta_sq", "A", "sigma_eps_sq"])

    print("\n=== Invariance check: phi and sigma_eps_sq across all three fixes ===")
    print(f"           A=1 fix     sigma_eta=1 fix     mu=1 fix")
    print(f"phi med:   {np.median(phi_A1):8.4f}        {np.median(phi_sig1):8.4f}        {np.median(phi_mu1):8.4f}")
    print(f"phi sd:    {np.std(phi_A1):8.4f}        {np.std(phi_sig1):8.4f}        {np.std(phi_mu1):8.4f}")
    print(f"sig_eps med: {np.median(se_A1):8.4f}      {np.median(se_sig1):8.4f}      {np.median(se_mu1):8.4f}")
    print(f"sig_eps sd:  {np.std(se_A1):8.4f}      {np.std(se_sig1):8.4f}      {np.std(se_mu1):8.4f}")

    # ============ Save outputs ============
    np.savez_compressed(out_dir / "v2_three_way_samples.npz",
                        A1_const=A1_const, sig1_const=sig1_const,
                        mu1_const=mu1_const,
                        y_obs=y_obs.numpy())

    result = {
        "tf_version": tf.__version__,
        "tfp_version": tfp.__version__,
        "truth": truth,
        "ridge_predictions": {
            "A1": ridge_A1, "sigma_eta1": ridge_sig1, "mu1": ridge_mu1,
        },
        "config": {
            "T": args.T, "num_chains": args.num_chains,
            "num_burnin": args.num_burnin, "num_results": args.num_results,
            "L": args.L, "step_size": args.step_size,
            "data_seed": args.data_seed, "base_seed": args.base_seed,
        },
        "A1_summary": A1_summary,
        "sig1_summary": sig1_summary,
        "mu1_summary": mu1_summary,
        "invariance": {
            "phi": {
                "A1":   {"median": float(np.median(phi_A1)),   "sd": float(np.std(phi_A1))},
                "sig1": {"median": float(np.median(phi_sig1)), "sd": float(np.std(phi_sig1))},
                "mu1":  {"median": float(np.median(phi_mu1)),  "sd": float(np.std(phi_mu1))},
            },
            "sigma_eps_sq": {
                "A1":   {"median": float(np.median(se_A1)),   "sd": float(np.std(se_A1))},
                "sig1": {"median": float(np.median(se_sig1)), "sd": float(np.std(se_sig1))},
                "mu1":  {"median": float(np.median(se_mu1)),  "sd": float(np.std(se_mu1))},
            },
        },
        "accept_rates": {
            "A1":   float(np.mean([a.mean() for a in accs_A1])),
            "sig1": float(np.mean([a.mean() for a in accs_sig1])),
            "mu1":  float(np.mean([a.mean() for a in accs_mu1])),
        },
    }
    with open(out_dir / "v2_three_way_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote: {out_dir}/v2_three_way_result.json")
    print(f"       {out_dir}/v2_three_way_samples.npz")


if __name__ == "__main__":
    raise SystemExit(main())
