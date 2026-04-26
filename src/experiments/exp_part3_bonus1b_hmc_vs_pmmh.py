"""
Experiment: L-HNN HMC (differentiable LEDH + OT-Sinkhorn) vs PMMH (bootstrap PF)
for the nonlinear state-space model from Andrieu et al. (2010), section 3.1.

Replaces the standard HMC leapfrog integrator with a Latent Hamiltonian Neural
Network (L-HNN, Dhulipala et al. 2022) so that the expensive particle-filter
backward pass is called only during an upfront training phase and for
accept/reject steps — NOT inside the leapfrog loop.

Key new metrics vs the original experiment
------------------------------------------
  ESS / grad-eval   — primary efficiency metric from the L-HNN paper (Table 1).
                      Counts ALL real particle-filter gradient evaluations:
                        pilot training  +  fallback steps during sampling.
  Gradient savings  — how many fewer PF gradient calls vs traditional HMC.
  Fallback rate     — fraction of leapfrog steps that triggered the online
                      error monitor and fell back to real gradients.

Comparison columns in the results table
----------------------------------------
  PMMH (bootstrap PF)         — random-walk MH baseline
  HMC  (LEDH + OT-Sinkhorn)   — standard HMC using full PF gradients
  L-HNN HMC                   — this experiment's new algorithm

Ablation (Question B)
---------------------
  Sweeps sinkhorn_epsilon × grad_window × n_lambda exactly as before, but
  each ablation config runs the L-HNN sampler.  The L-HNN trained on the
  *baseline* filter config is reused (``pretrained_lhnn``) to avoid
  retraining for every config — appropriate because the posterior geometry
  is the same; only the stochastic likelihood estimator changes.  A flag
  ``--retrain_ablation`` forces independent training per config.

Question B sampler (same output directory, different filenames)
-----------------------------------------------------------------
  ``--second_part --standard`` — each ablation cell runs ``run_hmc`` (classic
  HMC + LEDH likelihood), matching ``exp_part3_bonus1b_hmc_vs_pmmh.py``.  Skips the main L-HNN run when
  Question A is off (only PMMH + baseline HMC are needed).

  ``--second_part --hnn`` (or neither flag) — each cell runs ``run_lhnn_hmc``
  (default).  Reports: ``ablation_lhnn_hmc.txt``, ``ablation_summary_lhnn_hmc.png``.

  Standard mode writes: ``ablation_standard_hmc.txt``, ``ablation_summary_standard_hmc.png``.

Usage
-----
    python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh                                               # run both parts (default)
    python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh --first_part                                  # baseline comparison only (Question A)
    python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh --second_part                                 # ablation only (Question B)
    python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh --full_grid                                   # full 3×2×2 ablation (Question B)
    python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh --lhnn_epochs 5000 --lhnn_trajectories 50     # L-HNN hyper-parameters (Question B)
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.filters.bonus.hmc_pf import HMCResult, run_hmc          # kept for 3-way comparison
from src.filters.bonus.lhnn_hmc_pf import (
    LHNNConfig,
    LatentHNN,
    run_lhnn_hmc,
    ess_per_gradient,
)
from src.filters.bonus.pmmh import PMMHResult, bootstrap_pf_log_likelihood, run_pmmh
from src.models.ssm_katigawa import PMCMCNonlinearSSM

tfd = tfp.distributions

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    plt = None


# ---------------------------------------------------------------------------
# Data generation (unchanged)
# ---------------------------------------------------------------------------

def generate_data(
    T: int = 100,
    sigma_v_sq: float = 10.0,
    sigma_w_sq: float = 1.0,
    seed: int = 42,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simulate from the nonlinear SSM (Andrieu et al., Eqs. 14–15)."""
    tf.random.set_seed(seed)
    sv = tf.sqrt(tf.cast(sigma_v_sq, tf.float32))
    sw = tf.sqrt(tf.cast(sigma_w_sq, tf.float32))

    x = tf.random.normal([]) * tf.sqrt(tf.constant(5.0, tf.float32))
    xs, ys = [x], [x**2 / 20.0 + sw * tf.random.normal([])]

    for t in range(2, T + 1):
        t_f = tf.cast(t, tf.float32)
        x = (
            0.5 * x
            + 25.0 * x / (1.0 + x**2)
            + 8.0 * tf.cos(1.2 * t_f)
            + sv * tf.random.normal([])
        )
        ys.append(x**2 / 20.0 + sw * tf.random.normal([]))
        xs.append(x)

    return tf.stack(xs), tf.stack(ys)


# ---------------------------------------------------------------------------
# Target log-probability (unchanged)
# ---------------------------------------------------------------------------

def make_target_log_prob(y_obs: tf.Tensor, pf_log_lik_fn: Callable) -> Callable:
    """
    Unnormalised log-posterior for (log sigma_v^2, log sigma_w^2).

    Priors: sigma_v^2 ~ InvGamma(0.01, 0.01), sigma_w^2 ~ InvGamma(0.01, 0.01).
    Jacobian |d sigma^2 / d log sigma^2| = sigma^2 included.
    """
    prior_v = tfd.InverseGamma(concentration=0.01, scale=0.01)
    prior_w = tfd.InverseGamma(concentration=0.01, scale=0.01)

    def target(theta: tf.Tensor) -> tf.Tensor:
        log_sv2, log_sw2 = theta[0], theta[1]
        sv2, sw2 = tf.exp(log_sv2), tf.exp(log_sw2)
        lp_prior = prior_v.log_prob(sv2) + prior_w.log_prob(sw2)
        jacobian = log_sv2 + log_sw2
        ssm = PMCMCNonlinearSSM(sigma_v_sq=sv2, sigma_w_sq=sw2)
        ll = pf_log_lik_fn(ssm, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-1e6, tf.float32))
        result = lp_prior + jacobian + ll
        result = tf.math.real(result)
        result = tf.cast(result, tf.float32)
        return tf.where(tf.math.is_finite(result), result, tf.constant(-1e6, tf.float32))

    return target


# ---------------------------------------------------------------------------
# Diagnostics (unchanged helpers)
# ---------------------------------------------------------------------------

def compute_acf(chain: tf.Tensor, max_lag: int = 50) -> np.ndarray:
    n = chain.shape[0]
    c = chain - tf.reduce_mean(chain)
    var = tf.maximum(tf.reduce_sum(c**2) / tf.cast(n, tf.float32), 1e-12)
    acf = [1.0]
    for k in range(1, max_lag):
        cov_k = tf.reduce_sum(c[:-k] * c[k:]) / tf.cast(n, tf.float32)
        acf.append(float((cov_k / var).numpy()))
    return np.array(acf)


def rmse(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.sqrt(tf.reduce_mean((samples - true_val) ** 2)).numpy())


def mae(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.reduce_mean(tf.abs(samples - true_val)).numpy())


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a_sorted, b_sorted = np.sort(a), np.sort(b)
    combined = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, combined, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, combined, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    xs = np.sort(np.unique(np.concatenate([a, b])))
    a_s, b_s = np.sort(a), np.sort(b)
    fa = np.searchsorted(a_s, xs, side="right") / a_s.size
    fb = np.searchsorted(b_s, xs, side="right") / b_s.size
    return float(np.trapz(np.abs(fa - fb), xs))


def per_step_cost(total_time: float, n_samples: int, n_burnin: int) -> float:
    return total_time / (n_samples + n_burnin)


# ---------------------------------------------------------------------------
# Gradient statistics (unchanged)
# ---------------------------------------------------------------------------

def grad_stats_for_target(
    target_log_prob_fn: Callable,
    theta0: tf.Tensor,
    crn_seeds: list,
    horizon_fractions: Optional[list] = None,
    y_obs_full: Optional[tf.Tensor] = None,
    pf_log_lik_fn: Optional[Callable] = None,
) -> dict:
    grad_norms, grad_components = [], []
    for s in crn_seeds:
        tf.random.set_seed(s)
        q_t = tf.identity(theta0)
        with tf.GradientTape() as tape:
            tape.watch(q_t)
            lp = target_log_prob_fn(q_t)
        grad = tape.gradient(lp, q_t)
        if grad is None:
            grad = tf.zeros_like(q_t)
        mask = tf.reduce_all(tf.math.is_finite(grad))
        grad = tf.where(mask, grad, tf.zeros_like(grad))
        gnorm = tf.cast(tf.math.real(tf.linalg.norm(grad)), tf.float32)
        grad_norms.append(float(gnorm.numpy()))
        grad_components.append(grad)

    gnorms = np.asarray(grad_norms, dtype=np.float64)
    fin = np.isfinite(gnorms)
    gnorms_f = gnorms[fin]
    gcomp = np.asarray([g.numpy() for g in grad_components], dtype=np.float64)

    if gnorms_f.size == 0:
        gm = gs = gmx = gcv = gmed = gmad = gcv_mad = np.nan
        g_log10_med = g_log10_mad = np.nan
    else:
        gm = float(np.mean(gnorms_f))
        gs = float(np.std(gnorms_f))
        gmx = float(np.max(gnorms_f))
        gcv = float(gs / (gm + 1e-12))
        gmed = float(np.median(gnorms_f))
        gmad = float(np.median(np.abs(gnorms_f - gmed)))
        gcv_mad = float(gmad / (gmed + 1e-12))
        log10_g = np.log10(np.maximum(gnorms_f, 1e-300))
        g_log10_med = float(np.median(log10_g))
        g_log10_mad = float(np.median(np.abs(log10_g - g_log10_med)))

    result = {
        "grad_norm_mean": gm, "grad_norm_std": gs, "grad_norm_max": gmx,
        "grad_norm_median": gmed, "grad_norm_mad": gmad,
        "grad_log10_median": g_log10_med, "grad_log10_mad": g_log10_mad,
        "grad_cv": gcv, "grad_cv_mad": gcv_mad,
        "grad_mean_components": np.nanmean(gcomp, axis=0).tolist(),
        "grad_std_components": np.nanstd(gcomp, axis=0).tolist(),
        "horizon_variance": {},
    }

    if (
        horizon_fractions is not None
        and y_obs_full is not None
        and pf_log_lik_fn is not None
    ):
        T_full = int(y_obs_full.shape[0])
        for frac in horizon_fractions:
            T_h = max(1, int(T_full * frac))
            y_h = y_obs_full[:T_h]
            target_h = make_target_log_prob(y_h, pf_log_lik_fn)
            h_norms = []
            for s in crn_seeds:
                tf.random.set_seed(s)
                q_t = tf.identity(theta0)
                with tf.GradientTape() as tape:
                    tape.watch(q_t)
                    lp = target_h(q_t)
                grad = tape.gradient(lp, q_t)
                if grad is None:
                    grad = tf.zeros_like(q_t)
                mask = tf.reduce_all(tf.math.is_finite(grad))
                grad = tf.where(mask, grad, tf.zeros_like(grad))
                hn = tf.cast(tf.math.real(tf.linalg.norm(grad)), tf.float32)
                h_norms.append(float(hn.numpy()))
            h_arr = np.asarray(h_norms, dtype=np.float64)
            h_arr = h_arr[np.isfinite(h_arr)]
            if h_arr.size == 0:
                hm = hs = hcv = np.nan
            else:
                hm = float(np.mean(h_arr))
                hs = float(np.std(h_arr))
                hcv = float(hs / (hm + 1e-12))
            result["horizon_variance"][f"T={T_h}"] = {"mean": hm, "std": hs, "cv": hcv}

    return result


# ---------------------------------------------------------------------------
# Chain summary (unchanged)
# ---------------------------------------------------------------------------

def summarise_chain(
    samples: tf.Tensor,
    name: str,
    param_names: list,
    true_values: tf.Tensor,
) -> None:
    ess = tfp.mcmc.effective_sample_size(samples)
    for j, pname in enumerate(param_names):
        chain_j = samples[:, j]
        tf.print(
            f"  {name} {pname}:",
            "  mean =", tf.reduce_mean(chain_j),
            "  std =", tf.math.reduce_std(chain_j),
            "  ESS =", float(ess[j].numpy()),
            "  true =", true_values[j],
        )


# ---------------------------------------------------------------------------
# Plotting — updated for 3-way comparison
# ---------------------------------------------------------------------------

def plot_diagnostics(
    pmmh_samples: tf.Tensor,
    hmc_samples: tf.Tensor,
    lhnn_samples: tf.Tensor,
    true_values: tf.Tensor,
    param_names: list,
    out_dir: Path,
) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = len(param_names)
    max_lag = 50

    # ---------- Trace + posterior (3 chains) ----------
    fig, axes = plt.subplots(n_params, 2, figsize=(14, 5 * n_params))
    if n_params == 1:
        axes = axes[None, :]

    for j in range(n_params):
        pm_j  = pmmh_samples[:, j].numpy()
        hm_j  = hmc_samples[:, j].numpy()
        lh_j  = lhnn_samples[:, j].numpy()
        tv_j  = float(true_values[j].numpy())

        ax = axes[j, 0]
        ax.plot(pm_j, alpha=0.55, lw=0.8, label="PMMH")
        ax.plot(hm_j, alpha=0.55, lw=0.8, label="HMC-LEDH")
        ax.plot(lh_j, alpha=0.55, lw=0.8, label="L-HNN HMC", ls="--")
        ax.axhline(tv_j, color="k", ls="--", lw=1.2, label="true")
        ax.set_ylabel(param_names[j])
        ax.set_xlabel("iteration")
        ax.set_title(f"Trace — {param_names[j]}", fontweight="bold")
        ax.legend(fontsize=8)

        ax = axes[j, 1]
        ax.hist(pm_j, bins=40, density=True, alpha=0.45, label="PMMH")
        ax.hist(hm_j, bins=40, density=True, alpha=0.45, label="HMC-LEDH")
        ax.hist(lh_j, bins=40, density=True, alpha=0.45, label="L-HNN HMC")
        ax.axvline(tv_j, color="k", ls="--", lw=1.2, label="true")

        # L-HNN vs PMMH distributional agreement
        ks_lh  = ks_stat(pm_j, lh_j)
        w1_lh  = wasserstein1(pm_j, lh_j)
        ks_hmc = ks_stat(pm_j, hm_j)
        w1_hmc = wasserstein1(pm_j, hm_j)
        ax.set_title(
            f"Posterior — {param_names[j]}\n"
            f"L-HNN: KS={ks_lh:.3f} W1={w1_lh:.3f}  |  "
            f"HMC: KS={ks_hmc:.3f} W1={w1_hmc:.3f}",
            fontweight="bold", fontsize=8,
        )
        ax.set_xlabel(param_names[j])
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison" / "trace_posterior.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- ACF — 3 chains ----------
    fig, axes = plt.subplots(n_params, 3, figsize=(18, 5 * n_params))
    if n_params == 1:
        axes = axes[None, :]

    chains_labels = [
        (pmmh_samples, "PMMH"),
        (hmc_samples,  "HMC-LEDH"),
        (lhnn_samples, "L-HNN HMC"),
    ]
    for j in range(n_params):
        tv_j = float(true_values[j].numpy())
        for col, (chain_t, label) in enumerate(chains_labels):
            chain_arr = chain_t[:, j].numpy()
            acf_vals  = compute_acf(tf.constant(chain_arr), max_lag)
            ci        = 1.96 / np.sqrt(len(chain_arr))

            ax = axes[j, col]
            ax.bar(range(max_lag), acf_vals, width=0.6, alpha=0.7)
            ax.axhline(0,   color="k", lw=0.5)
            ax.axhline( ci, color="r", ls="--", lw=0.8, label="±1.96/√n")
            ax.axhline(-ci, color="r", ls="--", lw=0.8)
            ax.set_title(f"ACF — {label} — {param_names[j]}", fontweight="bold")
            ax.set_xlabel("lag")
            ax.set_ylabel("autocorrelation")
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison" / "acf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- ESS / grad comparison bar chart ----------
    # (populated by save_results, stored in a side file — see below)


def plot_ess_per_grad(
    methods: list,
    ess_per_grad_vals: list,
    out_dir: Path,
) -> None:
    """Bar chart comparing ESS/gradient-eval for each method."""
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(methods, ess_per_grad_vals, color=colors[: len(methods)], alpha=0.85)
    ax.set_ylabel("ESS / gradient evaluation")
    ax.set_title("Sampling efficiency: ESS per gradient evaluation\n(higher is better)",
                 fontweight="bold")
    for bar, val in zip(bars, ess_per_grad_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.3g}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison" / "ess_per_grad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_summary(
    hmc_grid_results: list,
    out_dir: Path,
    param_names: list,
    *,
    ablation_mode: str = "lhnn",
) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    is_lhnn = ablation_mode == "lhnn"
    fig_name = (
        "ablation_summary_lhnn_hmc.png" if is_lhnn else "ablation_summary_standard_hmc.png"
    )

    labels       = [r["config_short"]          for r in hmc_grid_results]
    ess_per_s    = [r["ess_per_s"]             for r in hmc_grid_results]
    ess_per_grad = [r.get("ess_per_grad", 0.0) for r in hmc_grid_results]
    bias_v       = [r["bias_true"][0]          for r in hmc_grid_results]
    bias_w       = [r["bias_true"][1]          for r in hmc_grid_results]
    grad_spread  = [r["grad_stats"]["grad_log10_mad"] for r in hmc_grid_results]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # (0,0) Bias vs ESS/s (sigma_v^2)
    # (0,1) Bias vs ESS/s (sigma_w^2)
    for ax, bias, title in [
        (axes[0, 0], bias_v, f"Bias vs ESS/s ({param_names[0]})"),
        (axes[0, 1], bias_w, f"Bias vs ESS/s ({param_names[1]})"),
    ]:
        ax.scatter(ess_per_s, bias, s=60, alpha=0.8, zorder=3)
        for lbl, x, y in zip(labels, ess_per_s, bias):
            ax.annotate(lbl, (x, y), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel("ESS / s")
        ax.set_ylabel("Abs bias to truth")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # (0,2) ESS / gradient-eval per config (the paper's primary metric)
    x_pos = np.arange(len(labels))
    axes[0, 2].bar(x_pos, ess_per_grad, alpha=0.75, color="#55A868")
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[0, 2].set_ylabel("ESS / gradient evaluation")
    axes[0, 2].set_title(
        "L-HNN ESS per gradient eval\n(primary metric)"
        if is_lhnn
        else "HMC ESS per gradient eval\n(ESS / (L+1) per iter)",
        fontweight="bold",
    )
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    # (1,0) log10-space gradient spread (MAD)
    axes[1, 0].bar(x_pos, grad_spread, alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("MAD of log10 ||grad||")
    axes[1, 0].set_title("Gradient spread in log10 space", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # (1,1) Multi-horizon gradient std
    horizon_data = {
        r["config_short"]: r["grad_stats"].get("horizon_variance", {})
        for r in hmc_grid_results
    }
    has_horizon = any(bool(v) for v in horizon_data.values())
    if has_horizon:
        ax4 = axes[1, 1]
        for cfg_label, hv in horizon_data.items():
            if not hv:
                continue
            t_keys = sorted(hv.keys(), key=lambda s: int(s.split("=")[1]))
            t_vals = [int(k.split("=")[1]) for k in t_keys]
            stds = [hv[k]["std"] for k in t_keys]
            ax4.plot(t_vals, stds, marker="o", label=cfg_label)
        ax4.set_xlabel("Horizon T")
        ax4.set_ylabel("Gradient norm std across seeds")
        ax4.set_title("Gradient variance vs horizon", fontweight="bold")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")

    # (1,2) Fallback rate per config (L-HNN only; zeros for standard HMC)
    fallback_rates = [r.get("fallback_rate", 0.0) for r in hmc_grid_results]
    axes[1, 2].bar(x_pos, fallback_rates, alpha=0.75, color="#C44E52")
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[1, 2].set_ylabel("Fallback rate")
    axes[1, 2].set_title(
        "Online error monitor fallback rate\n(fraction of leapfrog steps)"
        if is_lhnn
        else "L-HNN fallback (N/A for standard HMC)",
        fontweight="bold",
    )
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_dir / fig_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results saving — updated for 3-way comparison
# ---------------------------------------------------------------------------

def save_results(
    pmmh_result:  PMMHResult,
    hmc_result:   HMCResult,
    lhnn_result:  HMCResult,
    pmmh_samples: tf.Tensor,
    hmc_samples:  tf.Tensor,
    lhnn_samples: tf.Tensor,
    t_pmmh:       float,
    t_hmc:        float,
    t_lhnn:       float,
    n_samp:       int,
    n_burn:       int,
    true_values:  tf.Tensor,
    param_names:  list,
    out_dir:      Path,
    # --- new L-HNN cost breakdown ---
    lhnn_training_grad_evals:  int,
    lhnn_sampling_grad_evals:  int,
    lhnn_num_leapfrog_steps:   int,
) -> dict:
    """
    Write the 3-way comparison table and return a dict of key metrics for
    downstream use (e.g. the ESS/grad bar chart).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ess_pm   = tfp.mcmc.effective_sample_size(pmmh_samples)
    ess_hmc  = tfp.mcmc.effective_sample_size(hmc_samples)
    ess_lhnn = tfp.mcmc.effective_sample_size(lhnn_samples)

    mean_ess_pm   = float(tf.reduce_mean(ess_pm).numpy())
    mean_ess_hmc  = float(tf.reduce_mean(ess_hmc).numpy())
    mean_ess_lhnn = float(tf.reduce_mean(ess_lhnn).numpy())

    acc_pm   = float(pmmh_result.accept_rate.numpy())
    acc_hmc  = float(hmc_result.accept_rate.numpy())
    acc_lhnn = float(lhnn_result.accept_rate.numpy())

    cost_pm   = per_step_cost(t_pmmh,  n_samp, n_burn)
    cost_hmc  = per_step_cost(t_hmc,   n_samp, n_burn)
    cost_lhnn = per_step_cost(t_lhnn,  n_samp, n_burn)

    # ESS / gradient-evaluation  (the L-HNN paper's Table 1 metric)
    # PMMH uses random-walk proposals — no gradient evaluations at all.
    # Standard HMC: (L+1) gradient evals per iteration × total iterations.
    total_iters = n_samp + n_burn
    hmc_total_grads  = (lhnn_num_leapfrog_steps + 1) * total_iters
    lhnn_total_grads = lhnn_training_grad_evals + lhnn_sampling_grad_evals

    # ESS per gradient (mean over params)
    ess_per_grad_hmc  = mean_ess_hmc  / max(hmc_total_grads,  1)
    ess_per_grad_lhnn = mean_ess_lhnn / max(lhnn_total_grads, 1)

    # Fallback rate: fraction of leapfrog steps using real gradients
    lhnn_total_leapfrog = lhnn_num_leapfrog_steps * total_iters
    fallback_rate = lhnn_sampling_grad_evals / max(lhnn_total_leapfrog, 1)

    # Gradient savings vs traditional HMC
    saved     = hmc_total_grads - lhnn_total_grads
    saved_pct = 100.0 * saved / max(hmc_total_grads, 1)

    col_w = 34
    lines = [
        "=" * 85,
        "HMC (LEDH + OT-Sinkhorn)  vs  L-HNN HMC  vs  PMMH (Bootstrap PF)",
        "Model: Andrieu et al. (2010), Eq 14-15  |  Prior: InvGamma(0.01, 0.01)",
        "-" * 85,
        "Note on wall-clock vs gradient efficiency",
        "  L-HNN's advantage is in TARGET GRADIENT evaluations, not raw wall time.",
        "  On low-d problems the NN GradientTape per leapfrog step costs as much as",
        "  the LEDH call itself, so ESS/s may not improve even when ESS/grad does.",
        "  The speedup grows with d (higher-d posteriors) and with L (more leapfrog",
        "  steps per proposal), because the amortised training cost shrinks relative",
        "  to the sampling cost: savings = (L+1)×total_iters − training_evals.",
        "=" * 85,
        f"{'Metric':<{col_w}} {'PMMH':>14} {'HMC-LEDH':>14} {'L-HNN HMC':>14}",
        "-" * 85,
        f"{'Acceptance rate':<{col_w}} {acc_pm:>14.3f} {acc_hmc:>14.3f} {acc_lhnn:>14.3f}",
    ]

    for j, pname in enumerate(param_names):
        e_pm   = float(ess_pm[j].numpy())
        e_hmc  = float(ess_hmc[j].numpy())
        e_lhnn = float(ess_lhnn[j].numpy())
        lines.append(
            f"{'ESS (' + pname + ')':<{col_w}} {e_pm:>14.1f} {e_hmc:>14.1f} {e_lhnn:>14.1f}"
        )

    lines += [
        f"{'Mean ESS':<{col_w}} {mean_ess_pm:>14.1f} {mean_ess_hmc:>14.1f} {mean_ess_lhnn:>14.1f}",
        f"{'Runtime (s)':<{col_w}} {t_pmmh:>14.1f} {t_hmc:>14.1f} {t_lhnn:>14.1f}",
        f"{'ESS/s':<{col_w}} {mean_ess_pm/t_pmmh:>14.3f} {mean_ess_hmc/t_hmc:>14.3f} {mean_ess_lhnn/t_lhnn:>14.3f}",
        f"{'Cost per step (s/proposal)':<{col_w}} {cost_pm:>14.4f} {cost_hmc:>14.4f} {cost_lhnn:>14.4f}",
        "-" * 85,
    ]

    for j, pname in enumerate(param_names):
        m_pm   = float(tf.reduce_mean(pmmh_samples[:, j]).numpy())
        m_hmc  = float(tf.reduce_mean(hmc_samples[:, j]).numpy())
        m_lhnn = float(tf.reduce_mean(lhnn_samples[:, j]).numpy())
        tv = float(true_values[j].numpy())
        lines.append(
            f"{'Mean ' + pname:<{col_w}} {m_pm:>14.3f} {m_hmc:>14.3f} {m_lhnn:>14.3f}"
            f"  (true={tv:.3f})"
        )

    for j, pname in enumerate(param_names):
        s_pm   = float(tf.math.reduce_std(pmmh_samples[:, j]).numpy())
        s_hmc  = float(tf.math.reduce_std(hmc_samples[:, j]).numpy())
        s_lhnn = float(tf.math.reduce_std(lhnn_samples[:, j]).numpy())
        lines.append(
            f"{'Std ' + pname:<{col_w}} {s_pm:>14.3f} {s_hmc:>14.3f} {s_lhnn:>14.3f}"
        )

    # --- RMSE / MAE ---
    lines.append("-" * 85)
    lines.append("Quality vs ground truth:")
    for j, pname in enumerate(param_names):
        tv = float(true_values[j].numpy())
        lines.append(
            f"  RMSE ({pname:<16}) PMMH={rmse(pmmh_samples[:, j], tv):.4f}"
            f"  HMC={rmse(hmc_samples[:, j], tv):.4f}"
            f"  L-HNN={rmse(lhnn_samples[:, j], tv):.4f}"
        )
        lines.append(
            f"  MAE  ({pname:<16}) PMMH={mae(pmmh_samples[:, j], tv):.4f}"
            f"  HMC={mae(hmc_samples[:, j], tv):.4f}"
            f"  L-HNN={mae(lhnn_samples[:, j], tv):.4f}"
        )

    # --- Distributional agreement (vs PMMH as reference) ---
    lines.append("-" * 85)
    lines.append("Distributional agreement with PMMH (KS / W1 — lower is better):")
    for j, pname in enumerate(param_names):
        pm_np  = pmmh_samples[:, j].numpy()
        hm_np  = hmc_samples[:, j].numpy()
        lh_np  = lhnn_samples[:, j].numpy()
        ks_hmc  = ks_stat(pm_np, hm_np)
        ks_lhnn = ks_stat(pm_np, lh_np)
        w1_hmc  = wasserstein1(pm_np, hm_np)
        w1_lhnn = wasserstein1(pm_np, lh_np)
        lines.append(
            f"  {pname:<20}  HMC: KS={ks_hmc:.4f} W1={w1_hmc:.4f}"
            f"   L-HNN: KS={ks_lhnn:.4f} W1={w1_lhnn:.4f}"
        )

    # --- L-HNN gradient cost breakdown (the key new section) ---
    lines += [
        "-" * 85,
        "L-HNN gradient cost breakdown (Dhulipala et al. 2022, Table 1 metric):",
        f"  Training (pilot) gradient evals : {lhnn_training_grad_evals}",
        f"  Sampling fallback gradient evals : {lhnn_sampling_grad_evals}",
        f"  Total L-HNN gradient evals       : {lhnn_total_grads}",
        f"  Traditional HMC gradient evals   : {hmc_total_grads}  ((L+1)×iters = ({lhnn_num_leapfrog_steps}+1)×{total_iters})",
        f"  Gradient evals saved             : {saved} ({saved_pct:.1f}%)",
        f"  Fallback intensity (∇ / step)    : {fallback_rate:.3f}",
        f"  Break-even iters (train÷(L+1))   : {lhnn_training_grad_evals // (lhnn_num_leapfrog_steps + 1)}"
        f"  (L-HNN wins beyond this many iterations)",
        "-" * 85,
        "ESS per gradient evaluation (primary L-HNN efficiency metric):",
        f"  HMC-LEDH  : {ess_per_grad_hmc:.4e}",
        f"  L-HNN HMC : {ess_per_grad_lhnn:.4e}",
        f"  Speedup   : {ess_per_grad_lhnn / max(ess_per_grad_hmc, 1e-20):.2f}×",
        "=" * 85,
    ]

    report = "\n".join(lines)
    path = out_dir / "comparison" / "results.txt"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved results to {path}")

    return {
        "ess_per_grad_hmc":  ess_per_grad_hmc,
        "ess_per_grad_lhnn": ess_per_grad_lhnn,
        "fallback_rate":     fallback_rate,
        "saved_pct":         saved_pct,
    }


# ---------------------------------------------------------------------------
# Ablation save — updated for L-HNN metrics
# ---------------------------------------------------------------------------

def save_ablation_results(
    out_dir: Path,
    pmmh_samples: tf.Tensor,
    hmc_grid_results: list,
    true_values: tf.Tensor,
    param_names: list,
    *,
    ablation_mode: str = "lhnn",
) -> None:
    """
    ablation_mode: ``\"lhnn\"`` (default) or ``\"standard\"`` (``run_hmc`` cells).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    is_lhnn = ablation_mode == "lhnn"
    fname = (
        "ablation_lhnn_hmc.txt" if is_lhnn else "ablation_standard_hmc.txt"
    )
    path = out_dir / fname

    col_w = 26
    header = (
        f"{'config':<{col_w}} "
        f"{'accept':>8} {'ess_mean':>9} {'ess/s':>8} {'ess/grad':>10} "
        f"{'fallback':>9} {'cost/step':>10} "
        f"{'bias_v':>8} {'bias_w':>8} "
        f"{'lg_mdn':>8} {'lg_mad':>7}"
    )
    title = (
        "L-HNN HMC ablation: sinkhorn_epsilon × grad_window × n_lambda"
        if is_lhnn
        else "Standard HMC ablation: sinkhorn_epsilon × grad_window × n_lambda (run_hmc per cell)"
    )
    note_a = (
        "  Metric ess/grad = ESS / total-grad-evals (training + fallback) — Table 1 of Dhulipala et al. (2022)."
        if is_lhnn
        else "  Metric ess/grad = ESS / ((L+1) × MCMC iters) — full HMC gradient budget per iteration."
    )
    note_b = (
        "  fallback = fraction of leapfrog steps using real gradients (online error monitor)."
        if is_lhnn
        else "  fallback = 0 (no L-HNN online monitor; column shown for table alignment)."
    )
    lines = [
        "=" * 120,
        title,
        note_a,
        note_b,
        "=" * 120,
        header,
        "-" * 120,
    ]
    pmmh_mean = tf.reduce_mean(pmmh_samples, axis=0)

    for r in hmc_grid_results:
        bv, bw = r["bias_true"]
        g = r["grad_stats"]
        lines.append(
            f"{r['config_short']:<{col_w}} "
            f"{r['accept_rate']:>8.3f} {r['ess_mean']:>9.1f} {r['ess_per_s']:>8.3f} "
            f"{r.get('ess_per_grad', 0.0):>10.3e} "
            f"{r.get('fallback_rate', 0.0):>9.3f} {r['cost_per_step']:>10.5f} "
            f"{bv:>8.4f} {bw:>8.4f} "
            f"{g['grad_log10_median']:>8.3f} {g['grad_log10_mad']:>7.3f}"
        )

    # Multi-horizon section
    horizon_rows = []
    for r in hmc_grid_results:
        hv = r["grad_stats"].get("horizon_variance", {})
        if hv:
            row = f"  {r['config_short']}: " + "  ".join(
                f"T={k.split('=')[1]} cv={v['cv']:.3f}"
                for k, v in sorted(hv.items(), key=lambda x: int(x[0].split("=")[1]))
            )
            horizon_rows.append(row)
    if horizon_rows:
        lines.append("-" * 120)
        lines.append("Gradient CV across horizons:")
        lines.extend(horizon_rows)

    notes_tail = [
        "-" * 120,
        "Notes:",
    ]
    if is_lhnn:
        notes_tail += [
            "  ess/grad   = mean_ESS / (lhnn_training_evals + fallback_evals).",
            "  fallback   = sampling fallback steps / total leapfrog steps.",
            "  Ablation uses the L-HNN trained on baseline config (eps=2.0,gw=1,nl=5).",
            "  For independent L-HNN training per config use --retrain_ablation.",
        ]
    else:
        notes_tail += [
            "  ess/grad   = mean_ESS / ((L+1) × chain_length) for standard HMC.",
            "  fallback   = 0 (not applicable).",
        ]
    notes_tail.append("=" * 120)
    lines += notes_tail
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved ablation report to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_ablation_configs(full_grid: bool) -> list:
    if full_grid:
        return [
            {
                "sinkhorn_epsilon": float(eps),
                "grad_window": int(gw),
                "n_lambda": int(nl),
                "short": f"eps={eps},gw={gw},nl={nl}",
            }
            for eps, gw, nl in itertools.product([0.5, 1.0, 2.0], [1, 5], [3, 5])
        ]

    return [
        {"sinkhorn_epsilon": 0.5, "grad_window": 1, "n_lambda": 5, "short": "eps=0.5,gw=1,nl=5"},
        {"sinkhorn_epsilon": 1.0, "grad_window": 1, "n_lambda": 5, "short": "eps=1.0,gw=1,nl=5"},
        {"sinkhorn_epsilon": 2.0, "grad_window": 5, "n_lambda": 5, "short": "eps=2.0,gw=5,nl=5"},
        {"sinkhorn_epsilon": 2.0, "grad_window": 1, "n_lambda": 3, "short": "eps=2.0,gw=1,nl=3"},
        {"sinkhorn_epsilon": 1.0, "grad_window": 1, "n_lambda": 3, "short": "eps=1.0,gw=1,nl=3"},
        {"sinkhorn_epsilon": 2.0, "grad_window": 0, "n_lambda": 5, "short": "eps=2.0,gw=0,nl=5"},
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare L-HNN HMC (DifferentiableLEDH + OT) vs standard HMC vs PMMH."
    )
    p.add_argument("--first_part",  action="store_true",
                   help="Question A: 3-way baseline comparison.")
    p.add_argument("--second_part", action="store_true",
                   help="Question B: ablation over OT/gradient/differentiability.")
    p.add_argument("--full_grid",   action="store_true",
                   help="Run full 3×2×2 ablation (12 configs) instead of budget subset.")
    p.add_argument("--retrain_ablation", action="store_true",
                   help="Train a new L-HNN for each ablation config (slower but independent).")
    p.add_argument("--standard", action="store_true",
                   help="Question B: ablation runs run_hmc per cell (not L-HNN). Implies baseline HMC.")
    p.add_argument("--hnn", action="store_true",
                   help="Question B: ablation runs run_lhnn_hmc per cell (default if neither --standard nor --hnn).")
    # L-HNN hyper-parameters exposed for quick tuning from CLI
    p.add_argument("--lhnn_epochs",        type=int,   default=3_000,
                   help="L-HNN training epochs (default 3000).")
    p.add_argument("--lhnn_trajectories",  type=int,   default=50,
                   help="Pilot trajectories for training data (default 20).")
    p.add_argument("--lhnn_pilot_steps",   type=int,   default=50,
                   help="Leapfrog steps per pilot trajectory (default 30).")
    p.add_argument("--lhnn_hidden",        type=int,   default=256,
                   help="L-HNN hidden layer width (default 256).")
    p.add_argument("--lhnn_layers",        type=int,   default=3,
                   help="L-HNN number of hidden layers (default 3).")
    p.add_argument("--lhnn_error_thresh",  type=float, default=10.0,
                   help="Online error monitor threshold Δmax_hnn (default 10.0).")
    p.add_argument("--lhnn_cooldown",      type=int,   default=10,
                   help="Cooldown steps N_lf after fallback trigger (default 10).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    run_A = bool(args.first_part)
    run_B = bool(args.second_part)
    if not run_A and not run_B:
        run_A = run_B = True

    if bool(args.standard) and bool(args.hnn):
        raise SystemExit("Use at most one of --standard and --hnn for Question B ablation.")
    ablation_standard = bool(args.standard)
    # Default Question B mode is L-HNN unless --standard is set.
    need_main_hmc = run_A or (run_B and ablation_standard)
    need_main_lhnn = run_A or (run_B and not ablation_standard)

    tf.random.set_seed(123)

    # ---- Configuration ----
    # Gradient cost rationale
    # -----------------------
    # L-HNN training cost is FIXED regardless of n_samp: num_traj × (steps+1).
    # HMC sampling cost scales as (L+1) × (n_burn + n_samp).
    # For the L-HNN advantage to be visible we need:
    #   (L+1) × total_iters  >>  num_traj × (steps+1)
    #
    # Old settings:  (5+1)×250=1500  vs  30×51=1530  → identical, no savings shown.
    # New settings:  (10+1)×750=8250 vs  20×31=620   → 13× fewer gradient evals.
    #
    # Mixing improvement: L=10 gives longer trajectories; adapt_step_size=True
    # during burn-in tunes ε automatically to the target acceptance rate.
    T           = 50
    true_sv2    = 10.0
    true_sw2    = 1.0
    N_ledh      = 50
    N_bpf       = 500
    n_samp      = 500   # was 150 — more samples to amortise training cost and improve ESS
    n_burn      = 250   # was 100 — longer burn-in for step-size adaptation to stabilise
    L           = 10    # was  5  — longer trajectories; HMC cost ∝ L+1 so savings scale up
    param_names  = ["sigma_v^2", "sigma_w^2"]
    true_values  = tf.constant([true_sv2, true_sw2], tf.float32)
    out_dir      = Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH")

    print("=" * 70)
    print("  L-HNN HMC (LEDH + OT-Sinkhorn)  vs  PMMH (Bootstrap PF)")
    print(f"  T={T}  N_ledh={N_ledh}  N_bpf={N_bpf}")
    print(f"  HMC: L={L}  samples={n_samp}  burn={n_burn}")
    print(f"  True: sigma_v^2={true_sv2}  sigma_w^2={true_sw2}")
    print("=" * 70)

    _, y_obs    = generate_data(T, true_sv2, true_sw2, seed=42)
    init_state  = tf.stack([tf.math.log(8.0), tf.math.log(1.5)])

    # ---- L-HNN config from CLI ----
    lhnn_cfg = LHNNConfig(
        hidden_units               = args.lhnn_hidden,
        num_hidden                 = args.lhnn_layers,
        epochs                     = args.lhnn_epochs,
        # 20 traj × 31 evals = 620 training grads vs 8250 for HMC → 13× cheaper
        num_pilot_trajectories     = args.lhnn_trajectories,   # CLI default: 20
        pilot_steps_per_trajectory = args.lhnn_pilot_steps,    # CLI default: 30
        error_threshold            = args.lhnn_error_thresh,
        cooldown_steps             = args.lhnn_cooldown,
    )
    # Pilot ∇ budget hint (exact count comes from lhnn_diag after training)
    pilot_grad_hint = lhnn_cfg.num_pilot_trajectories * (
        lhnn_cfg.pilot_steps_per_trajectory + 1
    )

    # ---- Build likelihood callables ----
    ledh_filter = DifferentiableLEDHLogLikelihood(
        num_particles     = N_ledh,
        n_lambda          = 5,
        sinkhorn_epsilon  = 2.0,
        sinkhorn_iters    = 20,
        resample_threshold = 0.5,
        grad_window       = 1,
        jit_compile       = True,
    )

    def ledh_ll(ssm, y):
        return ledh_filter(ssm, y)

    def bpf_ll(ssm, y):
        return bootstrap_pf_log_likelihood(ssm, y, num_particles=N_bpf)

    target_hmc  = make_target_log_prob(y_obs, ledh_ll)
    target_pmmh = make_target_log_prob(y_obs, bpf_ll)

    # ==================================================================
    # PMMH (unchanged)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  Running PMMH (Bootstrap PF + Random-Walk MH)")
    print("=" * 70)
    t0 = time.time()
    pmmh_result = run_pmmh(
        target_log_prob_fn = target_pmmh,
        initial_state      = init_state,
        num_results        = n_samp,
        num_burnin         = n_burn,
        step_size          = 0.30,   # was 0.15 — acceptance was only 0.14, target ~0.23
        seed               = 200,
    )
    t_pmmh      = time.time() - t0
    pmmh_samples = tf.exp(pmmh_result.samples)
    print(f"\n  PMMH done in {t_pmmh:.1f}s  accept={float(pmmh_result.accept_rate):.3f}")
    if run_A:
        summarise_chain(pmmh_samples, "PMMH", param_names, true_values)

    # ==================================================================
    # Standard HMC (Question A, or Question B with --standard ablation)
    # ==================================================================
    if need_main_hmc:
        print("\n" + "=" * 70)
        print("  Running standard HMC (LEDH + OT-Sinkhorn) — reference only")
        print("=" * 70)
        t0 = time.time()
        hmc_result = run_hmc(
            target_log_prob_fn = target_hmc,
            initial_state      = init_state,
            num_results        = n_samp,
            num_burnin         = n_burn,
            step_size          = 0.005,
            num_leapfrog_steps = L,
            target_accept_prob = 0.65,
            seed               = 300,
            adapt_step_size    = True,   # tune ε during burn-in
        )
        t_hmc = time.time() - t0
        hmc_samples = tf.exp(hmc_result.samples)
        print(f"\n  HMC done in {t_hmc:.1f}s  accept={float(hmc_result.accept_rate):.3f}")
        if run_A:
            summarise_chain(hmc_samples, "HMC", param_names, true_values)
    else:
        hmc_result = None
        hmc_samples = None
        t_hmc = 0.0
        print("\n  [Skipping standard HMC — not needed for this run configuration]")

    # ==================================================================
    # L-HNN HMC (Question A, or Question B with L-HNN ablation [default])
    # ==================================================================
    trained_lhnn = None
    lhnn_diag = None
    training_grad_evals = 0
    lhnn_sampling_grad_evals = 0

    if need_main_lhnn:
        print("\n" + "=" * 70)
        print("  Running L-HNN HMC (LEDH target, neural leapfrog)")
        print(f"  Training: {lhnn_cfg.num_pilot_trajectories} trajectories × "
              f"{lhnn_cfg.pilot_steps_per_trajectory} steps ≈ "
              f"{pilot_grad_hint} pilot ∇ evals (one-time cost)")
        print(f"  Δmax_hnn={lhnn_cfg.error_threshold}  cooldown={lhnn_cfg.cooldown_steps} steps")
        print("=" * 70)

        t0 = time.time()
        lhnn_result, trained_lhnn, lhnn_diag = run_lhnn_hmc(
            target_log_prob_fn = target_hmc,
            initial_state      = init_state,
            num_results        = n_samp,
            num_burnin         = n_burn,
            step_size          = 0.005,
            num_leapfrog_steps = L,
            target_accept_prob = 0.45,   # 0.65 is optimal for deterministic HMC; stochastic
                                         # targets need a lower target or dual averaging drives
                                         # eps to the floor chasing an unachievable rate.
            seed               = 400,
            verbose            = True,
            adapt_step_size    = True,
            lhnn_config        = lhnn_cfg,
        )
        t_lhnn       = time.time() - t0
        lhnn_samples = tf.exp(lhnn_result.samples)

        training_grad_evals      = lhnn_diag.training_gradient_evals
        lhnn_sampling_grad_evals = lhnn_diag.sampling_real_gradient_evals

        print(f"\n  L-HNN HMC done in {t_lhnn:.1f}s  accept={float(lhnn_result.accept_rate):.3f}")
        if run_A:
            summarise_chain(lhnn_samples, "L-HNN HMC", param_names, true_values)
    else:
        lhnn_result = None
        lhnn_samples = None
        t_lhnn = 0.0
        print("\n  [Skipping L-HNN HMC — Question B uses --standard ablation only]")

    # ==================================================================
    # Question A: 3-way comparison report + plots
    # ==================================================================
    if run_A:
        key_metrics = save_results(
            pmmh_result  = pmmh_result,
            hmc_result   = hmc_result,
            lhnn_result  = lhnn_result,
            pmmh_samples = pmmh_samples,
            hmc_samples  = hmc_samples,
            lhnn_samples = lhnn_samples,
            t_pmmh       = t_pmmh,
            t_hmc        = t_hmc,
            t_lhnn       = t_lhnn,
            n_samp       = n_samp,
            n_burn       = n_burn,
            true_values  = true_values,
            param_names  = param_names,
            out_dir      = out_dir,
            lhnn_training_grad_evals  = training_grad_evals,
            lhnn_sampling_grad_evals  = lhnn_sampling_grad_evals,
            lhnn_num_leapfrog_steps   = L,
        )
        plot_diagnostics(
            pmmh_samples, hmc_samples, lhnn_samples,
            true_values, param_names, out_dir,
        )
        plot_ess_per_grad(
            methods            = ["HMC-LEDH", "L-HNN HMC"],
            ess_per_grad_vals  = [
                key_metrics["ess_per_grad_hmc"],
                key_metrics["ess_per_grad_lhnn"],
            ],
            out_dir = out_dir,
        )

    # ==================================================================
    # Question B: ablation (standard HMC per cell, or L-HNN HMC per cell)
    # ==================================================================
    if run_B:
        ablation_out = out_dir / "ablation"
        ablation_out.mkdir(parents=True, exist_ok=True)
        t_b0 = time.time()
        ablation_cfgs = _build_ablation_configs(full_grid=bool(args.full_grid))
        retrain = bool(args.retrain_ablation)
        ablation_mode = "standard" if ablation_standard else "lhnn"

        if ablation_standard and retrain:
            print("  [Note: --retrain_ablation applies to L-HNN only; ignored for --standard ablation.]")

        mode_str = "FULL 3×2×2 grid" if args.full_grid else "budget subset"
        sampler_str = "run_hmc (standard HMC)" if ablation_standard else "run_lhnn_hmc"
        retrain_str = (
            "N/A (standard HMC)"
            if ablation_standard
            else ("independent L-HNN per config" if retrain else "shared baseline L-HNN")
        )
        print(
            f"\n  Ablation mode: {mode_str} ({len(ablation_cfgs)} configs)  |  sampler: {sampler_str}  |  {retrain_str}."
        )

        grad_seeds    = [1000 + 23 * k for k in range(8)]
        horizon_fracs = [0.2, 0.5, 1.0]

        hmc_grid_results: list = []

        total_iters_base = n_samp + n_burn
        grad_stats_base = grad_stats_for_target(
            target_hmc, init_state, grad_seeds, horizon_fracs, y_obs, ledh_ll,
        )

        # ---- Baseline entry from the main run above (HMC or L-HNN) ----
        if ablation_standard:
            assert hmc_result is not None and hmc_samples is not None
            ess_base = tfp.mcmc.effective_sample_size(hmc_samples)
            ess_mean_base = float(tf.reduce_mean(ess_base).numpy())
            mean_base = tf.reduce_mean(hmc_samples, axis=0)
            total_grads_base = (L + 1) * total_iters_base
            hmc_grid_results.append({
                "config_short":   "eps=2.0,gw=1,nl=5 (baseline)",
                "accept_rate":    float(hmc_result.accept_rate.numpy()),
                "ess_mean":       ess_mean_base,
                "ess_per_s":      ess_mean_base / max(t_hmc, 1e-6),
                "ess_per_grad":   ess_mean_base / max(total_grads_base, 1),
                "fallback_rate":  0.0,
                "cost_per_step":  per_step_cost(t_hmc, n_samp, n_burn),
                "runtime_s":      t_hmc,
                "bias_true":      tf.abs(mean_base - true_values).numpy().tolist(),
                "bias_vs_pmmh":   tf.abs(mean_base - tf.reduce_mean(pmmh_samples, axis=0)).numpy().tolist(),
                "grad_stats":     grad_stats_base,
            })
        else:
            assert lhnn_result is not None and lhnn_samples is not None and lhnn_diag is not None
            ess_base = tfp.mcmc.effective_sample_size(lhnn_samples)
            ess_mean_base = float(tf.reduce_mean(ess_base).numpy())
            mean_base = tf.reduce_mean(lhnn_samples, axis=0)
            total_grads_base = training_grad_evals + lhnn_sampling_grad_evals
            hmc_grid_results.append({
                "config_short":   "eps=2.0,gw=1,nl=5 (baseline)",
                "accept_rate":    float(lhnn_result.accept_rate.numpy()),
                "ess_mean":       ess_mean_base,
                "ess_per_s":      ess_mean_base / max(t_lhnn, 1e-6),
                "ess_per_grad":   ess_mean_base / max(total_grads_base, 1),
                "fallback_rate":  lhnn_sampling_grad_evals / max(L * total_iters_base, 1),
                "cost_per_step":  per_step_cost(t_lhnn, n_samp, n_burn),
                "runtime_s":      t_lhnn,
                "bias_true":      tf.abs(mean_base - true_values).numpy().tolist(),
                "bias_vs_pmmh":   tf.abs(mean_base - tf.reduce_mean(pmmh_samples, axis=0)).numpy().tolist(),
                "grad_stats":     grad_stats_base,
            })

        n_samp_g = 60
        n_burn_g = 60
        total_ab = len(ablation_cfgs)
        total_iters_g = n_samp_g + n_burn_g
        hmc_grads_per_iter = L + 1

        for idx, cfg in enumerate(ablation_cfgs):
            idx_1 = idx + 1
            print("\n" + "-" * 70)
            print(f"  Ablation {idx_1}/{total_ab}: {cfg['short']}")
            print("-" * 70)

            filt = DifferentiableLEDHLogLikelihood(
                num_particles      = N_ledh,
                n_lambda           = cfg["n_lambda"],
                sinkhorn_epsilon   = cfg["sinkhorn_epsilon"],
                sinkhorn_iters     = 20,
                resample_threshold = 0.5,
                grad_window        = cfg["grad_window"],
                jit_compile        = True,
            )

            def _ll(ssm, y, _f=filt):
                return _f(ssm, y)

            target_cfg = make_target_log_prob(y_obs, _ll)

            g_stats = grad_stats_for_target(
                target_cfg, init_state, grad_seeds, horizon_fracs, y_obs, _ll,
            )

            if ablation_standard:
                t0 = time.time()
                res = run_hmc(
                    target_log_prob_fn = target_cfg,
                    initial_state      = init_state,
                    num_results        = n_samp_g,
                    num_burnin         = n_burn_g,
                    step_size          = 0.005,
                    num_leapfrog_steps = L,
                    target_accept_prob = 0.65,
                    seed               = 400 + 100 * idx,
                    verbose            = False,
                    adapt_step_size    = True,
                )
                t_cfg = time.time() - t0
                smp = tf.exp(res.samples)
                ess_c = tfp.mcmc.effective_sample_size(smp)
                ess_m = float(tf.reduce_mean(ess_c).numpy())
                mean_c = tf.reduce_mean(smp, axis=0)
                abl_total_grads = hmc_grads_per_iter * total_iters_g
                ess_per_grad_c = ess_m / max(abl_total_grads, 1)
                fallback_rate_c = 0.0
                print(
                    f"    accept={float(res.accept_rate):.3f}"
                    f"  ESS/s={ess_m / max(t_cfg, 1e-6):.2f}"
                    f"  lg_mdn={g_stats['grad_log10_median']:.2f}"
                    f"  lg_mad={g_stats['grad_log10_mad']:.2f}"
                    f"  runtime={t_cfg:.1f}s"
                    f"  ({idx_1}/{total_ab}, ablation elapsed {time.time() - t_b0:.0f}s)"
                )
                hmc_grid_results.append({
                    "config_short":  cfg["short"],
                    "accept_rate":   float(res.accept_rate.numpy()),
                    "ess_mean":      ess_m,
                    "ess_per_s":     ess_m / max(t_cfg, 1e-6),
                    "ess_per_grad":  ess_per_grad_c,
                    "fallback_rate": fallback_rate_c,
                    "cost_per_step": per_step_cost(t_cfg, n_samp_g, n_burn_g),
                    "runtime_s":     t_cfg,
                    "bias_true":     tf.abs(mean_c - true_values).numpy().tolist(),
                    "bias_vs_pmmh":  tf.abs(mean_c - tf.reduce_mean(pmmh_samples, axis=0)).numpy().tolist(),
                    "grad_stats":    g_stats,
                })
            else:
                if retrain:
                    abl_lhnn_cfg = LHNNConfig(
                        hidden_units               = lhnn_cfg.hidden_units,
                        num_hidden                 = lhnn_cfg.num_hidden,
                        epochs                     = max(lhnn_cfg.epochs // 2, 500),
                        num_pilot_trajectories     = lhnn_cfg.num_pilot_trajectories,
                        pilot_steps_per_trajectory = lhnn_cfg.pilot_steps_per_trajectory,
                        error_threshold            = lhnn_cfg.error_threshold,
                        cooldown_steps             = lhnn_cfg.cooldown_steps,
                    )
                    pretrained = None
                else:
                    abl_lhnn_cfg = lhnn_cfg
                    pretrained   = trained_lhnn

                t0 = time.time()
                res, _, abl_diag = run_lhnn_hmc(
                    target_log_prob_fn = target_cfg,
                    initial_state      = init_state,
                    num_results        = n_samp_g,
                    num_burnin         = n_burn_g,
                    step_size          = 0.005,
                    num_leapfrog_steps = L,
                    target_accept_prob = 0.45,
                    seed               = 400 + 100 * idx,
                    verbose            = False,
                    lhnn_config        = abl_lhnn_cfg,
                    pretrained_lhnn    = pretrained,
                )
                t_cfg = time.time() - t0

                smp   = tf.exp(res.samples)
                ess_c = tfp.mcmc.effective_sample_size(smp)
                ess_m = float(tf.reduce_mean(ess_c).numpy())
                mean_c = tf.reduce_mean(smp, axis=0)

                abl_total_grads = abl_diag.total_real_gradient_evals
                ess_per_grad_c  = ess_m / max(abl_total_grads, 1) if abl_total_grads > 0 else float("nan")
                fallback_rate_c = abl_diag.sampling_fallback_intensity

                print(
                    f"    accept={float(res.accept_rate):.3f}"
                    f"  ESS/s={ess_m / max(t_cfg, 1e-6):.2f}"
                    f"  lg_mdn={g_stats['grad_log10_median']:.2f}"
                    f"  lg_mad={g_stats['grad_log10_mad']:.2f}"
                    f"  runtime={t_cfg:.1f}s"
                    f"  ({idx_1}/{total_ab}, ablation elapsed {time.time() - t_b0:.0f}s)"
                )

                hmc_grid_results.append({
                    "config_short":  cfg["short"],
                    "accept_rate":   float(res.accept_rate.numpy()),
                    "ess_mean":      ess_m,
                    "ess_per_s":     ess_m / max(t_cfg, 1e-6),
                    "ess_per_grad":  ess_per_grad_c,
                    "fallback_rate": fallback_rate_c,
                    "cost_per_step": per_step_cost(t_cfg, n_samp_g, n_burn_g),
                    "runtime_s":     t_cfg,
                    "bias_true":     tf.abs(mean_c - true_values).numpy().tolist(),
                    "bias_vs_pmmh":  tf.abs(mean_c - tf.reduce_mean(pmmh_samples, axis=0)).numpy().tolist(),
                    "grad_stats":    g_stats,
                })

        save_ablation_results(
            ablation_out,
            pmmh_samples,
            hmc_grid_results,
            true_values,
            param_names,
            ablation_mode=ablation_mode,
        )
        plot_ablation_summary(
            hmc_grid_results,
            ablation_out,
            param_names,
            ablation_mode=ablation_mode,
        )
        print(f"\n  Question B ablation done in {time.time() - t_b0:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())