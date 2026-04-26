"""
Experiment: DPF-HMC vs Particle Gibbs vs PMMH on Gaussian SSL.

Compares three inference methods on the State-Space LSTM model
(Zheng et al. 2017, Example 1):

  (a) Particle Gibbs (PG)  — conditional SMC + M-step
  (b) DPF-HMC              — EM warm-up + HMC over emission params
  (c) PMMH                 — EM warm-up + bootstrap PF + M-step

Key change from v1: DPF-HMC now runs **actual HMC** over emission
parameters (C, b, R) using the differentiable LEDH filter to provide
gradients. The LSTM is frozen after warm-up.

Usage
-----
    python -m src.experiments.exp_part3_bonus3_ssl_comparison
    python -m src.experiments.exp_part3_bonus3_ssl_comparison --quick
    python -m src.experiments.exp_part3_bonus3_ssl_comparison --dynamics circle
    python -m src.experiments.exp_part3_bonus3_ssl_comparison --seed 123
    python -m src.experiments.exp_part3_bonus3_ssl_comparison --output /path/to/summary.txt
"""

from __future__ import annotations

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

from src.models.gaussian_ssl import GaussianSSL, generate_ssl_data
from src.filters.bonus.ssl_particle_gibbs import particle_gibbs_ssl
from src.filters.bonus.dpf_hmc_ssl import (
    dpf_hmc_inference,
    pmmh_ssl_inference,
    compute_filtered_trajectory,
    test_log_likelihood,
    filtering_rmse,
    EmissionParameterizer,
)


# =========================================================================
# Reporting
# =========================================================================

def format_results(
    dynamics: str,
    pg_rmse: float,
    dpf_rmse: float,
    pmmh_rmse: float,
    pg_test_ll: float,
    dpf_test_ll: float,
    pmmh_test_ll: float,
    pg_train_ll: float,
    dpf_train_ll: float,
    pmmh_train_ll: float,
    pg_time: float,
    pmmh_time: float,
    dpf_time: float,
    pg_iters: int,
    dpf_hmc_samples: int,
    pmmh_outer: int,
    dpf_accept_rate: float,
    ssl_pg: GaussianSSL,
    ssl_dpf: GaussianSSL,
    ssl_pmmh: GaussianSSL,
    dpf_post_C: tf.Tensor,
    dpf_post_b: tf.Tensor,
    dpf_post_R: tf.Tensor,
    *,
    run_header: str | None = None,
) -> str:
    """Build the formatted comparison report (same text as printed to stdout)."""
    R_true = 0.3 ** 2  # noise_std^2
    lines: list[str] = []
    if run_header:
        lines.append(run_header.rstrip())
        lines.append("")

    lines.extend([
        "\n" + "=" * 80,
        f"  Results: Gaussian SSL ({dynamics} trajectory)",
        "=" * 80,
        "",
        "--- A. Filtering RMSE (lower is better) ---",
        f"  {'Method':<20s} {'RMSE':>10s}",
        "  " + "-" * 32,
        f"  {'Particle Gibbs':<20s} {pg_rmse:>10.4f}",
        f"  {'DPF-HMC':<20s} {dpf_rmse:>10.4f}",
        f"  {'PMMH':<20s} {pmmh_rmse:>10.4f}",
        "",
        "--- B. Test Log-Likelihood (higher is better) ---",
        f"  {'Method':<20s} {'Test LL':>10s}",
        "  " + "-" * 32,
        f"  {'Particle Gibbs':<20s} {pg_test_ll:>10.2f}",
        f"  {'DPF-HMC':<20s} {dpf_test_ll:>10.2f}",
        f"  {'PMMH':<20s} {pmmh_test_ll:>10.2f}",
        "",
        "--- C. Training Log-Likelihood (final) ---",
        f"  {'Method':<20s} {'Train LL':>10s}",
        "  " + "-" * 32,
        f"  {'Particle Gibbs':<20s} {pg_train_ll:>10.2f}",
        f"  {'DPF-HMC':<20s} {dpf_train_ll:>10.2f}",
        f"  {'PMMH':<20s} {pmmh_train_ll:>10.2f}",
        "",
        "--- D. Wall-Clock Time ---",
        f"  {'Method':<20s} {'Total (s)':>10s} {'Per iter (s)':>12s}",
        "  " + "-" * 44,
        f"  {'Particle Gibbs':<20s} {pg_time:>10.1f} "
        f"{pg_time / pg_iters:>12.2f}",
        f"  {'DPF-HMC':<20s} {dpf_time:>10.1f} "
        f"{dpf_time / max(dpf_hmc_samples, 1):>12.2f}",
        f"  {'PMMH':<20s} {pmmh_time:>10.1f} "
        f"{pmmh_time / pmmh_outer:>12.2f}",
        "",
        "--- E. HMC Diagnostics (DPF-HMC only) ---",
        f"  Acceptance rate: {dpf_accept_rate:.3f}",
        f"  Posterior mean C:\n    {dpf_post_C.numpy()}",
        f"  Posterior mean b: {dpf_post_b.numpy()}",
        f"  Posterior mean R_diag: {dpf_post_R.numpy()}",
        f"  True R_diag: [{R_true:.4f}, {R_true:.4f}]",
        "",
        "--- F. Emission Parameter Recovery ---",
        f"  {'Method':<20s} {'||C - I||':>10s} {'||R - 0.09I||':>14s}",
        "  " + "-" * 46,
    ])
    for name, ssl_m in [
        ("Particle Gibbs", ssl_pg),
        ("DPF-HMC (post.mean)", ssl_dpf),
        ("PMMH", ssl_pmmh),
    ]:
        C_err = float(tf.norm(ssl_m.C - tf.eye(2)))
        R_err = float(tf.norm(ssl_m.R_diag - R_true))
        lines.append(f"  {name:<20s} {C_err:>10.4f} {R_err:>14.4f}")
    lines.extend(["", "=" * 80])
    return "\n".join(lines)




# =========================================================================
# Plotting
# =========================================================================

def plot_hmc_trace(hmc_samples: tf.Tensor, fix_C: bool, save_path: Path):
    """HMC trace plots for each emission parameter."""
    samples = hmc_samples.numpy()
    n_samples, dim = samples.shape

    if fix_C:
        labels = [r"$b_1$", r"$b_2$", r"$\log R_1$", r"$\log R_2$"]
    else:
        labels = [f"param {i}" for i in range(dim)]

    fig, axes = plt.subplots(dim, 1, figsize=(10, 2.5 * dim), sharex=True)
    if dim == 1:
        axes = [axes]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(samples[:, i], linewidth=0.5, color="C0", alpha=0.8)
        ax.axhline(np.mean(samples[:, i]), color="C3", linestyle="--",
                    linewidth=1, label=f"mean = {np.mean(samples[:, i]):.3f}")
        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("HMC iteration (post burn-in)", fontsize=11)
    fig.suptitle("HMC Trace Plots — DPF-HMC Emission Parameters", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_posterior_histograms(hmc_samples: tf.Tensor, fix_C: bool,
                              R_true: float, save_path: Path):
    """Posterior histograms for R_diag with true values marked."""
    samples = hmc_samples.numpy()

    if fix_C:
        # samples columns: b1, b2, log_R1, log_R2
        log_R = samples[:, 2:]  # (n, 2)
        R_diag = np.exp(log_R)
        param_names = [r"$R_1$", r"$R_2$"]
    else:
        # last obs_dim columns are log_R_diag
        obs_dim = 2
        log_R = samples[:, -obs_dim:]
        R_diag = np.exp(log_R)
        param_names = [r"$R_1$", r"$R_2$"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(R_diag[:, i], bins=30, density=True, color="C0", alpha=0.7,
                edgecolor="white")
        ax.axvline(R_true, color="C3", linestyle="--", linewidth=2,
                   label=f"True = {R_true:.3f}")
        ax.axvline(np.mean(R_diag[:, i]), color="C2", linestyle="-",
                   linewidth=2, label=f"Mean = {np.mean(R_diag[:, i]):.3f}")
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle(r"HMC Posterior: $R_{\mathrm{diag}}$ vs True $R = 0.09 I$",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_filtered_trajectories(z_true: tf.Tensor, pg_z: tf.Tensor,
                                dpf_z: tf.Tensor, pmmh_z: tf.Tensor,
                                save_path: Path,
                                dpf_posterior_trajs: np.ndarray | None = None):
    """Filtered trajectory vs truth for all three methods (both dims).

    Parameters
    ----------
    dpf_posterior_trajs : (n_draws, T, d_z) array, optional
        Filtered trajectories from multiple HMC posterior draws.
        If provided, a 2-sigma shaded band is drawn around the DPF-HMC mean.
    """
    z_true_np = z_true.numpy()
    pg_np = pg_z.numpy()
    dpf_np = dpf_z.numpy()
    pmmh_np = pmmh_z.numpy()
    T = z_true_np.shape[0]
    t_axis = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    dim_labels = [r"$z_1$", r"$z_2$"]

    for d, ax in enumerate(axes):
        # DPF-HMC 2-sigma posterior band (plot first so it's behind lines)
        if dpf_posterior_trajs is not None:
            post_mean = np.mean(dpf_posterior_trajs[:, :, d], axis=0)
            post_std = np.std(dpf_posterior_trajs[:, :, d], axis=0)
            ax.fill_between(
                t_axis, post_mean - 2 * post_std, post_mean + 2 * post_std,
                color="red", alpha=0.15, label=r"DPF-HMC $\pm 2\sigma$",
            )

        ax.plot(t_axis, z_true_np[:, d], "k-", linewidth=2, label="True",
                alpha=0.9)
        ax.plot(t_axis, pg_np[:, d], "-", linewidth=1.2, label="PG",
                alpha=0.8, color="C0")
        ax.plot(t_axis, dpf_np[:, d], "-", linewidth=1.2, label="DPF-HMC",
                alpha=0.8, color="red")
        ax.plot(t_axis, pmmh_np[:, d], "-", linewidth=1.2, label="PMMH",
                alpha=0.8, color="C2")
        ax.set_ylabel(dim_labels[d], fontsize=12)
        ax.legend(loc="upper right", fontsize=9, ncol=5)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step $t$", fontsize=11)
    fig.suptitle("Filtered Latent Trajectories vs Ground Truth", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =========================================================================
# Main comparison
# =========================================================================

def run_comparison(
    dynamics: str,
    T_train: int,
    T_test: int,
    n_particles: int,
    pg_iters: int,
    n_warmup: int,
    dpf_hmc_samples: int,
    dpf_hmc_burnin: int,
    pmmh_outer: int,
    n_m_steps: int,
    hmc_step_size: float,
    n_leapfrog: int,
    adapt_step_size: bool,
    fix_C: bool,
    quick: bool,
    seed: int = 42,
    output_path: Path | None = None,
    run_header: str | None = None,
):
    # ---- Generate data ----
    T_total = T_train + T_test
    z_true, x_obs = generate_ssl_data(
        T=T_total, state_dim=2, obs_dim=2,
        dynamics=dynamics, noise_std=0.3, seed=seed,
    )
    x_train = x_obs[:T_train]
    x_test = x_obs[T_train:]
    z_train = z_true[:T_train]
    z_test = z_true[T_train:]

    print(f"\n  Data: dynamics={dynamics}, T_train={T_train}, T_test={T_test}")
    print(f"  z range: [{float(tf.reduce_min(z_true)):.2f}, "
          f"{float(tf.reduce_max(z_true)):.2f}]")

    # ---- 1. Particle Gibbs ----
    tf.random.set_seed(seed + 1)
    print(f"\n{'=' * 60}")
    print("  [1/3] Particle Gibbs (Zheng(17))")
    print(f"{'=' * 60}")
    ssl_pg = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=32)
    _ = ssl_pg.transition_params(
        ssl_pg.get_initial_lstm_state(1), tf.zeros([1, 2])
    )
    
    t0 = time.perf_counter()
    pg_result = particle_gibbs_ssl(
        ssl_pg, x_train,
        n_particles=n_particles,
        n_iterations=pg_iters,
        n_m_steps=n_m_steps,
        verbose=True,
    )
    pg_time = time.perf_counter() - t0
    
    pg_z_final = pg_result.z_samples[-1]
    pg_rmse = float(filtering_rmse(pg_z_final, z_train))
    pg_test_ll = float(test_log_likelihood(ssl_pg, x_test, z_test))
    pg_train_ll = float(pg_result.log_marginal[-1])

    print(f"\n  PG: RMSE={pg_rmse:.4f}  test_LL={pg_test_ll:.2f}  "
          f"time={pg_time:.1f}s")
    
    # ---- 2. DPF-HMC ----
    tf.random.set_seed(seed + 2)
    print(f"\n{'=' * 60}")
    print("  [2/3] DPF-HMC (Differentiable PF + actual HMC)")
    print(f"{'=' * 60}")
    ssl_dpf = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=32)
    
    t0 = time.perf_counter()
    dpf_result = dpf_hmc_inference(
        ssl_dpf, x_train,
        n_warmup=n_warmup,
        n_m_steps=n_m_steps,
        n_hmc_samples=dpf_hmc_samples,
        n_hmc_burnin=dpf_hmc_burnin,
        hmc_step_size=hmc_step_size,
        n_leapfrog=n_leapfrog,
        adapt_step_size=adapt_step_size,
        n_particles=n_particles,
        sinkhorn_epsilon=2.0,
        fix_C=fix_C,
        verbose=True,
    )
    dpf_time = time.perf_counter() - t0

    # Filtered trajectory using posterior mean emission params
    dpf_z_filtered = compute_filtered_trajectory(
        ssl_dpf, x_train, dpf_result.z_ref
    )
    dpf_rmse = float(filtering_rmse(dpf_z_filtered, z_train))
    dpf_test_ll = float(test_log_likelihood(ssl_dpf, x_test, z_test))
    dpf_train_ll = float(dpf_result.hmc_log_probs[-1]) if len(
        dpf_result.hmc_log_probs) > 0 else float("-inf")

    print(f"\n  DPF-HMC: RMSE={dpf_rmse:.4f}  test_LL={dpf_test_ll:.2f}  "
          f"accept={float(dpf_result.hmc_accept_rate):.3f}  "
          f"time={dpf_time:.1f}s")

    # ---- 3. PMMH ----
    tf.random.set_seed(seed + 3)
    print(f"\n{'=' * 60}")
    print("  [3/3] PMMH (Bootstrap PF)")
    print(f"{'=' * 60}")
    ssl_pmmh = GaussianSSL(state_dim=2, obs_dim=2, lstm_units=32)
    _ = ssl_pmmh.transition_params(
        ssl_pmmh.get_initial_lstm_state(1), tf.zeros([1, 2])
    )
    obs_var = tf.math.reduce_variance(x_train, axis=0)
    ssl_pmmh.log_R_diag.assign(
        tf.math.log(tf.maximum(obs_var * 0.1, tf.constant(1e-4)))
    )

    t0 = time.perf_counter()
    pmmh_result = pmmh_ssl_inference(
        ssl_pmmh, x_train,
        n_warmup=n_warmup,
        n_m_steps=n_m_steps,
        n_outer=pmmh_outer,
        n_particles=n_particles,
        verbose=True,
    )
    pmmh_time = time.perf_counter() - t0

    pmmh_z_final = pmmh_result.z_samples[-1]
    pmmh_rmse = float(filtering_rmse(pmmh_z_final, z_train))
    pmmh_test_ll = float(test_log_likelihood(ssl_pmmh, x_test, z_test))
    pmmh_train_ll = pmmh_result.log_lls[-1] if pmmh_result.log_lls else float(
        "-inf"
    )

    print(f"\n  PMMH: RMSE={pmmh_rmse:.4f}  test_LL={pmmh_test_ll:.2f}  "
          f"time={pmmh_time:.1f}s")

    # ---- Generate plots ----
    if output_path is not None:
        plot_dir = Path(output_path).parent
    else:
        plot_dir = Path(__file__).resolve().parents[2] / "reports" / "8_BonusQ3_SSL_Comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    R_true = 0.3 ** 2  # 0.09
    print("\n  Generating plots...")
    plot_hmc_trace(dpf_result.hmc_samples, fix_C,
                   plot_dir / f"hmc_trace_{dynamics}.png")
    plot_posterior_histograms(dpf_result.hmc_samples, fix_C, R_true,
                              plot_dir / f"hmc_posterior_R_{dynamics}.png")

    # ---- DPF-HMC posterior trajectory bands ----
    # Run filtering for a subsample of HMC draws to get posterior uncertainty
    n_draws = min(30, dpf_result.hmc_samples.shape[0])
    draw_indices = np.linspace(
        0, dpf_result.hmc_samples.shape[0] - 1, n_draws, dtype=int
    )
    param = EmissionParameterizer(obs_dim=2, state_dim=2, fix_C=fix_C)
    param.set_fixed_C(dpf_result.posterior_mean_C)
    # Save current emission params to restore after
    saved_b = tf.identity(ssl_dpf.b)
    saved_logR = tf.identity(ssl_dpf.log_R_diag)

    posterior_trajs = []
    print(f"  Computing {n_draws} posterior trajectory draws...")
    for i, idx in enumerate(draw_indices):
        param.apply_to_ssl(ssl_dpf, dpf_result.hmc_samples[idx])
        z_draw = compute_filtered_trajectory(
            ssl_dpf, x_train, dpf_result.z_ref, n_particles=30
        )
        posterior_trajs.append(z_draw.numpy())
    dpf_posterior_trajs = np.stack(posterior_trajs)  # (n_draws, T, d_z)

    # Restore posterior mean params
    ssl_dpf.b.assign(saved_b)
    ssl_dpf.log_R_diag.assign(saved_logR)
    print(f"  Posterior band: {n_draws} draws, shape {dpf_posterior_trajs.shape}")

    plot_filtered_trajectories(z_train, pg_z_final, dpf_z_filtered,
                                pmmh_z_final,
                                plot_dir / f"filtered_trajectories_{dynamics}.png",
                                dpf_posterior_trajs=dpf_posterior_trajs)

    # ---- Print / save results ----
    report = format_results(
        dynamics=dynamics,
        pg_rmse=pg_rmse,
        dpf_rmse=dpf_rmse,
        pmmh_rmse=pmmh_rmse,
        pg_test_ll=pg_test_ll,
        dpf_test_ll=dpf_test_ll,
        pmmh_test_ll=pmmh_test_ll,
        pg_train_ll=pg_train_ll,
        dpf_train_ll=dpf_train_ll,
        pmmh_train_ll=pmmh_train_ll,
        pg_time=pg_time,
        pmmh_time=pmmh_time,
        dpf_time=dpf_time,
        pg_iters=pg_iters,
        dpf_hmc_samples=dpf_hmc_samples,
        pmmh_outer=pmmh_outer,
        dpf_accept_rate=float(dpf_result.hmc_accept_rate),
        ssl_pg=ssl_pg,
        ssl_dpf=ssl_dpf,
        ssl_pmmh=ssl_pmmh,
        dpf_post_C=dpf_result.posterior_mean_C,
        dpf_post_b=dpf_result.posterior_mean_b,
        dpf_post_R=dpf_result.posterior_mean_R,
        run_header=run_header,
    )
    print(report)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")
        print(f"\n  Saved results to {output_path.resolve()}")


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(
        description="DPF-HMC vs Particle Gibbs vs PMMH on Gaussian SSL"
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dynamics", type=str, default="sine",
                   choices=["line", "sine", "circle", "swiss_roll"])
    p.add_argument("--n_particles", type=int, default=50)
    p.add_argument("--pg_iters", type=int, default=50)
    p.add_argument("--n_warmup", type=int, default=50)
    p.add_argument("--dpf_hmc_samples", type=int, default=300)
    p.add_argument("--dpf_hmc_burnin", type=int, default=100)
    p.add_argument("--pmmh_outer", type=int, default=50)
    p.add_argument("--hmc_step_size", type=float, default=0.05)
    p.add_argument("--n_leapfrog", type=int, default=5)
    p.add_argument("--adapt", action="store_true",
                   help="Enable HMC step-size adaptation (off by default)")
    p.add_argument("--sample_C", action="store_true",
                   help="Include C in HMC sampling (default: fix C at warm-up value)")
    p.add_argument("--seed", type=int, default=42,
                   help="Master RNG seed for reproducibility (default: 42)")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the results summary (.txt). "
             "Default: reports/8_BonusQ3_SSL_Comparison/results_<dynamics>.txt",
    )
    args = p.parse_args()

    seed = int(args.seed)
    tf.random.set_seed(seed)

    T_train = 100
    T_test = 40
    n_m_steps = 5

    if args.quick:
        T_train = 60
        T_test = 20
        args.pg_iters = 30
        args.n_warmup = 5
        args.dpf_hmc_samples = 50
        args.dpf_hmc_burnin = 30
        args.pmmh_outer = 15
        args.n_particles = 30

    print("=" * 70)
    print("  DPF-HMC vs Particle Gibbs vs PMMH on Gaussian SSL")
    print(f"  dynamics={args.dynamics}  T_train={T_train}  T_test={T_test}  "
          f"N={args.n_particles}  seed={seed}")
    print(f"  PG_iters={args.pg_iters}  "
          f"DPF_HMC_samples={args.dpf_hmc_samples}  "
          f"PMMH_outer={args.pmmh_outer}")
    adapt = args.adapt
    fix_C = not args.sample_C
    print(f"  HMC: init_step={args.hmc_step_size}  "
          f"L={args.n_leapfrog}  "
          f"burnin={args.dpf_hmc_burnin}  "
          f"adapt={adapt}  fix_C={fix_C}")
    print("=" * 70)

    if args.output:
        out_path = Path(args.output)
    else:
        repo_reports = Path(__file__).resolve().parents[2] / "reports"
        out_path = (
            repo_reports / "8_BonusQ3_SSL_Comparison"
            / f"results_{args.dynamics}.txt"
        )

    run_header = (
        f"8_BonusQ3_SSL_Comparison\n"
        f"  dynamics={args.dynamics}  T_train={T_train}  T_test={T_test}  "
        f"N={args.n_particles}  seed={seed}  quick={args.quick}\n"
        f"  pg_iters={args.pg_iters}  dpf_hmc_samples={args.dpf_hmc_samples}  "
        f"dpf_hmc_burnin={args.dpf_hmc_burnin}  pmmh_outer={args.pmmh_outer}\n"
        f"  hmc_step_size={args.hmc_step_size}  n_leapfrog={args.n_leapfrog}  "
        f"adapt={adapt}  fix_C={fix_C}"
    )

    run_comparison(
        dynamics=args.dynamics,
        T_train=T_train,
        T_test=T_test,
        n_particles=args.n_particles,
        pg_iters=args.pg_iters,
        n_warmup=args.n_warmup,
        dpf_hmc_samples=args.dpf_hmc_samples,
        dpf_hmc_burnin=args.dpf_hmc_burnin,
        pmmh_outer=args.pmmh_outer,
        n_m_steps=n_m_steps,
        hmc_step_size=args.hmc_step_size,
        n_leapfrog=args.n_leapfrog,
        adapt_step_size=adapt,
        fix_C=fix_C,
        quick=args.quick,
        seed=seed,
        output_path=out_path,
        run_header=run_header,
    )

    print("\n  Experiment complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

