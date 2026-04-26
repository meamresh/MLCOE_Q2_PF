"""
Experiment: Stochastic Particle Flow with stiffness mitigation (Dai & Daum, 2021).

Replicates the main numerical example and Table 1-style comparison from:

    Dai & Daum (2021), "Stiffness Mitigation in Stochastic Particle Flow Filters"

Features
--------
- 2D bearing-only localization with two sensors
- Straight-line homotopy β(λ) = λ vs optimal β*(λ)
- Stochastic particle flow driven by diffusion Q
- Pure TensorFlow / TensorFlow Probability implementation (no NumPy/SciPy
  in the core filter computation)

This script is intentionally self-contained and geared towards reproducible
experiments rather than being a general-purpose API.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.filters.spf_dai_daum import (  # noqa: E402
    SPFConfig,
    DaiDaumStochasticParticleFlow,
    compute_M,
    compute_objective_J,
    cond_number,
    run_particle_flow,
    solve_optimal_beta_tf,
)
from src.models.ssm_dai_daum_bearing_only import (  # noqa: E402
    DaiDaumBearingSSM,
    DTYPE,
)


tfd = tfp.distributions


# Use non-interactive backend for batch runs
try:  # pragma: no cover - plotting backend
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting optional
    plt = None


def compute_stiffness_ratio(
    beta_grid: tf.Tensor,
    x_ref: tf.Tensor,
    z: tf.Tensor,
    ssm: DaiDaumBearingSSM,
) -> tf.Tensor:
    """
    Compute stiffness ratio R_stiff(λ) along a homotopy trajectory.

    We use the nuclear-norm-based condition number κ*(M) = tr(M) tr(M^{-1})
    and normalize by its value at λ=0 for comparability:

        R_stiff(λ_i) = κ*(M(β(λ_i))) / κ*(M(β(0)))
    """
    beta_grid = tf.cast(beta_grid, dtype=DTYPE)
    kappa_values: List[tf.Tensor] = []
    for i in range(int(beta_grid.shape[0])):
        beta_val = beta_grid[i]
        M = compute_M(ssm, x_ref, beta_val, z)
        kappa = cond_number(M)
        kappa_values.append(kappa)

    kappa_tensor = tf.stack(kappa_values)
    kappa0 = kappa_tensor[0]
    kappa0 = tf.where(kappa0 > 0, kappa0, tf.constant(1.0, dtype=DTYPE))
    return kappa_tensor / kappa0


def tf_interp_1d(
    x_new: tf.Tensor,
    x_old: tf.Tensor,
    y_old: tf.Tensor,
) -> tf.Tensor:
    """
    Pure TensorFlow 1D linear interpolation.
    """
    x_new = tf.cast(tf.convert_to_tensor(x_new), dtype=DTYPE)
    x_old = tf.cast(tf.convert_to_tensor(x_old), dtype=DTYPE)
    y_old = tf.cast(tf.convert_to_tensor(y_old), dtype=DTYPE)

    indices = tf.searchsorted(x_old, x_new, side="right")
    indices = tf.clip_by_value(indices, 1, tf.shape(x_old)[0] - 1)

    idx_low = indices - 1
    idx_high = indices

    x_low = tf.gather(x_old, idx_low)
    x_high = tf.gather(x_old, idx_high)
    y_low = tf.gather(y_old, idx_low)
    y_high = tf.gather(y_old, idx_high)

    dx = x_high - x_low
    dx = tf.where(tf.abs(dx) < 1e-12, tf.ones_like(dx) * 1e-12, dx)
    weights = (x_new - x_low) / dx
    y_new = y_low + (y_high - y_low) * weights
    return y_new


def plot_figure2(
    beta_opt_lam: tf.Tensor,
    beta_opt: tf.Tensor,
    beta_opt_dot: tf.Tensor,
    x_ref: tf.Tensor,
    z: tf.Tensor,
    out_path: Path,
    ssm: DaiDaumBearingSSM,
) -> None:
    """
    Reproduce Figure 2 from the paper:
    (a) β(λ) vs β*(λ)
    (b) Error e = β* - λ
    (c) Control u*(λ) = dβ*/dλ
    (d) Stiffness ratio R_stiff
    """
    if plt is None:  # plotting not available
        return

    lam_fine = tf.linspace(tf.constant(0.0, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE), 200)
    beta_linear = lam_fine

    beta_opt_fine = tf_interp_1d(lam_fine, beta_opt_lam, beta_opt)
    beta_opt_dot_fine = tf_interp_1d(lam_fine, beta_opt_lam, beta_opt_dot)

    # Error e = β* - λ
    error = beta_opt_fine - lam_fine

    # Stiffness ratios for linear and optimal homotopies
    R_stiff_linear = compute_stiffness_ratio(beta_linear, x_ref, z, ssm)
    R_stiff_optimal = compute_stiffness_ratio(beta_opt_fine, x_ref, z, ssm)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) β(λ)
    axes[0, 0].plot(
        lam_fine.numpy(),
        lam_fine.numpy(),
        "k--",
        label=r"$\beta(\lambda) = \lambda$",
        linewidth=2,
    )
    axes[0, 0].plot(
        beta_opt_lam.numpy(),
        beta_opt.numpy(),
        "r-",
        label=r"optimal $\beta^*(\lambda)$",
        linewidth=2,
    )
    axes[0, 0].set_xlabel(r"$\lambda$", fontsize=12)
    axes[0, 0].set_ylabel(r"$\beta(\lambda)$", fontsize=12)
    axes[0, 0].set_title("(a) Homotopy Functions", fontsize=13, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])

    # (b) Error e = β* - λ
    axes[0, 1].plot(lam_fine.numpy(), error.numpy(), "b-", linewidth=2)
    axes[0, 1].set_xlabel(r"$\lambda$", fontsize=12)
    axes[0, 1].set_ylabel(r"$e = \beta^*(\lambda) - \lambda$", fontsize=12)
    axes[0, 1].set_title("(b) Error from Linear", fontsize=13, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].axhline(y=0.0, color="k", linestyle="--", alpha=0.3)

    # (c) Control u*(λ)
    axes[1, 0].plot(beta_opt_lam.numpy(), beta_opt_dot.numpy(), "g-", linewidth=2)
    axes[1, 0].set_xlabel(r"$\lambda$", fontsize=12)
    axes[1, 0].set_ylabel(r"$u^*(\lambda) = d\beta^*/d\lambda$", fontsize=12)
    axes[1, 0].set_title("(c) Optimal Control", fontsize=13, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 1])

    # (d) Stiffness ratio (log scale)
    axes[1, 1].semilogy(
        lam_fine.numpy(),
        R_stiff_linear.numpy(),
        "g--",
        label=r"$\beta(\lambda) = \lambda$",
        linewidth=2,
    )
    axes[1, 1].semilogy(
        lam_fine.numpy(),
        R_stiff_optimal.numpy(),
        "r-",
        label=r"optimal $\beta^*(\lambda)$",
        linewidth=2,
    )
    axes[1, 1].set_xlabel(r"$\lambda$", fontsize=12)
    axes[1, 1].set_ylabel(r"$R_{\mathrm{stiff}}$", fontsize=12)
    axes[1, 1].set_title("(d) Stiffness Ratio", fontsize=13, fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, which="both")
    axes[1, 1].set_xlim([0, 1])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved Figure 2 to {out_path}")
    plt.close(fig)


def plot_figure3(
    mc_times_linear: List[float],
    mc_times_optimal: List[float],
    out_path: Path,
) -> None:
    """
    Reproduce Figure 3 from the paper:
    Computing time comparison between linear and optimal homotopy.
    """
    if plt is None:  # plotting not available
        return

    mc_indices = list(range(1, len(mc_times_linear) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        mc_indices,
        mc_times_optimal,
        "r-",
        marker="^",
        markersize=6,
        label=r"optimal $\beta^*(\lambda)$",
        linewidth=1.5,
    )
    ax.plot(
        mc_indices,
        mc_times_linear,
        "k-",
        marker=".",
        markersize=6,
        label=r"$\beta(\lambda) = \lambda$",
        linewidth=1.5,
    )

    mean_opt = (
        float(
            tf.reduce_mean(tf.constant(mc_times_optimal, dtype=DTYPE)).numpy()
        )
        if mc_times_optimal
        else 0.0
    )
    mean_lin = (
        float(
            tf.reduce_mean(tf.constant(mc_times_linear, dtype=DTYPE)).numpy()
        )
        if mc_times_linear
        else 0.0
    )
    ax.axhline(
        y=mean_opt,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"average for optimal = {mean_opt:.2f}s",
    )
    ax.axhline(
        y=mean_lin,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"average for linear = {mean_lin:.2f}s",
    )

    ax.set_xlabel("Monte Carlo run index", fontsize=12)
    ax.set_ylabel("Computing time (seconds)", fontsize=12)
    ax.set_title(
        "Comparison of computing time for Example 1", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper left", fancybox=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved Figure 3 to {out_path}")
    plt.close(fig)


def run_experiment(
    config: SPFConfig,
) -> Tuple[List[Tuple[float, float, float, float]], float, float, List[float], List[float]]:
    """
    Run the Dai–Daum stochastic particle flow experiment.

    Returns
    -------
    results : list of tuples
        [(rmse_lin, rmse_opt, trP_lin, trP_opt), ...] for each MC run.
    J_linear : float
        Objective J(β=λ) for straight-line homotopy.
    J_optimal : float
        Objective J(β*) for optimal homotopy.
    times_lin : list of float
        Filtering times for linear homotopy.
    times_opt : list of float
        Filtering times for optimal homotopy.
    time_training : float
        Time spent solving for optimal β*(λ).
    """
    # Use float64 for numerical stability
    tf.keras.backend.set_floatx("float64")

    # State-space model (same structure as other experiments)
    ssm = DaiDaumBearingSSM()
    x_true = ssm.x_true
    h_true, _ = ssm.measurement_model(x_true)

    # Measurement noise distribution
    noise_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros(2, dtype=DTYPE),
        scale_tril=tf.linalg.cholesky(ssm.R),
    )

    # Sample a nominal measurement
    v_nominal = noise_dist.sample()
    z_nominal = h_true + v_nominal

    # Reference point
    x_ref = ssm.prior_mean

    # Solve for optimal β*(λ)
    t_bvp_start = time.perf_counter()
    beta_opt_lam, beta_opt, beta_opt_dot, J_optimal = solve_optimal_beta_tf(
        x_ref, z_nominal, config, ssm
    )
    time_training = time.perf_counter() - t_bvp_start

    # Objective for straight-line homotopy β(λ) = λ
    lam_linear = tf.linspace(
        tf.constant(0.0, dtype=DTYPE),
        tf.constant(1.0, dtype=DTYPE),
        config.bvp_mesh_points,
    )
    beta_linear = lam_linear
    beta_dot_linear = tf.ones_like(lam_linear, dtype=DTYPE)
    J_linear = compute_objective_J(
        beta_linear,
        beta_dot_linear,
        x_ref,
        z_nominal,
        tf.constant(config.mu, dtype=DTYPE),
        ssm,
    )

    # Figure 2 (homotopy / stiffness) – saved to reports directory
    fig_dir = Path("reports/4_Stochastic_Particle_Flow/Dai_Daum")
    plot_figure2(
        beta_opt_lam,
        beta_opt,
        beta_opt_dot,
        x_ref,
        z_nominal,
        out_path=fig_dir / "figure2_spf_homotopy_stiffness.png",
        ssm=ssm,
    )

    results: List[Tuple[float, float, float, float]] = []
    times_lin: List[float] = []
    times_opt: List[float] = []

    # Monte Carlo runs
    n_mc = config.lambda_steps // 10  # default: 20 runs if lambda_steps=200

    print("Running Monte Carlo simulations with TensorFlow...")
    print("=" * 70)
    print(f"{'MC':>3} | {'RMSE(β)':>10} | {'RMSE(β*)':>10} | {'tr(Pβ)':>12} | {'tr(Pβ*)':>12}")
    print("=" * 70)

    # Filters (same interface as other experiments)
    filter_lin = DaiDaumStochasticParticleFlow(
        ssm, ssm.prior_mean, ssm.prior_cov, config, homotopy_mode="linear"
    )
    filter_opt = DaiDaumStochasticParticleFlow(
        ssm, ssm.prior_mean, ssm.prior_cov, config, homotopy_mode="optimal"
    )

    for mc in range(1, n_mc + 1):
        v_mc = noise_dist.sample()
        z_mc = h_true + v_mc

        # Straight-line homotopy
        t0 = time.perf_counter()
        filter_lin.update(z_mc)
        t1 = time.perf_counter()
        x_hat_lin = tf.cast(filter_lin.state, DTYPE)
        P_hat_lin = tf.cast(filter_lin.covariance, DTYPE)
        rmse_lin = float(
            tf.sqrt(tf.reduce_mean((x_hat_lin - x_true) ** 2)).numpy()
        )
        trP_lin = float(tf.linalg.trace(P_hat_lin).numpy())

        # Optimal homotopy β*(λ)
        t2 = time.perf_counter()
        filter_opt.update(
            z_mc,
            beta_opt_lam=beta_opt_lam,
            beta_opt=beta_opt,
            beta_opt_dot=beta_opt_dot,
        )
        t3 = time.perf_counter()
        x_hat_opt = tf.cast(filter_opt.state, DTYPE)
        P_hat_opt = tf.cast(filter_opt.covariance, DTYPE)
        rmse_opt = float(
            tf.sqrt(tf.reduce_mean((x_hat_opt - x_true) ** 2)).numpy()
        )
        trP_opt = float(tf.linalg.trace(P_hat_opt).numpy())

        results.append((rmse_lin, rmse_opt, trP_lin, trP_opt))
        times_lin.append(t1 - t0)
        times_opt.append(t3 - t2)

        print(
            f"{mc:3d} | {rmse_lin:10.4f} | {rmse_opt:10.4f} | "
            f"{trP_lin:12.1f} | {trP_opt:12.1f}"
        )

    # Figure 3: computing time comparison
    plot_figure3(
        times_lin,
        times_opt,
        out_path=fig_dir / "figure3_spf_timing.png",
    )

    return results, float(J_linear.numpy()), float(J_optimal.numpy()), times_lin, times_opt, time_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stochastic Particle Flow experiment (Dai & Daum, 2021)."
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=50,
        help="Number of particles (default: 50, paper uses 50).",
    )
    parser.add_argument(
        "--lambda-steps",
        type=int,
        default=200,
        help="Number of Euler–Maruyama steps in λ (default: 200).",
    )
    parser.add_argument(
        "--bvp-mesh",
        type=int,
        default=50,
        help="Number of mesh points for β(λ) discretization (default: 50).",
    )
    parser.add_argument(
        "--bvp-iters",
        type=int,
        default=1600,
        help="Maximum iterations per μ step when solving for β*(λ) (default: 1000).",
    )
    parser.add_argument(
        "--bvp-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer in β*(λ) optimization (default: 1e-3).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.2,
        help="Regularization parameter μ in J(β) (default: 0.2).",
    )
    parser.add_argument(
        "--no-mu-cont",
        action="store_true",
        help="Disable μ-continuation (not recommended).",
    )
    parser.add_argument(
        "--mu-steps",
        type=int,
        default=8,
        help="Number of μ-continuation steps (default: 8).",
    )

    args = parser.parse_args(argv)

    # Reproducibility
    tf.random.set_seed(0)

    config = SPFConfig(
        n_particles=args.particles,
        lambda_steps=args.lambda_steps,
        bvp_mesh_points=args.bvp_mesh,
        bvp_max_iter=args.bvp_iters,
        bvp_learning_rate=args.bvp_lr,
        mu=args.mu,
        use_mu_continuation=not args.no_mu_cont,
        mu_steps=args.mu_steps,
    )

    print("=" * 70)
    print("Stochastic Particle Flow Filters with Stiffness Mitigation")
    print("Dai & Daum (2021) - Bearing-only localization example")
    print("=" * 70)
    print(f"Particles       : {config.n_particles}")
    print(f"λ steps         : {config.lambda_steps}")
    print(f"BVP mesh points : {config.bvp_mesh_points}")
    print(f"BVP iters       : {config.bvp_max_iter}")
    print(f"BVP lr          : {config.bvp_learning_rate}")
    print(f"μ               : {config.mu}")
    print(f"μ-continuation  : {config.use_mu_continuation} (steps={config.mu_steps})")
    print()

    t_main_start = time.perf_counter()
    results, J_linear, J_optimal, times_lin, times_opt, time_training = run_experiment(config)
    time_total = time.perf_counter() - t_main_start

    print(f"Objective J(β=λ)    : {J_linear:.4f}")
    print(f"Objective J(β*)     : {J_optimal:.4f}")
    if J_optimal < J_linear:
        reduction = (J_linear - J_optimal) / J_linear * 100.0
        print(f"  Reduction in J    : {reduction:.1f}%")
    else:
        print("  WARNING: J(β*) >= J(β=λ)")
    print()

    # Convert to TensorFlow for summary stats
    results_tf = tf.constant(results, dtype=DTYPE)
    rmse_lin = results_tf[:, 0]
    rmse_opt = results_tf[:, 1]
    trP_lin = results_tf[:, 2]
    trP_opt = results_tf[:, 3]

    mean_rmse_lin = tf.reduce_mean(rmse_lin).numpy().item()
    mean_rmse_opt = tf.reduce_mean(rmse_opt).numpy().item()
    mean_trP_lin = tf.reduce_mean(trP_lin).numpy().item()
    mean_trP_opt = tf.reduce_mean(trP_opt).numpy().item()

    print("=" * 70)
    print("Monte Carlo summary (approximate Table 1 style)")
    print("=" * 70)
    print(f"Mean RMSE  (β=λ) : {mean_rmse_lin:.4f}")
    print(f"Mean RMSE  (β*)  : {mean_rmse_opt:.4f}")
    print(f"Mean tr(P) (β=λ) : {mean_trP_lin:.4f}")
    print(f"Mean tr(P) (β*)  : {mean_trP_opt:.4f}")
    if mean_rmse_lin > 0:
        rmse_red = (mean_rmse_lin - mean_rmse_opt) / mean_rmse_lin * 100.0
    else:
        rmse_red = 0.0
    if mean_trP_lin > 0:
        trP_red = (mean_trP_lin - mean_trP_opt) / mean_trP_lin * 100.0
    else:
        trP_red = 0.0
    print(f"RMSE reduction    : {rmse_red:.1f}%")
    print(f"tr(P) reduction   : {trP_red:.1f}%")
    print("=" * 70)

    print()
    print("=" * 70)
    print("Timing Summary")
    print("=" * 70)
    print(f"Total experiment time : {time_total:.2f} s")
    print(f"Training (BVP solver) : {time_training:.2f} s")
    print(f"Filtering (Linear β)  : {sum(times_lin):.2f} s (total over {len(times_lin)} MC runs)")
    print(f"Filtering (Optimal β) : {sum(times_opt):.2f} s (total over {len(times_opt)} MC runs)")
    print("=" * 70)


    # ------------------------------------------------------------------
    # Save per-MC table + summary to text file
    # ------------------------------------------------------------------
    out_dir = Path("reports/4_Stochastic_Particle_Flow/Dai_Daum")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spf_dai_daum_results.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Stochastic Particle Flow Filters with Stiffness Mitigation\n")
        f.write("Dai & Daum (2021) - Bearing-only localization example\n")
        f.write("=" * 70 + "\n")
        f.write(f"Particles       : {config.n_particles}\n")
        f.write(f"λ steps         : {config.lambda_steps}\n")
        f.write(f"BVP mesh points : {config.bvp_mesh_points}\n")
        f.write(f"BVP iters       : {config.bvp_max_iter}\n")
        f.write(f"BVP lr          : {config.bvp_learning_rate}\n")
        f.write(f"μ               : {config.mu}\n")
        f.write(f"μ-continuation  : {config.use_mu_continuation} (steps={config.mu_steps})\n\n")

        f.write(f"Objective J(β=λ)    : {J_linear:.4f}\n")
        f.write(f"Objective J(β*)     : {J_optimal:.4f}\n")
        if J_optimal < J_linear:
            f.write(
                f"  Reduction in J    : "
                f"{(J_linear - J_optimal) / J_linear * 100.0:.1f}%\n"
            )
        else:
            f.write("  WARNING: J(β*) >= J(β=λ)\n")
        f.write("\n")

        f.write("Monte Carlo results (per run)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'MC':>3} | {'RMSE(β)':>10} | {'RMSE(β*)':>10} | "
                f"{'tr(Pβ)':>12} | {'tr(Pβ*)':>12}\n")
        f.write("=" * 70 + "\n")
        for mc, (rm_l, rm_o, tr_l, tr_o) in enumerate(results, start=1):
            f.write(
                f"{mc:3d} | {rm_l:10.4f} | {rm_o:10.4f} | "
                f"{tr_l:12.1f} | {tr_o:12.1f}\n"
            )
        f.write("\n")

        f.write("Monte Carlo summary (approximate Table 1 style)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Mean RMSE  (β=λ) : {mean_rmse_lin:.4f}\n")
        f.write(f"Mean RMSE  (β*)  : {mean_rmse_opt:.4f}\n")
        f.write(f"Mean tr(P) (β=λ) : {mean_trP_lin:.4f}\n")
        f.write(f"Mean tr(P) (β*)  : {mean_trP_opt:.4f}\n")
        f.write(f"RMSE reduction    : {rmse_red:.1f}%\n")
        f.write(f"tr(P) reduction   : {trP_red:.1f}%\n")
        f.write("=" * 70 + "\n\n")

        f.write("Timing Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total experiment time : {time_total:.2f} s\n")
        f.write(f"Training (BVP solver) : {time_training:.2f} s\n")
        f.write(f"Filtering (Linear β)  : {sum(times_lin):.2f} s (total over {len(times_lin)} MC runs)\n")
        f.write(f"Filtering (Optimal β) : {sum(times_opt):.2f} s (total over {len(times_opt)} MC runs)\n")
        f.write("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

