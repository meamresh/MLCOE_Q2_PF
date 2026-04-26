"""
Experiment: Kalman Filter Optimality and Numerical Stability Analysis.

This script analyzes the filtering performance and numerical stability of the
Kalman filter implementation by comparing Riccati vs Joseph-stabilized updates.

Key analyses:
    1. Filtering accuracy (RMSE) and consistency (NEES)
    2. Numerical stability (condition numbers, symmetry, PD checks)
    3. Riccati vs Joseph covariance update comparison
"""

from __future__ import annotations

import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Project imports
sys.path.append('.')
from src.data.generators import generate_lgssm_from_yaml
from src.filters.kalman import KalmanFilter
from src.metrics.accuracy import (
    compute_rmse, 
    compute_mae, 
    compute_nees,
    compute_per_dimension_rmse
)
from src.metrics.stability import (
    compute_condition_numbers,
    check_symmetry,
    check_positive_definite,
    compute_frobenius_norm_difference,
    compute_trace,
    compute_log_determinant
)


def run_stability_analysis(config_path: str = "configs/ssm_linear.yaml") -> dict:
    """
    Run comprehensive Kalman filter stability analysis.

    Compares standard Riccati and Joseph-stabilized covariance updates across
    multiple numerical stability and accuracy metrics.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file defining the LGSSM.

    Returns
    -------
    results : dict
        Dictionary containing all analysis results, including:
        - model: LGSSM instance
        - X_true: ground truth states (N, nx)
        - Y_obs: observations (N, ny)
        - results_riccati: Riccati filter outputs
        - results_joseph: Joseph filter outputs
        - rmse: RMSE metrics for both filters
        - nees: NEES consistency metrics
        - condition_numbers: κ(P) for filtered and predicted covariances
        - symmetry: symmetry violation metrics
        - eigenvalues: minimum eigenvalues (PD check)
        - covariance_difference: ||P_Riccati - P_Joseph||_F
    """
    print("=" * 70)
    print("KALMAN FILTER OPTIMALITY AND NUMERICAL STABILITY ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Generate synthetic data
    # -------------------------------------------------------------------------
    print("\n[1/9] Generating synthetic data...")
    model, X_true, Y_obs, data_dict = generate_lgssm_from_yaml(config_path)
    N = tf.shape(X_true)[0]
    nx = model.nx
    print(f"      Time steps: {N}")
    print(f"      State dimension: {nx}")
    print(f"      Observation dimension: {model.ny}")

    # -------------------------------------------------------------------------
    # 2. Initialize filters
    # -------------------------------------------------------------------------
    print("\n[2/9] Initializing Kalman filters...")
    # Reshape m0 to (n, 1) if needed
    try:
        if model.m0.shape.ndims == 1:
            x0_reshaped = tf.reshape(model.m0, [-1, 1])
        else:
            x0_reshaped = model.m0
    except (AttributeError, ValueError):
        x0_rank = tf.rank(model.m0)
        x0_reshaped = tf.cond(tf.equal(x0_rank, 1),
                             lambda: tf.reshape(model.m0, [-1, 1]),
                             lambda: model.m0)
    kf_riccati = KalmanFilter(A=model.A, C=model.C, x0=x0_reshaped, P0=model.P0,
                              Q=model.Q, R=model.R)
    kf_joseph = KalmanFilter(A=model.A, C=model.C, x0=x0_reshaped, P0=model.P0,
                            Q=model.Q, R=model.R)
    print("      ✓ Riccati filter initialized")
    print("      ✓ Joseph filter initialized")

    # -------------------------------------------------------------------------
    # 3. Run filtering
    # -------------------------------------------------------------------------
    print("\n[3/9] Running Kalman filter...")
    print("      → Standard Riccati update...")
    results_riccati = kf_riccati.filter(Y_obs, joseph=False, use_solve=True)
    print("      → Joseph-stabilized update...")
    results_joseph = kf_joseph.filter(Y_obs, joseph=True, use_solve=True)
    print("      ✓ Filtering complete")

    # -------------------------------------------------------------------------
    # 4. Compute accuracy metrics (RMSE, MAE, per-dimension RMSE)
    # -------------------------------------------------------------------------
    print("\n[4/9] Computing accuracy metrics...")
    rmse_riccati_filt = compute_rmse(results_riccati["x_filt"], X_true)
    rmse_joseph_filt = compute_rmse(results_joseph["x_filt"], X_true)
    rmse_riccati_pred = compute_rmse(results_riccati["x_pred"], X_true)
    rmse_joseph_pred = compute_rmse(results_joseph["x_pred"], X_true)

    mae_riccati = compute_mae(results_riccati["x_filt"], X_true)
    mae_joseph = compute_mae(results_joseph["x_filt"], X_true)

    rmse_per_dim_riccati = compute_per_dimension_rmse(results_riccati["x_filt"], X_true)
    rmse_per_dim_joseph = compute_per_dimension_rmse(results_joseph["x_filt"], X_true)

    print(f"      Riccati - Filtered RMSE: {rmse_riccati_filt:.6f}")
    print(f"      Riccati - Predicted RMSE: {rmse_riccati_pred:.6f}")
    print(f"      Joseph  - Filtered RMSE: {rmse_joseph_filt:.6f}")
    print(f"      Joseph  - Predicted RMSE: {rmse_joseph_pred:.6f}")

    # -------------------------------------------------------------------------
    # 5. Compute NEES (consistency check)
    # -------------------------------------------------------------------------
    print("\n[5/9] Computing NEES (consistency check)...")
    nees_riccati = compute_nees(results_riccati["x_filt"], 
                                results_riccati["P_filt"], X_true)
    nees_joseph = compute_nees(results_joseph["x_filt"],
                               results_joseph["P_filt"], X_true)

    mean_nees_riccati = tf.reduce_mean(nees_riccati)
    mean_nees_joseph = tf.reduce_mean(nees_joseph)

    print(f"      Riccati - Mean NEES: {float(mean_nees_riccati):.4f} (expected ≈ {nx})")
    print(f"      Joseph  - Mean NEES: {float(mean_nees_joseph):.4f} (expected ≈ {nx})")

    # -------------------------------------------------------------------------
    # 6. Compute condition numbers
    # -------------------------------------------------------------------------
    print("\n[6/9] Computing condition numbers...")
    cond_riccati_filt = compute_condition_numbers(results_riccati["P_filt"])
    cond_joseph_filt = compute_condition_numbers(results_joseph["P_filt"])
    cond_riccati_pred = compute_condition_numbers(results_riccati["P_pred"])
    cond_joseph_pred = compute_condition_numbers(results_joseph["P_pred"])

    print(f"      Riccati - Filtered κ(P): mean={float(tf.reduce_mean(cond_riccati_filt)):.2e}, "
          f"max={float(tf.reduce_max(cond_riccati_filt)):.2e}")
    print(f"      Joseph  - Filtered κ(P): mean={float(tf.reduce_mean(cond_joseph_filt)):.2e}, "
          f"max={float(tf.reduce_max(cond_joseph_filt)):.2e}")

    # -------------------------------------------------------------------------
    # 7. Check symmetry preservation
    # -------------------------------------------------------------------------
    print("\n[7/9] Checking covariance symmetry...")
    sym_riccati = check_symmetry(results_riccati["P_filt"])
    sym_joseph = check_symmetry(results_joseph["P_filt"])

    print(f"      Riccati - max|P - P^T|: {float(tf.reduce_max(sym_riccati)):.2e}")
    print(f"      Joseph  - max|P - P^T|: {float(tf.reduce_max(sym_joseph)):.2e}")

    # -------------------------------------------------------------------------
    # 8. Check positive definiteness
    # -------------------------------------------------------------------------
    print("\n[8/9] Checking positive definiteness...")
    eigmin_riccati, pd_riccati = check_positive_definite(results_riccati["P_filt"])
    eigmin_joseph, pd_joseph = check_positive_definite(results_joseph["P_filt"])

    pd_riccati_sum = tf.reduce_sum(tf.cast(~pd_riccati, tf.int32))
    pd_joseph_sum = tf.reduce_sum(tf.cast(~pd_joseph, tf.int32))
    print(f"      Riccati - min λ_min: {float(tf.reduce_min(eigmin_riccati)):.2e}, "
          f"PD violations: {int(pd_riccati_sum)}/{N}")
    print(f"      Joseph  - min λ_min: {float(tf.reduce_min(eigmin_joseph)):.2e}, "
          f"PD violations: {int(pd_joseph_sum)}/{N}")

    # -------------------------------------------------------------------------
    # 9. Compare Riccati vs Joseph covariances
    # -------------------------------------------------------------------------
    print("\n[9/9] Comparing Riccati vs Joseph covariances...")
    P_diff = compute_frobenius_norm_difference(results_riccati["P_filt"], 
                                                results_joseph["P_filt"])

    print(f"      max ||P_Riccati - P_Joseph||_F: {float(tf.reduce_max(P_diff)):.2e}")
    print(f"      mean ||P_Riccati - P_Joseph||_F: {float(tf.reduce_mean(P_diff)):.2e}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Package results
    # -------------------------------------------------------------------------
    results = {
        "model": model,
        "X_true": X_true,
        "Y_obs": Y_obs,
        "results_riccati": results_riccati,
        "results_joseph": results_joseph,
        "rmse": {
            "riccati_filt": rmse_riccati_filt,
            "joseph_filt": rmse_joseph_filt,
            "riccati_pred": rmse_riccati_pred,
            "joseph_pred": rmse_joseph_pred,
            "riccati_per_dim": rmse_per_dim_riccati,
            "joseph_per_dim": rmse_per_dim_joseph,
        },
        "mae": {
            "riccati": mae_riccati,
            "joseph": mae_joseph,
        },
        "nees": {
            "riccati": nees_riccati,
            "joseph": nees_joseph,
            "mean_riccati": mean_nees_riccati,
            "mean_joseph": mean_nees_joseph,
        },
        "condition_numbers": {
            "riccati_filt": cond_riccati_filt,
            "joseph_filt": cond_joseph_filt,
            "riccati_pred": cond_riccati_pred,
            "joseph_pred": cond_joseph_pred,
        },
        "symmetry": {
            "riccati": sym_riccati,
            "joseph": sym_joseph,
        },
        "eigenvalues": {
            "riccati": eigmin_riccati,
            "joseph": eigmin_joseph,
        },
        "covariance_difference": P_diff,
        "pd_riccati": pd_riccati,
        "pd_joseph": pd_joseph,
    }

    return results


def generate_analysis_plots(results: dict, output_dir: str = "reports/1_LinearGaussianSSM/figures"):
    """
    Generate comprehensive visualization of stability analysis.

    Creates two publication-quality figures:
        1. State estimates with 2σ uncertainty bounds
        2. Numerical stability metrics (NEES, κ(P), eigenvalues, differences)

    Parameters
    ----------
    results : dict
        Results dictionary from run_stability_analysis().
    output_dir : str
        Directory to save figures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_true = results["X_true"]
    Y_obs = results["Y_obs"]
    N = int(tf.shape(X_true)[0])
    t = tf.range(N, dtype=tf.float32).numpy()

    # Extract results and convert to numpy for plotting
    x_filt_r = tf.squeeze(results["results_riccati"]["x_filt"]).numpy()
    x_filt_j = tf.squeeze(results["results_joseph"]["x_filt"]).numpy()
    P_filt_r = results["results_riccati"]["P_filt"]
    P_filt_j = results["results_joseph"]["P_filt"]
    X_true_np = X_true.numpy()
    Y_obs_np = Y_obs.numpy()

    # =========================================================================
    # Figure 1: State estimates and 2σ bounds
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Kalman Filter: State Estimates and Uncertainty", 
                 fontsize=14, fontweight='bold')

    state_labels = ['x [pos]', 'vx [vel]', 'y [pos]', 'vy [vel]']

    for i in range(4):
        ax = axes[i // 2, i % 2]

        # True state
        ax.plot(t, X_true_np[:, i], 'k-', label='True', linewidth=1.5, alpha=0.8)

        # Riccati estimate with 2σ bounds
        std_r = tf.sqrt([P_filt_r[k, i, i] for k in range(N)]).numpy()
        ax.plot(t, x_filt_r[:, i], 'b-', label='Riccati', linewidth=1.2)
        ax.fill_between(t, x_filt_r[:, i] - 2*std_r, x_filt_r[:, i] + 2*std_r,
                        alpha=0.2, color='b', label='Riccati 2σ')

        # Joseph estimate
        std_j = tf.sqrt([P_filt_j[k, i, i] for k in range(N)]).numpy()
        ax.plot(t, x_filt_j[:, i], 'r--', label='Joseph', linewidth=1.2, alpha=0.8)

        # Add observations for position states
        if i in [0, 2]:
            obs_idx = 0 if i == 0 else 1
            ax.scatter(t[::5], Y_obs_np[::5, obs_idx], c='gray', s=10, alpha=0.5,
                      label='Observations', zorder=1)

        ax.set_xlabel('Time step')
        ax.set_ylabel(state_labels[i])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/kf_stability_states.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/kf_stability_states.png")

    # =========================================================================
    # Figure 2: Numerical stability metrics
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Numerical Stability Analysis", fontsize=14, fontweight='bold')

    # Convert tensors to numpy for plotting
    nees_riccati = results["nees"]["riccati"].numpy()
    nees_joseph = results["nees"]["joseph"].numpy()
    cond_riccati_filt = results["condition_numbers"]["riccati_filt"].numpy()
    cond_joseph_filt = results["condition_numbers"]["joseph_filt"].numpy()
    eigmin_riccati = results["eigenvalues"]["riccati"].numpy()
    eigmin_joseph = results["eigenvalues"]["joseph"].numpy()
    cov_diff = results["covariance_difference"].numpy()

    # NEES
    ax = axes[0, 0]
    ax.plot(t, nees_riccati, 'b-', label='Riccati', alpha=0.7)
    ax.plot(t, nees_joseph, 'r--', label='Joseph', alpha=0.7)
    ax.axhline(4, color='k', linestyle=':', label='Expected (nx=4)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('NEES')
    ax.set_title('Normalized Estimation Error Squared')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Condition numbers (log scale)
    ax = axes[0, 1]
    ax.semilogy(t, cond_riccati_filt, 'b-', 
                label='Riccati (filt)', alpha=0.7)
    ax.semilogy(t, cond_joseph_filt, 'r--',
                label='Joseph (filt)', alpha=0.7)
    ax.set_xlabel('Time step')
    ax.set_ylabel('κ(P)')
    ax.set_title('Condition Number of Filtered Covariance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Minimum eigenvalues
    ax = axes[1, 0]
    ax.plot(t, eigmin_riccati, 'b-', label='Riccati', alpha=0.7)
    ax.plot(t, eigmin_joseph, 'r--', label='Joseph', alpha=0.7)
    ax.axhline(0, color='k', linestyle=':')
    ax.set_xlabel('Time step')
    ax.set_ylabel('λ_min(P)')
    ax.set_title('Minimum Eigenvalue (PD check)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frobenius norm difference
    ax = axes[1, 1]
    ax.semilogy(t, cov_diff, 'g-', alpha=0.7)
    ax.set_xlabel('Time step')
    ax.set_ylabel('||P_Riccati - P_Joseph||_F')
    ax.set_title('Covariance Difference: Riccati vs Joseph')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/kf_stability_metrics.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/kf_stability_metrics.png")

    plt.close('all')


def save_summary_table(results: dict, 
                        output_path: str = "reports/1_LinearGaussianSSM/stability_summary.csv"):
    """
    Save summary statistics to text file.

    Parameters
    ----------
    results : dict
        Results from run_stability_analysis().
    output_path : str
        Path to save text file.

    Returns
    -------
    summary : dict
        Summary dictionary.
    """
    summary = {
        "Metric": [
            "RMSE (filtered)",
            "RMSE (predicted)",
            "MAE (filtered)",
            "Mean NEES",
            "Mean κ(P_filt)",
            "Max κ(P_filt)",
            "Max |P - P^T|",
            "Min λ_min(P)",
            "PD violations",
        ],
        "Riccati": [
            results["rmse"]["riccati_filt"],
            results["rmse"]["riccati_pred"],
            results["mae"]["riccati"],
            float(results["nees"]["mean_riccati"]),
            float(tf.reduce_mean(results["condition_numbers"]["riccati_filt"])),
            float(tf.reduce_max(results["condition_numbers"]["riccati_filt"])),
            float(tf.reduce_max(results["symmetry"]["riccati"])),
            float(tf.reduce_min(results["eigenvalues"]["riccati"])),
            int(tf.reduce_sum(tf.cast(results["eigenvalues"]["riccati"] <= 0, tf.int32))),
        ],
        "Joseph": [
            results["rmse"]["joseph_filt"],
            results["rmse"]["joseph_pred"],
            results["mae"]["joseph"],
            float(results["nees"]["mean_joseph"]),
            float(tf.reduce_mean(results["condition_numbers"]["joseph_filt"])),
            float(tf.reduce_max(results["condition_numbers"]["joseph_filt"])),
            float(tf.reduce_max(results["symmetry"]["joseph"])),
            float(tf.reduce_min(results["eigenvalues"]["joseph"])),
            int(tf.reduce_sum(tf.cast(results["eigenvalues"]["joseph"] <= 0, tf.int32))),
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("Metric,Riccati,Joseph\n")
        for i in range(len(summary["Metric"])):
            f.write(f"{summary['Metric'][i]},{summary['Riccati'][i]},{summary['Joseph'][i]}\n")
    print(f"\n✓ Saved summary table: {output_path}")

    return summary


if __name__ == "__main__":
    # Run analysis
    #import multiprocessing as mp
    #mp.set_start_method("spawn", force=True)
    
    results = run_stability_analysis("configs/ssm_linear.yaml")

    # Generate plots
    generate_analysis_plots(results)

    # Save summary
    summary = save_summary_table(results)

    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print("Metric\t\t\t\tRiccati\t\tJoseph")
    print("-" * 70)
    for i in range(len(summary["Metric"])):
        print(f"{summary['Metric'][i]:<30}\t{summary['Riccati'][i]:.6f}\t\t{summary['Joseph'][i]:.6f}")
    print("=" * 70)