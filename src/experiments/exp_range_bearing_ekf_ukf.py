"""
Experiment: Range-Bearing Localization with EKF and UKF.

This experiment compares Extended Kalman Filter (EKF) and Unscented Kalman
Filter (UKF) performance for robot localization using range-bearing
measurements to known landmarks. Includes parameter tuning via grid search.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ssm_range_bearing import RangeBearingSSM
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter

# ---------------------------------------------------------------------
# Report directory structure
# ---------------------------------------------------------------------
BASE_REPORT_DIR = Path("reports/range_bearing")
EXPERIMENT_DIR = BASE_REPORT_DIR / "experiments"
TUNING_DIR = BASE_REPORT_DIR / "tuning"

def simulate_localization_with_params(scenario: str = 'moderate',
                                     num_steps: int = 100,
                                     alpha: float = 0.001,
                                     beta: float = 2.0,
                                     kappa: float = 0,
                                     seed: int | None = None) -> dict:
    """
    Run simulation with specific UKF parameters.

    Parameters
    ----------
    scenario : str
        'moderate' or 'strong' nonlinearity scenario.
    num_steps : int
        Number of simulation steps.
    alpha : float
        UKF alpha parameter.
    beta : float
        UKF beta parameter.
    kappa : float
        UKF kappa parameter.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing states, covariances, landmarks, and scenario.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    if scenario == 'moderate':
        dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        initial_uncertainty = tf.eye(3, dtype=tf.float32) * 0.5

        landmarks = tf.constant([
            [5.0, 5.0], [-5.0, 5.0],
            [5.0, -5.0], [-5.0, -5.0]
        ], dtype=tf.float32)
    elif scenario == 'strong':
        dt = 0.5
        Q = tf.eye(3, dtype=tf.float32) * 0.1
        R = tf.eye(2, dtype=tf.float32) * 0.5
        initial_uncertainty = tf.eye(3, dtype=tf.float32) * 2.0

        landmarks = tf.constant([
            [1.0, 1.0], [-1.0, -1.0]
        ], dtype=tf.float32)
    else:
        raise ValueError("Unknown scenario")

    ssm = RangeBearingSSM(dt=dt, process_noise=Q, meas_noise=R)
    true_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    initial_state = tf.constant([0.5, -0.5, 0.2], dtype=tf.float32)

    ekf = ExtendedKalmanFilter(ssm, initial_state, initial_uncertainty)
    ukf = UnscentedKalmanFilter(ssm, initial_state, initial_uncertainty,
                                alpha=alpha, beta=beta, kappa=kappa)

    true_states = []
    ekf_states = []
    ukf_states = []
    ekf_covs = []
    ukf_covs = []

    for step in range(num_steps):
        step_tf = tf.cast(step, tf.float32)
        v = 1.5 + 0.5 * tf.sin(step_tf * 0.15)
        omega = 0.6 + 0.3 * tf.cos(step_tf * 0.12)
        control = tf.stack([v, omega])

        true_state = ssm.motion_model(true_state, control)[0]
        true_state = (true_state +
                     tf.random.normal([3], mean=0.0, stddev=0.1,
                                     dtype=tf.float32))

        true_meas = ssm.measurement_model(true_state, landmarks)[0]
        meas_std = tf.sqrt(tf.linalg.diag_part(ssm.R))
        meas_noise = tf.random.normal([tf.shape(landmarks)[0], 2],
                                      mean=0.0, stddev=meas_std,
                                      dtype=tf.float32)
        measured = true_meas + meas_noise

        ekf.predict(control)
        ekf.update(measured, landmarks)

        ukf.predict(control)
        ukf.update(measured, landmarks)

        true_states.append(true_state)
        ekf_states.append(tf.identity(ekf.state))
        ukf_states.append(tf.identity(ukf.state))
        ekf_covs.append(tf.identity(ekf.covariance))
        ukf_covs.append(tf.identity(ukf.covariance))

    ts = tf.stack(true_states)
    es = tf.stack(ekf_states)
    us = tf.stack(ukf_states)
    ec = tf.stack(ekf_covs)
    uc = tf.stack(ukf_covs)

    return {
        'true_states': ts.numpy(),
        'ekf_states': es.numpy(),
        'ukf_states': us.numpy(),
        'ekf_covariances': ec.numpy(),
        'ukf_covariances': uc.numpy(),
        'landmarks': landmarks.numpy(),
        'scenario': scenario
    }


def grid_search_ukf_parameters(scenario: str = 'strong',
                               num_steps: int = 100,
                               alpha_values: list[float] | None = None,
                               beta_values: list[float] | None = None,
                               kappa_values: list[float] | None = None,
                               verbose: bool = True) -> dict:
    """
    Perform grid search over UKF parameters (alpha, beta, kappa).

    Parameters
    ----------
    scenario : str
        'moderate' or 'strong' nonlinearity scenario.
    num_steps : int
        Number of simulation steps.
    alpha_values : list[float], optional
        List of alpha values to test. Defaults to [0.001, 0.01, 0.1, 0.5, 1.0].
    beta_values : list[float], optional
        List of beta values to test. Defaults to [2.0].
    kappa_values : list[float], optional
        List of kappa values to test. Defaults to [0, -1, 1].
    verbose : bool
        Print progress information.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'best_params': Best (alpha, beta, kappa) combination
        - 'best_rmse': Best position RMSE
        - 'all_results': List of all tested configurations with metrics
        - 'true_states': True states used in evaluation
        - 'scenario': Scenario name
    """
    if alpha_values is None:
        alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0]
    if beta_values is None:
        beta_values = [2.0]
    if kappa_values is None:
        kappa_values = [0, -1, 1]

    if scenario == 'moderate':
        dt = 0.1
        Q = tf.eye(3, dtype=tf.float32) * 0.01
        R = tf.eye(2, dtype=tf.float32) * 0.05
        initial_uncertainty = tf.eye(3, dtype=tf.float32) * 0.5

        landmarks = tf.constant([
            [5.0, 5.0], [-5.0, 5.0],
            [5.0, -5.0], [-5.0, -5.0]
        ], dtype=tf.float32)
    elif scenario == 'strong':
        dt = 0.5
        Q = tf.eye(3, dtype=tf.float32) * 0.1
        R = tf.eye(2, dtype=tf.float32) * 0.5
        initial_uncertainty = tf.eye(3, dtype=tf.float32) * 2.0

        landmarks = tf.constant([
            [1.0, 1.0], [-1.0, -1.0]
        ], dtype=tf.float32)
    else:
        raise ValueError("Unknown scenario")

    ssm = RangeBearingSSM(dt=dt, process_noise=Q, meas_noise=R)

    # Generate true trajectory once (for consistency across runs)
    tf.random.set_seed(42)

    true_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    true_states = []
    measurements = []
    controls = []

    for step in range(num_steps):
        step_tf = tf.cast(step, tf.float32)
        v = 1.5 + 0.5 * tf.sin(step_tf * 0.15)
        omega = 0.6 + 0.3 * tf.cos(step_tf * 0.12)
        control = tf.stack([v, omega])

        true_state = ssm.motion_model(true_state, control)[0]
        true_state = (true_state +
                     tf.random.normal([3], mean=0.0, stddev=0.1,
                                     dtype=tf.float32))

        true_meas = ssm.measurement_model(true_state, landmarks)[0]
        meas_std = tf.sqrt(tf.linalg.diag_part(ssm.R))
        meas_noise = tf.random.normal([tf.shape(landmarks)[0], 2],
                                      mean=0.0, stddev=meas_std,
                                      dtype=tf.float32)
        measured = true_meas + meas_noise

        true_states.append(true_state.numpy())
        measurements.append(measured.numpy())
        controls.append(control.numpy())

    true_states = tf.constant(true_states, dtype=tf.float32)
    measurements_array = measurements
    controls_array = controls

    # Grid search
    all_results = []
    best_rmse = float('inf')
    best_params = None

    total_configs = len(alpha_values) * len(beta_values) * len(kappa_values)
    config_num = 0

    if verbose:
        print(f"\nTesting {total_configs} parameter combinations...")
        print("-" * 80)

    for alpha in alpha_values:
        for beta in beta_values:
            for kappa in kappa_values:
                config_num += 1

                # Initialize UKF with current parameters
                initial_state = tf.constant([0.5, -0.5, 0.2], dtype=tf.float32)
                ukf = UnscentedKalmanFilter(ssm, initial_state,
                                           initial_uncertainty,
                                           alpha=alpha, beta=beta, kappa=kappa)

                ukf_states = []

                # Run filter
                try:
                    for step in range(num_steps):
                        control = tf.constant(controls_array[step],
                                            dtype=tf.float32)
                        measured = tf.constant(measurements_array[step],
                                             dtype=tf.float32)

                        ukf.predict(control)
                        ukf.update(measured, landmarks)

                        ukf_states.append(tf.identity(ukf.state).numpy())

                    ukf_states = tf.constant(ukf_states, dtype=tf.float32)

                    # Compute metrics
                    pos_error = tf.norm(ukf_states[:, :2] -
                                       true_states[:, :2], axis=1)
                    pos_rmse = tf.sqrt(tf.reduce_mean(pos_error ** 2)).numpy()
                    pos_final = pos_error[-1].numpy()

                    heading_diff = ukf_states[:, 2] - true_states[:, 2]
                    heading_error = tf.abs(tf.math.atan2(tf.sin(heading_diff),
                                                         tf.cos(heading_diff)))
                    heading_rmse = tf.sqrt(tf.reduce_mean(heading_error ** 2)).numpy()
                    heading_final = heading_error[-1].numpy()

                    # Store results
                    result = {
                        'alpha': alpha,
                        'beta': beta,
                        'kappa': kappa,
                        'pos_rmse': pos_rmse,
                        'pos_final_error': pos_final,
                        'heading_rmse': heading_rmse,
                        'heading_final_error': heading_final,
                        'success': True
                    }

                    all_results.append(result)

                    # Update best
                    if pos_rmse < best_rmse:
                        best_rmse = pos_rmse
                        best_params = (alpha, beta, kappa)

                    if verbose:
                        print(f"[{config_num}/{total_configs}] "
                             f"α={alpha:.3f}, β={beta:.1f}, κ={kappa:+.0f} | "
                             f"Pos RMSE: {pos_rmse:.4f} m, "
                             f"Heading RMSE: {heading_rmse:.4f} rad")

                except Exception as e:
                    if verbose:
                        print(f"[{config_num}/{total_configs}] "
                             f"α={alpha:.3f}, β={beta:.1f}, κ={kappa:+.0f} | "
                             f"FAILED: {str(e)}")

                    result = {
                        'alpha': alpha,
                        'beta': beta,
                        'kappa': kappa,
                        'pos_rmse': float('inf'),
                        'pos_final_error': float('inf'),
                        'heading_rmse': float('inf'),
                        'heading_final_error': float('inf'),
                        'success': False
                    }
                    all_results.append(result)

    if verbose:
        print("-" * 80)
        print(f"\n✓ Best parameters: α={best_params[0]:.3f}, "
             f"β={best_params[1]:.1f}, κ={best_params[2]:+.0f}")
        print(f"✓ Best position RMSE: {best_rmse:.4f} m")

    # Sort results by position RMSE
    all_results_sorted = sorted(all_results, key=lambda x: x['pos_rmse'])

    return {
        'best_params': best_params,
        'best_rmse': best_rmse,
        'all_results': all_results_sorted,
        'true_states': true_states.numpy(),
        'scenario': scenario
    }


def plot_results(results: dict, output_dir: str, title_add: str = "") -> None:
    """
    Plot comprehensive comparison of EKF vs UKF results.

    Parameters
    ----------
    results : dict
        Dictionary containing:
        - 'true_states': True states (N, 3)
        - 'ekf_states': EKF estimates (N, 3)
        - 'ukf_states': UKF estimates (N, 3)
        - 'ekf_covariances': EKF covariances (N, 3, 3)
        - 'ukf_covariances': UKF covariances (N, 3, 3)
        - 'landmarks': Landmark positions (M, 2)
        - 'scenario': Scenario name string
    output_dir : str
        Directory to save plots.
    title_add : str, optional
        Additional text to append to title.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"EKF vs UKF ({results['scenario'].title()})" + title_add,
                 fontsize=14)

    # Trajectory
    ax = axes[0, 0]
    ax.plot(results['true_states'][:, 0], results['true_states'][:, 1],
           'k--', label='True', alpha=0.7, linewidth=2)
    ax.plot(results['ekf_states'][:, 0], results['ekf_states'][:, 1],
           'b-', label='EKF', linewidth=1.5)
    ax.plot(results['ukf_states'][:, 0], results['ukf_states'][:, 1],
           'r-', label='UKF', linewidth=1.5)
    ax.scatter(results['landmarks'][:, 0], results['landmarks'][:, 1],
              c='g', marker='^', s=100, label='Landmarks', zorder=5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectory')
    ax.axis('equal')

    # Position error - using TensorFlow operations
    true_pos = results['true_states'][:, :2]
    ekf_pos = results['ekf_states'][:, :2]
    ukf_pos = results['ukf_states'][:, :2]

    ekf_error_tf = tf.norm(ekf_pos - true_pos, axis=1).numpy()
    ukf_error_tf = tf.norm(ukf_pos - true_pos, axis=1).numpy()

    ax = axes[0, 1]
    ax.plot(ekf_error_tf, 'b-', label='EKF', linewidth=1.5)
    ax.plot(ukf_error_tf, 'r-', label='UKF', linewidth=1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Error [m]')
    ax.set_title('Position Error')

    # Heading error - using TensorFlow operations
    true_heading = results['true_states'][:, 2]
    ekf_heading = results['ekf_states'][:, 2]
    ukf_heading = results['ukf_states'][:, 2]

    ekf_diff = ekf_heading - true_heading
    ukf_diff = ukf_heading - true_heading

    ekf_heading_error = tf.abs(tf.math.atan2(tf.sin(ekf_diff),
                                             tf.cos(ekf_diff))).numpy()
    ukf_heading_error = tf.abs(tf.math.atan2(tf.sin(ukf_diff),
                                             tf.cos(ukf_diff))).numpy()

    ax = axes[1, 0]
    ax.plot(ekf_heading_error, 'b-', label='EKF', linewidth=1.5)
    ax.plot(ukf_heading_error, 'r-', label='UKF', linewidth=1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Error [rad]')
    ax.set_title('Heading Error')

    # Covariance trace - using TensorFlow operations
    ekf_trace = tf.linalg.trace(results['ekf_covariances']).numpy()
    ukf_trace = tf.linalg.trace(results['ukf_covariances']).numpy()

    ax = axes[1, 1]
    ax.plot(ekf_trace, 'b-', label='EKF', linewidth=1.5)
    ax.plot(ukf_trace, 'r-', label='UKF', linewidth=1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Trace')
    ax.set_title('Uncertainty (Covariance Trace)')

    plt.tight_layout()
    fig_path = os.path.join(output_dir,
                           f"ekf_ukf_comparison_{results['scenario']}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {fig_path}")
    plt.close()

    # Print statistics - using TensorFlow operations
    ekf_error_mean = tf.reduce_mean(ekf_error_tf).numpy()
    ukf_error_mean = tf.reduce_mean(ukf_error_tf).numpy()

    print(f"\n=== {results['scenario'].title()} Nonlinearity ===")
    print(f"EKF Final Pos Error: {ekf_error_tf[-1]:.4f} m")
    print(f"UKF Final Pos Error: {ukf_error_tf[-1]:.4f} m")
    print(f"EKF Final Heading Error: {ekf_heading_error[-1]:.4f} rad")
    print(f"UKF Final Heading Error: {ukf_heading_error[-1]:.4f} rad")
    print(f"EKF Mean Pos Error: {ekf_error_mean:.4f} m")
    print(f"UKF Mean Pos Error: {ukf_error_mean:.4f} m")

    per = 100 * (ekf_error_mean - ukf_error_mean) / ekf_error_mean

    if per < 0:
        perc = -per
        print(f"EKF Wins: {perc:.2f} % Better")
    else:
        print(f"UKF Wins: {per:.2f} % Better")

    # Save metrics to CSV
    csv_path = os.path.join(output_dir,
                           f"ekf_ukf_metrics_{results['scenario']}.csv")
    with open(csv_path, 'w') as f:
        f.write("metric,ekf,ukf\n")
        f.write(f"final_pos_error,{ekf_error_tf[-1]:.6f},{ukf_error_tf[-1]:.6f}\n")
        f.write(f"final_heading_error,{ekf_heading_error[-1]:.6f},{ukf_heading_error[-1]:.6f}\n")
        f.write(f"mean_pos_error,{ekf_error_mean:.6f},{ukf_error_mean:.6f}\n")
        if per < 0:
            perc = -per
            f.write(f"EKF Wins, {perc:.2f} % Better\n")
        else:
            f.write(f"UKF Wins, {per:.2f} % Better\n")
    print(f"✓ Saved metrics: {csv_path}")


def plot_grid_search_results(grid_results: dict, output_dir: str,
                             top_n: int = 10) -> None:
    """
    Visualize grid search results.

    Parameters
    ----------
    grid_results : dict
        Output from grid_search_ukf_parameters.
    output_dir : str
        Directory to save plots.
    top_n : int
        Number of top configurations to display.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = grid_results['all_results']
    successful_results = [r for r in results if r['success']]

    if len(successful_results) == 0:
        print("No successful configurations to plot!")
        return

    # Filter top N
    top_results = successful_results[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'UKF Parameter Grid Search Results '
                f'({grid_results["scenario"].title()})',
                fontsize=14, fontweight='bold')

    # 1. Position RMSE comparison
    ax = axes[0, 0]
    labels = [f"α={r['alpha']:.3f}\nβ={r['beta']:.1f}\nκ={r['kappa']:+.0f}"
             for r in top_results]
    pos_rmse = [r['pos_rmse'] for r in top_results]
    colors = ['green' if i == 0 else 'steelblue'
             for i in range(len(top_results))]

    bars = ax.bar(range(len(top_results)), pos_rmse, color=colors, alpha=0.7)
    ax.set_xticks(range(len(top_results)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Position RMSE [m]', fontweight='bold')
    ax.set_title(f'Top {top_n} Configurations - Position RMSE')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight best
    bars[0].set_edgecolor('darkgreen')
    bars[0].set_linewidth(2.5)

    # 2. Heading RMSE comparison
    ax = axes[0, 1]
    heading_rmse = [r['heading_rmse'] for r in top_results]
    bars = ax.bar(range(len(top_results)), heading_rmse,
                 color=colors, alpha=0.7)
    ax.set_xticks(range(len(top_results)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Heading RMSE [rad]', fontweight='bold')
    ax.set_title(f'Top {top_n} Configurations - Heading RMSE')
    ax.grid(True, alpha=0.3, axis='y')

    bars[0].set_edgecolor('darkgreen')
    bars[0].set_linewidth(2.5)

    # 3. Alpha vs Performance
    ax = axes[1, 0]
    alpha_groups = {}
    for r in successful_results:
        alpha = r['alpha']
        if alpha not in alpha_groups:
            alpha_groups[alpha] = []
        alpha_groups[alpha].append(r['pos_rmse'])

    alphas = sorted(alpha_groups.keys())
    mean_rmse = [tf.reduce_mean(alpha_groups[a]).numpy() for a in alphas]
    std_rmse = [tf.math.reduce_std(alpha_groups[a]).numpy() for a in alphas]

    ax.errorbar(alphas, mean_rmse, yerr=std_rmse, marker='o', capsize=5,
               linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Alpha (α)', fontweight='bold')
    ax.set_ylabel('Mean Position RMSE [m]', fontweight='bold')
    ax.set_title('Effect of Alpha on Performance')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 4. Parameter sensitivity summary
    ax = axes[1, 1]

    # Create text summary
    best = top_results[0]
    summary_text = (
        f"Best Configuration:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"α (alpha)  = {best['alpha']:.4f}\n"
        f"β (beta)   = {best['beta']:.1f}\n"
        f"κ (kappa)  = {best['kappa']:+.0f}\n"
        f"\nPerformance Metrics:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Position RMSE:  {best['pos_rmse']:.4f} m\n"
        f"Position Final: {best['pos_final_error']:.4f} m\n"
        f"Heading RMSE:   {best['heading_rmse']:.4f} rad\n"
        f"Heading Final:  {best['heading_final_error']:.4f} rad\n"
        f"\nTop 3 Configurations:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    for i, r in enumerate(top_results[:3], 1):
        summary_text += (f"\n{i}. α={r['alpha']:.3f}, β={r['beta']:.1f}, "
                        f"κ={r['kappa']:+.0f}")
        summary_text += f"\n   RMSE: {r['pos_rmse']:.4f} m"

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.axis('off')

    plt.tight_layout()
    fig_path = os.path.join(output_dir,
                           f"grid_search_{grid_results['scenario']}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {fig_path}")
    plt.close()

    # Save grid search results to CSV
    csv_path = os.path.join(output_dir,
                           f"grid_search_results_{grid_results['scenario']}.csv")
    with open(csv_path, 'w') as f:
        f.write("rank,alpha,beta,kappa,pos_rmse,pos_final_error,heading_rmse,heading_final_error\n")
        for i, r in enumerate(top_results, 1):
            f.write(f"{i},{r['alpha']:.6f},{r['beta']:.1f},{r['kappa']:.0f},"
                   f"{r['pos_rmse']:.6f},{r['pos_final_error']:.6f},"
                   f"{r['heading_rmse']:.6f},{r['heading_final_error']:.6f}\n")
    print(f"✓ Saved grid search results: {csv_path}")

    # Print detailed table
    print("\n" + "="*80)
    print(f"{'Rank':<6} {'Alpha':<8} {'Beta':<6} {'Kappa':<6} "
         f"{'Pos RMSE':<12} {'Head RMSE':<12}")
    print("="*80)
    for i, r in enumerate(top_results, 1):
        print(f"{i:<6} {r['alpha']:<8.4f} {r['beta']:<6.1f} "
             f"{r['kappa']:<+6.0f} {r['pos_rmse']:<12.4f} "
             f"{r['heading_rmse']:<12.4f}")
    print("="*80)


def run_experiment(scenario: str = 'moderate', num_steps: int = 100,
                  alpha: float = 1, beta: float = 1.5, kappa: float = -1.0,
                  output_dir: Path = EXPERIMENT_DIR,
                  seed: int | None = None) -> dict:
    """
    Run range-bearing localization experiment.

    Parameters
    ----------
    scenario : str
        'moderate' or 'strong' nonlinearity scenario.
    num_steps : int
        Number of simulation steps.
    alpha : float
        UKF alpha parameter.
    beta : float
        UKF beta parameter.
    kappa : float
        UKF kappa parameter.
    output_dir : str
        Directory to save results.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Experiment results dictionary.
    """
    print("=" * 70)
    print("RANGE-BEARING LOCALIZATION: EKF vs UKF")
    print("=" * 70)

    results = simulate_localization_with_params(
        scenario=scenario,
        num_steps=num_steps,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        seed=seed
    )

    plot_results(results, output_dir)

    return results


def run_parameter_tuning(scenario: str = "strong", num_steps: int = 100,
                        alpha_values: list[float] | None = None,
                        beta_values: list[float] | None = None,
                        kappa_values: list[float] | None = None,
                        output_dir: Path = TUNING_DIR) -> dict:
    """
    Run UKF parameter tuning via grid search.

    Parameters
    ----------
    scenario : str
        'moderate' or 'strong' nonlinearity scenario.
    num_steps : int
        Number of simulation steps.
    alpha_values : list[float], optional
        List of alpha values to test.
    beta_values : list[float], optional
        List of beta values to test.
    kappa_values : list[float], optional
        List of kappa values to test.
    output_dir : str
        Directory to save results.

    Returns
    -------
    grid_results : dict
        Grid search results dictionary.
    """
    print("\n" + "="*80)
    print("UKF PARAMETER TUNING - GRID SEARCH")
    print("="*80)

    if alpha_values is None:
        alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    if beta_values is None:
        beta_values = [1, 1.5, 2.0]
    if kappa_values is None:
        kappa_values = [-1, 0, 1]

    # Run grid search
    print(f"\nScenario: {scenario} nonlinearity")
    grid_results = grid_search_ukf_parameters(
        scenario=scenario,
        num_steps=num_steps,
        alpha_values=alpha_values,
        beta_values=beta_values,
        kappa_values=kappa_values,
        verbose=True
    )

    # Visualize results
    plot_grid_search_results(grid_results, output_dir, top_n=10)

    # Run with best parameters
    print("\n" + "="*80)
    print("RUNNING SIMULATION WITH BEST PARAMETERS")
    print("="*80)

    best_alpha, best_beta, best_kappa = grid_results['best_params']
    print(f"Using: α={best_alpha:.4f}, β={best_beta:.1f}, "
         f"κ={best_kappa:+.0f}\n")

    # Run optimized simulation
    results_optimized = simulate_localization_with_params(
        scenario=scenario,
        num_steps=num_steps,
        alpha=best_alpha,
        beta=best_beta,
        kappa=best_kappa,
        seed=42
    )
    plot_results(results_optimized, output_dir,
                 title_add=f" - Optimized (α={best_alpha:.3f})")

    return grid_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Range-bearing localization experiment')
    parser.add_argument('--mode', type=str, default='experiment',
                       choices=['experiment', 'tuning'],
                       help='Run mode: experiment or tuning')
    parser.add_argument('--scenario', type=str, default='moderate',
                       choices=['moderate', 'strong'],
                       help='Nonlinearity scenario')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of simulation steps')
    parser.add_argument('--alpha', type=float, default=0.001,
                       help='UKF alpha parameter')
    parser.add_argument('--beta', type=float, default=2.0,
                       help='UKF beta parameter')
    parser.add_argument('--kappa', type=float, default=0.0,
                       help='UKF kappa parameter')
    #parser.add_argument('--output_dir', type=str,
                       #default='reports/range_bearing',
                       #help='Output directory for results')
    args = parser.parse_args()

    if args.mode == 'experiment':
        run_experiment(
            scenario=args.scenario,
            num_steps=args.num_steps,
            alpha=args.alpha,
            beta=args.beta,
            kappa=args.kappa,
            output_dir=EXPERIMENT_DIR
        )
    else:
        run_parameter_tuning(
            scenario=args.scenario,
            num_steps=args.num_steps,
            output_dir=TUNING_DIR
        )

#python3 -m src.experiments.exp_range_bearing_ekf_ukf --mode tuning --scenario strong