"""
Experiment: EKF/UKF Failure Mode Analysis.

This experiment analyzes failure modes of Extended Kalman Filter (EKF) and
Unscented Kalman Filter (UKF) for range-bearing localization, including:
- EKF linearization error analysis
- UKF sigma point collapse detection
- Filter consistency via NEES
- Visualization of linearization breakdown and sigma point behavior
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ssm_range_bearing import RangeBearingSSM
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.metrics.accuracy import compute_nees

tfd = tfp.distributions

# Output directory
OUTPUT_DIR = Path("reports/range_bearing/linearization_sigma_pt_failures")


def analyze_ekf_linearization_error(ssm: RangeBearingSSM,
                                     state: tf.Tensor,
                                     landmarks: tf.Tensor,
                                     uncertainty_scales: list[float] | None = None
                                     ) -> list[dict]:
    """
    Analyze EKF linearization accuracy at different uncertainty levels.

    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    state : tf.Tensor
        State vector of shape (3,).
    landmarks : tf.Tensor
        Landmark positions of shape (num_landmarks, 2).
    uncertainty_scales : list[float], optional
        List of uncertainty scales to test. Defaults to [0.1, 0.5, 1.0, 2.0].

    Returns
    -------
    results : list[dict]
        List of dictionaries containing analysis results for each scale.
    """
    if uncertainty_scales is None:
        uncertainty_scales = [0.1, 0.5, 1.0, 2.0]

    results = []

    for scale in uncertainty_scales:
        cov = tf.eye(3, dtype=tf.float32) * float(scale)

        # True nonlinear propagation via Monte Carlo
        scale_tf = tf.cast(tf.sqrt(scale), tf.float32)
        samples = tf.random.normal([500, 3], mean=state, stddev=scale_tf)
        true_meas = ssm.measurement_model(samples, landmarks)
        true_meas_flat = tf.reshape(true_meas, [500, -1])
        true_cov = tfp.stats.covariance(true_meas_flat)

        # EKF's linearized approximation
        pred_meas = ssm.measurement_model(state[tf.newaxis, :], landmarks)[0]
        H = ssm.measurement_jacobian(state[tf.newaxis, :], landmarks)[0]
        R = ssm.full_measurement_cov(tf.shape(landmarks)[0])
        linear_cov = H @ cov @ tf.transpose(H) + R

        # Diagonal variance error (most important for filtering)
        true_vars = tf.linalg.diag_part(true_cov)
        linear_vars = tf.linalg.diag_part(linear_cov)
        var_error = tf.reduce_mean(tf.abs(true_vars - linear_vars) /
                                   (true_vars + 1e-6))

        results.append({
            'scale': scale,
            'var_error': float(var_error),
            'true_std': true_vars.numpy(),
            'linear_std': linear_vars.numpy()
        })

    return results


def diagnose_sigma_point_collapse(ukf: UnscentedKalmanFilter,
                                   landmarks: tf.Tensor) -> dict:
    """
    Detect sigma point collapse by checking spread and variance.

    Parameters
    ----------
    ukf : UnscentedKalmanFilter
        UKF instance.
    landmarks : tf.Tensor
        Landmark positions of shape (num_landmarks, 2).

    Returns
    -------
    collapse_info : dict
        Dictionary containing sigma point analysis results.
    """
    sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)

    sigma_spread = tf.math.reduce_std(sigma_points, axis=0)
    meas_sigma = ukf.ssm.measurement_model(sigma_points, landmarks)
    meas_flat = tf.reshape(meas_sigma, [tf.shape(sigma_points)[0], -1])

    meas_std = tf.math.reduce_std(meas_flat, axis=0)
    _, S = ukf.unscented_transform(meas_flat,
                                   ukf.ssm.full_measurement_cov(
                                       tf.shape(landmarks)[0]))
    pred_std = tf.sqrt(tf.linalg.diag_part(S))

    collapse_ratio = meas_std / (pred_std + 1e-10)

    return {
        'sigma_points': sigma_points.numpy(),
        'meas_sigma_points': meas_flat.numpy(),
        'collapse_ratio': collapse_ratio.numpy(),
        'pred_std': pred_std.numpy(),
        'empirical_std': meas_std.numpy()
    }


def run_trajectory_nees(ssm: RangeBearingSSM,
                        landmarks: tf.Tensor,
                        initial_state: tf.Tensor,
                        initial_cov: tf.Tensor,
                        n_steps: int = 10) -> dict:
    """
    Run a short trajectory and compute NEES to check filter consistency.

    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    landmarks : tf.Tensor
        Landmark positions of shape (num_landmarks, 2).
    initial_state : tf.Tensor
        Initial state of shape (3,).
    initial_cov : tf.Tensor
        Initial covariance of shape (3, 3).
    n_steps : int
        Number of time steps.

    Returns
    -------
    results : dict
        Dictionary containing NEES results and interpretation.
    """
    true_state = initial_state
    control = tf.constant([0.5, 0.1], dtype=tf.float32)

    ekf = ExtendedKalmanFilter(ssm, initial_state, initial_cov)
    ukf = UnscentedKalmanFilter(ssm, initial_state, initial_cov,
                                 alpha=0.3, beta=2.0, kappa=0.0)

    ekf_states = []
    ukf_states = []
    ekf_covs = []
    ukf_covs = []
    true_states = []

    for t in range(n_steps):
        true_state = ssm.motion_model(true_state[tf.newaxis, :],
                                       control[tf.newaxis, :])[0]
        true_meas = ssm.measurement_model(true_state[tf.newaxis, :],
                                          landmarks)[0]
        noisy_meas = true_meas + tf.random.normal(tf.shape(true_meas),
                                                  stddev=0.1)

        ekf.predict(control)
        ekf.update(noisy_meas, landmarks)
        ekf_states.append(tf.identity(ekf.state))
        ekf_covs.append(tf.identity(ekf.covariance))

        ukf.predict(control)
        ukf.update(noisy_meas, landmarks)
        ukf_states.append(tf.identity(ukf.state))
        ukf_covs.append(tf.identity(ukf.covariance))

        true_states.append(true_state)

    # Stack for NEES computation
    ekf_states_stack = tf.stack(ekf_states)
    ukf_states_stack = tf.stack(ukf_states)
    ekf_covs_stack = tf.stack(ekf_covs)
    ukf_covs_stack = tf.stack(ukf_covs)
    true_states_stack = tf.stack(true_states)

    # Compute NEES
    ekf_nees = compute_nees(ekf_states_stack, ekf_covs_stack,
                           true_states_stack)
    ukf_nees = compute_nees(ukf_states_stack, ukf_covs_stack,
                           true_states_stack)

    avg_nees_ekf = float(tf.reduce_mean(ekf_nees))
    avg_nees_ukf = float(tf.reduce_mean(ukf_nees))

    # Chi-squared bounds for 3D state (95% CI)
    chi2_dist = tfd.Chi2(df=3.0)
    nees_lower = float(chi2_dist.quantile(0.025))
    nees_upper = float(chi2_dist.quantile(0.975))

    return {
        'ekf_nees': ekf_nees.numpy(),
        'ukf_nees': ukf_nees.numpy(),
        'avg_nees_ekf': avg_nees_ekf,
        'avg_nees_ukf': avg_nees_ukf,
        'nees_lower': nees_lower,
        'nees_upper': nees_upper,
        'n_steps': n_steps
    }


def run_failure_analysis() -> tuple:
    """
    Run text-based analysis of EKF and UKF failure modes.

    Returns
    -------
    ssm : RangeBearingSSM
        State-space model instance.
    initial_state : tf.Tensor
        Initial state.
    initial_cov : tf.Tensor
        Initial covariance.
    scenarios : list
        List of scenario tuples (name, landmarks).
    analysis_results : dict
        Dictionary containing all analysis results.
    """
    ssm = RangeBearingSSM(
        dt=0.1,
        process_noise=tf.eye(3, dtype=tf.float32) * 0.01,
        meas_noise=tf.eye(2, dtype=tf.float32) * 0.1
    )

    scenarios = [
        ("Moderate Nonlinearity (Distant Landmark)",
         tf.constant([[10.0, 10.0]], dtype=tf.float32)),
        ("Strong Nonlinearity (Close Landmark)",
         tf.constant([[0.5, 0.2]], dtype=tf.float32))
    ]

    initial_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    initial_cov = tf.eye(3, dtype=tf.float32) * 1.0

    analysis_results = {}

    for name, landmarks in scenarios:
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"Landmark position: {landmarks.numpy()}")
        print('='*70)

        # 1. EKF Linearization Analysis
        print("\n[1] EKF LINEARIZATION ERROR ANALYSIS")
        ekf_errors = analyze_ekf_linearization_error(ssm, initial_state,
                                                      landmarks)

        print("\n  Summary (variance error: higher = worse linearization):")
        for res in ekf_errors:
            print(f"    σ={res['scale']:.1f}: Var Error={res['var_error']:.3f}")

        # 2. UKF Sigma Point Collapse Analysis
        print("\n[2] UKF SIGMA POINT COLLAPSE ANALYSIS")
        ukf = UnscentedKalmanFilter(ssm, initial_state, initial_cov,
                                     alpha=0.3, beta=2.0, kappa=0.0)
        collapse_info = diagnose_sigma_point_collapse(ukf, landmarks)

        collapse_ratio = collapse_info['collapse_ratio']
        print(f"\n  Collapse Ratios: {collapse_ratio}")
        print(f"  Interpretation:")
        if tf.reduce_any(tf.less(collapse_ratio, 0.1)):
            print(f"    ⚠️ SIGMA POINT COLLAPSE DETECTED (ratio < 0.1)!")
        elif tf.reduce_any(tf.less(collapse_ratio, 0.3)):
            print(f"    ⚠️ Partial collapse (ratio < 0.3) - marginal spread")
        else:
            print(f"    ✓ Good sigma point spread (ratio > 0.3)")

        # 3. Run a short trajectory for NEES
        print("\n[3] FILTER CONSISTENCY (NEES)")
        nees_results = run_trajectory_nees(ssm, landmarks, initial_state,
                                          initial_cov)

        print(f"  Average NEES over {nees_results['n_steps']} steps:")
        print(f"    EKF: {nees_results['avg_nees_ekf']:.2f} "
              f"(expected ~3.0 for 3D state)")
        print(f"    UKF: {nees_results['avg_nees_ukf']:.2f}")
        print(f"  Interpretation (95% CI: "
              f"[{nees_results['nees_lower']:.2f}, "
              f"{nees_results['nees_upper']:.2f}]):")

        if nees_results['avg_nees_ekf'] > nees_results['nees_upper']:
            print(f"    EKF: ⚠️ Filter diverged or inconsistent")
        elif nees_results['avg_nees_ekf'] < nees_results['nees_lower']:
            print(f"    EKF: ⚠️ Overly conservative (underconfident)")
        else:
            print(f"    EKF: ✓ Consistent")

        if nees_results['avg_nees_ukf'] > nees_results['nees_upper']:
            print(f"    UKF: ⚠️ Filter diverged or inconsistent")
        elif nees_results['avg_nees_ukf'] < nees_results['nees_lower']:
            print(f"    UKF: ⚠️ Overly conservative")
        else:
            print(f"    UKF: ✓ Consistent")

        analysis_results[name] = {
            'ekf_errors': ekf_errors,
            'collapse_info': collapse_info,
            'nees_results': nees_results,
            'landmarks': landmarks
        }

    return ssm, initial_state, initial_cov, scenarios, analysis_results


def plot_ekf_linearization_curve(ssm: RangeBearingSSM,
                                   state: tf.Tensor,
                                   landmarks_moderate: tf.Tensor,
                                   landmarks_strong: tf.Tensor,
                                   output_dir: Path) -> None:
    """
    Plot EKF linearization error vs uncertainty for both scenarios.

    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    state : tf.Tensor
        State vector.
    landmarks_moderate : tf.Tensor
        Moderate scenario landmarks.
    landmarks_strong : tf.Tensor
        Strong scenario landmarks.
    output_dir : Path
        Output directory for saving plots.
    """
    uncertainty_scales = tf.linspace(0.1, 3.0, 15).numpy()

    errors_moderate = []
    errors_strong = []

    print("Computing linearization errors for curve...")
    for scale in uncertainty_scales:
        cov = tf.eye(3, dtype=tf.float32) * float(scale)

        # Moderate case
        scale_tf = tf.cast(tf.sqrt(scale), tf.float32)
        samples = tf.random.normal([500, 3], mean=state, stddev=scale_tf)
        true_meas = ssm.measurement_model(samples, landmarks_moderate)
        true_meas_flat = tf.reshape(true_meas, [500, -1])
        true_cov = tfp.stats.covariance(true_meas_flat)

        H = ssm.measurement_jacobian(state[tf.newaxis, :],
                                     landmarks_moderate)[0]
        R = ssm.full_measurement_cov(tf.shape(landmarks_moderate)[0])
        linear_cov = H @ cov @ tf.transpose(H) + R

        true_vars = tf.linalg.diag_part(true_cov)
        linear_vars = tf.linalg.diag_part(linear_cov)
        var_error = tf.reduce_mean(tf.abs(true_vars - linear_vars) /
                                   (true_vars + 1e-6))
        errors_moderate.append(float(var_error))

        # Strong case
        samples = tf.random.normal([500, 3], mean=state, stddev=scale_tf)
        true_meas = ssm.measurement_model(samples, landmarks_strong)
        true_meas_flat = tf.reshape(true_meas, [500, -1])
        true_cov = tfp.stats.covariance(true_meas_flat)

        H = ssm.measurement_jacobian(state[tf.newaxis, :],
                                    landmarks_strong)[0]
        R = ssm.full_measurement_cov(tf.shape(landmarks_strong)[0])
        linear_cov = H @ cov @ tf.transpose(H) + R

        true_vars = tf.linalg.diag_part(true_cov)
        linear_vars = tf.linalg.diag_part(linear_cov)
        var_error = tf.reduce_mean(tf.abs(true_vars - linear_vars) /
                                   (true_vars + 1e-6))
        errors_strong.append(float(var_error))

    plt.figure(figsize=(10, 6))
    plt.plot(uncertainty_scales, errors_moderate, 'b-o', linewidth=2,
             label='Moderate (Distant Landmark)', markersize=6)
    plt.plot(uncertainty_scales, errors_strong, 'r-s', linewidth=2,
             label='Strong (Close Landmark)', markersize=6)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5,
                label='Critical threshold')
    plt.xlabel('State Uncertainty (σ)', fontsize=12)
    plt.ylabel('Variance Error (higher = worse)', fontsize=12)
    plt.title('EKF Linearization Accuracy Breakdown', fontsize=14,
              fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add annotation
    max_idx = len(errors_strong) - 3
    if errors_strong[max_idx] > 0.5:
        plt.annotate('EKF linearization fails\n(>50% error)',
                    xy=(uncertainty_scales[max_idx],
                        errors_strong[max_idx]),
                    xytext=(1.5, max(errors_strong) * 0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    fig_path = output_dir / 'ekf_linearization_curve.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()

    # Save data to CSV
    csv_path = output_dir / 'ekf_linearization_curve.csv'
    with open(csv_path, 'w') as f:
        f.write("uncertainty_scale,error_moderate,error_strong\n")
        for i, scale in enumerate(uncertainty_scales):
            f.write(f"{scale:.6f},{errors_moderate[i]:.6f},"
                   f"{errors_strong[i]:.6f}\n")
    print(f"✓ Saved: {csv_path}")


def plot_simple_sigma_point_visualization(ukf: UnscentedKalmanFilter,
                                          landmarks: tf.Tensor,
                                          true_meas: tf.Tensor,
                                          scenario_name: str,
                                          output_dir: Path) -> None:
    """
    Simple 2D plot: robot, landmark, sigma points, and predicted measurements.

    Parameters
    ----------
    ukf : UnscentedKalmanFilter
        UKF instance.
    landmarks : tf.Tensor
        Landmark positions.
    true_meas : tf.Tensor
        True measurement.
    scenario_name : str
        Scenario name for title and filename.
    output_dir : Path
        Output directory for saving plots.
    """
    sigma_points = ukf.generate_sigma_points(ukf.state, ukf.covariance)
    sigma_np = sigma_points.numpy()

    meas_sigma = ukf.ssm.measurement_model(sigma_points, landmarks)
    meas_flat = meas_sigma.numpy().reshape(meas_sigma.shape[0], -1)

    robot = ukf.state.numpy()[:2]
    landmark = landmarks.numpy()[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 1. Robot and landmark
    ax.plot(robot[0], robot[1], 'bo', markersize=12, label='Robot estimate')
    ax.plot(landmark[0], landmark[1], 'r*', markersize=16, label='Landmark')

    # 2. Sigma points
    ax.plot(sigma_np[:, 0], sigma_np[:, 1], 'bs', markersize=8, alpha=0.7,
           label='Sigma points')

    # 3. Arrows to predicted measurements (scaled)
    scale = 0.5
    for i in range(len(sigma_np)):
        start = sigma_np[i, :2]
        # Use tf for angle calculations
        angle_shift = tf.constant(3.14159, dtype=tf.float32)  # pi
        end_x = start[0] + scale * meas_flat[i, 0]  # range
        end_y = start[1] + scale * (meas_flat[i, 1] + float(angle_shift))  # bearing
        ax.arrow(start[0], start[1], end_x - start[0], end_y - start[1],
                 head_width=0.05, head_length=0.05, fc='orange', ec='orange',
                 alpha=0.6)

    # 4. Predicted measurements (shifted)
    pred_x = sigma_np[:, 0] + scale * meas_flat[:, 0]
    pred_y = sigma_np[:, 1] + scale * (meas_flat[:, 1] + float(angle_shift))
    ax.plot(pred_x, pred_y, 'o', color='orange', markersize=6, alpha=0.7,
           label='Predicted measurements')

    # 5. True measurement (shifted)
    true_x = robot[0] + scale * true_meas[0]
    true_y = robot[1] + scale * (true_meas[1] + float(angle_shift))
    ax.plot(true_x, true_y, 'rx', markersize=12, linewidth=3,
           label='True measurement')

    # 6. Add collapse info
    collapse_info = diagnose_sigma_point_collapse(ukf, landmarks)
    collapse_ratio = collapse_info['collapse_ratio']
    sigma_spread = tf.math.reduce_std(sigma_points, axis=0).numpy()[:2]

    ax.text(0.02, 0.98, f'Sigma spread: [{sigma_spread[0]:.3f}, '
           f'{sigma_spread[1]:.3f}]',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.text(0.02, 0.90, f'Collapse ratio: {collapse_ratio}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Sigma Point Collapse: {scenario_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    filename = f'simple_sigma_point_{scenario_name.lower().replace(" ", "_")}.png'
    fig_path = output_dir / filename
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()

    # Save sigma point data to CSV
    csv_path = output_dir / f'sigma_points_{scenario_name.lower().replace(" ", "_")}.csv'
    with open(csv_path, 'w') as f:
        f.write("sigma_point_idx,x,y,theta,meas_range,meas_bearing\n")
        for i in range(len(sigma_np)):
            f.write(f"{i},{sigma_np[i,0]:.6f},{sigma_np[i,1]:.6f},"
                   f"{sigma_np[i,2]:.6f},{meas_flat[i,0]:.6f},"
                   f"{meas_flat[i,1]:.6f}\n")
    print(f"✓ Saved: {csv_path}")


def save_analysis_results(analysis_results: dict, output_dir: Path) -> None:
    """
    Save analysis results to CSV files.

    Parameters
    ----------
    analysis_results : dict
        Dictionary containing analysis results for each scenario.
    output_dir : Path
        Output directory for saving CSV files.
    """
    for scenario_name, results in analysis_results.items():
        # Save EKF linearization errors
        csv_path = output_dir / f'ekf_linearization_{scenario_name.lower().replace(" ", "_")}.csv'
        with open(csv_path, 'w') as f:
            f.write("uncertainty_scale,variance_error,true_std_range,true_std_bearing,"
                   "linear_std_range,linear_std_bearing\n")
            for err in results['ekf_errors']:
                f.write(f"{err['scale']:.6f},{err['var_error']:.6f},"
                       f"{err['true_std'][0]:.6f},{err['true_std'][1]:.6f},"
                       f"{err['linear_std'][0]:.6f},{err['linear_std'][1]:.6f}\n")
        print(f"✓ Saved: {csv_path}")

        # Save sigma point collapse info
        csv_path = output_dir / f'sigma_collapse_{scenario_name.lower().replace(" ", "_")}.csv'
        collapse_info = results['collapse_info']
        with open(csv_path, 'w') as f:
            f.write("collapse_ratio_range,collapse_ratio_bearing,"
                   "pred_std_range,pred_std_bearing,empirical_std_range,"
                   "empirical_std_bearing\n")
            f.write(f"{collapse_info['collapse_ratio'][0]:.6f},"
                   f"{collapse_info['collapse_ratio'][1]:.6f},"
                   f"{collapse_info['pred_std'][0]:.6f},"
                   f"{collapse_info['pred_std'][1]:.6f},"
                   f"{collapse_info['empirical_std'][0]:.6f},"
                   f"{collapse_info['empirical_std'][1]:.6f}\n")
        print(f"✓ Saved: {csv_path}")

        # Save NEES results
        csv_path = output_dir / f'nees_{scenario_name.lower().replace(" ", "_")}.csv'
        nees_results = results['nees_results']
        with open(csv_path, 'w') as f:
            f.write("step,ekf_nees,ukf_nees\n")
            for i in range(nees_results['n_steps']):
                f.write(f"{i},{nees_results['ekf_nees'][i]:.6f},"
                       f"{nees_results['ukf_nees'][i]:.6f}\n")
            f.write(f"\nsummary\n")
            f.write(f"avg_nees_ekf,{nees_results['avg_nees_ekf']:.6f}\n")
            f.write(f"avg_nees_ukf,{nees_results['avg_nees_ukf']:.6f}\n")
            f.write(f"nees_lower,{nees_results['nees_lower']:.6f}\n")
            f.write(f"nees_upper,{nees_results['nees_upper']:.6f}\n")
        print(f"✓ Saved: {csv_path}")


def run_final_analysis() -> None:
    """
    Run complete failure mode analysis: text + plots + CSV outputs.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("RUNNING TEXT ANALYSIS")
    print("="*70)

    # Run text analysis
    ssm, initial_state, initial_cov, scenarios, analysis_results = (
        run_failure_analysis())

    # Save analysis results to CSV
    print("\n" + "="*70)
    print("SAVING ANALYSIS RESULTS TO CSV")
    print("="*70)
    save_analysis_results(analysis_results, OUTPUT_DIR)

    print("\n" + "="*70)
    print("GENERATING EKF LINEARIZATION CURVE")
    print("="*70)

    # Plot EKF linearization curve
    landmarks_moderate = tf.constant([[10.0, 10.0]], dtype=tf.float32)
    landmarks_strong = tf.constant([[0.5, 0.2]], dtype=tf.float32)
    plot_ekf_linearization_curve(ssm, initial_state, landmarks_moderate,
                                 landmarks_strong, OUTPUT_DIR)

    print("\n" + "="*70)
    print("GENERATING SIMPLE SIGMA POINT VISUALIZATIONS")
    print("="*70)

    # For each scenario, generate one clean sigma point plot
    for name, landmarks in scenarios:
        print(f"\nCreating simple sigma point plot for: {name}")

        # Simulate one true measurement
        true_state = initial_state
        control = tf.constant([0.5, 0.1], dtype=tf.float32)
        true_state = ssm.motion_model(true_state[tf.newaxis, :],
                                      control[tf.newaxis, :])[0]
        true_meas = ssm.measurement_model(true_state[tf.newaxis, :],
                                         landmarks)[0][0]

        # Create UKF for this scenario
        ukf = UnscentedKalmanFilter(ssm, initial_state, initial_cov,
                                    alpha=0.3, beta=2.0, kappa=0.0)

        # Generate plot
        plot_simple_sigma_point_visualization(ukf, landmarks, true_meas,
                                              name, OUTPUT_DIR)

    print("\n" + "="*70)
    print("✓ FINAL ANALYSIS COMPLETE!")
    print("  - Text: EKF error, UKF collapse, NEES")
    print("  - Plot: EKF linearization curve")
    print("  - Plots: Simple sigma point collapse (no trajectories)")
    print(f"  - All results saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    run_final_analysis()

