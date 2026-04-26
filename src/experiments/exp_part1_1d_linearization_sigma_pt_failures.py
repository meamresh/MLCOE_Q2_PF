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

# Global seed for deterministic runs of this experiment
tf.random.set_seed(42)

# Output directory
OUTPUT_DIR = Path("reports/2_Nonlinear_NonGaussianSSM/linearization_sigma_pt_failures")


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
            'true_std': true_vars.numpy().tolist(),
            'linear_std': linear_vars.numpy().tolist()
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
        'sigma_points': sigma_points.numpy().tolist(),
        'meas_sigma_points': meas_flat.numpy().tolist(),
        'collapse_ratio': collapse_ratio.numpy().tolist(),
        'pred_std': pred_std.numpy().tolist(),
        'empirical_std': meas_std.numpy().tolist()
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
        'ekf_nees': ekf_nees.numpy().tolist(),
        'ukf_nees': ukf_nees.numpy().tolist(),
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


def plot_linearization_breakdown_2d(ssm: RangeBearingSSM,
                                      state: tf.Tensor,
                                      landmark: tf.Tensor,
                                      scenario_name: str,
                                      output_dir: Path) -> None:
    """
    Visualize how the EKF's linear approximation breaks down.

    Shows:
    - True nonlinear measurement function (Monte Carlo samples)
    - EKF's linearized approximation (tangent plane)
    - Covariance ellipses: true vs linearized

    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    state : tf.Tensor
        Nominal state [x, y, theta].
    landmark : tf.Tensor
        Landmark position [1, 2].
    scenario_name : str
        Name for labeling.
    output_dir : Path
        Output directory.
    """
    # Sample states around the nominal
    n_samples = 1000
    uncertainty = 1.0  # std dev for each state dimension
    samples = tf.random.normal([n_samples, 3], dtype=tf.float32) * uncertainty + state

    # True nonlinear measurements
    true_meas = ssm.measurement_model(samples, landmark)
    true_meas_flat = tf.reshape(true_meas, [n_samples, -1])

    # Nominal measurement and Jacobian
    nom_meas = ssm.measurement_model(state[tf.newaxis, :], landmark)[0]
    nom_meas_flat = tf.reshape(nom_meas, [-1])
    H = ssm.measurement_jacobian(state[tf.newaxis, :], landmark)[0]

    # Linearized measurements: z_lin = h(x0) + H @ (x - x0)
    delta_x = samples - state[tf.newaxis, :]  # [N, 3]
    linear_meas = nom_meas_flat[tf.newaxis, :] + tf.matmul(
        delta_x, H, transpose_b=True
    )  # [N, 2]

    # Compute covariances
    true_cov = tfp.stats.covariance(true_meas_flat)
    linear_cov = tfp.stats.covariance(linear_meas)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left plot: Scatter comparison ---
    ax = axes[0]
    true_flat_np = true_meas_flat.numpy()
    linear_meas_np = linear_meas.numpy()
    nom_flat_np = nom_meas_flat.numpy()
    ax.scatter(true_flat_np[:, 0], true_flat_np[:, 1], alpha=0.3, s=10,
               c='blue', label='True nonlinear')
    ax.scatter(linear_meas_np[:, 0], linear_meas_np[:, 1], alpha=0.3, s=10,
               c='red', label='Linearized (EKF)')
    ax.plot(nom_flat_np[0], nom_flat_np[1], 'k*', markersize=15,
            label='Nominal', zorder=5)

    # Add covariance ellipses
    from matplotlib.patches import Ellipse

    _pi_180 = 180.0 / 3.141592653589793

    def draw_ellipse(ax, mean, cov, color, label, linestyle='-'):
        cov = tf.cast(cov, tf.float32)
        eigvals, eigvecs = tf.linalg.eigh(cov)
        angle = float(tf.math.atan2(eigvecs[1, 0], eigvecs[0, 0]).numpy()) * _pi_180
        widths = 2 * 2.0 * tf.sqrt(eigvals)
        width, height = float(widths[0].numpy()), float(widths[1].numpy())
        mean_np = mean.numpy() if hasattr(mean, 'numpy') else mean
        ellipse = Ellipse(mean_np, width, height, angle=angle, fill=False,
                          edgecolor=color, linewidth=2, linestyle=linestyle,
                          label=label)
        ax.add_patch(ellipse)

    true_mean = tf.reduce_mean(true_meas_flat, axis=0)
    linear_mean = tf.reduce_mean(linear_meas, axis=0)
    draw_ellipse(ax, true_mean, true_cov, 'blue', 'True 2σ ellipse')
    draw_ellipse(ax, linear_mean, linear_cov, 'red', 'Linearized 2σ ellipse', '--')

    ax.set_xlabel('Range', fontsize=12)
    ax.set_ylabel('Bearing (rad)', fontsize=12)
    ax.set_title(f'Measurement Space: {scenario_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Right plot: Covariance magnitude comparison ---
    ax2 = axes[1]
    labels = ['Range Var', 'Bearing Var', 'Covariance']
    true_vals = [
        float(true_cov[0, 0].numpy()),
        float(true_cov[1, 1].numpy()),
        float(true_cov[0, 1].numpy()),
    ]
    lin_vals = [
        float(linear_cov[0, 0].numpy()),
        float(linear_cov[1, 1].numpy()),
        float(linear_cov[0, 1].numpy()),
    ]

    x_pos = list(range(len(labels)))
    width = 0.35
    ax2.bar([p - width/2 for p in x_pos], true_vals, width, label='True (MC)', color='blue', alpha=0.7)
    ax2.bar([p + width/2 for p in x_pos], lin_vals, width, label='Linearized', color='red', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Variance / Covariance', fontsize=12)
    ax2.set_title('Covariance Component Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage error annotations
    for i, (tv, lv) in enumerate(zip(true_vals, lin_vals)):
        if abs(tv) > 1e-6:
            pct_err = abs(lv - tv) / abs(tv) * 100
            ax2.annotate(f'{pct_err:.1f}% err', xy=(i, max(tv, lv) * 1.05),
                         ha='center', fontsize=9, color='darkred')

    plt.tight_layout()
    fname = f'linearization_breakdown_{scenario_name.lower().replace(" ", "_")}.png'
    fig_path = output_dir / fname
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()


def plot_sigma_point_propagation(ssm: RangeBearingSSM,
                                   ukf: UnscentedKalmanFilter,
                                   landmark: tf.Tensor,
                                   scenario_name: str,
                                   output_dir: Path) -> None:
    """
    Visualize sigma point propagation from state space to measurement space.

    Shows:
    - Sigma points in state space (x, y)
    - Propagated sigma points in measurement space (range, bearing)
    - UKF weighted mean (Σ w_i·z_i) vs arithmetic mean of 7 sigma pts vs MC mean
    - Collapse detection

    Note: The UKF mean in measurement space is the *weighted* mean with weights
    w_i (ukf.wm). For alpha=0.3, n=3, λ = α²(n+κ)-n < 0 so the center sigma
    point has negative weight; the weighted mean can then lie *outside* the
    convex hull of the 7 points. It is correct; the arithmetic mean (1/7)Σz_i
    is shown for comparison.

    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    ukf : UnscentedKalmanFilter
        UKF instance.
    landmark : tf.Tensor
        Landmark [1, 2].
    scenario_name : str
        Scenario label.
    output_dir : Path
        Output directory.
    """
    # Generate sigma points
    sigma_pts = ukf.generate_sigma_points(ukf.state, ukf.covariance)
    n_sigma = int(tf.shape(sigma_pts)[0])

    # Propagate through measurement model
    meas_sigma = ssm.measurement_model(sigma_pts, landmark)
    meas_sigma_flat = tf.reshape(meas_sigma, [n_sigma, -1])

    # UKF weighted mean (correct: Σ w_i * z_i; center weight can be negative when λ<0)
    ukf_meas_mean = tf.reduce_sum(
        ukf.wm[:, tf.newaxis] * meas_sigma_flat, axis=0
    )
    # Arithmetic mean of sigma point measurements (for comparison; not used by UKF)
    sigma_pts_arithmetic_mean = tf.reduce_mean(meas_sigma_flat, axis=0)

    # Monte Carlo ground truth
    n_mc = 2000
    L = tf.linalg.cholesky(
        ukf.covariance + 1e-6 * tf.eye(3, dtype=tf.float32)
    )
    mc_noise = tf.random.normal([n_mc, 3], dtype=tf.float32)
    mc_samples = ukf.state[tf.newaxis, :] + tf.matmul(
        mc_noise, L, transpose_b=True
    )
    mc_meas = ssm.measurement_model(mc_samples, landmark)
    mc_meas_flat = tf.reshape(mc_meas, [n_mc, -1])
    mc_mean = tf.reduce_mean(mc_meas_flat, axis=0)

    sigma_pts_np = sigma_pts.numpy()
    meas_sigma_flat_np = meas_sigma_flat.numpy()
    mc_samples_np = mc_samples.numpy()
    mc_meas_flat_np = mc_meas_flat.numpy()
    state_np = ukf.state.numpy()
    ukf_meas_mean_np = ukf_meas_mean.numpy()
    sigma_arithmetic_mean_np = sigma_pts_arithmetic_mean.numpy()
    mc_mean_np = mc_mean.numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Subplot 1: State space (x, y) ---
    ax1 = axes[0]
    ax1.scatter(mc_samples_np[:, 0], mc_samples_np[:, 1], alpha=0.1, s=5, c='gray',
                label='MC samples')
    ax1.scatter(sigma_pts_np[:, 0], sigma_pts_np[:, 1], s=100, c='blue', marker='o',
                edgecolors='black', label='Sigma points', zorder=5)
    ax1.plot(state_np[0], state_np[1], 'k*', markersize=15, label='Mean', zorder=6)
    lm0 = landmark.numpy()[0]
    ax1.plot(lm0[0], lm0[1], 'r^', markersize=12, label='Landmark')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('State Space', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # --- Subplot 2: Measurement space ---
    ax2 = axes[1]
    ax2.scatter(mc_meas_flat_np[:, 0], mc_meas_flat_np[:, 1], alpha=0.1, s=5, c='gray',
                label='MC measurements')
    ax2.scatter(meas_sigma_flat_np[:, 0], meas_sigma_flat_np[:, 1], s=100, c='blue',
                marker='o', edgecolors='black', label='Sigma pt meas', zorder=5)
    ax2.plot(ukf_meas_mean_np[0], ukf_meas_mean_np[1], 'b*', markersize=15,
             label='UKF weighted mean (Σ w_i·z_i)', zorder=6)
    ax2.plot(sigma_arithmetic_mean_np[0], sigma_arithmetic_mean_np[1], 'co',
             markersize=10, markeredgecolor='black', label='Arith. mean of 7 sigma pts', zorder=6)
    ax2.plot(mc_mean_np[0], mc_mean_np[1], 'g*', markersize=15, label='MC mean', zorder=6)
    ax2.set_xlabel('Range', fontsize=11)
    ax2.set_ylabel('Bearing (rad)', fontsize=11)
    ax2.set_title('Measurement Space', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Subplot 3: Sigma point spread analysis ---
    ax3 = axes[2]
    # Compute spreads (tf)
    sigma_spread_meas = tf.math.reduce_std(meas_sigma_flat, axis=0)
    mc_spread_meas = tf.math.reduce_std(mc_meas_flat, axis=0)
    collapse_ratio_tf = sigma_spread_meas / (mc_spread_meas + 1e-10)
    sigma_spread_meas_np = sigma_spread_meas.numpy()
    mc_spread_meas_np = mc_spread_meas.numpy()
    collapse_ratio_np = collapse_ratio_tf.numpy()

    bar_width = 0.25
    x = [0, 1]
    ax3.bar([p - bar_width for p in x], sigma_spread_meas_np, bar_width, label='Sigma pt spread',
            color='blue', alpha=0.7)
    ax3.bar(x, mc_spread_meas_np, bar_width, label='True spread (MC)',
            color='gray', alpha=0.7)
    ax3.bar([p + bar_width for p in x], collapse_ratio_np, bar_width, label='Collapse ratio',
            color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Range (m)', 'Bearing (rad)'])
    ax3.set_ylabel('Standard Deviation (m or rad) / Ratio', fontsize=11)
    ax3.set_title('Sigma Point Collapse Analysis', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal ratio')

    # Add collapse warning
    if tf.reduce_any(collapse_ratio_tf < 0.5).numpy():
        ax3.text(0.5, 0.95, '⚠ SIGMA POINT COLLAPSE', transform=ax3.transAxes,
                 fontsize=11, color='red', fontweight='bold', ha='center',
                 va='top', bbox=dict(facecolor='yellow', alpha=0.8))

    plt.suptitle(f'UKF Sigma Point Propagation: {scenario_name}', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f'sigma_pt_propagation_{scenario_name.lower().replace(" ", "_")}.png'
    fig_path = output_dir / fname
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()


def plot_nonlinearity_heatmap(ssm: RangeBearingSSM,
                                landmark: tf.Tensor,
                                output_dir: Path) -> None:
    """
    Heatmap of measurement nonlinearity (Jacobian condition number) across state space.

    High condition number = strong nonlinearity = EKF/UKF likely to fail.

    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    landmark : tf.Tensor
        Landmark [1, 2].
    output_dir : Path
        Output directory.
    """
    # Grid over x, y (fixing theta=0)
    x_range = tf.linspace(-2.0, 12.0, 50)
    y_range = tf.linspace(-2.0, 12.0, 50)
    X, Y = tf.meshgrid(x_range, y_range)
    X_np = X.numpy()
    Y_np = Y.numpy()

    landmark_0 = landmark[0]
    lm0 = float(landmark_0[0].numpy())
    lm1 = float(landmark_0[1].numpy())

    cond_numbers_np = [[0.0] * 50 for _ in range(50)]
    range_to_landmark_np = [[0.0] * 50 for _ in range(50)]

    for i in range(50):
        for j in range(50):
            xi = float(X_np[i, j])
            yj = float(Y_np[i, j])
            state = tf.constant([xi, yj, 0.0], dtype=tf.float32)
            H = ssm.measurement_jacobian(state[tf.newaxis, :], landmark)[0]

            # Condition number of H (higher = more ill-conditioned = nonlinear)
            try:
                s = tf.linalg.svd(H, compute_uv=False)
                cond = tf.reduce_max(s) / tf.reduce_min(s)
                cond_val = min(float(cond.numpy()), 100.0)
            except Exception:
                cond_val = float('inf')
            cond_numbers_np[i][j] = cond_val

            # Distance to landmark
            r_val = tf.sqrt((xi - lm0)**2 + (yj - lm1)**2)
            range_to_landmark_np[i][j] = float(r_val.numpy())

    landmark_np = landmark.numpy()[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Heatmap 1: Jacobian condition number ---
    ax1 = axes[0]
    im1 = ax1.contourf(X_np, Y_np, cond_numbers_np, levels=20, cmap='hot')
    ax1.plot(landmark_np[0], landmark_np[1], 'c*', markersize=15, label='Landmark')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('Jacobian Condition Number\n(Higher = Stronger Nonlinearity)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Condition Number')
    ax1.legend(loc='upper right')
    ax1.axis('equal')

    # --- Heatmap 2: Range to landmark ---
    ax2 = axes[1]
    im2 = ax2.contourf(X_np, Y_np, range_to_landmark_np, levels=20, cmap='viridis')
    ax2.plot(landmark_np[0], landmark_np[1], 'r*', markersize=15, label='Landmark')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('Distance to Landmark\n(Closer = Stronger Nonlinearity)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Range (m)')
    ax2.legend(loc='upper right')
    ax2.axis('equal')

    # Add contour lines showing critical zones
    ax1.contour(X_np, Y_np, cond_numbers_np, levels=[10, 20, 50], colors='white',
                linestyles='--', linewidths=1.5)
    ax2.contour(X_np, Y_np, range_to_landmark_np, levels=[0.5, 1.0, 2.0], colors='white',
                linestyles='--', linewidths=1.5)

    plt.suptitle('Nonlinearity Analysis: Where EKF/UKF Fail', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = output_dir / 'nonlinearity_heatmap.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()


def plot_nees_comparison(analysis_results: dict, output_dir: Path) -> None:
    """
    Plot NEES time series comparing EKF vs UKF for both scenarios.

    Parameters
    ----------
    analysis_results : dict
        Results from run_failure_analysis.
    output_dir : Path
        Output directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (scenario_name, results) in enumerate(analysis_results.items()):
        ax = axes[idx]
        nees = results['nees_results']
        steps = list(range(nees['n_steps']))

        ax.plot(steps, nees['ekf_nees'], 'b-o', linewidth=2, markersize=6,
                label=f"EKF (avg={nees['avg_nees_ekf']:.2f})")
        ax.plot(steps, nees['ukf_nees'], 'r-s', linewidth=2, markersize=6,
                label=f"UKF (avg={nees['avg_nees_ukf']:.2f})")

        # 95% confidence bounds
        ax.axhline(y=nees['nees_lower'], color='green', linestyle='--',
                   alpha=0.7, label=f"95% CI [{nees['nees_lower']:.2f}, {nees['nees_upper']:.2f}]")
        ax.axhline(y=nees['nees_upper'], color='green', linestyle='--', alpha=0.7)
        ax.fill_between(steps, nees['nees_lower'], nees['nees_upper'],
                        color='green', alpha=0.1)

        # Expected value
        ax.axhline(y=3.0, color='gray', linestyle=':', alpha=0.7, label='Expected (n=3)')

        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('NEES', fontsize=11)
        ax.set_title(scenario_name, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(nees['ekf_nees']), max(nees['ukf_nees'])) * 1.2)

    plt.suptitle('Filter Consistency (NEES): EKF vs UKF', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = output_dir / 'nees_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()


def plot_report_strong_nonlinearity(ssm: RangeBearingSSM,
                                    state: tf.Tensor,
                                    initial_cov: tf.Tensor,
                                    landmark_strong: tf.Tensor,
                                    output_dir: Path) -> None:
    """
    Report figure: Enhanced physical view showing sigma points and range-bearing rays.

    Two-panel comparison: Distant (moderate) vs Close (strong nonlinearity).
    Shows robot, landmark, sigma points, and rays to landmark for each sigma point.
    The spread of rays visually shows collapse under strong nonlinearity.
    """
    from matplotlib.patches import Ellipse, Circle
    from matplotlib.lines import Line2D

    _pi_180 = 180.0 / 3.141592653589793

    # Two scenarios
    landmark_distant = tf.constant([[8.0, 6.0]], dtype=tf.float32)
    landmark_close = tf.constant([[0.5, 0.3]], dtype=tf.float32)

    uncertainty = 1.0
    cov = tf.eye(3, dtype=tf.float32) * float(uncertainty ** 2)

    state_np = state.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (lm, title_suffix, scenario_label) in enumerate([
        (landmark_distant, 'Distant Landmark (Moderate Nonlinearity)', 'Moderate'),
        (landmark_close, 'Close Landmark (Strong Nonlinearity)', 'Strong')
    ]):
        ax = axes[ax_idx]
        lm_np = lm.numpy()[0]

        # Create UKF and generate sigma points
        ukf = UnscentedKalmanFilter(ssm, state, cov, alpha=0.3, beta=2.0, kappa=0.0)
        sigma_pts = ukf.generate_sigma_points(ukf.state, ukf.covariance).numpy()

        # Compute measurements for sigma points
        meas_sigma = ssm.measurement_model(
            tf.constant(sigma_pts, dtype=tf.float32), lm)
        n_sig = sigma_pts.shape[0]
        meas_sigma_np = tf.reshape(meas_sigma, [n_sig, -1]).numpy()

        # State uncertainty ellipse (2D projection)
        cov_2d = cov[:2, :2]
        eigvals, eigvecs = tf.linalg.eigh(cov_2d)
        angle_deg = float(tf.math.atan2(eigvecs[1, 0], eigvecs[0, 0]).numpy()) * _pi_180
        eigvals_safe = tf.maximum(eigvals, 1e-8)
        w = float(2 * 2.0 * tf.sqrt(eigvals_safe[0]).numpy())
        h = float(2 * 2.0 * tf.sqrt(eigvals_safe[1]).numpy())
        ell = Ellipse(state_np[:2], w, h, angle=angle_deg, fill=True,
                      facecolor='lightblue', edgecolor='blue', linewidth=2,
                      alpha=0.3, label='State uncertainty (2σ)')
        ax.add_patch(ell)

        # Draw rays from each sigma point to landmark
        colors = plt.cm.Oranges(tf.linspace(0.4, 0.9, len(sigma_pts)).numpy())
        for i, (sp, color) in enumerate(zip(sigma_pts, colors)):
            ax.plot([sp[0], lm_np[0]], [sp[1], lm_np[1]],
                    color=color, linewidth=1.5, alpha=0.7, zorder=2)

        # Sigma points
        ax.scatter(sigma_pts[:, 0], sigma_pts[:, 1], s=100, c='orange',
                   marker='o', edgecolors='black', linewidths=1.5,
                   label=f'{len(sigma_pts)} sigma points', zorder=4)

        # Robot (mean state)
        ax.plot(state_np[0], state_np[1], 'ko', markersize=14, zorder=5)
        ax.annotate('Robot', (state_np[0], state_np[1]), textcoords='offset points',
                    xytext=(10, 10), fontsize=11, fontweight='bold')

        # Landmark
        ax.plot(lm_np[0], lm_np[1], 'r^', markersize=16, markeredgewidth=2,
                markeredgecolor='darkred', zorder=5)
        ax.annotate('Landmark', (lm_np[0], lm_np[1]), textcoords='offset points',
                    xytext=(10, -15), fontsize=11, fontweight='bold', color='darkred')

        # Compute range spread (std of ranges from sigma points)
        sigma_pts_t = tf.constant(sigma_pts, dtype=tf.float32)
        ranges = tf.sqrt(
            (sigma_pts_t[:, 0] - lm_np[0])**2 + (sigma_pts_t[:, 1] - lm_np[1])**2
        )
        range_spread = float(tf.math.reduce_std(ranges).numpy())
        range_mean = float(tf.reduce_mean(ranges).numpy())

        # Compute bearing spread
        bearings = tf.math.atan2(
            lm_np[1] - sigma_pts_t[:, 1], lm_np[0] - sigma_pts_t[:, 0]
        )
        bearing_spread = float(tf.math.reduce_std(bearings).numpy())

        # Distance from robot to landmark
        dist_to_lm = float(tf.sqrt(
            (state_np[0] - lm_np[0])**2 + (state_np[1] - lm_np[1])**2
        ).numpy())

        # Metrics box
        bearing_deg = bearing_spread * _pi_180
        metrics_text = (
            f"Distance to landmark: {dist_to_lm:.1f} m\n"
            f"Range spread (σ): {range_spread:.3f} m\n"
            f"Bearing spread (σ): {bearing_deg:.1f}°"
        )

        # Interpretation
        if scenario_label == 'Moderate':
            interpretation = "\n→ Rays spread nicely (OK)"
            box_color = 'lightgreen'
        else:
            interpretation = "\n→ Rays cluster/converge (COLLAPSE)"
            box_color = 'lightyellow'

        ax.text(0.02, 0.98, metrics_text + interpretation,
                transform=ax.transAxes, fontsize=11, va='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9, edgecolor='gray'))

        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title(f'({chr(97+ax_idx)}) {title_suffix}', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Set axis limits based on scenario
        if scenario_label == 'Moderate':
            ax.set_xlim(-4, 12)
            ax.set_ylim(-3, 10)
        else:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-2, 2)

    # Custom legend for rays
    custom_lines = [
        Line2D([0], [0], color='orange', linewidth=2, label='Range-bearing rays'),
    ]

    plt.suptitle('Sigma Point Spread: Moderate vs Strong Nonlinearity',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = output_dir / 'report_strong_nonlinearity.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved (report figure): {fig_path}")
    plt.close()


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


def plot_ekf_failure_via_filter(ssm: RangeBearingSSM,
                                landmarks_moderate: tf.Tensor,
                                landmarks_strong: tf.Tensor,
                                output_dir: Path) -> None:
    """
    EKF linearization failure: actual EKF vs true posterior (MC).

    Compares EKF Gaussian posterior with true posterior from particle/MC
    for moderate (distant) and strong (close) nonlinearity.
    """
    from matplotlib.patches import Ellipse

    _pi_180 = 180.0 / 3.141592653589793

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scenarios = [
        ("Moderate (Distant)", landmarks_moderate, 1.0),
        ("Strong (Close)", landmarks_strong, 2.5),
    ]

    for scenario_idx, (title, landmarks, uncertainty) in enumerate(scenarios):
        initial_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        initial_cov = tf.eye(3, dtype=tf.float32) * (uncertainty ** 2)

        control = tf.constant([0.3, 0.05], dtype=tf.float32)
        true_state = ssm.motion_model(initial_state[tf.newaxis, :],
                                      control[tf.newaxis, :])[0]

        true_meas = ssm.measurement_model(true_state[tf.newaxis, :], landmarks)[0]
        noisy_meas = true_meas + tf.random.normal(tf.shape(true_meas), stddev=0.1)

        ekf = ExtendedKalmanFilter(ssm, initial_state, initial_cov)
        ekf.predict(control)
        ekf.update(noisy_meas, landmarks)

        ekf_mean = ekf.state
        ekf_cov = ekf.covariance

        # True posterior via importance sampling (MC)
        n_particles = 5000
        L_Q = tf.linalg.cholesky(ssm.Q + 1e-6 * tf.eye(3, dtype=tf.float32))

        x0_samples = tf.random.normal(
            [n_particles, 3], mean=initial_state,
            stddev=tf.sqrt(tf.constant(uncertainty, dtype=tf.float32)))
        x_pred_samples = ssm.motion_model(
            x0_samples,
            tf.tile(control[tf.newaxis, :], [n_particles, 1]))
        process_noise = tf.matmul(
            tf.random.normal([n_particles, 3]), L_Q, transpose_b=True)
        x_pred_samples = x_pred_samples + process_noise

        z_samples = ssm.measurement_model(x_pred_samples, landmarks)
        z_samples_flat = tf.reshape(z_samples, [n_particles, -1])

        meas_dim = tf.shape(z_samples_flat)[1]
        R = ssm.full_measurement_cov(tf.shape(landmarks)[0])
        R_inv = tf.linalg.inv(R + 1e-6 * tf.eye(meas_dim, dtype=tf.float32))

        diff = tf.reshape(noisy_meas, [-1])[tf.newaxis, :] - z_samples_flat
        mahal = tf.reduce_sum(diff @ R_inv * diff, axis=1)
        log_weights = -0.5 * mahal

        max_log_weight = tf.reduce_max(log_weights)
        weights = tf.exp(log_weights - max_log_weight)
        weights = weights / tf.reduce_sum(weights)

        true_mean = tf.reduce_sum(weights[:, tf.newaxis] * x_pred_samples, axis=0)
        centered = x_pred_samples - true_mean
        weighted_centered = centered * tf.sqrt(weights[:, tf.newaxis])
        true_cov = tf.matmul(weighted_centered, weighted_centered, transpose_a=True)

        # State space plot
        ax_state = axes[scenario_idx, 0]
        particles_np = x_pred_samples.numpy()
        weights_np = weights.numpy()

        ax_state.scatter(particles_np[:, 0], particles_np[:, 1],
                        c=weights_np, s=5, alpha=0.3, cmap='Greys')

        def draw_cov_ellipse(ax, mean, cov, color, label, linestyle='-'):
            cov_2d = tf.cast(cov[:2, :2], tf.float32)
            mean_2d = mean[:2].numpy() if hasattr(mean, 'numpy') else mean[:2]
            eigvals, eigvecs = tf.linalg.eigh(cov_2d)
            eigvals = tf.maximum(eigvals, 1e-8)
            angle_deg = float(tf.math.atan2(eigvecs[1, 0], eigvecs[0, 0]).numpy()) * _pi_180
            width = float(2 * 2.0 * tf.sqrt(eigvals[0]).numpy())
            height = float(2 * 2.0 * tf.sqrt(eigvals[1]).numpy())
            ell = Ellipse(mean_2d, width, height, angle=angle_deg, fill=False,
                          edgecolor=color, linewidth=2, linestyle=linestyle,
                          label=label)
            ax.add_patch(ell)

        draw_cov_ellipse(ax_state, ekf_mean, ekf_cov, 'red', 'EKF posterior', '--')
        draw_cov_ellipse(ax_state, true_mean, true_cov, 'blue', 'True posterior', '-')

        ax_state.plot(ekf_mean.numpy()[0], ekf_mean.numpy()[1], 'r*',
                      markersize=15, label='EKF mean', zorder=10)
        ax_state.plot(true_mean.numpy()[0], true_mean.numpy()[1], 'b*',
                      markersize=15, label='True mean', zorder=10)
        ax_state.plot(true_state.numpy()[0], true_state.numpy()[1], 'go',
                      markersize=12, label='True state', zorder=10)
        ax_state.plot(landmarks.numpy()[0, 0], landmarks.numpy()[0, 1],
                      'r^', markersize=12, label='Landmark')

        ax_state.set_xlabel('x (m)', fontsize=11)
        ax_state.set_ylabel('y (m)', fontsize=11)
        ax_state.set_title(f'({chr(97+scenario_idx)}1) {title}: State Space',
                          fontsize=11, fontweight='bold')
        ax_state.legend(fontsize=9, loc='best')
        ax_state.grid(True, alpha=0.3)
        ax_state.axis('equal')

        # Covariance comparison
        ax_cov = axes[scenario_idx, 1]
        ekf_vars = tf.linalg.diag_part(ekf_cov[:2, :2]).numpy()
        true_vars = tf.linalg.diag_part(true_cov[:2, :2]).numpy()

        x_pos = range(2)
        bar_w = 0.35
        ax_cov.bar([p - bar_w/2 for p in x_pos], true_vars, bar_w,
                   label='True posterior', color='blue', alpha=0.7)
        ax_cov.bar([p + bar_w/2 for p in x_pos], ekf_vars, bar_w,
                   label='EKF posterior', color='red', alpha=0.7)
        ax_cov.set_xticks(list(x_pos))
        ax_cov.set_xticklabels(['x variance', 'y variance'])
        ax_cov.set_ylabel('Variance', fontsize=11)
        ax_cov.set_title(f'({chr(97+scenario_idx)}2) {title}: Uncertainty',
                        fontsize=11, fontweight='bold')
        ax_cov.legend(fontsize=9)
        ax_cov.grid(True, alpha=0.3, axis='y')

        mean_error = float(tf.sqrt(tf.reduce_sum((ekf_mean[:2] - true_mean[:2])**2)))
        cov_error = float(tf.reduce_mean(
            tf.abs(ekf_vars - true_vars) / (true_vars + 1e-6)
        ).numpy())
        error_text = f"Mean error: {mean_error:.3f} m\nCov error: {cov_error:.1%}"
        ax_cov.text(0.98, 0.98, error_text, transform=ax_cov.transAxes,
                    fontsize=10, va='top', ha='right',
                    bbox=dict(boxstyle='round',
                              facecolor='lightyellow' if cov_error > 0.3 else 'lightgreen',
                              alpha=0.9))

    plt.suptitle('EKF Linearization Failure: True vs EKF Posterior',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = output_dir / 'ekf_linearization_failure.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()


def plot_sigma_point_collapse_minimal(ssm: RangeBearingSSM,
                                      state: tf.Tensor,
                                      landmarks_moderate: tf.Tensor,
                                      landmarks_strong: tf.Tensor,
                                      output_dir: Path) -> None:
    """
    UKF sigma point collapse: state space → measurement space.

    Compares moderate (distant) vs strong (close) nonlinearity;
    shows sigma point spread in state vs measurement space.
    """
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    scenarios = [
        ("Moderate (Distant, σ=1.0, α=0.3)", landmarks_moderate, 1.0, 0.3),
        ("Strong (Close, σ=2.5, α=0.1)", landmarks_strong, 2.5, 0.1),
    ]

    for scenario_idx, (title, landmarks, uncertainty, alpha) in enumerate(scenarios):
        cov = tf.eye(3, dtype=tf.float32) * (uncertainty ** 2)
        ukf = UnscentedKalmanFilter(ssm, state, cov, alpha=alpha, beta=2.0, kappa=0.0)

        sigma_pts = ukf.generate_sigma_points(ukf.state, ukf.covariance)
        n_sigma = tf.shape(sigma_pts)[0]
        meas_sigma = ssm.measurement_model(sigma_pts, landmarks)
        meas_sigma_flat = tf.reshape(meas_sigma, [n_sigma, -1])

        ukf_meas_mean = tf.reduce_sum(ukf.wm[:, tf.newaxis] * meas_sigma_flat, axis=0)

        n_mc = 2000
        L = tf.linalg.cholesky(cov + 1e-6 * tf.eye(3, dtype=tf.float32))
        noise = tf.random.normal([n_mc, 3], dtype=tf.float32)
        mc_samples = state + tf.matmul(noise, L, transpose_b=True)
        mc_meas = ssm.measurement_model(mc_samples, landmarks)
        mc_meas_flat = tf.reshape(mc_meas, [n_mc, -1])
        mc_mean = tf.reduce_mean(mc_meas_flat, axis=0)

        sigma_pts_np = sigma_pts.numpy()
        meas_sigma_np = meas_sigma_flat.numpy()
        mc_samples_np = mc_samples.numpy()
        mc_meas_np = mc_meas_flat.numpy()
        state_np = state.numpy()
        lm_np = landmarks.numpy()[0]
        ukf_mean_np = ukf_meas_mean.numpy()
        mc_mean_np = mc_mean.numpy()

        # State space
        ax_state = axes[scenario_idx, 0]
        ax_state.scatter(mc_samples_np[:, 0], mc_samples_np[:, 1],
                        alpha=0.1, s=5, c='gray', label='MC samples')
        for sp in sigma_pts_np:
            ax_state.plot([sp[0], lm_np[0]], [sp[1], lm_np[1]],
                         'orange', alpha=0.3, linewidth=1, zorder=1)
        ax_state.scatter(sigma_pts_np[:, 0], sigma_pts_np[:, 1],
                        s=100, c='blue', marker='o', edgecolors='black',
                        label=f'{len(sigma_pts_np)} sigma points', zorder=5)
        ax_state.plot(state_np[0], state_np[1], 'k*', markersize=15,
                      label='Mean', zorder=6)
        ax_state.plot(lm_np[0], lm_np[1], 'r^', markersize=12, label='Landmark')
        circ = Circle((state_np[0], state_np[1]), 2 * uncertainty,
                      fill=False, edgecolor='blue', linestyle='--',
                      linewidth=1.5, alpha=0.5, label='2σ uncertainty')
        ax_state.add_patch(circ)
        ax_state.set_xlabel('x (m)', fontsize=11)
        ax_state.set_ylabel('y (m)', fontsize=11)
        ax_state.set_title(f'({chr(97+scenario_idx)}1) {title}: State Space',
                          fontsize=11, fontweight='bold')
        ax_state.legend(loc='upper right', fontsize=9)
        ax_state.grid(True, alpha=0.3)
        ax_state.axis('equal')

        # Measurement space
        ax_meas = axes[scenario_idx, 1]
        ax_meas.scatter(mc_meas_np[:, 0], mc_meas_np[:, 1],
                       alpha=0.1, s=5, c='gray', label='MC measurements')
        ax_meas.scatter(meas_sigma_np[:, 0], meas_sigma_np[:, 1],
                       s=100, c='blue', marker='o', edgecolors='black',
                       label='Sigma pt meas', zorder=5)
        ax_meas.plot(ukf_mean_np[0], ukf_mean_np[1], 'b*',
                     markersize=15, label='UKF mean', zorder=6)
        ax_meas.plot(mc_mean_np[0], mc_mean_np[1], 'g*',
                     markersize=15, label='MC mean', zorder=6)

        sigma_spread = tf.math.reduce_std(meas_sigma_flat, axis=0)
        mc_spread = tf.math.reduce_std(mc_meas_flat, axis=0)
        collapse_ratio = sigma_spread / (mc_spread + 1e-10)
        mean_error = float(tf.sqrt(tf.reduce_sum((ukf_meas_mean - mc_mean)**2)))

        metrics_text = (
            f"Collapse ratio:\n"
            f"  Range:   {float(collapse_ratio[0]):.3f}\n"
            f"  Bearing: {float(collapse_ratio[1]):.3f}\n"
            f"Mean error: {mean_error:.3f}\n"
            f"α = {alpha:.2f}"
        )
        if tf.reduce_any(collapse_ratio < 0.2):
            box_color = 'salmon'
            metrics_text += "\n⚠ SEVERE COLLAPSE"
        elif tf.reduce_any(collapse_ratio < 0.4):
            box_color = 'lightyellow'
            metrics_text += "\n⚠ COLLAPSE"
        else:
            box_color = 'lightgreen'
            metrics_text += "\n✓ Good spread"

        ax_meas.text(0.98, 0.02, metrics_text, transform=ax_meas.transAxes,
                    fontsize=10, va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor=box_color,
                              alpha=0.9, edgecolor='gray'))

        ax_meas.set_xlabel('Range (m)', fontsize=11)
        ax_meas.set_ylabel('Bearing (rad)', fontsize=11)
        ax_meas.set_title(f'({chr(97+scenario_idx)}2) {title}: Measurement Space',
                         fontsize=11, fontweight='bold')
        ax_meas.legend(loc='upper right', fontsize=9)
        ax_meas.grid(True, alpha=0.3)

    plt.suptitle('UKF Sigma Point Collapse: State → Measurement Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = output_dir / 'ukf_sigma_point_collapse.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()


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


def save_text_analysis_summary(analysis_results: dict, output_dir: Path) -> None:
    """
    Save human-readable text summary of the analysis to a .txt file.

    This mirrors the key console output from run_failure_analysis so that
    the textual diagnostics are preserved alongside the CSV files.
    """
    summary_path = output_dir / 'text_analysis_summary.txt'
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("EKF / UKF LINEARIZATION & SIGMA-POINT FAILURE ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    for scenario_name, results in analysis_results.items():
        lines.append("=" * 70)
        lines.append(scenario_name)
        # Landmarks (if present)
        landmarks = results.get('landmarks')
        try:
            if hasattr(landmarks, 'numpy'):
                lm_str = str(landmarks.numpy())
            else:
                lm_str = str(landmarks)
            lines.append(f"Landmark position: {lm_str}")
        except Exception:
            pass
        lines.append("=" * 70)
        lines.append("")

        # 1. EKF linearization error summary
        lines.append("[1] EKF LINEARIZATION ERROR ANALYSIS")
        ekf_errors = results.get('ekf_errors', [])
        if ekf_errors:
            lines.append("  Summary (variance error: higher = worse linearization):")
            for res in ekf_errors:
                scale = res.get('scale')
                var_error = res.get('var_error')
                lines.append(f"    σ={scale:.1f}: Var Error={var_error:.3f}")
        lines.append("")

        # 2. UKF sigma point collapse summary
        lines.append("[2] UKF SIGMA POINT COLLAPSE ANALYSIS")
        collapse_info = results.get('collapse_info', {})
        collapse_ratio = collapse_info.get('collapse_ratio', [])
        lines.append(f"  Collapse Ratios: {collapse_ratio}")
        lines.append("  Interpretation:")
        # collapse_ratio is a list of floats
        if any(r < 0.1 for r in collapse_ratio):
            lines.append("    SIGMA POINT COLLAPSE DETECTED (ratio < 0.1)!")
        elif any(r < 0.3 for r in collapse_ratio):
            lines.append("    Partial collapse (ratio < 0.3) - marginal spread")
        else:
            lines.append("    Good sigma point spread (ratio > 0.3)")
        lines.append("")

        # 3. NEES-based consistency summary
        lines.append("[3] FILTER CONSISTENCY (NEES)")
        nees_results = results.get('nees_results', {})
        n_steps = nees_results.get('n_steps', 0)
        avg_nees_ekf = nees_results.get('avg_nees_ekf', float('nan'))
        avg_nees_ukf = nees_results.get('avg_nees_ukf', float('nan'))
        nees_lower = nees_results.get('nees_lower', float('nan'))
        nees_upper = nees_results.get('nees_upper', float('nan'))

        lines.append(f"  Average NEES over {n_steps} steps:")
        lines.append(f"    EKF: {avg_nees_ekf:.2f} (expected ~3.0 for 3D state)")
        lines.append(f"    UKF: {avg_nees_ukf:.2f}")
        lines.append(f"  Interpretation (95% CI: [{nees_lower:.2f}, {nees_upper:.2f}]):")

        # EKF consistency
        if avg_nees_ekf > nees_upper:
            lines.append("    EKF: Filter diverged or inconsistent")
        elif avg_nees_ekf < nees_lower:
            lines.append("    EKF: Overly conservative (underconfident)")
        else:
            lines.append("    EKF: Consistent")

        # UKF consistency
        if avg_nees_ukf > nees_upper:
            lines.append("    UKF: Filter diverged or inconsistent")
        elif avg_nees_ukf < nees_lower:
            lines.append("    UKF: Overly conservative")
        else:
            lines.append("    UKF: Consistent")

        lines.append("")
        lines.append("")

    with open(summary_path, 'w') as f:
        for line in lines:
            f.write(line + "\n")

    print(f"✓ Saved text analysis summary: {summary_path}")


def run_final_analysis() -> None:
    """
    Run complete failure mode analysis: text + plots + CSV outputs.

    Generates comprehensive visualizations addressing:
    - Linearization accuracy limits (EKF)
    - Sigma point failures under strong nonlinearity (UKF)
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("RUNNING TEXT ANALYSIS")
    print("="*70)

    # Run text analysis
    ssm, initial_state, initial_cov, scenarios, analysis_results = (
        run_failure_analysis())

    # Save analysis results to CSV and human-readable text
    print("\n" + "="*70)
    print("SAVING ANALYSIS RESULTS TO CSV/TXT")
    print("="*70)
    save_analysis_results(analysis_results, OUTPUT_DIR)
    save_text_analysis_summary(analysis_results, OUTPUT_DIR)

    # Define landmarks for plotting
    landmarks_moderate = tf.constant([[10.0, 10.0]], dtype=tf.float32)
    landmarks_strong = tf.constant([[0.5, 0.2]], dtype=tf.float32)

    # -------------------------------------------------------------------------
    # REPORT FIGURE: Strong nonlinearity only (use this in your report)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING REPORT FIGURE (Strong Nonlinearity)")
    print("="*70)

    plot_report_strong_nonlinearity(
        ssm, initial_state, initial_cov, landmarks_strong, OUTPUT_DIR
    )

    # -------------------------------------------------------------------------
    # MINIMAL FAILURE ANALYSIS: EKF actual vs true posterior, UKF collapse
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING MINIMAL FAILURE PLOTS (EKF actual + UKF collapse)")
    print("="*70)

    landmarks_strong_close = tf.constant([[0.3, 0.15]], dtype=tf.float32)
    plot_ekf_failure_via_filter(
        ssm, landmarks_moderate, landmarks_strong_close, OUTPUT_DIR
    )
    plot_sigma_point_collapse_minimal(
        ssm, initial_state, landmarks_moderate, landmarks_strong_close,
        OUTPUT_DIR
    )

    # -------------------------------------------------------------------------
    # NEW VISUALIZATION 1: Linearization Breakdown (True vs Linear in meas space)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING LINEARIZATION BREAKDOWN PLOTS")
    print("="*70)

    plot_linearization_breakdown_2d(ssm, initial_state, landmarks_moderate,
                                    "Moderate (Distant)", OUTPUT_DIR)
    plot_linearization_breakdown_2d(ssm, initial_state, landmarks_strong,
                                    "Strong (Close)", OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # NEW VISUALIZATION 2: Sigma Point Propagation (State → Measurement Space)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING SIGMA POINT PROPAGATION PLOTS")
    print("="*70)

    for name, landmarks in scenarios:
        ukf = UnscentedKalmanFilter(ssm, initial_state, initial_cov,
                                    alpha=0.3, beta=2.0, kappa=0.0)
        plot_sigma_point_propagation(ssm, ukf, landmarks, name, OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # NEW VISUALIZATION 3: Nonlinearity Heatmap (Jacobian condition number)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING NONLINEARITY HEATMAP")
    print("="*70)

    # Use close landmark to show where nonlinearity is strongest
    plot_nonlinearity_heatmap(ssm, landmarks_strong, OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # NEW VISUALIZATION 4: NEES Comparison (Filter Consistency Over Time)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING NEES COMPARISON PLOT")
    print("="*70)

    plot_nees_comparison(analysis_results, OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # EXISTING: EKF Linearization Curve (Error vs Uncertainty)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING EKF LINEARIZATION CURVE")
    print("="*70)

    plot_ekf_linearization_curve(ssm, initial_state, landmarks_moderate,
                                 landmarks_strong, OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # EXISTING: Simple Sigma Point Visualization
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING SIMPLE SIGMA POINT VISUALIZATIONS")
    print("="*70)

    for name, landmarks in scenarios:
        print(f"\nCreating simple sigma point plot for: {name}")

        true_state = initial_state
        control = tf.constant([0.5, 0.1], dtype=tf.float32)
        true_state = ssm.motion_model(true_state[tf.newaxis, :],
                                      control[tf.newaxis, :])[0]
        true_meas = ssm.measurement_model(true_state[tf.newaxis, :],
                                         landmarks)[0][0]

        ukf = UnscentedKalmanFilter(ssm, initial_state, initial_cov,
                                    alpha=0.3, beta=2.0, kappa=0.0)

        plot_simple_sigma_point_visualization(ukf, landmarks, true_meas,
                                              name, OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("✓ FINAL ANALYSIS COMPLETE!")
    print("="*70)
    print("  Generated visualizations:")
    print("  ★ REPORT FIGURE: report_strong_nonlinearity.png (use in report)")
    print("  ★ MINIMAL: ekf_linearization_failure.png (EKF vs true posterior)")
    print("  ★ MINIMAL: ukf_sigma_point_collapse.png (state → meas space)")
    print("  1. Linearization Breakdown: True vs linearized distributions")
    print("  2. Sigma Point Propagation: State → measurement space mapping")
    print("  3. Nonlinearity Heatmap: Jacobian condition number across space")
    print("  4. NEES Comparison: Filter consistency (EKF vs UKF)")
    print("  5. EKF Linearization Curve: Error vs uncertainty scale")
    print("  6. Simple Sigma Point Plots: Per-scenario collapse analysis")
    print(f"\n  All results saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    run_final_analysis()