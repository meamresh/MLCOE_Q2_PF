"""
Replicate Hu & van Leeuwen (2021) Figures 2 and 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.filters.pff_kernel import (
    scalar_kernel_flow,
    diagonal_kernel_flow,
    assimilate_pff,
)
from src.models.ssm_lorenz96 import Lorenz96Model

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})


# =============================================================================
# Figure 2: Kernel schematic (TF for computation; .numpy() for plotting)
# =============================================================================

def _get_extended_line_tf(p1: tf.Tensor, p2: tf.Tensor, extend_factor: float = 2.0):
    """Extended line through p1, p2. Returns start, end as tensors."""
    direction = p2 - p1
    center = (p1 + p2) / 2.0
    norm_d = tf.sqrt(tf.reduce_sum(direction ** 2)) + 1e-12
    half_length = norm_d * tf.cast(extend_factor / 2.0, tf.float32)
    unit_dir = direction / norm_d
    start = center - half_length * unit_dir
    end = center + half_length * unit_dir
    return start, end


def fig2_kernel_schematic():
    """
    Replicate Figure 2: Schematic comparison of matrix vs scalar kernels.
    Shows repulsion forces for two particle configurations. TF for computation.
    """
    print("\n" + "=" * 70)
    print("REPLICATING FIGURE 2: Kernel Force Comparison")
    print("=" * 70)

    sigmax_sq = tf.constant(2.0, dtype=tf.float32)
    sigmay_sq = tf.constant(2.0, dtype=tf.float32)
    alpha_inv = 5.0
    sigma_sq_np = tf.stack([sigmax_sq, sigmay_sq])

    def K_matrix(p1: tf.Tensor, p2: tf.Tensor) -> tf.Tensor:
        return tf.stack([
            tf.exp(-((p1[0] - p2[0]) ** 2) / sigmax_sq),
            tf.exp(-((p1[1] - p2[1]) ** 2) / sigmay_sq),
        ])

    sigma_sq = tf.constant(2.0, dtype=tf.float32)

    def K_scalar(p1: tf.Tensor, p2: tf.Tensor) -> tf.Tensor:
        return tf.exp(-tf.reduce_sum((p1 - p2) ** 2) / (1.4 * sigma_sq))

    p1 = tf.constant([9.0, 9.6], dtype=tf.float32)
    p2 = tf.constant([10.4, 10.4], dtype=tf.float32)

    matrix_force1 = alpha_inv * 2.0 * ((p1 - p2) / sigma_sq_np) * K_matrix(p1, p2)
    matrix_force2 = -alpha_inv * 2.0 * ((p1 - p2) / sigma_sq_np) * K_matrix(p1, p2)
    scalar_force1 = alpha_inv * 2.0 * (p1 - p2) * K_scalar(p1, p2) / sigma_sq
    scalar_force2 = -alpha_inv * 2.0 * (p1 - p2) * K_scalar(p1, p2) / sigma_sq

    p3 = tf.constant([6.0, 9.6], dtype=tf.float32)
    p4 = tf.constant([14.0, 10.4], dtype=tf.float32)

    matrix_force3 = alpha_inv * 2.0 * ((p3 - p4) / sigma_sq_np) * K_matrix(p3, p4)
    matrix_force4 = -alpha_inv * 2.0 * ((p3 - p4) / sigma_sq_np) * K_matrix(p3, p4)
    scalar_force3 = alpha_inv * 2.0 * (p3 - p4) * K_scalar(p3, p4) / sigma_sq
    scalar_force4 = -alpha_inv * 2.0 * (p3 - p4) * K_scalar(p3, p4) / sigma_sq

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    x_min, x_max = 3, 18
    y_min, y_max = 7, 13
    headwidth = 0.5
    labels = ['(a)', '(b)', '(c)', '(d)']
    kernel_labels = ['matrix', 'scalar', 'matrix', 'scalar']

    p1_np = p1.numpy()
    p2_np = p2.numpy()
    p3_np = p3.numpy()
    p4_np = p4.numpy()
    mf1_np = matrix_force1.numpy()
    mf2_np = matrix_force2.numpy()
    sf1_np = scalar_force1.numpy()
    sf2_np = scalar_force2.numpy()
    mf3_np = matrix_force3.numpy()
    mf4_np = matrix_force4.numpy()
    r_sigmax = float(tf.sqrt(sigmax_sq).numpy())

    for idx, ax in enumerate(axs.flat):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
        ax.annotate('', xy=(x_max - 0.5, y_min), xytext=(x_min, y_min), arrowprops=arrow_props)
        ax.annotate('', xy=(x_min, y_max - 0.5), xytext=(x_min, y_min), arrowprops=arrow_props)
        ax.text(x_max - 1, y_min - 1, 'x₁', fontsize=20, ha='center')
        ax.text(x_min - 1, y_max - 1, 'x₂', fontsize=20, va='center')
        ax.text(-0.1, 1.1, labels[idx], transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top', ha='left')
        ax.text(0.8, 0.15, kernel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='right')

    start1, end1 = _get_extended_line_tf(p1, p2, 6.0)
    start1_np, end1_np = start1.numpy(), end1.numpy()
    start2, end2 = _get_extended_line_tf(p3, p4, 1.5)
    start2_np, end2_np = start2.numpy(), end2.numpy()

    ax = axs[0, 0]
    ax.scatter(*p1_np, color='black', s=300, zorder=5)
    ax.scatter(*p2_np, color='black', s=300, zorder=5)
    ax.arrow(p1_np[0], p1_np[1], mf1_np[0], 0, color='black',
             head_width=headwidth, length_includes_head=True)
    ax.arrow(p1_np[0], p1_np[1], 0, mf1_np[1], color='black',
             head_width=headwidth, length_includes_head=True)
    ax.arrow(p2_np[0], p2_np[1], mf2_np[0], 0, color='black',
             head_width=headwidth, length_includes_head=True)
    ax.arrow(p2_np[0], p2_np[1], 0, mf2_np[1], color='black',
             head_width=headwidth, length_includes_head=True)
    for center in [p1_np, p2_np]:
        circle = plt.Circle(center, r_sigmax, color='gray', alpha=0.2)
        ax.add_patch(circle)
    ax.axhline(y=(p1_np[1] + p2_np[1]) / 2, color='black', linestyle='--', linewidth=1)
    ax.plot([start1_np[0], end1_np[0]], [start1_np[1], end1_np[1]],
            color='black', linestyle='--', linewidth=1, zorder=3)

    ax = axs[0, 1]
    ax.scatter(*p1_np, color='black', s=300, zorder=5)
    ax.scatter(*p2_np, color='black', s=300, zorder=5)
    ax.arrow(p1_np[0], p1_np[1], sf1_np[0], sf1_np[1], color='black',
             head_width=headwidth, length_includes_head=True)
    ax.arrow(p2_np[0], p2_np[1], sf2_np[0], sf2_np[1], color='black',
             head_width=headwidth, length_includes_head=True)
    for center in [p1_np, p2_np]:
        circle = plt.Circle(center, r_sigmax, color='gray', alpha=0.2)
        ax.add_patch(circle)
    ax.axhline(y=(p1_np[1] + p2_np[1]) / 2, color='black', linestyle='--', linewidth=1)
    ax.plot([start1_np[0], end1_np[0]], [start1_np[1], end1_np[1]],
            color='black', linestyle='--', linewidth=1, zorder=3)

    ax = axs[1, 0]
    ax.scatter(*p3_np, color='black', s=300, zorder=5)
    ax.scatter(*p4_np, color='black', s=300, zorder=5)
    ax.arrow(p3_np[0], p3_np[1], 0, mf3_np[1], color='black',
             head_width=headwidth, length_includes_head=True)
    ax.arrow(p4_np[0], p4_np[1], 0, mf4_np[1], color='black',
             head_width=headwidth, length_includes_head=True)
    for center in [p3_np, p4_np]:
        circle = plt.Circle(center, r_sigmax, color='gray', alpha=0.2)
        ax.add_patch(circle)
    ax.axhline(y=(p3_np[1] + p4_np[1]) / 2, color='black', linestyle='--', linewidth=1)
    ax.plot([start2_np[0], end2_np[0]], [start2_np[1], end2_np[1]],
            color='black', linestyle='--', linewidth=1, zorder=3)

    ax = axs[1, 1]
    ax.scatter(*p3_np, color='black', s=300, zorder=5)
    ax.scatter(*p4_np, color='black', s=300, zorder=5)
    for center in [p3_np, p4_np]:
        circle = plt.Circle(center, r_sigmax, color='gray', alpha=0.2)
        ax.add_patch(circle)
    ax.axhline(y=(p3_np[1] + p4_np[1]) / 2, color='black', linestyle='--', linewidth=1)
    ax.plot([start2_np[0], end2_np[0]], [start2_np[1], end2_np[1]],
            color='black', linestyle='--', linewidth=1, zorder=3)

    plt.tight_layout()
    out_dir = Path("reports/3_Deterministic_Kernel_Flow/Hu(21)")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'figure2_kernel_schematic.pdf', bbox_inches='tight', dpi=300)
    #plt.savefig(out_dir / 'figure2_kernel_schematic.png', bbox_inches='tight', dpi=150)
    print("✓ Saved: figure2_kernel_schematic.pdf")
    #print("✓ Saved: figure2_kernel_schematic.png")
    plt.close()


# =============================================================================
# Figure 3: High-dimensional L96 (TF-only)
# =============================================================================

def _localization_matrix_l96_tf(D: int, r_in: float) -> tf.Tensor:
    """Gaussian localization C_ij = exp(-d_ij^2 / r_in^2) with periodic L96 distance."""
    idx = tf.range(D, dtype=tf.float32)
    dist = tf.abs(idx[:, None] - idx[None, :])
    dist = tf.minimum(dist, tf.cast(D, tf.float32) - dist)
    C = tf.exp(-(dist ** 2) / (r_in ** 2))
    return C


def fig3_high_dimensional_lorenz_96():
    """
    Replicate Figure 3: High-dimensional L96 experiment (D=1000).
    Scalar kernel collapse vs matrix kernel preservation. TensorFlow throughout.
    """
    print("\n" + "=" * 70)
    print("REPLICATING FIGURE 3: High-Dimensional Lorenz-96 (TensorFlow)")
    print("=" * 70)

    N = 20
    D = 1000
    F = 8.0
    obs_noise_var = 0.5
    obs_noise_std = float(tf.sqrt(tf.constant(obs_noise_var, dtype=tf.float32)).numpy())

    print(f"Initializing Lorenz-96 (D={D}, N={N})...")
    model = Lorenz96Model(dim=D, F=F, dt=0.01)

    tf.random.set_seed(42)
    print("Spinning up truth to chaotic attractor...")
    x_true = tf.ones(D, dtype=tf.float32) * F
    x_true = x_true + tf.random.normal([D], stddev=0.5, dtype=tf.float32)
    for _ in range(200):
        x_true = model.step(x_true)

    print("Finding suitable truth snapshot...")
    for _ in range(500):
        if x_true[19] > 4.0:
            break
        x_true = model.step(x_true)

    print(f"Truth snapshot: x[19]={float(x_true[18].numpy()):.2f}, x[20]={float(x_true[19].numpy()):.2f}")

    prior_mean_climo = tf.ones(D, dtype=tf.float32) * F
    prior_std = 2.0
    print("Generating prior ensemble...")
    tf.random.set_seed(123)
    prior = tf.random.normal([N, D], stddev=prior_std, dtype=tf.float32) + prior_mean_climo

    prior_mean_ens = tf.reduce_mean(prior, axis=0)
    prior_rmse = tf.sqrt(tf.reduce_mean((prior_mean_ens - x_true) ** 2))
    print(f"\nPrior RMSE: {float(prior_rmse.numpy()):.3f}")

    print("\nBuilding localized sample covariance (paper Eq. 29)...")
    Xc = prior - prior_mean_ens[tf.newaxis, :]
    prior_cov_sample = tf.matmul(Xc, Xc, transpose_a=True) / tf.cast(N - 1, tf.float32)
    r_in = 4.0
    C_tf = _localization_matrix_l96_tf(D, r_in)
    prior_cov = prior_cov_sample * C_tf
    prior_cov = 0.5 * (prior_cov + tf.transpose(prior_cov))
    mean_var = tf.reduce_mean(tf.linalg.diag_part(prior_cov))
    prior_cov = prior_cov + tf.eye(D, dtype=tf.float32) * (1e-6 * mean_var + 1e-8)
    prior_mean = prior_mean_ens

    obs_indices = tf.range(3, D, 4, dtype=tf.int32)
    tf.random.set_seed(456)
    obs = tf.gather(x_true, obs_indices)
    obs = obs + tf.random.normal(tf.shape(obs), stddev=obs_noise_std, dtype=tf.float32)
    print(f"Observations: {tf.shape(obs_indices)[0].numpy()} out of {D} dimensions")
    print(f"Observation for x[20]: {float(obs[4].numpy()):.2f}")

    prior_mean_obs = tf.gather(prior_mean_ens, obs_indices)
    innovation_rmse = tf.sqrt(tf.reduce_mean((prior_mean_obs - obs) ** 2))
    print(f"Innovation RMSE: {float(innovation_rmse):.3f}")

    print("\nRunning Matrix Kernel PFF...")
    post_matrix, traj_mat, diag_mat = assimilate_pff(
        tf.identity(prior),
        obs, obs_indices, obs_noise_std,
        prior_mean, prior_cov,
        kernel_type='diagonal',
        max_steps=500,
        initial_step_size=0.5,
        convergence_tol=1e-5,
    )
    print(f"  Converged: {diag_mat['converged'].numpy()}, Steps: {diag_mat['n_steps'].numpy()}")

    print("Running Scalar Kernel PFF...")
    post_scalar, traj_scal, diag_scal = assimilate_pff(
        tf.identity(prior),
        obs, obs_indices, obs_noise_std,
        prior_mean, prior_cov,
        kernel_type='scalar',
        max_steps=500,
        initial_step_size=0.5,
        convergence_tol=1e-5,
    )
    print(f"  Converged: {diag_scal['converged'].numpy()}, Steps: {diag_scal['n_steps'].numpy()}")

    x_true_np = x_true.numpy()
    prior_np = prior.numpy()
    post_matrix_np = post_matrix.numpy()
    post_scalar_np = post_scalar.numpy()
    obs_indices_np = obs_indices.numpy()

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    idx_u = 18
    idx_o = 19

    prior_spread_obs = float(tf.math.reduce_std(prior_np[:, idx_o]).numpy())
    post_mat_spread_obs = float(tf.math.reduce_std(post_matrix_np[:, idx_o]).numpy())
    post_scal_spread_obs = float(tf.math.reduce_std(post_scalar_np[:, idx_o]).numpy())

    print(f"\nObserved dimension (x[20]) spread:")
    print(f"  Prior:  {prior_spread_obs:.3f}")
    print(f"  Matrix: {post_mat_spread_obs:.3f} (diversity ratio: {post_mat_spread_obs/prior_spread_obs:.3f})")
    print(f"  Scalar: {post_scal_spread_obs:.3f} (diversity ratio: {post_scal_spread_obs/prior_spread_obs:.3f})")

    prior_mean_np = tf.reduce_mean(prior, axis=0).numpy()
    post_matrix_mean_np = tf.reduce_mean(post_matrix, axis=0).numpy()
    post_scalar_mean_np = tf.reduce_mean(post_scalar, axis=0).numpy()

    rmse_prior = float(tf.sqrt(tf.reduce_mean((prior_mean_np - x_true_np) ** 2)).numpy())
    rmse_matrix = float(tf.sqrt(tf.reduce_mean((post_matrix_mean_np - x_true_np) ** 2)).numpy())
    rmse_scalar = float(tf.sqrt(tf.reduce_mean((post_scalar_mean_np - x_true_np) ** 2)).numpy())

    rmse_prior_obs = float(tf.sqrt(tf.reduce_mean((prior_mean_np[obs_indices_np] - x_true_np[obs_indices_np]) ** 2)).numpy())
    rmse_matrix_obs = float(tf.sqrt(tf.reduce_mean((post_matrix_mean_np[obs_indices_np] - x_true_np[obs_indices_np]) ** 2)).numpy())
    rmse_scalar_obs = float(tf.sqrt(tf.reduce_mean((post_scalar_mean_np[obs_indices_np] - x_true_np[obs_indices_np]) ** 2)).numpy())

    unobs_mask = tf.reduce_all(
        tf.not_equal(tf.range(D)[:, None], tf.cast(obs_indices[None, :], tf.int32)), axis=1
    )
    unobs_indices = tf.reshape(tf.where(unobs_mask), [-1]).numpy()

    rmse_prior_unobs = float(tf.sqrt(tf.reduce_mean((prior_mean_np[unobs_indices] - x_true_np[unobs_indices]) ** 2)).numpy())
    rmse_matrix_unobs = float(tf.sqrt(tf.reduce_mean((post_matrix_mean_np[unobs_indices] - x_true_np[unobs_indices]) ** 2)).numpy())
    rmse_scalar_unobs = float(tf.sqrt(tf.reduce_mean((post_scalar_mean_np[unobs_indices] - x_true_np[unobs_indices]) ** 2)).numpy())

    print(f"\nRMSE (full state):")
    print(f"  Prior:  {rmse_prior:.3f}")
    print(f"  Matrix: {rmse_matrix:.3f} (improvement: {(rmse_prior-rmse_matrix)/rmse_prior*100:.1f}%)")
    print(f"  Scalar: {rmse_scalar:.3f} (improvement: {(rmse_prior-rmse_scalar)/rmse_prior*100:.1f}%)")
    print(f"\nRMSE (observed dimensions only):")
    print(f"  Prior:  {rmse_prior_obs:.3f}")
    print(f"  Matrix: {rmse_matrix_obs:.3f}")
    print(f"  Scalar: {rmse_scalar_obs:.3f}")
    print(f"\nRMSE (unobserved dimensions only):")
    print(f"  Prior:  {rmse_prior_unobs:.3f}")
    print(f"  Matrix: {rmse_matrix_unobs:.3f}")
    print(f"  Scalar: {rmse_scalar_unobs:.3f}")

    print("\n" + "=" * 70)
    print("CREATING FIGURE")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def plot_result(ax, post, title, is_collapsed):
        ax.scatter(
            prior_np[:, idx_u], prior_np[:, idx_o],
            facecolors='none', edgecolors='k', s=100, linewidths=2,
            label='Prior', zorder=3,
        )
        ax.scatter(
            post[:, idx_u], post[:, idx_o],
            c='r', s=100, label='Posterior', zorder=4, alpha=0.8,
        )
        obs_val_np = float(obs[4].numpy())
        # Observation line (noisy measurement)
        ax.axhline(obs_val_np, color='b', linestyle='--', linewidth=2, label='Observation', zorder=2)
        # Posterior mean in observed dim: where particles actually collapse to (Bayesian update)
        post_mean_obs = float(tf.reduce_mean(post[:, idx_o]).numpy())
        ax.axhline(post_mean_obs, color='darkgreen', linestyle='-', linewidth=1.5,
                   label='Posterior mean (x₂₀)', zorder=2)
        collapse_text = " (Collapsed)" if is_collapsed else ""
        ax.set_title(f"{title} Kernel{collapse_text}", fontweight='bold')
        ax.set_xlabel("Unobserved x₁₉", fontsize=14)
        ax.set_ylabel("Observed x₂₀", fontsize=14)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        x_vals = list(prior_np[:, idx_u].flatten()) + list(post[:, idx_u].flatten())
        y_vals = list(prior_np[:, idx_o].flatten()) + list(post[:, idx_o].flatten()) + [obs_val_np, post_mean_obs]
        pad = 4.0
        ax.set_xlim(min(x_vals) - pad, max(x_vals) + pad)
        ax.set_ylim(min(y_vals) - pad, max(y_vals) + pad)

    plot_result(axes[0], post_matrix_np, "Matrix-Valued", False)
    plot_result(axes[1], post_scalar_np, "Scalar", True)

    plt.suptitle(
        f"Replication of Figure 3 (Hu & van Leeuwen 2021)\n"
        f"Effect of Kernel Choice in High Dimensions (D={D}, N={N})",
        fontsize=15, fontweight='bold',
    )
    plt.tight_layout()
    out_dir = Path("reports/3_Deterministic_Kernel_Flow/Hu(21)")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'figure3_L96.pdf', bbox_inches='tight', dpi=300)
    #plt.savefig(out_dir / 'figure3_high_dimensional_L96_tensorflow.png', bbox_inches='tight', dpi=150)
    print("\n✓ Saved: figure3_L96.pdf")
    #print("✓ Saved: figure3_high_dimensional_L96_tensorflow.png")
    plt.close()


# =============================================================================
# Sequential linear experiment (TF-only)
# =============================================================================

def replicate_linear_sequential_rmse(
    *,
    D: int = 1000,
    N: int = 20,
    F: float = 8.0,
    model_dt: float = 0.01,
    spinup_steps: int = 1000,
    total_steps: int = 500,
    assim_interval: int = 20,
    obs_stride: int = 4,
    obs_start: int = 3,
    obs_noise_var: float = 0.5,
    r_in: float = 4.0,
    pff_max_steps: int = 200,
    pff_step_size: float = 0.05,
    convergence_tol: float = 1e-5,
):
    """
    Sequential linear-observation experiment (paper Section 3/4).
    RMSE time series for noDA, PFF scalar, PFF matrix kernel. TensorFlow only.
    """
    print("\n" + "=" * 70)
    print("REPLICATING SEQUENTIAL LINEAR EXPERIMENT (RMSE TIME SERIES)")
    print("=" * 70)
    print(f"D={D}, N={N}, total_steps={total_steps}, assim_interval={assim_interval}")

    obs_noise_std = float(tf.sqrt(tf.constant(obs_noise_var, dtype=tf.float32)).numpy())
    model = Lorenz96Model(dim=D, F=F, dt=model_dt)

    x0 = tf.ones(D, dtype=tf.float32) * F
    indices_plus = tf.range(4, D, 5)
    x0 = tf.tensor_scatter_nd_add(
        x0,
        tf.reshape(indices_plus, [-1, 1]),
        tf.ones(tf.shape(indices_plus), dtype=tf.float32),
    )
    x_true = x0

    print("Spinning up truth to chaotic attractor...")
    for _ in range(spinup_steps):
        x_true = model.step(x_true)

    pert_std = float(tf.sqrt(tf.constant(2.0, dtype=tf.float32)).numpy())
    tf.random.set_seed(123)
    ens0 = x_true[tf.newaxis, :] + tf.random.normal([N, D], stddev=pert_std, dtype=tf.float32)

    ens_noda = tf.identity(ens0)
    ens_scalar = tf.identity(ens0)
    ens_matrix = tf.identity(ens0)

    obs_indices = tf.range(obs_start, D, obs_stride, dtype=tf.int32)
    unobs_mask = tf.reduce_all(
        tf.not_equal(tf.range(D)[:, None], tf.cast(obs_indices[None, :], tf.int32)), axis=1
    )
    unobs_indices_tf = tf.reshape(tf.where(unobs_mask), [-1])

    print("Precomputing Gaussian localization matrix (paper Eq. 29)...")
    C_tf = _localization_matrix_l96_tf(D, r_in)

    def rmse_components(ens: tf.Tensor, truth: tf.Tensor):
        mean = tf.reduce_mean(ens, axis=0)
        diff = mean - truth
        rmse_all = tf.sqrt(tf.reduce_mean(diff ** 2))
        rmse_obs = tf.sqrt(tf.reduce_mean(tf.gather(diff, obs_indices) ** 2))
        rmse_unobs = tf.sqrt(tf.reduce_mean(tf.gather(diff, unobs_indices_tf) ** 2))
        return float(rmse_all.numpy()), float(rmse_obs.numpy()), float(rmse_unobs.numpy())

    rmse = {
        "noDA": {"all": [], "obs": [], "unobs": []},
        "scalar": {"all": [], "obs": [], "unobs": []},
        "matrix": {"all": [], "obs": [], "unobs": []},
    }

    for key, ens in [("noDA", ens_noda), ("scalar", ens_scalar), ("matrix", ens_matrix)]:
        a, b, c = rmse_components(ens, x_true)
        rmse[key]["all"].append(a)
        rmse[key]["obs"].append(b)
        rmse[key]["unobs"].append(c)

    print("Running forecast/analysis cycles...")
    tf.random.set_seed(456)
    for t in range(1, total_steps + 1):
        x_true = model.step(x_true)
        ens_noda = model.step(ens_noda)
        ens_scalar = model.step(ens_scalar)
        ens_matrix = model.step(ens_matrix)

        if (t % assim_interval) == 0:
            y = tf.gather(x_true, obs_indices)
            y = y + tf.random.normal(tf.shape(y), stddev=obs_noise_std, dtype=tf.float32)

            def analyze(ens: tf.Tensor, kernel_type: str):
                prior_mean = tf.reduce_mean(ens, axis=0)
                Xc = ens - prior_mean[tf.newaxis, :]
                B = tf.matmul(Xc, Xc, transpose_a=True) / tf.cast(N - 1, tf.float32)
                B = B * C_tf
                B = 0.5 * (B + tf.transpose(B))
                mean_var = tf.reduce_mean(tf.linalg.diag_part(B))
                B = B + tf.eye(D, dtype=tf.float32) * (1e-6 * mean_var + 1e-8)
                post, _, _ = assimilate_pff(
                    ens, y, obs_indices, obs_noise_std,
                    prior_mean, B,
                    kernel_type=kernel_type,
                    max_steps=pff_max_steps,
                    initial_step_size=pff_step_size,
                    convergence_tol=convergence_tol,
                )
                return post

            ens_matrix = analyze(ens_matrix, "diagonal")
            ens_scalar = analyze(ens_scalar, "scalar")

        for key, ens in [("noDA", ens_noda), ("scalar", ens_scalar), ("matrix", ens_matrix)]:
            a, b, c = rmse_components(ens, x_true)
            rmse[key]["all"].append(a)
            rmse[key]["obs"].append(b)
            rmse[key]["unobs"].append(c)

        if t % 100 == 0:
            print(f"  t={t}/{total_steps}: RMSE(all) noDA={rmse['noDA']['all'][-1]:.3f}  "
                  f"matrix={rmse['matrix']['all'][-1]:.3f}  scalar={rmse['scalar']['all'][-1]:.3f}")

    steps_list = list(range(total_steps + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    ax1.plot(steps_list, rmse["noDA"]["obs"], "k--", label="noDA (obs)")
    ax1.plot(steps_list, rmse["matrix"]["obs"], color="tab:blue", label="PFF matrix (obs)")
    ax1.plot(steps_list, rmse["scalar"]["obs"], color="tab:orange", label="PFF scalar (obs)")
    ax1.set_ylabel("RMSE (observed)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.plot(steps_list, rmse["noDA"]["unobs"], "k--", label="noDA (unobs)")
    ax2.plot(steps_list, rmse["matrix"]["unobs"], color="tab:blue", label="PFF matrix (unobs)")
    ax2.plot(steps_list, rmse["scalar"]["unobs"], color="tab:orange", label="PFF scalar (unobs)")
    ax2.set_xlabel("Model time step")
    ax2.set_ylabel("RMSE (unobserved)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    plt.suptitle(
        f"Sequential linear experiment (Hu & van Leeuwen 2021 style)\n"
        f"D={D}, N={N}, R=εI with ε={obs_noise_var}, assim every {assim_interval} steps",
    )
    plt.tight_layout()
    out_dir = Path("reports/3_Deterministic_Kernel_Flow/Hu(21)")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rmse_time_series_linear_D{D}_N{N}_T{total_steps}.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Saved RMSE plot: {out_path}")
    return rmse


# Output directory for figures and stats (same as Hu(21) report)
OUTPUT_DIR = Path("reports/3_Deterministic_Kernel_Flow/Hu(21)")


class _Tee:
    """Write to both a file and the original stream (e.g. stdout)."""

    def __init__(self, file_handle, stream):
        self._file = file_handle
        self._stream = stream

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUTPUT_DIR / "run_stats.txt"

    with open(stats_path, "w") as f:
        tee = _Tee(f, sys.stdout)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            print("\n" + "=" * 70)
            print("REPLICATING HU & VAN LEEUWEN (2021) FIGURES")
            print("TensorFlow implementation (no NumPy)")
            print("=" * 70)

            fig2_kernel_schematic()
            fig3_high_dimensional_lorenz_96()
            replicate_linear_sequential_rmse(total_steps=500)

            print("\n" + "=" * 70)
            print("REPLICATION COMPLETE")
            print("=" * 70)
        finally:
            sys.stdout = old_stdout

    print(f"✓ Terminal stats saved to: {stats_path}")

#python3 -m src.experiments.exp_hu_vanleeuwen_fig2_fig3