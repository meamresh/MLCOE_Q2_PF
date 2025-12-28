"""
Experiment: Particle Filter Degeneracy Diagnostics.

This experiment analyzes particle filter degeneracy by tracking and visualizing:
  1. Effective Sample Size (ESS) over time
  2. Weight distribution (histogram + cumulative) at most degenerate time
  3. Weight entropy over time
  4. Particle spread in 2D state space

All results are saved to CSV and PNG files in the reports directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ssm_range_bearing import RangeBearingSSM
from src.filters.particle_filter import ParticleFilter
from src.metrics.particle_filter_metrics import (
    compute_effective_sample_size,
    compute_weight_entropy,
    compute_weight_variance
)

tfd = tfp.distributions

# ---------------------------------------------------------------------
# Report directory structure
# ---------------------------------------------------------------------
OUTPUT_DIR = Path("reports/range_bearing/particle_degeneracy")


# ============================================================================
# DIAGNOSTIC VISUALIZATION CLASS
# ============================================================================

class ParticleFilterDiagnostics:
    """
    Diagnostic visualization for particle filter degeneracy.
    
    Tracks and visualizes:
      1. Effective Sample Size (ESS) over time
      2. Weight distribution (histogram + cumulative) at most degenerate time
      3. Weight entropy over time
      4. Particle spread in 2D state space
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    ess_history : list
        Effective sample size at each time step.
    entropy_history : list
        Normalized weight entropy at each time step.
    weight_variance_history : list
        Weight variance at each time step.
    resample_events : list
        Time steps where resampling occurred.
    max_weight_history : list
        Maximum particle weight at each time step.
    particle_snapshots : list
        Particle states at interesting time steps.
    weight_snapshots : list
        Particle weights at interesting time steps.
    state_snapshots : list
        State estimates at interesting time steps.
    snapshot_timesteps : list
        Time steps corresponding to snapshots.
    time_steps : list
        All time steps recorded.
    """

    def __init__(self):
        self.ess_history = []
        self.entropy_history = []
        self.weight_variance_history = []
        self.resample_events = []
        self.max_weight_history = []

        # Snapshots taken around interesting times (near resampling / periodically)
        self.particle_snapshots = []
        self.weight_snapshots = []
        self.state_snapshots = []
        self.snapshot_timesteps = []

        self.time_steps = []

    # ---------- update from PF loop ----------

    def update(self, pf: ParticleFilter, time_step: int, resampled: bool = False) -> None:
        """
        Update diagnostic metrics at each time step.

        Parameters
        ----------
        pf : ParticleFilter
            ParticleFilter instance.
        time_step : int
            Current time step.
        resampled : bool, optional
            Whether resampling happened at this step. Defaults to False.
        """
        weights = pf.weights
        particles = pf.particles
        state = pf.state

        ess = compute_effective_sample_size(weights)
        entropy = compute_weight_entropy(weights)
        weight_var = compute_weight_variance(weights)
        max_w = tf.reduce_max(weights)

        self.ess_history.append(ess.numpy())
        self.entropy_history.append(entropy.numpy())
        self.weight_variance_history.append(weight_var.numpy())
        self.max_weight_history.append(max_w.numpy())
        self.time_steps.append(time_step)

        if resampled:
            self.resample_events.append(time_step)

    # ---------- plotting ----------

    def plot_all_diagnostics(self, num_particles: int, resample_threshold: float = 0.5,
                             figsize: tuple[int, int] = (18, 12), save_path: Path | None = None) -> plt.Figure:
        """
        Create 4-panel diagnostic figure.

        Parameters
        ----------
        num_particles : int
            Number of particles.
        resample_threshold : float, optional
            Resampling threshold (0-1). Defaults to 0.5.
        figsize : tuple[int, int], optional
            Figure size. Defaults to (18, 12).
        save_path : Path, optional
            Path to save the figure. Defaults to None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            3, 2,
            height_ratios=[1.0, 1.0, 1.2],
            hspace=0.35, wspace=0.3
        )

        # 1. ESS over time
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_ess(ax1, num_particles, resample_threshold)

        # 2. Entropy over time
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_entropy(ax2)

        # 3. Weight distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_weight_distribution(ax3, ax4)

        # 4. Particle spread
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_particle_spread(ax5)

        plt.suptitle(
            'Particle Filter Degeneracy Diagnostics',
            fontsize=16, fontweight='bold', y=0.995
        )

        if save_path:
            os.makedirs(save_path.parent, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Diagnostic plot saved to {save_path}")

        plt.close()
        return fig

    def _plot_ess(self, ax: plt.Axes, num_particles: int, resample_threshold: float) -> None:
        """1. ESS over time with resampling markers."""
        ax.plot(self.time_steps, self.ess_history,
                'b-', linewidth=2, label='ESS')

        threshold = resample_threshold * num_particles
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Resample Threshold ({resample_threshold:.0%})')

        if self.resample_events:
            resample_ess = [
                self.ess_history[self.time_steps.index(t)]
                for t in self.resample_events if t in self.time_steps
            ]
            ax.scatter(self.resample_events, resample_ess,
                       c='red', s=80, marker='x', linewidths=2,
                       zorder=10, label='Resampling')

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('ESS', fontsize=12)
        ax.set_title('1. Effective Sample Size (ESS) Over Time',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim(0, num_particles * 1.1)

        mean_ess = tf.reduce_mean(tf.constant(self.ess_history, dtype=tf.float32)).numpy()
        min_ess = tf.reduce_min(tf.constant(self.ess_history, dtype=tf.float32)).numpy()
        ax.text(
            0.75, 0.25,
            f'Mean ESS: {mean_ess:.1f}\nMin ESS: {min_ess:.1f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    def _plot_entropy(self, ax: plt.Axes) -> None:
        """2. Normalized weight entropy over time."""
        ax.plot(self.time_steps, self.entropy_history,
                'g-', linewidth=2, label='Entropy')

        if self.resample_events:
            resample_ent = [
                self.entropy_history[self.time_steps.index(t)]
                for t in self.resample_events if t in self.time_steps
            ]
            ax.scatter(self.resample_events, resample_ent,
                       c='red', s=80, marker='x', linewidths=2,
                       zorder=10, label='Resampling')

        ax.axhline(y=1.0, color='blue', linestyle=':', linewidth=1.5,
                   alpha=0.5, label='Uniform (no degeneracy)')
        ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5,
                   alpha=0.5, label='Moderate degeneracy')

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Normalized Entropy', fontsize=12)
        ax.set_title('3. Weight Entropy Over Time',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=9)

        mean_ent = tf.reduce_mean(tf.constant(self.entropy_history, dtype=tf.float32)).numpy()
        min_ent = tf.reduce_min(tf.constant(self.entropy_history, dtype=tf.float32)).numpy()
        ax.text(
            0.75, 0.3,
            f'Mean: {mean_ent:.3f}\nMin: {min_ent:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        )

    def _plot_weight_distribution(self, ax_hist: plt.Axes, ax_cumsum: plt.Axes) -> None:
        """3. Weight histogram + cumulative distribution at most degenerate snapshot."""
        if len(self.weight_snapshots) == 0:
            return

        # Pick snapshot with minimum ESS (most degenerate)
        ess_values = [compute_effective_sample_size(tf.constant(w, dtype=tf.float32)).numpy() 
                     for w in self.weight_snapshots]
        idx = int(tf.argmin(tf.constant(ess_values, dtype=tf.float32)).numpy())
        weights = tf.constant(self.weight_snapshots[idx], dtype=tf.float32)
        ess_snapshot = ess_values[idx]

        weights_np = weights.numpy()

        # Histogram (log y-scale)
        ax_hist.hist(weights_np, bins=50, edgecolor='black',
                     alpha=0.7, color='steelblue')
        ax_hist.set_xlabel('Weight Value', fontsize=12)
        ax_hist.set_ylabel('Frequency (log scale)', fontsize=12)
        ax_hist.set_title(
            f'2. Weight Distribution (Most Degenerate Snapshot, ESS={ess_snapshot:.1f})',
            fontsize=13, fontweight='bold'
        )
        ax_hist.set_yscale('log')
        ax_hist.grid(True, alpha=0.3, axis='y')

        n_effective = int(compute_effective_sample_size(weights).numpy())
        max_weight = tf.reduce_max(weights).numpy()
        mean_weight = 1.0 / tf.cast(tf.size(weights), tf.float32).numpy()
        ax_hist.text(
            0.98, 0.98,
            f'Max weight: {max_weight:.4f}\n'
            f'N_eff: {n_effective}\n'
            f'Ratio max/mean: {max_weight/mean_weight:.1f}×',
            transform=ax_hist.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
        )

        # Cumulative distribution
        sorted_w = tf.sort(weights, direction='DESCENDING')
        cumsum_w = tf.cumsum(sorted_w)
        cumsum_w_np = cumsum_w.numpy()
        
        ax_cumsum.plot(cumsum_w_np, linewidth=2, color='darkblue')
        ax_cumsum.axhline(y=0.5, color='orange', linestyle='--',
                          linewidth=2, label='50% weight')
        ax_cumsum.axhline(y=0.95, color='red', linestyle='--',
                          linewidth=2, label='95% weight')

        # Find indices using TensorFlow where operation (more robust than searchsorted)
        cumsum_tf = tf.constant(cumsum_w_np, dtype=tf.float32)
        num_particles = tf.cast(tf.size(cumsum_tf), tf.int32)
        
        # Find first index where cumsum >= threshold using tf.where
        # Create boolean mask for values >= threshold
        mask_50 = cumsum_tf >= 0.5
        mask_95 = cumsum_tf >= 0.95
        
        # Find indices where condition is true
        indices_50 = tf.where(mask_50)
        indices_95 = tf.where(mask_95)
        
        # Get first index (if any exists), otherwise use last index
        if tf.size(indices_50) > 0:
            idx_50 = tf.cast(indices_50[0, 0], tf.int32)
        else:
            idx_50 = num_particles - 1
            
        if tf.size(indices_95) > 0:
            idx_95 = tf.cast(indices_95[0, 0], tf.int32)
        else:
            idx_95 = num_particles - 1
        
        # Clamp to valid range
        idx_50 = tf.clip_by_value(idx_50, 0, num_particles - 1)
        idx_95 = tf.clip_by_value(idx_95, 0, num_particles - 1)
        
        n_50 = int(idx_50.numpy()) + 1
        n_95 = int(idx_95.numpy()) + 1
        
        # Final bounds check
        n_50 = min(max(n_50, 1), len(cumsum_w_np))
        n_95 = min(max(n_95, 1), len(cumsum_w_np))
        
        ax_cumsum.axvline(x=n_50, color='orange', linestyle=':', alpha=0.5)
        ax_cumsum.axvline(x=n_95, color='red', linestyle=':', alpha=0.5)

        ax_cumsum.set_xlabel('Number of Particles (sorted by weight)', fontsize=12)
        ax_cumsum.set_ylabel('Cumulative Weight', fontsize=12)
        ax_cumsum.set_title('Cumulative Weight Distribution',
                            fontsize=13, fontweight='bold')
        ax_cumsum.grid(True, alpha=0.3)
        ax_cumsum.legend(loc='lower right', fontsize=10)
        ax_cumsum.set_ylim(0, 1.05)

        ax_cumsum.text(
            0.02, 0.98,
            f'Top {n_50} particles = 50% weight\n'
            f'Top {n_95} particles = 95% weight',
            transform=ax_cumsum.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
        )

    def _plot_particle_spread(self, ax: plt.Axes) -> None:
        """4. Particle spread in (x,y) at several snapshots."""
        if len(self.particle_snapshots) == 0:
            return

        n_snapshots = min(5, len(self.particle_snapshots))
        indices = tf.linspace(
            0.0, tf.cast(len(self.particle_snapshots) - 1, tf.float32),
            n_snapshots
        ).numpy().astype(int)
        colors = plt.cm.viridis(tf.linspace(0.0, 1.0, n_snapshots).numpy())

        scatter = None
        for i, idx in enumerate(indices):
            particles = tf.constant(self.particle_snapshots[idx], dtype=tf.float32)
            weights = tf.constant(self.weight_snapshots[idx], dtype=tf.float32)
            state = tf.constant(self.state_snapshots[idx], dtype=tf.float32)
            time_step = self.snapshot_timesteps[idx]

            particles_np = particles.numpy()
            weights_np = weights.numpy()
            state_np = state.numpy()

            sizes = (weights_np / tf.reduce_max(weights).numpy()) * 100.0 + 10.0

            scatter = ax.scatter(
                particles_np[:, 0], particles_np[:, 1],
                c=weights_np, s=sizes, cmap='hot',
                alpha=0.5, edgecolors='none',
                vmin=0, vmax=tf.reduce_max(weights).numpy() * 1.2
            )
            ax.scatter(
                state_np[0], state_np[1], c=[colors[i]], s=300,
                marker='*', edgecolors='black', linewidths=2,
                label=f't={time_step}', zorder=10
            )

            if i == n_snapshots - 1:
                self._plot_covariance_ellipse(ax, particles, weights, colors[i])

        ax.set_xlabel('x position [m]', fontsize=12)
        ax.set_ylabel('y position [m]', fontsize=12)
        ax.set_title('4. Particle Spread in State Space (Position Only)',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=n_snapshots)

        if scatter is not None:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label('Particle Weight', fontsize=11)

    def _plot_covariance_ellipse(self, ax: plt.Axes, particles: tf.Tensor,
                                  weights: tf.Tensor, color: tuple, n_std: float = 2.0) -> None:
        """Draw weighted covariance ellipse for particle cloud in (x,y)."""
        # Compute weighted mean
        weights_expanded = tf.expand_dims(weights, axis=1)
        mean = tf.reduce_sum(weights_expanded * particles[:, :2], axis=0)
        
        # Compute weighted covariance
        diff = particles[:, :2] - mean
        weighted_diff = weights_expanded * diff
        cov = tf.matmul(weighted_diff, diff, transpose_a=True)
        
        # Ensure symmetry
        cov = 0.5 * (cov + tf.transpose(cov))
        
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = tf.linalg.eigh(cov)
        
        # Get angle from first eigenvector
        angle_rad = tf.math.atan2(eigvecs[1, 0], eigvecs[0, 0])
        angle_deg = tf.cast(angle_rad * 180.0 / tf.constant(3.141592653589793, dtype=tf.float32), tf.float32)
        
        # Get ellipse dimensions
        width = 2 * n_std * tf.sqrt(eigvals[1])
        height = 2 * n_std * tf.sqrt(eigvals[0])
        
        mean_np = mean.numpy()
        width_np = width.numpy()
        height_np = height.numpy()
        angle_deg_np = angle_deg.numpy()

        ellipse = Ellipse(
            xy=mean_np, width=width_np, height=height_np, angle=angle_deg_np,
            facecolor='none', edgecolor=color, linewidth=3,
            linestyle='--', label=f'{n_std}σ ellipse', zorder=5
        )
        ax.add_patch(ellipse)

    # ---------- summary and CSV export ----------

    def print_summary(self) -> None:
        """Print summary statistics of collected diagnostics."""
        print("\n" + "=" * 60)
        print("PARTICLE FILTER DEGENERACY SUMMARY")
        print("=" * 60)

        if self.ess_history:
            ess_tf = tf.constant(self.ess_history, dtype=tf.float32)
            print("\nEffective Sample Size (ESS):")
            print(f"  Mean: {tf.reduce_mean(ess_tf).numpy():.2f}")
            print(f"  Min:  {tf.reduce_min(ess_tf).numpy():.2f}")
            print(f"  Max:  {tf.reduce_max(ess_tf).numpy():.2f}")
            print(f"  Std:  {tf.math.reduce_std(ess_tf).numpy():.2f}")

        if self.entropy_history:
            ent_tf = tf.constant(self.entropy_history, dtype=tf.float32)
            print("\nNormalized Weight Entropy:")
            print(f"  Mean: {tf.reduce_mean(ent_tf).numpy():.4f}")
            print(f"  Min:  {tf.reduce_min(ent_tf).numpy():.4f}")
            print(f"  Max:  {tf.reduce_max(ent_tf).numpy():.4f}")

        print(f"\nResampling Events: {len(self.resample_events)}")
        print(f"Total Time Steps:  {len(self.time_steps)}")
        if self.time_steps:
            rate = len(self.resample_events) / len(self.time_steps) * 100.0
        else:
            rate = 0.0
        print(f"Resampling Rate:   {rate:.1f}%")

        if self.max_weight_history:
            max_w_tf = tf.constant(self.max_weight_history, dtype=tf.float32)
            print("\nMaximum Particle Weight:")
            print(f"  Mean: {tf.reduce_mean(max_w_tf).numpy():.6f}")
            print(f"  Max:  {tf.reduce_max(max_w_tf).numpy():.6f}")

        print("=" * 60 + "\n")

    def save_to_csv(self, output_dir: Path) -> None:
        """
        Save diagnostic data to CSV files.

        Parameters
        ----------
        output_dir : Path
            Directory to save CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save time series data
        csv_path = output_dir / 'diagnostics_timeseries.csv'
        with open(csv_path, 'w') as f:
            f.write("time_step,ess,entropy,weight_variance,max_weight\n")
            for i, t in enumerate(self.time_steps):
                f.write(f"{t},{self.ess_history[i]:.6f},{self.entropy_history[i]:.6f},"
                       f"{self.weight_variance_history[i]:.6f},{self.max_weight_history[i]:.6f}\n")
        print(f"✓ Saved: {csv_path}")

        # Save resampling events
        csv_path = output_dir / 'resampling_events.csv'
        with open(csv_path, 'w') as f:
            f.write("time_step\n")
            for t in self.resample_events:
                f.write(f"{t}\n")
        print(f"✓ Saved: {csv_path}")


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def generate_circular_trajectory(num_steps: int = 150, dt: float = 0.1,
                                seed: int | None = None) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Generate a simple curved trajectory for testing.

    Parameters
    ----------
    num_steps : int, optional
        Number of time steps. Defaults to 150.
    dt : float, optional
        Time step duration. Defaults to 0.1.
    seed : int, optional
        Random seed for reproducibility. Defaults to None.

    Returns
    -------
    trajectory : tf.Tensor
        True trajectory of shape (num_steps + 1, 3).
    controls : tf.Tensor
        Control inputs of shape (num_steps, 2).
    """
    if seed is not None:
        tf.random.set_seed(seed)

    trajectory = []
    controls = []

    state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    trajectory.append(state)

    for t in range(num_steps):
        if t < 40:
            v, omega = 1.0, 0.15
        elif t < 80:
            v, omega = 1.2, -0.1
        elif t < 120:
            v, omega = 0.8, 0.2
        else:
            v, omega = 1.0, 0.05

        controls.append([v, omega])

        # Update state
        state = tf.stack([
            state[0] + v * dt * tf.cos(state[2]),
            state[1] + v * dt * tf.sin(state[2]),
            state[2] + omega * dt
        ])
        
        # Add process noise
        noise = tf.random.normal([3], stddev=0.05, dtype=tf.float32)
        state = state + noise

        trajectory.append(state)

    return tf.stack(trajectory), tf.constant(controls, dtype=tf.float32)


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_particle_filter_with_diagnostics(
    num_steps: int = 150,
    num_particles: int = 1000,
    resample_threshold: float = 0.5,
    seed: int | None = None,
    output_dir: Path | None = None
) -> tuple[ParticleFilterDiagnostics, ParticleFilter, tf.Tensor]:
    """
    Run particle filter simulation with degeneracy diagnostics.

    Parameters
    ----------
    num_steps : int, optional
        Number of simulation steps. Defaults to 150.
    num_particles : int, optional
        Number of particles. Defaults to 1000.
    resample_threshold : float, optional
        Resampling threshold (0-1). Defaults to 0.5.
    seed : int, optional
        Random seed for reproducibility. Defaults to None.
    output_dir : Path, optional
        Directory to save results. Defaults to OUTPUT_DIR.

    Returns
    -------
    diagnostics : ParticleFilterDiagnostics
        Diagnostic data and visualizations.
    pf : ParticleFilter
        The particle filter instance.
    true_trajectory : tf.Tensor
        True trajectory of shape (num_steps + 1, 3).
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    if seed is not None:
        tf.random.set_seed(seed)

    print("Setting up particle filter simulation...")

    dt = 0.1
    process_noise = tf.eye(3, dtype=tf.float32) * 0.01
    meas_noise = tf.eye(2, dtype=tf.float32) * 0.1
    ssm = RangeBearingSSM(dt=dt, process_noise=process_noise, meas_noise=meas_noise)

    # Landmarks
    landmarks = tf.constant(
        [[0.5, 0.5], [5.0, 5.0], [1.0, -1.0]],
        dtype=tf.float32
    )

    print("Generating true trajectory...")
    true_trajectory, controls = generate_circular_trajectory(num_steps, dt, seed=seed)

    initial_state = true_trajectory[0] + tf.random.normal([3], stddev=0.5, dtype=tf.float32)
    initial_cov = tf.eye(3, dtype=tf.float32) * 0.5

    pf = ParticleFilter(
        ssm, initial_state, initial_cov,
        num_particles=num_particles,
        resample_threshold=resample_threshold
    )

    diagnostics = ParticleFilterDiagnostics()

    print(f"Running particle filter for {num_steps} steps...")
    for t in range(num_steps):
        if t % 20 == 0:
            print(f"  Step {t}/{num_steps}")

        true_state = true_trajectory[t]
        control = controls[t]

        # Measurement from true state
        meas = ssm.measurement_model(
            true_state[tf.newaxis, :],
            landmarks
        )[0]
        meas = meas + tf.random.normal(tf.shape(meas), stddev=0.1, dtype=tf.float32)

        # Pre-update snapshot (for weight distribution)
        weights_before = tf.identity(pf.weights)
        ess_before = pf._effective_sample_size()

        # Predict + update
        pf.predict(control)
        state, cov, residual, resampled = pf.update(meas, landmarks)

        # Store pre-update snapshot when close to threshold
        if ess_before < resample_threshold * num_particles * 1.1:
            diagnostics.particle_snapshots.append(pf.particles.numpy().copy())
            diagnostics.weight_snapshots.append(weights_before.numpy().copy())
            diagnostics.state_snapshots.append(pf.state.numpy().copy())
            diagnostics.snapshot_timesteps.append(t)

        diagnostics.update(pf, t, resampled)

    print("Simulation complete!")
    diagnostics.print_summary()

    # Save CSV files
    diagnostics.save_to_csv(output_dir)

    # Generate and save diagnostic plots
    print("Generating diagnostic plots...")
    plot_path = output_dir / 'particle_filter_diagnostics.png'
    diagnostics.plot_all_diagnostics(
        num_particles=num_particles,
        resample_threshold=resample_threshold,
        save_path=plot_path
    )

    return diagnostics, pf, true_trajectory


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    diagnostics, pf, true_traj = run_particle_filter_with_diagnostics(
        num_steps=150,
        num_particles=1000,
        resample_threshold=0.5,
        seed=42,
        output_dir=OUTPUT_DIR
    )
    
    print("\nDiagnostic visualization complete.")
    print(f"Results saved to: {OUTPUT_DIR}")

