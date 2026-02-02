"""
Experiment: Runtime and Memory Profiling for EKF, UKF, and Particle Filter.

This experiment provides comprehensive runtime and memory profiling for all
three filter types, comparing their performance characteristics including:
- Computation time (predict/update steps)
- Memory usage (CPU and GPU)
- Accuracy metrics (RMSE, NEES, NIS)
- Filter consistency
- Scaling analysis for particle filters

All results are saved to CSV and PNG files in the reports directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import psutil

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ssm_range_bearing import RangeBearingSSM
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.filters.particle_filter import ParticleFilter
from src.metrics.accuracy import compute_rmse, compute_nees, compute_nis, compute_nis

tfd = tfp.distributions

# ---------------------------------------------------------------------
# Report directory structure
# ---------------------------------------------------------------------
OUTPUT_DIR = Path("reports/2_Nonlinear_NonGaussianSSM/EKF_UKF_PF_Comparison")


# ============================================================================
# PROFILING AND MEMORY TRACKING UTILITIES
# ============================================================================

@dataclass
class ProfileMetrics:
    """Storage for profiling metrics."""
    predict_times: List[float]
    update_times: List[float]
    total_time: float
    peak_memory_mb: float
    gpu_memory_mb: float = 0.0
    
    def mean_predict_time(self) -> float:
        """Compute mean predict time in seconds."""
        if not self.predict_times:
            return 0.0
        return float(tf.reduce_mean(tf.constant(self.predict_times, dtype=tf.float32)).numpy())
    
    def mean_update_time(self) -> float:
        """Compute mean update time in seconds."""
        if not self.update_times:
            return 0.0
        return float(tf.reduce_mean(tf.constant(self.update_times, dtype=tf.float32)).numpy())
    
    def total_step_time(self) -> float:
        """Compute total step time (predict + update)."""
        return self.mean_predict_time() + self.mean_update_time()


class MemoryTracker:
    """Track CPU and GPU memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0.0
        self.has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    
    def get_current_memory_mb(self) -> float:
        """Get current CPU memory in MB."""
        try:
            return float(self.process.memory_info().rss / 1024 / 1024)
        except:
            return 0.0
    
    def get_gpu_memory_mb(self) -> float:
        """Get current GPU memory in MB (if available)."""
        if not self.has_gpu:
            return 0.0
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                return float(mem_info['current'] / 1024 / 1024)
        except:
            return 0.0
    
    def update_peak(self):
        """Update peak memory usage."""
        current = self.get_current_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
    
    def reset(self):
        """Reset peak memory tracking."""
        self.peak_memory = self.get_current_memory_mb()


@contextmanager
def profile_filter_run(filter_name: str):
    """Context manager for profiling a complete filter run."""
    tracker = MemoryTracker()
    tracker.reset()
    
    predict_times = []
    update_times = []
    
    start_time = time.perf_counter()
    
    class Profiler:
        def __init__(self):
            self.predict_times = predict_times
            self.update_times = update_times
            self.tracker = tracker
        
        @contextmanager
        def time_predict(self):
            tracker.update_peak()
            t0 = time.perf_counter()
            yield
            self.predict_times.append(time.perf_counter() - t0)
        
        @contextmanager
        def time_update(self):
            tracker.update_peak()
            t0 = time.perf_counter()
            yield
            self.update_times.append(time.perf_counter() - t0)
    
    profiler = Profiler()
    
    try:
        yield profiler
    finally:
        total_time = time.perf_counter() - start_time
        peak_memory = tracker.peak_memory
        gpu_memory = tracker.get_gpu_memory_mb()
        
        metrics = ProfileMetrics(
            predict_times=predict_times,
            update_times=update_times,
            total_time=total_time,
            peak_memory_mb=peak_memory,
            gpu_memory_mb=gpu_memory
        )
        
        print(f"\n{filter_name} Profile:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg predict: {metrics.mean_predict_time()*1000:.2f}ms")
        print(f"  Avg update: {metrics.mean_update_time()*1000:.2f}ms")
        print(f"  Peak CPU RAM: {peak_memory:.1f}MB")
        if gpu_memory > 0:
            print(f"  GPU RAM: {gpu_memory:.1f}MB")


# ============================================================================
# EVALUATION METRICS (Using Existing Metrics Module)
# ============================================================================

class FilterEvaluator:
    """Comprehensive filter evaluation metrics using TensorFlow."""
    
    @staticmethod
    def convergence_time(errors: tf.Tensor, threshold: float = 0.1) -> int:
        """
        Compute convergence time: first step where error stays below threshold.
        
        Returns -1 if never converges.
        """
        errors = tf.cast(errors, tf.float32)
        n_steps = tf.shape(errors)[0]
        
        for i in tf.range(n_steps):
            remaining = errors[i:]
            if tf.reduce_all(remaining < threshold):
                return int(i.numpy())
        return -1
    
    @staticmethod
    def compute_consistency_percentage(nees_array: tf.Tensor, 
                                      state_dim: int, 
                                      alpha: float = 0.05) -> float:
        """
        Compute percentage of NEES values within confidence bounds.
        
        Uses TensorFlow Probability for chi-squared distribution.
        """
        nees_array = tf.cast(nees_array, tf.float32)
        
        # Chi-squared distribution with state_dim degrees of freedom
        chi2_dist = tfd.Chi2(df=float(state_dim))
        
        # Compute confidence bounds
        lower = chi2_dist.quantile(alpha/2)
        upper = chi2_dist.quantile(1 - alpha/2)
        
        # Filter valid NEES values
        valid_mask = tf.math.is_finite(nees_array)
        valid_nees = tf.boolean_mask(nees_array, valid_mask)
        
        # Count values within bounds
        within_bounds = tf.logical_and(valid_nees >= lower, valid_nees <= upper)
        count_within = tf.reduce_sum(tf.cast(within_bounds, tf.float32))
        total_count = tf.cast(tf.shape(valid_nees)[0], tf.float32)
        
        percentage = (count_within / total_count) * 100.0 if total_count > 0 else 0.0
        
        return float(percentage.numpy())


# ============================================================================
# ENHANCED SIMULATION WITH PROFILING
# ============================================================================

def simulate_with_profiling(ssm: RangeBearingSSM, landmarks: tf.Tensor, 
                           controls: List[tf.Tensor], measurements: List[tf.Tensor], 
                           true_states: tf.Tensor, initial_state: tf.Tensor, 
                           initial_covariance: tf.Tensor, num_particles: int = 3000) -> Dict:
    """
    Run all filters with comprehensive profiling and metric collection.
    
    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    landmarks : tf.Tensor
        Landmark positions.
    controls : List[tf.Tensor]
        Control inputs for each time step.
    measurements : List[tf.Tensor]
        Measurements for each time step.
    true_states : tf.Tensor
        True states for each time step.
    initial_state : tf.Tensor
        Initial state estimate.
    initial_covariance : tf.Tensor
        Initial covariance matrix.
    num_particles : int, optional
        Number of particles for PF. Defaults to 3000.
    
    Returns
    -------
    results : Dict
        Dictionary containing filter results and profiles.
    """
    results = {}
    
    # ========== EKF ==========
    print("\n" + "="*60)
    print("Running EKF with profiling...")
    print("="*60)
    
    ekf = ExtendedKalmanFilter(ssm, initial_state, initial_covariance)
    
    ekf_states = []
    ekf_covs = []
    ekf_innovations = []
    ekf_innovation_covs = []
    
    with profile_filter_run("EKF") as profiler:
        for step in range(len(controls)):
            with profiler.time_predict():
                ekf.predict(controls[step])
            
            with profiler.time_update():
                _, _, residual = ekf.update(measurements[step], landmarks)
            
            ekf_states.append(tf.identity(ekf.state))
            ekf_covs.append(tf.identity(ekf.covariance))
            ekf_innovations.append(residual)
            
            # Compute innovation covariance (for NIS)
            H = ssm.measurement_jacobian(ekf.state, landmarks)[0]
            S = H @ ekf.covariance @ tf.transpose(H) + ssm.full_measurement_cov(tf.shape(landmarks)[0])
            ekf_innovation_covs.append(S)
    
    results['ekf'] = {
        'states': tf.stack(ekf_states),
        'covariances': tf.stack(ekf_covs),
        'innovations': ekf_innovations,
        'innovation_covs': ekf_innovation_covs,
        'profile': ProfileMetrics(
            predict_times=profiler.predict_times,
            update_times=profiler.update_times,
            total_time=sum(profiler.predict_times) + sum(profiler.update_times),
            peak_memory_mb=profiler.tracker.peak_memory
        )
    }
    
    # ========== UKF ==========
    print("\n" + "="*60)
    print("Running UKF with profiling...")
    print("="*60)
    
    ukf = UnscentedKalmanFilter(ssm, initial_state, initial_covariance,
                                alpha=1e-1, beta=1.0, kappa=0)
    
    ukf_states = []
    ukf_covs = []
    ukf_innovations = []
    ukf_innovation_covs = []
    
    with profile_filter_run("UKF") as profiler:
        for step in range(len(controls)):
            with profiler.time_predict():
                ukf.predict(controls[step])
            
            with profiler.time_update():
                _, _, residual = ukf.update(measurements[step], landmarks)
            
            ukf_states.append(tf.identity(ukf.state))
            ukf_covs.append(tf.identity(ukf.covariance))
            ukf_innovations.append(residual)
            
            # Approximate innovation covariance
            num_landmarks = tf.shape(landmarks)[0]
            H_approx = ssm.measurement_jacobian(ukf.state, landmarks)[0]
            S = H_approx @ ukf.covariance @ tf.transpose(H_approx) + ssm.full_measurement_cov(num_landmarks)
            ukf_innovation_covs.append(S)
    
    results['ukf'] = {
        'states': tf.stack(ukf_states),
        'covariances': tf.stack(ukf_covs),
        'innovations': ukf_innovations,
        'innovation_covs': ukf_innovation_covs,
        'profile': ProfileMetrics(
            predict_times=profiler.predict_times,
            update_times=profiler.update_times,
            total_time=sum(profiler.predict_times) + sum(profiler.update_times),
            peak_memory_mb=profiler.tracker.peak_memory
        )
    }
    
    # ========== PF ==========
    print("\n" + "="*60)
    print(f"Running PF ({num_particles} particles) with profiling...")
    print("="*60)
    
    pf = ParticleFilter(ssm, initial_state, initial_covariance,
                       num_particles=num_particles)
    
    pf_states = []
    pf_covs = []
    pf_innovations = []
    pf_innovation_covs = []
    
    with profile_filter_run("PF") as profiler:
        for step in range(len(controls)):
            with profiler.time_predict():
                pf.predict(controls[step])
            
            with profiler.time_update():
                _, _, residual, _ = pf.update(measurements[step], landmarks)
            
            pf_states.append(tf.identity(pf.state))
            pf_covs.append(tf.identity(pf.covariance))
            pf_innovations.append(residual)
            
            # Simplified innovation covariance
            num_landmarks = tf.shape(landmarks)[0]
            S_full = tf.eye(2 * num_landmarks, dtype=tf.float32) * 0.01
            pf_innovation_covs.append(S_full)
    
    results['pf'] = {
        'states': tf.stack(pf_states),
        'covariances': tf.stack(pf_covs),
        'innovations': pf_innovations,
        'innovation_covs': pf_innovation_covs,
        'profile': ProfileMetrics(
            predict_times=profiler.predict_times,
            update_times=profiler.update_times,
            total_time=sum(profiler.predict_times) + sum(profiler.update_times),
            peak_memory_mb=profiler.tracker.peak_memory
        )
    }
    
    return results


def comprehensive_evaluation(results: Dict, true_states: tf.Tensor, 
                            state_dim: int = 3) -> Dict:
    """
    Compute all evaluation metrics for all filters.
    
    Uses existing metrics functions from accuracy module.
    
    Returns
    -------
    summary_data : Dict
        Dictionary with metrics for each filter.
    """
    true_states = tf.cast(true_states, dtype=tf.float32)
    evaluator = FilterEvaluator()
    summary_data = {}
    
    for filter_name, data in results.items():
        estimates = data['states']
        covariances = data['covariances']
        profile = data['profile']
        
        # RMSE (using existing function)
        pos_rmse = compute_rmse(estimates[:, :2], true_states[:, :2])
        
        # Position and heading errors for detailed metrics
        pos_errors = tf.linalg.norm(estimates[:, :2] - true_states[:, :2], axis=1)
        heading_diff = estimates[:, 2] - true_states[:, 2]
        heading_errors = tf.abs(tf.math.atan2(tf.sin(heading_diff), tf.cos(heading_diff)))
        heading_rmse = tf.sqrt(tf.reduce_mean(heading_errors**2))
        
        # NEES (using existing function)
        nees_values = compute_nees(estimates, covariances, true_states)
        mean_nees = tf.reduce_mean(nees_values)
        
        # NIS (using existing function)
        # compute_nis expects: innovations as list of (n_landmarks, n_meas) tensors
        # and innovation_covariances as list of lists, where each inner list has
        # one covariance matrix per landmark
        if len(data['innovations']) > 0:
            # Check shape of first innovation to determine format
            first_innov = data['innovations'][0]
            innov_rank = tf.rank(first_innov)
            
            # If innovation is flattened (1D), reshape it
            if int(innov_rank.numpy()) == 1:
                # Flattened: (2*n_landmarks,) -> reshape to (n_landmarks, 2)
                innov_size = int(tf.shape(first_innov)[0].numpy())
                num_landmarks = innov_size // 2
                innovations_reshaped = []
                for innov in data['innovations']:
                    # Ensure we have the right shape: (num_landmarks, 2)
                    innov_reshaped = tf.reshape(innov, [num_landmarks, 2])
                    innovations_reshaped.append(innov_reshaped)
            else:
                # Already in correct format: (n_landmarks, 2)
                innovations_reshaped = data['innovations']
                num_landmarks = int(tf.shape(first_innov)[0].numpy())
        else:
            num_landmarks = 1
            innovations_reshaped = data['innovations']
        
        # Convert innovation_covs to per-landmark format
        innovation_covs_per_landmark = []
        for full_cov in data['innovation_covs']:
            # Split full covariance into per-landmark covariances
            # Full cov is (2*n_landmarks, 2*n_landmarks), need (n_landmarks, 2, 2)
            landmark_covs = []
            for i in range(num_landmarks):
                start_idx = i * 2
                end_idx = (i + 1) * 2
                landmark_cov = full_cov[start_idx:end_idx, start_idx:end_idx]
                landmark_covs.append(landmark_cov)
            innovation_covs_per_landmark.append(landmark_covs)
        
        nis_array, nis_lower, nis_upper = compute_nis(
            innovations_reshaped, 
            innovation_covs_per_landmark,
            n_meas_per_landmark=2
        )
        mean_nis = tf.reduce_mean(nis_array)
        
        # Convergence
        conv_time = evaluator.convergence_time(pos_errors)
        
        # Consistency
        consistency = evaluator.compute_consistency_percentage(nees_values, state_dim)
        
        # Final errors
        final_pos_error = pos_errors[-1]
        final_heading_error = heading_errors[-1]
        
        summary_data[filter_name] = {
            'Filter': filter_name.upper(),
            'Pos RMSE [m]': pos_rmse,
            'Head RMSE [rad]': float(heading_rmse.numpy()),
            'Final Pos Error [m]': float(final_pos_error.numpy()),
            'Final Head Error [rad]': float(final_heading_error.numpy()),
            'Mean NEES': float(mean_nees.numpy()),
            'Consistency [%]': consistency,
            'Mean NIS': float(mean_nis.numpy()),
            'Convergence Step': conv_time if conv_time >= 0 else 'Never',
            'Total Time [s]': profile.total_time,
            'Avg Predict [ms]': profile.mean_predict_time() * 1000,
            'Avg Update [ms]': profile.mean_update_time() * 1000,
            'Peak RAM [MB]': profile.peak_memory_mb,
            'GPU RAM [MB]': profile.gpu_memory_mb
        }
    
    return summary_data


def print_evaluation_table(summary_data: Dict) -> None:
    """Print evaluation results in a formatted table."""
    print("\n" + "="*140)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*140)
    
    # Print header
    header = f"{'Filter':<8} {'Pos RMSE [m]':<14} {'Head RMSE':<12} {'Final Pos':<12} {'Final Head':<12} " \
             f"{'Mean NEES':<11} {'Consist[%]':<11} {'Mean NIS':<10} {'Conv Step':<11} " \
             f"{'Time[s]':<9} {'Pred[ms]':<10} {'Upd[ms]':<10} {'RAM[MB]':<9}"
    print(header)
    print("="*140)
    
    # Print data
    for filter_name, data in summary_data.items():
        row = f"{data['Filter']:<8} " \
              f"{data['Pos RMSE [m]']:<14.4f} " \
              f"{data['Head RMSE [rad]']:<12.4f} " \
              f"{data['Final Pos Error [m]']:<12.4f} " \
              f"{data['Final Head Error [rad]']:<12.4f} " \
              f"{data['Mean NEES']:<11.2f} " \
              f"{data['Consistency [%]']:<11.1f} " \
              f"{data['Mean NIS']:<10.2f} " \
              f"{str(data['Convergence Step']):<11} " \
              f"{data['Total Time [s]']:<9.3f} " \
              f"{data['Avg Predict [ms]']:<10.2f} " \
              f"{data['Avg Update [ms]']:<10.2f} " \
              f"{data['Peak RAM [MB]']:<9.1f}"
        print(row)
    
    print("="*140)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comprehensive_comparison(results: Dict, true_states: tf.Tensor, 
                                 landmarks: tf.Tensor, eval_data: Dict,
                                 save_path: Path | None = None) -> None:
    """Create comprehensive comparison plots."""
    
    os.makedirs(save_path.parent if save_path else OUTPUT_DIR, exist_ok=True)
    
    # Convert tensors for plotting
    true_states_np = true_states.numpy()
    landmarks_np = landmarks.numpy()
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {'ekf': 'blue', 'ukf': 'red', 'pf': 'green'}
    
    # 1. Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(true_states_np[:, 0], true_states_np[:, 1], 'k--', label='True', linewidth=2.5, alpha=0.7)
    for name, data in results.items():
        states_np = data['states'].numpy()
        ax1.plot(states_np[:, 0], states_np[:, 1], 
                color=colors[name], label=name.upper(), linewidth=1.5, alpha=0.8)
    ax1.scatter(landmarks_np[:, 0], landmarks_np[:, 1], c='orange', marker='^', 
               s=150, label='Landmarks', zorder=5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Trajectory Comparison')
    ax1.axis('equal')
    
    # 2. Position Error Over Time
    ax2 = fig.add_subplot(gs[0, 1])
    evaluator = FilterEvaluator()
    for name, data in results.items():
        pos_errors = tf.linalg.norm(data['states'][:, :2] - true_states[:, :2], axis=1)
        errors_np = pos_errors.numpy()
        ax2.plot(errors_np, color=colors[name], label=name.upper(), linewidth=1.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error [m]')
    ax2.set_title('Position Error Trajectory')
    
    # 3. NEES Over Time
    ax3 = fig.add_subplot(gs[0, 2])
    for name, data in results.items():
        nees_values = compute_nees(data['states'], data['covariances'], true_states)
        nees_np = nees_values.numpy()
        valid_nees = nees_np[tf.math.is_finite(nees_np).numpy()]
        ax3.plot(valid_nees, color=colors[name], label=name.upper(), alpha=0.7)
    ax3.axhline(3, color='black', linestyle='--', label='Expected (dim=3)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('NEES')
    ax3.set_title('Filter Consistency (NEES)')
    
    # 4. Computation Time Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    filters = [name.upper() for name in results.keys()]
    predict_times = [results[name]['profile'].mean_predict_time() * 1000 
                    for name in results.keys()]
    update_times = [results[name]['profile'].mean_update_time() * 1000 
                   for name in results.keys()]
    
    x_pos = tf.range(len(filters), dtype=tf.float32).numpy()
    width = 0.35
    ax4.bar(x_pos - width/2, predict_times, width, label='Predict', alpha=0.8)
    ax4.bar(x_pos + width/2, update_times, width, label='Update', alpha=0.8)
    ax4.set_ylabel('Time [ms]')
    ax4.set_title('Average Step Time')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(filters)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Memory Usage
    ax5 = fig.add_subplot(gs[1, 1])
    memory = [results[name]['profile'].peak_memory_mb for name in results.keys()]
    bars = ax5.bar(filters, memory, color=[colors[name] for name in results.keys()], alpha=0.7)
    ax5.set_ylabel('Peak Memory [MB]')
    ax5.set_title('Memory Usage')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 6. Error Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    error_data = []
    for name in results.keys():
        pos_errors = tf.linalg.norm(results[name]['states'][:, :2] - true_states[:, :2], axis=1)
        errors_np = pos_errors.numpy()
        error_data.append(errors_np)
    bp = ax6.boxplot(error_data, tick_labels=filters, patch_artist=True)
    for patch, name in zip(bp['boxes'], results.keys()):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylabel('Position Error [m]')
    ax6.set_title('Error Distribution')
    
    # 7. Performance Table
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.axis('off')
    
    # Prepare table data
    table_data = []
    for name in ['ekf', 'ukf', 'pf']:
        if name in eval_data:
            data = eval_data[name]
            table_data.append([
                data['Filter'],
                f"{data['Pos RMSE [m]']:.3f}",
                f"{data['Mean NEES']:.2f}",
                f"{data['Consistency [%]']:.1f}",
                f"{data['Total Time [s]']:.3f}",
                f"{data['Peak RAM [MB]']:.1f}"
            ])
    
    if table_data:
        table = ax7.table(cellText=table_data,
                         colLabels=['Filter', 'Pos RMSE [m]', 'Mean NEES', 'Consistency [%]', 
                                   'Total Time [s]', 'Peak RAM [MB]'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(len(table_data)):
            filter_name = table_data[i][0].lower()
            if filter_name in colors:
                table[(i+1, 0)].set_facecolor(colors[filter_name])
                table[(i+1, 0)].set_alpha(0.3)
    
    ax7.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    # 8. Winner Summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Find winners
    if eval_data:
        best_accuracy = min(eval_data.items(), key=lambda x: x[1]['Pos RMSE [m]'])[0].upper()
        best_speed = min(eval_data.items(), key=lambda x: x[1]['Total Time [s]'])[0].upper()
        best_memory = min(eval_data.items(), key=lambda x: x[1]['Peak RAM [MB]'])[0].upper()
        best_consistency = max(eval_data.items(), key=lambda x: x[1]['Consistency [%]'])[0].upper()
        
        lines = [
            "WINNERS",
            "=" * 30,
            "",
            f"Best Accuracy: {best_accuracy}",
            "",
            f"Best Speed: {best_speed}",
            "",
            f"Best Memory: {best_memory}",
            "",
            f"Best Consistency: {best_consistency}",
        ]
        
        y_position = 0.95
        line_height = 0.06
        
        for line in lines:
            ax8.text(0.1, y_position, line, transform=ax8.transAxes,
                    fontsize=11, verticalalignment='top',
                    fontfamily='monospace')
            y_position -= line_height
    
    plt.suptitle('Comprehensive Filter Comparison', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_comprehensive_evaluation(num_steps: int = 200, num_particles: int = 10000,
                               seed: int | None = None, output_dir: Path | None = None) -> Dict:
    """
    Run complete evaluation with all metrics.
    
    Parameters
    ----------
    num_steps : int, optional
        Number of simulation steps. Defaults to 200.
    num_particles : int, optional
        Number of particles for PF. Defaults to 10000.
    seed : int, optional
        Random seed. Defaults to None.
    output_dir : Path, optional
        Output directory. Defaults to OUTPUT_DIR.
    
    Returns
    -------
    results : Dict
        Complete results dictionary.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SSM EVALUATION FRAMEWORK (TensorFlow Native)")
    print("="*80)
    
    if seed is not None:
        tf.random.set_seed(seed)
    
    # Scenario parameters
    landmarks = tf.constant([[-1.5, -1.5]], dtype=tf.float32)
    dt = 0.1
    Q = tf.eye(3, dtype=tf.float32) * 0.03
    R = tf.eye(2, dtype=tf.float32) * 0.01
    initial_uncertainty = tf.eye(3, dtype=tf.float32) * 1.0
    
    ssm = RangeBearingSSM(dt=dt, process_noise=Q, meas_noise=R)
    
    # Generate trajectory
    print("\nGenerating ground truth trajectory...")
    true_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    true_states = []
    controls = []
    measurements = []
    
    for step in range(num_steps):
        step_tf = tf.cast(step, tf.float32)
        v = 1.5 + 0.5 * tf.sin(step_tf * 0.15)
        omega = 0.6 + 0.3 * tf.cos(step_tf * 0.12)
        control = tf.stack([v, omega])
        controls.append(control)
        
        true_state = ssm.motion_model(true_state[tf.newaxis, :], control[tf.newaxis, :])[0]
        process_noise = tf.random.normal([3], mean=0.0, stddev=0.17, dtype=tf.float32)
        true_state = true_state + process_noise
        true_states.append(true_state)
        
        true_meas = ssm.measurement_model(true_state[tf.newaxis, :], landmarks)[0]
        meas_std = tf.sqrt(tf.linalg.diag_part(ssm.R))
        meas_noise = tf.random.normal([tf.shape(landmarks)[0], 2], mean=0.0, 
                                     stddev=meas_std, dtype=tf.float32)
        measured = true_meas + meas_noise
        measurements.append(measured)
    
    true_states_tensor = tf.stack(true_states)
    initial_state = tf.constant([0.3, -0.4, 0.15], dtype=tf.float32)
    
    # Run filters with profiling
    results = simulate_with_profiling(
        ssm, landmarks, controls, measurements, true_states_tensor,
        initial_state, initial_uncertainty, num_particles
    )
    
    # Comprehensive evaluation
    print("\n" + "="*80)
    print("COMPUTING EVALUATION METRICS...")
    print("="*80)
    
    eval_data = comprehensive_evaluation(results, true_states_tensor)
    
    # Print results
    print_evaluation_table(eval_data)
    
    # Save results to CSV
    csv_path = output_dir / 'evaluation_results.csv'
    with open(csv_path, 'w') as f:
        # Write header
        if eval_data:
            header = list(eval_data[list(eval_data.keys())[0]].keys())
            f.write(','.join(header) + '\n')
            # Write data
            for filter_name, data in eval_data.items():
                row = [str(data[key]) for key in header]
                f.write(','.join(row) + '\n')
    print(f"\n✓ Saved: {csv_path}")
    
    # Print insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if 'ekf' in eval_data and 'ukf' in eval_data and 'pf' in eval_data:
        ekf_time = eval_data['ekf']['Total Time [s]']
        ukf_time = eval_data['ukf']['Total Time [s]']
        pf_time = eval_data['pf']['Total Time [s]']
        
        print(f"\n1. SPEED:")
        print(f"   EKF: 1.00x (baseline)")
        print(f"   UKF: {ukf_time/ekf_time:.2f}x")
        print(f"   PF:  {pf_time/ekf_time:.2f}x")
        
        ekf_rmse = eval_data['ekf']['Pos RMSE [m]']
        ukf_rmse = eval_data['ukf']['Pos RMSE [m]']
        pf_rmse = eval_data['pf']['Pos RMSE [m]']
        
        print(f"\n2. ACCURACY:")
        print(f"   EKF: {ekf_rmse:.4f}m (baseline)")
        print(f"   UKF: {ukf_rmse:.4f}m ({(ekf_rmse-ukf_rmse)/ekf_rmse*100:+.1f}%)")
        print(f"   PF:  {pf_rmse:.4f}m ({(ekf_rmse-pf_rmse)/ekf_rmse*100:+.1f}%)")
        
        print("\n3. CONSISTENCY (NEES should be ~3 for 3D state):")
        for name in ['ekf', 'ukf', 'pf']:
            data = eval_data[name]
            print(f"   {data['Filter']}: {data['Mean NEES']:.2f} "
                  f"({data['Consistency [%]']:.1f}% within bounds)")
    
    print("\n" + "="*80)
    
    # Visualize
    plot_path = output_dir / 'comprehensive_comparison.png'
    plot_comprehensive_comparison(results, true_states_tensor, landmarks, eval_data, plot_path)
    
    return {'results': results, 'eval_data': eval_data, 'true_states': true_states_tensor}


def particle_count_scaling_study(num_trials: int = 5, num_steps: int = 200,
                                output_dir: Path | None = None) -> Dict:
    """
    Comprehensive study of PF performance vs particle count.
    
    Tests multiple particle counts with multiple trials for statistical reliability.
    
    Parameters
    ----------
    num_trials : int, optional
        Number of trials per particle count (for variance analysis). Defaults to 5.
    num_steps : int, optional
        Number of simulation steps. Defaults to 200.
    output_dir : Path, optional
        Output directory. Defaults to OUTPUT_DIR.
    
    Returns
    -------
    results_by_count : Dict
        Dictionary with results for each particle count.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PARTICLE FILTER SCALING STUDY (TensorFlow Native)")
    print("="*80)
    print("Testing particle counts: [500, 1000, 2000, 5000, 10000, 20000]")
    print(f"Running {num_trials} trials per configuration for statistical reliability")
    print("This will take several minutes...")
    print("="*80)
    
    # Setup scenario
    tf.random.set_seed(42)
    
    landmarks = tf.constant([[-1.5, -1.5]], dtype=tf.float32)
    dt = 0.1
    Q = tf.eye(3, dtype=tf.float32) * 0.03
    R = tf.eye(2, dtype=tf.float32) * 0.01
    initial_uncertainty = tf.eye(3, dtype=tf.float32) * 1.0
    
    ssm = RangeBearingSSM(dt=dt, process_noise=Q, meas_noise=R)
    
    # Generate trajectory once
    print("\nGenerating ground truth trajectory...")
    true_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    true_states = []
    controls = []
    measurements = []
    
    for step in range(num_steps):
        step_tf = tf.cast(step, tf.float32)
        v = 1.5 + 0.5 * tf.sin(step_tf * 0.15)
        omega = 0.6 + 0.3 * tf.cos(step_tf * 0.12)
        control = tf.stack([v, omega])
        controls.append(control)
        
        true_state = ssm.motion_model(true_state[tf.newaxis, :], control[tf.newaxis, :])[0]
        process_noise = tf.random.normal([3], mean=0.0, stddev=0.17, dtype=tf.float32)
        true_state = true_state + process_noise
        true_states.append(true_state)
        
        true_meas = ssm.measurement_model(true_state[tf.newaxis, :], landmarks)[0]
        meas_std = tf.sqrt(tf.linalg.diag_part(ssm.R))
        meas_noise = tf.random.normal([tf.shape(landmarks)[0], 2], mean=0.0, 
                                     stddev=meas_std, dtype=tf.float32)
        measured = true_meas + meas_noise
        measurements.append(measured)
    
    true_states_tensor = tf.stack(true_states)
    initial_state = tf.constant([0.3, -0.4, 0.15], dtype=tf.float32)
    
    # Test different particle counts with multiple trials
    particle_counts = [500, 1000, 2000, 5000, 10000, 20000]
    results_by_count = {}
    
    evaluator = FilterEvaluator()
    
    for n_particles in particle_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {n_particles} particles ({num_trials} trials)...")
        print(f"{'='*60}")
        
        # Store results from all trials
        trial_pos_rmse = []
        trial_head_rmse = []
        trial_mean_nees = []
        trial_consistency = []
        trial_total_time = []
        trial_predict_time = []
        trial_update_time = []
        trial_peak_memory = []
        trial_pos_errors = []
        trial_states = []
        
        for trial in range(num_trials):
            # Use different seed for each trial
            seed = 42 + trial * 100 + n_particles
            tf.random.set_seed(seed)
            
            print(f"  Trial {trial+1}/{num_trials}...", end=' ')
            
            pf = ParticleFilter(ssm, initial_state, initial_uncertainty,
                               num_particles=n_particles)
            
            pf_states = []
            pf_covs = []
            predict_times = []
            update_times = []
            
            tracker = MemoryTracker()
            tracker.reset()
            
            start_time = time.perf_counter()
            
            for step in range(len(controls)):
                tracker.update_peak()
                
                # Time predict
                t0 = time.perf_counter()
                pf.predict(controls[step])
                predict_times.append(time.perf_counter() - t0)
                
                # Time update
                t0 = time.perf_counter()
                pf.update(measurements[step], landmarks)
                update_times.append(time.perf_counter() - t0)
                
                pf_states.append(tf.identity(pf.state))
                pf_covs.append(tf.identity(pf.covariance))
            
            total_time = time.perf_counter() - start_time
            
            # Compute metrics for this trial
            pf_states_tensor = tf.stack(pf_states)
            pf_covs_tensor = tf.stack(pf_covs)
            
            # RMSE
            pos_rmse = compute_rmse(pf_states_tensor[:, :2], true_states_tensor[:, :2])
            heading_diff = pf_states_tensor[:, 2] - true_states_tensor[:, 2]
            heading_errors = tf.abs(tf.math.atan2(tf.sin(heading_diff), tf.cos(heading_diff)))
            heading_rmse = float(tf.sqrt(tf.reduce_mean(heading_errors**2)).numpy())
            pos_errors = tf.linalg.norm(pf_states_tensor[:, :2] - true_states_tensor[:, :2], axis=1)
            
            # NEES
            nees_values = compute_nees(pf_states_tensor, pf_covs_tensor, true_states_tensor)
            
            # Consistency
            consistency = evaluator.compute_consistency_percentage(nees_values, 3)
            
            # Store trial results
            trial_pos_rmse.append(pos_rmse)
            trial_head_rmse.append(heading_rmse)
            trial_mean_nees.append(float(tf.reduce_mean(nees_values).numpy()))
            trial_consistency.append(consistency)
            trial_total_time.append(total_time)
            trial_predict_time.append(float(tf.reduce_mean(tf.constant(predict_times, dtype=tf.float32)).numpy()))
            trial_update_time.append(float(tf.reduce_mean(tf.constant(update_times, dtype=tf.float32)).numpy()))
            trial_peak_memory.append(tracker.peak_memory)
            trial_pos_errors.append(pos_errors)
            trial_states.append(pf_states_tensor)
            
            print(f"RMSE: {pos_rmse:.4f}m")
        
        # Compute statistics across trials using TensorFlow
        pos_rmse_tensor = tf.constant(trial_pos_rmse, dtype=tf.float32)
        mean_nees_tensor = tf.constant(trial_mean_nees, dtype=tf.float32)
        total_time_tensor = tf.constant(trial_total_time, dtype=tf.float32)
        
        results_by_count[n_particles] = {
            'pos_rmse_mean': float(tf.reduce_mean(pos_rmse_tensor).numpy()),
            'pos_rmse_std': float(tf.math.reduce_std(pos_rmse_tensor).numpy()),
            'pos_rmse_min': float(tf.reduce_min(pos_rmse_tensor).numpy()),
            'pos_rmse_max': float(tf.reduce_max(pos_rmse_tensor).numpy()),
            'head_rmse': float(tf.reduce_mean(trial_head_rmse)),
            'mean_nees': float(tf.reduce_mean(mean_nees_tensor).numpy()),
            'nees_std': float(tf.math.reduce_std(mean_nees_tensor).numpy()),
            'consistency': float(tf.reduce_mean(trial_consistency)),
            'total_time': float(tf.reduce_mean(total_time_tensor).numpy()),
            'time_std': float(tf.math.reduce_std(total_time_tensor).numpy()),
            'avg_predict_time': float(tf.reduce_mean(trial_predict_time)),
            'avg_update_time': float(tf.reduce_mean(trial_update_time)),
            'peak_memory': float(tf.reduce_mean(trial_peak_memory)),
            'pos_errors': trial_pos_errors[0],  # Use first trial for visualization
            'states': trial_states[0],  # Use first trial for visualization
            'all_rmse': trial_pos_rmse  # Keep all for variance plot
        }
        
        print(f"\n  Summary over {num_trials} trials:")
        print(f"    Pos RMSE: {results_by_count[n_particles]['pos_rmse_mean']:.4f} ± {results_by_count[n_particles]['pos_rmse_std']:.4f}m")
        print(f"    Range: [{results_by_count[n_particles]['pos_rmse_min']:.4f}, {results_by_count[n_particles]['pos_rmse_max']:.4f}]m")
        print(f"    NEES: {results_by_count[n_particles]['mean_nees']:.2f} ± {results_by_count[n_particles]['nees_std']:.2f}")
        print(f"    Consistency: {results_by_count[n_particles]['consistency']:.1f}%")
        print(f"    Time: {results_by_count[n_particles]['total_time']:.3f} ± {results_by_count[n_particles]['time_std']:.3f}s")
    
    # Save results to CSV
    csv_path = output_dir / 'particle_scaling_results.csv'
    with open(csv_path, 'w') as f:
        f.write("particles,pos_rmse_mean,pos_rmse_std,pos_rmse_min,pos_rmse_max,mean_nees,nees_std,consistency,total_time,time_std,avg_predict_time,avg_update_time,peak_memory\n")
        for n, r in results_by_count.items():
            f.write(f"{n},{r['pos_rmse_mean']:.6f},{r['pos_rmse_std']:.6f},"
                   f"{r['pos_rmse_min']:.6f},{r['pos_rmse_max']:.6f},"
                   f"{r['mean_nees']:.6f},{r['nees_std']:.6f},"
                   f"{r['consistency']:.6f},{r['total_time']:.6f},"
                   f"{r['time_std']:.6f},{r['avg_predict_time']:.6f},"
                   f"{r['avg_update_time']:.6f},{r['peak_memory']:.6f}\n")
    print(f"\n✓ Saved: {csv_path}")
    
    # Plot comprehensive comparison
    plot_particle_scaling_analysis(results_by_count, particle_counts, true_states_tensor, landmarks, output_dir)
    
    return results_by_count


def plot_particle_scaling_analysis(results: Dict, particle_counts: List[int], 
                                   true_states: tf.Tensor, landmarks: tf.Tensor,
                                   output_dir: Path) -> None:
    """
    Create comprehensive visualization of particle count scaling.
    Updated to show mean ± std from multiple trials.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors for plotting
    true_states_np = true_states.numpy()
    landmarks_np = landmarks.numpy()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    # Extract data for plotting (using means)
    pos_rmse_mean = [results[n]['pos_rmse_mean'] for n in particle_counts]
    pos_rmse_std = [results[n]['pos_rmse_std'] for n in particle_counts]
    pos_rmse_min = [results[n]['pos_rmse_min'] for n in particle_counts]
    pos_rmse_max = [results[n]['pos_rmse_max'] for n in particle_counts]
    mean_nees = [results[n]['mean_nees'] for n in particle_counts]
    nees_std = [results[n]['nees_std'] for n in particle_counts]
    consistency = [results[n]['consistency'] for n in particle_counts]
    total_time = [results[n]['total_time'] for n in particle_counts]
    time_std = [results[n]['time_std'] for n in particle_counts]
    avg_step_time = [(results[n]['avg_predict_time'] + results[n]['avg_update_time']) * 1000 
                     for n in particle_counts]
    peak_memory = [results[n]['peak_memory'] for n in particle_counts]
    
    # Color scheme
    colors = plt.cm.viridis(tf.linspace(0.2, 0.9, len(particle_counts)).numpy())
    
    # 1. Position RMSE vs Particle Count (with error bars)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(particle_counts, pos_rmse_mean, yerr=pos_rmse_std, 
                fmt='o-', linewidth=2, markersize=8, color='steelblue',
                capsize=5, capthick=2, label='Mean ± Std')
    ax1.fill_between(particle_counts, pos_rmse_min, pos_rmse_max, 
                     alpha=0.2, color='steelblue', label='Min-Max Range')
    ax1.set_xlabel('Number of Particles', fontweight='bold')
    ax1.set_ylabel('Position RMSE [m]', fontweight='bold')
    ax1.set_title('Accuracy vs Particle Count', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.legend()
    
    # 2. RMSE Variance (Box Plot)
    ax2 = fig.add_subplot(gs[0, 1])
    rmse_data = [results[n]['all_rmse'] for n in particle_counts]
    bp = ax2.boxplot(rmse_data, tick_labels=[str(n) for n in particle_counts], 
                    patch_artist=True, showfliers=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xlabel('Number of Particles', fontweight='bold')
    ax2.set_ylabel('Position RMSE [m]', fontweight='bold')
    ax2.set_title('RMSE Distribution Across Trials', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. NEES vs Particle Count
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(particle_counts, mean_nees, 'o-', linewidth=2, markersize=8, color='coral')
    ax3.axhline(3, color='green', linestyle='--', linewidth=2, label='Ideal (state_dim=3)')
    ax3.set_xlabel('Number of Particles', fontweight='bold')
    ax3.set_ylabel('Mean NEES', fontweight='bold')
    ax3.set_title('Consistency vs Particle Count', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.legend()
    
    for i, (n, nees) in enumerate(zip(particle_counts, mean_nees)):
        ax3.annotate(f'{nees:.2f}', (n, nees), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=8)
    
    # 4. Consistency Percentage
    ax4 = fig.add_subplot(gs[0, 3])
    bars = ax4.bar(range(len(particle_counts)), consistency, color=colors, alpha=0.8)
    ax4.axhline(95, color='green', linestyle='--', linewidth=2, label='Target (95%)')
    ax4.set_xlabel('Number of Particles', fontweight='bold')
    ax4.set_ylabel('Consistency [%]', fontweight='bold')
    ax4.set_title('Filter Consistency', fontweight='bold')
    ax4.set_xticks(range(len(particle_counts)))
    ax4.set_xticklabels([str(n) for n in particle_counts], rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    for bar, val in zip(bars, consistency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 5. Coefficient of Variation (Stability Metric)
    ax5 = fig.add_subplot(gs[1, 0])
    cv = [std/mean*100 if mean > 0 else 0 for mean, std in zip(pos_rmse_mean, pos_rmse_std)]
    ax5.plot(particle_counts, cv, 'o-', linewidth=2, markersize=8, color='purple')
    ax5.set_xlabel('Number of Particles', fontweight='bold')
    ax5.set_ylabel('Coefficient of Variation [%]', fontweight='bold')
    ax5.set_title('Result Stability (Lower = Better)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    
    for i, (n, cv_val) in enumerate(zip(particle_counts, cv)):
        ax5.annotate(f'{cv_val:.1f}%', (n, cv_val), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=8)
    
    # 6. Computation Time (with error bars)
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.errorbar(particle_counts, total_time, yerr=time_std,
                fmt='o-', linewidth=2, markersize=8, color='crimson',
                capsize=5, capthick=2)
    ax6.set_xlabel('Number of Particles', fontweight='bold')
    ax6.set_ylabel('Total Time [s]', fontweight='bold')
    ax6.set_title('Runtime vs Particle Count', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    
    # 7. Average Step Time
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(particle_counts, avg_step_time, 'o-', linewidth=2, markersize=8, color='darkorange')
    ax7.set_xlabel('Number of Particles', fontweight='bold')
    ax7.set_ylabel('Avg Step Time [ms]', fontweight='bold')
    ax7.set_title('Per-Step Latency', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.set_xscale('log')
    
    for i, (n, t) in enumerate(zip(particle_counts, avg_step_time)):
        ax7.annotate(f'{t:.2f}ms', (n, t), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=8)
    
    # 8. Memory Usage
    ax8 = fig.add_subplot(gs[1, 3])
    bars = ax8.bar(range(len(particle_counts)), peak_memory, color=colors, alpha=0.8)
    ax8.set_xlabel('Number of Particles', fontweight='bold')
    ax8.set_ylabel('Peak Memory [MB]', fontweight='bold')
    ax8.set_title('Memory Usage', fontweight='bold')
    ax8.set_xticks(range(len(particle_counts)))
    ax8.set_xticklabels([str(n) for n in particle_counts], rotation=45)
    ax8.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, peak_memory):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 9. Trajectories Comparison
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9.plot(true_states_np[:, 0], true_states_np[:, 1], 'k--', 
            label='True', linewidth=2.5, alpha=0.7)
    for i, n in enumerate(particle_counts):
        states_np = results[n]['states'].numpy()
        ax9.plot(states_np[:, 0], states_np[:, 1], 
                color=colors[i], label=f'{n}', 
                linewidth=1.5, alpha=0.6)
    ax9.scatter(landmarks_np[:, 0], landmarks_np[:, 1], c='orange', 
               marker='^', s=150, label='LM', zorder=5)
    ax9.legend(fontsize=8, ncol=3)
    ax9.grid(True, alpha=0.3)
    ax9.set_xlabel('X [m]')
    ax9.set_ylabel('Y [m]')
    ax9.set_title('Trajectory Comparison')
    ax9.axis('equal')
    
    # 10. Error Evolution
    ax10 = fig.add_subplot(gs[2, 2:])
    for i, n in enumerate(particle_counts):
        errors_np = results[n]['pos_errors'].numpy()
        ax10.plot(errors_np, color=colors[i], label=f'{n}', 
                linewidth=1.5, alpha=0.7)
    ax10.legend(title='Particles', fontsize=8, ncol=3)
    ax10.grid(True, alpha=0.3)
    ax10.set_xlabel('Time Step')
    ax10.set_ylabel('Position Error [m]')
    ax10.set_title('Error Evolution Over Time')
    
    plt.suptitle('Particle Filter Scaling Analysis (Multi-Trial Statistics)', 
                fontsize=16, fontweight='bold')
    
    plot_path = output_dir / 'particle_scaling_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*120)
    print("PARTICLE SCALING SUMMARY TABLE (Mean ± Std over multiple trials)")
    print("="*120)
    
    header = f"{'Particles':<10} {'RMSE [m]':<20} {'RMSE Range':<25} {'NEES':<15} {'Consist[%]':<12} {'Time[s]':<20} {'Memory[MB]':<11}"
    print(header)
    print("="*120)
    
    for n in particle_counts:
        r = results[n]
        row = f"{n:<10} " \
              f"{r['pos_rmse_mean']:.4f} ± {r['pos_rmse_std']:.4f}{'':>4} " \
              f"[{r['pos_rmse_min']:.4f}, {r['pos_rmse_max']:.4f}]{'':>4} " \
              f"{r['mean_nees']:.2f} ± {r['nees_std']:.2f}{'':>2} " \
              f"{r['consistency']:.1f}{'':>7} " \
              f"{r['total_time']:.3f} ± {r['time_std']:.3f}{'':>6} " \
              f"{r['peak_memory']:.0f}"
        print(row)
    
    print("="*120)
    
    # Variance analysis
    print("\n" + "="*80)
    print("VARIANCE ANALYSIS (Coefficient of Variation)")
    print("="*80)
    for n, mean, std in zip(particle_counts, pos_rmse_mean, pos_rmse_std):
        cv_val = std / mean * 100 if mean > 0 else 0
        stability = "Excellent" if cv_val < 5 else "Good" if cv_val < 10 else "Fair" if cv_val < 20 else "Poor"
        print(f"  {n:5d} particles: CV = {cv_val:5.1f}% ({stability})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runtime and Memory Profiling Experiments for EKF, UKF, and Particle Filter'
    )
    parser.add_argument(
        '--scaling',
        action='store_true',
        help='Run particle count scaling study'
    )
    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help='Run comprehensive filter comparison'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=5,
        help='Number of trials per particle count for scaling study (default: 5)'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=200,
        help='Number of simulation steps (default: 200)'
    )
    parser.add_argument(
        '--num-particles',
        type=int,
        default=10000,
        help='Number of particles for comprehensive evaluation (default: 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # If no flags specified, run both
    if not args.scaling and not args.comprehensive:
        print("No experiment specified. Running both scaling and comprehensive evaluations.")
        print("Use --scaling or --comprehensive to run specific experiments.\n")
        particle_count_scaling_study(
            num_trials=args.num_trials,
            num_steps=args.num_steps
        )
        run_comprehensive_evaluation(
            num_steps=args.num_steps,
            num_particles=args.num_particles,
            seed=args.seed
        )
    else:
        if args.scaling:
            print("Running particle count scaling study...\n")
            particle_count_scaling_study(
                num_trials=args.num_trials,
                num_steps=args.num_steps
            )
        
        if args.comprehensive:
            print("Running comprehensive filter comparison...\n")
            run_comprehensive_evaluation(
                num_steps=args.num_steps,
                num_particles=args.num_particles,
                seed=args.seed
            )

# Run scaling study with 10 trials
#python3 -m src.experiments.exp_runtime_memory --scaling --num-trials 10

# Run comprehensive evaluation with 5000 particles
#python3 -m src.experiments.exp_runtime_memory --comprehensive --num-particles 5000

# Run both with custom parameters
#python3 -m src.experiments.exp_runtime_memory --scaling --comprehensive --num-steps 150 --seed 123