"""
Monte Carlo Filter Comparison for Multi-Target Acoustic Tracking.

Compares: EKF, UKF, PF, PF-PF-LEDH, PF-PF-EDH, LEDH, EDH.
Uses RMSE (position RMSE over time) for evaluation.

GPU Support
-----------
This experiment supports GPU acceleration. Use --gpu flag to enable.
TensorFlow will automatically use available GPUs when enabled.

Usage:
    python -m src.experiments.exp_Li... --gpu              # Use first GPU
    python -m src.experiments.exp_Li... --gpu --gpu_id 1   # Use specific GPU
    python -m src.experiments.exp_Li... --cpu              # Force CPU only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

from src.models.ssm_multi_target_acoustic import MultiTargetAcousticSSM
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.filters.particle_filter import ParticleFilter
from src.filters.pfpf_filter import PFPFLEDHFilter, PFPFEDHFilter
from src.filters.ledh import LEDH
from src.filters.edh import EDH
from src.metrics.accuracy import compute_rmse as _compute_rmse_shared

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# GPU Configuration
# =============================================================================


def configure_gpu(use_gpu: bool = True, gpu_id: int = 0, memory_growth: bool = True) -> str:
    """
    Configure TensorFlow GPU settings.
    
    Parameters
    ----------
    use_gpu : bool
        Whether to use GPU (if available).
    gpu_id : int
        Which GPU to use (if multiple available).
    memory_growth : bool
        Enable memory growth to avoid OOM errors.
        
    Returns
    -------
    str
        Description of the device being used.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not use_gpu or not gpus:
        # Force CPU-only mode
        tf.config.set_visible_devices([], 'GPU')
        return "CPU"
    
    try:
        # Select specific GPU if multiple available
        if gpu_id < len(gpus):
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            selected_gpu = gpus[gpu_id]
        else:
            selected_gpu = gpus[0]
            
        # Enable memory growth to prevent TF from allocating all GPU memory
        if memory_growth:
            tf.config.experimental.set_memory_growth(selected_gpu, True)
            
        # Get GPU name for logging
        gpu_details = tf.config.experimental.get_device_details(selected_gpu)
        gpu_name = gpu_details.get('device_name', f'GPU:{gpu_id}')
        
        return f"GPU: {gpu_name}"
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        print("Falling back to CPU")
        tf.config.set_visible_devices([], 'GPU')
        return "CPU (fallback)"


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns
    -------
    dict
        Device information including GPUs, CPUs, and memory.
    """
    info = {
        'gpus': [],
        'cpus': tf.config.list_physical_devices('CPU'),
        'gpu_available': False,
        'cuda_version': None,
    }
    
    gpus = tf.config.list_physical_devices('GPU')
    info['gpu_available'] = len(gpus) > 0
    
    for i, gpu in enumerate(gpus):
        try:
            details = tf.config.experimental.get_device_details(gpu)
            info['gpus'].append({
                'id': i,
                'name': details.get('device_name', f'GPU:{i}'),
                'compute_capability': details.get('compute_capability', 'unknown'),
            })
        except Exception:
            info['gpus'].append({'id': i, 'name': f'GPU:{i}'})
    
    # Try to get CUDA version
    try:
        info['cuda_version'] = tf.sysconfig.get_build_info().get('cuda_version', 'unknown')
    except Exception:
        pass
        
    return info


def print_device_info(device_str: str):
    """Print device configuration information."""
    info = get_device_info()
    print(f"Device: {device_str}")
    if info['gpu_available']:
        print(f"Available GPUs: {len(info['gpus'])}")
        for gpu in info['gpus']:
            print(f"  [{gpu['id']}] {gpu['name']}")
        if info['cuda_version']:
            print(f"CUDA Version: {info['cuda_version']}")
    else:
        print("No GPU available - using CPU")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Monte Carlo Multi-Target Filter Comparison (TF only, RMSE)'
    )
    parser.add_argument('--n_trajectories', type=int, default=50)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=20)

    parser.add_argument('--num_targets', type=int, default=4)
    parser.add_argument('--num_sensors', type=int, default=25)
    parser.add_argument('--area_size', type=float, default=40.0)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--psi', type=float, default=10.0)
    parser.add_argument('--d0', type=float, default=0.1)
    parser.add_argument('--sigma_w', type=float, default=0.1)
    parser.add_argument('--process_noise_scale', type=float, default=1.5)
    parser.add_argument('--sensor_grid_size', type=int, default=5)

    parser.add_argument('--pf_particles', type=int, default=5000)
    parser.add_argument('--n_particles', type=int, default=500)
    parser.add_argument('--ukf_alpha', type=float, default=0.1)
    parser.add_argument('--ukf_beta', type=float, default=1.0)
    parser.add_argument('--n_lambda', type=int, default=5)
    parser.add_argument('--n_lambda_ledh', type=int, default=29)
    parser.add_argument('--n_lambda_edh', type=int, default=29)
    parser.add_argument('--filter_type', type=str, default='ekf', choices=['ekf', 'ukf'])
    parser.add_argument('--edh_redraw', action='store_true')

    parser.add_argument(
        '--filters', nargs='+', type=str,
        default=['ekf', 'ukf', 'pf', 'pfpf_ledh', 'pfpf_edh', 'ledh', 'edh']
    )
    parser.add_argument('--output_dir', type=str, default='reports/3_Deterministic_Kernel_Flow/Li(17)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show_inner_progress', action='store_true')
    parser.add_argument('--plot_particles', action='store_true')
    parser.add_argument('--plot_std', action='store_true')
    
    # GPU configuration
    parser.add_argument('--gpu', action='store_true', 
                        help='Enable GPU acceleration (if available)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU-only mode (overrides --gpu)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    parser.add_argument('--no_memory_growth', action='store_true',
                        help='Disable GPU memory growth (allocate all GPU memory)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision (float16) for faster GPU computation')

    return parser.parse_args()


def compute_rmse(estimates: tf.Tensor, true_states: tf.Tensor) -> float:
    """Root mean square error (full state) using shared utility."""
    return _compute_rmse_shared(estimates, true_states)


def compute_pos_rmse_over_time(
    estimates: tf.Tensor, true_states: tf.Tensor, num_targets: int
) -> tf.Tensor:
    """Position RMSE at each time step (averaged over targets). Shape (T+1,)."""
    estimates = tf.cast(estimates, tf.float32)
    true_states = tf.cast(true_states, tf.float32)
    T_plus_1 = tf.shape(estimates)[0]
    est_reshaped = tf.reshape(estimates, [T_plus_1, num_targets, 4])
    true_reshaped = tf.reshape(true_states, [T_plus_1, num_targets, 4])
    pos_est = est_reshaped[:, :, :2]
    pos_true = true_reshaped[:, :, :2]
    per_target = tf.sqrt(
        tf.reduce_sum(tf.square(pos_est - pos_true), axis=2) + 1e-12
    )
    per_step = tf.reduce_mean(per_target, axis=1)
    return per_step


def plot_ess_vs_steps(summary, filters, n_steps, out_path, show_std=False):
    """Plot average ESS vs time steps for filters that report ESS."""
    plt.figure(figsize=(10, 6))
    ess_filters = ['pf', 'pfpf_ledh', 'pfpf_edh', 'ledh', 'edh']

    for f in filters:
        if f not in ess_filters:
            continue
        info = summary.get(f, {})
        ess_mean = info.get('ess_mean_time_series')
        ess_std = info.get('ess_std_time_series')
        if ess_mean is None:
            continue
        steps = list(range(1, n_steps + 1))
        plt.plot(steps, ess_mean, label=f.upper(), linewidth=2)
        if show_std and ess_std:
            plt.fill_between(
                steps,
                [a - b for a, b in zip(ess_mean, ess_std)],
                [a + b for a, b in zip(ess_mean, ess_std)],
                alpha=0.2,
            )
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Effective Sample Size (ESS)', fontsize=12)
    plt.title('Average ESS vs Time Steps', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_rmse_vs_steps(summary, filters, n_steps, out_path, show_std=False):
    """Plot average position RMSE vs time steps."""
    plt.figure(figsize=(10, 6))
    colors = {
        'ekf': 'blue', 'ukf': 'green', 'pf': 'red',
        'pfpf_ledh': 'purple', 'pfpf_edh': 'orange',
        'ledh': 'cyan', 'edh': 'magenta',
    }

    for f in filters:
        info = summary.get(f, {})
        rmse_mean = info.get('pos_rmse_mean_time_series')
        rmse_std = info.get('pos_rmse_std_time_series')
        if rmse_mean is None:
            continue
        steps = list(range(len(rmse_mean)))
        label = f.upper().replace('_', '-')
        plt.plot(steps, rmse_mean, label=label, linewidth=2, color=colors.get(f, 'black'))
        if show_std and rmse_std:
            plt.fill_between(
                steps,
                [a - b for a, b in zip(rmse_mean, rmse_std)],
                [a + b for a, b in zip(rmse_mean, rmse_std)],
                alpha=0.2,
                color=colors.get(f, 'black'),
            )
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Position RMSE (m)', fontsize=12)
    plt.title('Average Position RMSE vs Time Steps', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_trajectories_multitarget(true_states, estimates, num_targets, out_path, filter_name):
    """Plot true vs estimated trajectories (2D position)."""
    true_states = tf.constant(true_states, dtype=tf.float32)
    estimates = tf.constant(estimates, dtype=tf.float32)
    T_plus_1 = tf.shape(true_states)[0]
    true_reshaped = tf.reshape(true_states, [T_plus_1, num_targets, 4])
    est_reshaped = tf.reshape(estimates, [T_plus_1, num_targets, 4])
    true_reshaped = true_reshaped.numpy()
    est_reshaped = est_reshaped.numpy()

    plt.figure(figsize=(6, 6))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    for c in range(num_targets):
        true_pos = true_reshaped[:, c, :2]
        est_pos = est_reshaped[:, c, :2]
        col = colors[c % len(colors)]
        plt.plot(true_pos[:, 0], true_pos[:, 1], color=col, linewidth=2,
                 label=f'Target {c} True' if c == 0 else None)
        plt.plot(est_pos[:, 0], est_pos[:, 1], color=col, linestyle='--', linewidth=1.5,
                 label=f'Target {c} Est' if c == 0 else None)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Trajectories - {filter_name.upper()}')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_trajectory(ssm_gen: MultiTargetAcousticSSM, initial_state: tf.Tensor, n_steps: int):
    """Generate ground truth trajectory and measurements using TF only."""
    state = tf.reshape(initial_state, [1, -1])
    states_list = [state]
    measurements_list = []

    for t in range(n_steps):
        state_flat = tf.reshape(state, [-1])
        state_next = ssm_gen.motion_model(state_flat, control=None)
        if state_next.shape.rank == 1:
            state_next = state_next[tf.newaxis, :]
        noise = ssm_gen.sample_process_noise((), use_gen=True)
        noise_flat = tf.reshape(noise, [1, -1])
        state_next = state_next + noise_flat
        state = state_next
        states_list.append(state_next)

        z = ssm_gen.measurement_model(state_next, landmarks=None)
        if z.shape.rank == 1:
            z = z[tf.newaxis, :]
        noise_meas = tf.random.normal(
            tf.shape(z), mean=0.0, stddev=ssm_gen.sigma_w, dtype=ssm_gen.dtype
        )
        measurements_list.append(z + noise_meas)

    states = tf.concat(states_list, axis=0)
    measurements = tf.concat(measurements_list, axis=0)
    return states, measurements


def run_filter(
    filter_obj,
    n_steps: int,
    measurements: tf.Tensor,
    sensor_positions: tf.Tensor,
    store_particles: bool,
    filter_name: str | None,
    show_progress: bool,
):
    """Run a single filter on one trajectory. Returns Python types for serialization."""
    if hasattr(filter_obj, 'state'):
        s0 = filter_obj.state
    elif hasattr(filter_obj, 'x_hat'):
        s0 = filter_obj.x_hat
    else:
        raise ValueError('Filter has no state attribute')

    s0 = tf.reshape(s0, [-1])
    state_dim = int(s0.shape[0])
    estimates_list = [s0.numpy().flatten()]
    has_weights = hasattr(filter_obj, 'weights')
    has_particles_attr = hasattr(filter_obj, 'particles')
    has_particles = has_weights and has_particles_attr
    ess_history = [] if has_particles else None
    particle_history = [] if (has_particles_attr and store_particles) else None

    control = tf.zeros([2], dtype=tf.float32)
    t0 = time.time()
    iterator = range(n_steps)
    if show_progress and HAS_TQDM:
        iterator = tqdm(iterator, desc=f'{filter_name or "Filter"}', leave=False, total=n_steps)

    for t in iterator:
        filter_obj.predict(control)
        filter_obj.update(measurements[t], sensor_positions)

        if hasattr(filter_obj, 'state'):
            st = filter_obj.state
        elif hasattr(filter_obj, 'x_hat'):
            st = filter_obj.x_hat
        else:
            raise ValueError('Filter has no state attribute')
        st = tf.reshape(st, [-1])
        estimates_list.append(st.numpy().flatten())

        if has_particles:
            if hasattr(filter_obj, 'ess_before_resample'):
                ess_history.append(float(filter_obj.ess_before_resample.numpy()))
            else:
                try:
                    w = tf.reshape(filter_obj.weights, [-1])
                    w_sum = tf.reduce_sum(w)
                    w = tf.cond(
                        tf.greater(tf.abs(w_sum - 1.0), 1e-6),
                        lambda: tf.cond(
                            tf.logical_and(tf.greater(w_sum, 0), tf.math.is_finite(w_sum)),
                            lambda: w / w_sum,
                            lambda: tf.ones_like(w) / tf.cast(tf.size(w), tf.float32),
                        ),
                        lambda: w,
                    )
                    ess_val = 1.0 / (tf.reduce_sum(tf.square(w)) + 1e-15)
                    ess_history.append(float(ess_val.numpy()))
                except Exception:
                    ess_history.append(float('nan'))

        if store_particles and has_particles_attr:
            try:
                p = filter_obj.particles.numpy()
                particle_history.append(p.copy())
            except Exception:
                particle_history.append(None)

    exec_time = time.time() - t0
    estimates = tf.constant(estimates_list, dtype=tf.float32)
    return estimates, ess_history, exec_time, particle_history


def run_single_trajectory(
    args_or_dict,
    ssm_gen: MultiTargetAcousticSSM,
    ssm_filter: MultiTargetAcousticSSM,
    sensor_positions: tf.Tensor,
    initial_state: tf.Tensor,
    traj_idx: int,
    run_idx: int,
):
    """Run all selected filters on one trajectory."""
    if isinstance(args_or_dict, dict):
        d = args_or_dict
        filters = d['filters']
        output_dir = d['output_dir']
        plot_particles = d['plot_particles']
        show_inner_progress = d['show_inner_progress']
        n_particles = d['n_particles']
        pf_particles = d['pf_particles']
        ukf_alpha = d['ukf_alpha']
        ukf_beta = d['ukf_beta']
        n_lambda = d['n_lambda']
        n_lambda_ledh = d.get('n_lambda_ledh', n_lambda)
        n_lambda_edh = d.get('n_lambda_edh', n_lambda)
        filter_type = d['filter_type']
        edh_redraw = d['edh_redraw']
        n_steps = d['n_steps']
        num_targets = d['num_targets']
    else:
        args = args_or_dict
        filters = args.filters
        output_dir = args.output_dir
        plot_particles = args.plot_particles
        show_inner_progress = args.show_inner_progress
        n_particles = args.n_particles
        pf_particles = args.pf_particles
        ukf_alpha = args.ukf_alpha
        ukf_beta = args.ukf_beta
        n_lambda = args.n_lambda
        n_lambda_ledh = getattr(args, 'n_lambda_ledh', None) or n_lambda
        n_lambda_edh = getattr(args, 'n_lambda_edh', None) or n_lambda
        filter_type = args.filter_type
        edh_redraw = args.edh_redraw
        n_steps = args.n_steps
        num_targets = args.num_targets

    true_states, measurements = generate_trajectory(ssm_gen, initial_state, n_steps)
    results = {'true_states': true_states.numpy()}
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    store_particles_global = (traj_idx == 0 and run_idx == 0 and plot_particles)

    x0 = tf.constant(initial_state, dtype=tf.float32)
    state_dim = ssm_filter.state_dim

    for filter_name in filters:
        if filter_name == 'ukf':
            P0 = tf.eye(state_dim, dtype=tf.float32) * 0.1
        else:
            P0 = tf.eye(state_dim, dtype=tf.float32) * 1.0

        if filter_name == 'ekf':
            fobj = ExtendedKalmanFilter(ssm_filter, x0, P0)
        elif filter_name == 'ukf':
            fobj = UnscentedKalmanFilter(
                ssm_filter, x0, P0,
                alpha=0.1, beta=ukf_beta, kappa=0.0,
            )
        elif filter_name == 'pf':
            fobj = ParticleFilter(
                ssm_filter, x0, P0,
                num_particles=pf_particles,
            )
        elif filter_name == 'pfpf_ledh':
            fobj = PFPFLEDHFilter(
                ssm_filter, x0, P0,
                num_particles=n_particles,
                n_lambda=n_lambda_ledh,
                filter_type=filter_type,
                ukf_alpha=ukf_alpha,
                ukf_beta=ukf_beta,
                show_progress=show_inner_progress,
            )
        elif filter_name == 'pfpf_edh':
            fobj = PFPFEDHFilter(
                ssm_filter, x0, P0,
                num_particles=n_particles,
                n_lambda=n_lambda_edh,
                filter_type=filter_type,
                ukf_alpha=ukf_alpha,
                ukf_beta=ukf_beta,
                show_progress=show_inner_progress,
            )
        elif filter_name == 'ledh':
            fobj = LEDH(
                ssm_filter, x0, P0,
                num_particles=n_particles,
                n_lambda=n_lambda_ledh,
                filter_type=filter_type,
                ukf_alpha=ukf_alpha,
                show_progress=show_inner_progress,
                redraw_particles=False,
            )
        elif filter_name == 'edh':
            fobj = EDH(
                ssm_filter, x0, P0,
                num_particles=n_particles,
                n_lambda=n_lambda_edh,
                filter_type=filter_type,
                ukf_alpha=ukf_alpha,
                show_progress=show_inner_progress,
                redraw_particles=edh_redraw,
            )
        else:
            raise ValueError(f'Unknown filter: {filter_name}')

        store_particles = store_particles_global and filter_name in [
            'pf', 'pfpf_ledh', 'pfpf_edh', 'ledh', 'edh'
        ]
        estimates, ess_history, exec_time, particle_history = run_filter(
            fobj, n_steps, measurements, sensor_positions, store_particles,
            filter_name=filter_name, show_progress=show_inner_progress,
        )

        true_t = tf.constant(results['true_states'], dtype=tf.float32)
        rmse = compute_rmse(estimates, true_t)
        pos_rmse = compute_pos_rmse_over_time(estimates, true_t, num_targets)
        pos_rmse_list = pos_rmse.numpy().tolist()

        results[filter_name] = {
            'estimates': estimates.numpy(),
            'rmse': rmse,
            'pos_rmse': pos_rmse_list,
            'ess': ess_history,
            'exec_time': exec_time,
            'particles': particle_history,
        }

        if traj_idx == 0 and run_idx == 0:
            traj_plot_dir = save_dir / 'traj_plots'
            traj_plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = traj_plot_dir / f'traj0_run0_{filter_name}.png'
            plot_trajectories_multitarget(
                results['true_states'], estimates.numpy(),
                num_targets, plot_path, filter_name,
            )

    return results


def run_monte_carlo(args):
    """Main Monte Carlo loop (sequential, TF-only, RMSE only, GPU-compatible)."""
    tf.random.set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Get device info for logging
    device_info = get_device_info()
    use_gpu = args.gpu and not args.cpu and device_info['gpu_available']

    ssm_gen = MultiTargetAcousticSSM(
        num_targets=args.num_targets,
        num_sensors=args.num_sensors,
        area_size=args.area_size,
        dt=args.dt,
        psi=args.psi,
        d0=args.d0,
        sigma_w=args.sigma_w,
        process_noise_scale=1.0,
        sensor_grid_size=args.sensor_grid_size,
    )
    ssm_filter = MultiTargetAcousticSSM(
        num_targets=args.num_targets,
        num_sensors=args.num_sensors,
        area_size=args.area_size,
        dt=args.dt,
        psi=args.psi,
        d0=args.d0,
        sigma_w=args.sigma_w,
        process_noise_scale=args.process_noise_scale,
        sensor_grid_size=args.sensor_grid_size,
    )
    sensor_positions = ssm_gen.sensor_positions

    aggregate = {
        f: {'rmse': [], 'pos_rmse': [], 'ess': [], 'exec_time': []}
        for f in args.filters
    }
    total_runs = args.n_trajectories * args.n_runs
    pbar = tqdm(total=total_runs, desc='MC runs', ncols=120) if HAS_TQDM else None

    for traj_idx in range(args.n_trajectories):
        initial_state = ssm_gen.sample_initial_state(
            num_samples=1, seed=args.seed + traj_idx
        )
        if isinstance(initial_state, tf.Tensor):
            initial_state = tf.reshape(initial_state, [-1])
        else:
            initial_state = tf.constant(initial_state, dtype=tf.float32)

        for run_idx in range(args.n_runs):
            seed_offset = traj_idx * args.n_runs + run_idx
            tf.random.set_seed(args.seed + seed_offset)
            results = run_single_trajectory(
                args, ssm_gen, ssm_filter, sensor_positions,
                initial_state, traj_idx, run_idx,
            )
            for f in args.filters:
                info = results.get(f)
                if info is None:
                    continue
                aggregate[f]['rmse'].append(info['rmse'])
                aggregate[f]['pos_rmse'].append(info['pos_rmse'])
                if info['ess'] is not None:
                    aggregate[f]['ess'].append(info['ess'])
                aggregate[f]['exec_time'].append(info['exec_time'])
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    summary = {}
    for f in args.filters:
        rmse_list = aggregate[f]['rmse']
        exec_list = aggregate[f]['exec_time']
        rmse_t = tf.constant(rmse_list, dtype=tf.float32) if rmse_list else tf.zeros([0])
        exec_t = tf.constant(exec_list, dtype=tf.float32) if exec_list else tf.zeros([0])

        summary[f] = {
            'rmse_mean': float(tf.reduce_mean(rmse_t).numpy()) if rmse_list else None,
            'rmse_std': float(tf.math.reduce_std(rmse_t).numpy()) if rmse_list else None,
            'exec_time_mean': float(tf.reduce_mean(exec_t).numpy()) if exec_list else None,
            'exec_time_std': float(tf.math.reduce_std(exec_t).numpy()) if exec_list else None,
            'pos_rmse_mean_time_series': None,
            'pos_rmse_std_time_series': None,
            'ess_mean_time_series': None,
            'ess_std_time_series': None,
        }

        if aggregate[f]['pos_rmse']:
            stacked = tf.constant(aggregate[f]['pos_rmse'], dtype=tf.float32)
            summary[f]['pos_rmse_mean_time_series'] = tf.reduce_mean(stacked, axis=0).numpy().tolist()
            summary[f]['pos_rmse_std_time_series'] = tf.math.reduce_std(stacked, axis=0).numpy().tolist()

        if aggregate[f]['ess']:
            lens = [len(e) for e in aggregate[f]['ess']]
            max_len = max(lens)
            padded = []
            for e in aggregate[f]['ess']:
                pad = [float('nan')] * (max_len - len(e))
                padded.append(e + pad)
            stacked_ess = tf.constant(padded, dtype=tf.float32)
            nan_mask = tf.math.is_nan(stacked_ess)
            valid = tf.where(nan_mask, tf.zeros_like(stacked_ess), stacked_ess)
            count = tf.reduce_sum(tf.cast(~nan_mask, tf.float32), axis=0)
            mean_ess = tf.reduce_sum(valid, axis=0) / tf.maximum(count, 1.0)
            summary[f]['ess_mean_time_series'] = mean_ess.numpy().tolist()
            diff_sq = tf.square(stacked_ess - mean_ess[tf.newaxis, :])
            var_ess = tf.reduce_sum(tf.where(nan_mask, tf.zeros_like(diff_sq), diff_sq), axis=0)
            var_ess = var_ess / tf.maximum(count, 1.0)
            summary[f]['ess_std_time_series'] = tf.sqrt(tf.maximum(var_ess, 0.0)).numpy().tolist()

    # Add experiment configuration to summary
    summary['_config'] = {
        'n_trajectories': args.n_trajectories,
        'n_runs': args.n_runs,
        'n_steps': args.n_steps,
        'num_targets': args.num_targets,
        'num_sensors': args.num_sensors,
        'n_particles': args.n_particles,
        'pf_particles': args.pf_particles,
        'gpu_used': use_gpu,
        'device': 'GPU' if use_gpu else 'CPU',
        'gpu_info': device_info['gpus'] if use_gpu else None,
    }
    
    with open(outdir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    # Build and save Monte Carlo summary as text
    device_label = 'GPU' if use_gpu else 'CPU'
    lines = [
        '',
        '=' * 80,
        f'Monte Carlo Results Summary (RMSE only, TF, {device_label})',
        '=' * 80,
    ]
    for f in args.filters:
        info = summary[f]
        rmse_mean = info['rmse_mean']
        rmse_std = info['rmse_std']
        time_mean = info['exec_time_mean']
        time_std = info['exec_time_std']
        ess_ts = info.get('ess_mean_time_series')
        if ess_ts:
            ess_t = tf.constant(ess_ts, dtype=tf.float32)
            mask = ~tf.math.is_nan(ess_t)
            cnt = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
            avg_ess = float((tf.reduce_sum(tf.where(mask, ess_t, tf.zeros_like(ess_t))) / cnt).numpy())
        else:
            avg_ess = None
        if rmse_mean is not None:
            line = f'{f.upper():12s}: RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}  '
            if avg_ess is not None:
                line += f'ESS: {avg_ess:.2f}  '
            line += f'Time: {time_mean:.4f} ± {time_std:.4f} s'
            lines.append(line)
        else:
            lines.append(f'{f.upper():12s}: No data')
    lines.extend([
        '=' * 80,
        f'Results saved to: {outdir}',
    ])
    summary_txt = '\n'.join(lines)
    summary_txt_path = outdir / 'summary.txt'
    with open(summary_txt_path, 'w') as fh:
        fh.write(summary_txt)
        fh.write('\n')

    print(summary_txt)

    ess_filters = [f for f in args.filters if f in ['pf', 'pfpf_ledh', 'pfpf_edh', 'ledh', 'edh']]
    if ess_filters:
        ess_plot_path = outdir / 'ess_vs_steps.png'
        plot_ess_vs_steps(summary, args.filters, args.n_steps, ess_plot_path, show_std=args.plot_std)
        print(f'ESS vs steps: {ess_plot_path}')

    rmse_plot_path = outdir / 'rmse_vs_steps.png'
    plot_rmse_vs_steps(summary, args.filters, args.n_steps, rmse_plot_path, show_std=args.plot_std)
    print(f'RMSE vs steps: {rmse_plot_path}')

    return summary


if __name__ == '__main__':
    args = parse_args()
    
    # Configure GPU/CPU before any TensorFlow operations
    use_gpu = args.gpu and not args.cpu
    device_str = configure_gpu(
        use_gpu=use_gpu,
        gpu_id=args.gpu_id,
        memory_growth=not args.no_memory_growth
    )
    
    # Enable mixed precision if requested (can speed up GPU computation)
    if args.mixed_precision and use_gpu:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision (float16) enabled")
        except Exception as e:
            print(f"Mixed precision not available: {e}")
    
    print('=' * 80)
    print('Monte Carlo Multi-Target Filter Comparison (TF only, RMSE)')
    print('=' * 80)
    print_device_info(device_str)
    print('-' * 80)
    print(f'Filters: {", ".join(args.filters)}')
    print(f'Targets: {args.num_targets}, Sensors: {args.num_sensors}')
    print(f'Trajectories: {args.n_trajectories}, Runs: {args.n_runs}, Steps: {args.n_steps}')
    print(f'Particles: {args.n_particles}, PF particles: {args.pf_particles}')
    print(f'Lambda steps - LEDH: {args.n_lambda_ledh}, EDH: {args.n_lambda_edh}')
    print('=' * 80)
    
    # Run with timing
    start_time = time.time()
    run_monte_carlo(args)
    total_time = time.time() - start_time
    print(f'\nTotal experiment time: {total_time:.2f}s ({total_time/60:.2f} min)')
    print(f'Device used: {device_str}')
