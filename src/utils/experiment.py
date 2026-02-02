"""
Experiment utilities for filter evaluation.

This module provides shared functionality for running filter experiments,
including trajectory generation, filter evaluation, and result aggregation.

"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import tensorflow as tf

from src.metrics.accuracy import compute_rmse


@dataclass
class FilterResult:
    """Container for filter evaluation results."""
    filter_name: str
    estimates: tf.Tensor
    rmse: float
    execution_time: float
    success: bool = True
    error_message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def run_filter_on_trajectory(
    filter_obj,
    ssm,
    true_states: tf.Tensor,
    measurements: tf.Tensor,
    controls: Optional[tf.Tensor] = None,
    observation_available: Optional[tf.Tensor] = None,
    sensor_positions: Optional[tf.Tensor] = None,
    landmarks: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, float]:
    """
    Run a filter on a pre-generated trajectory.
    
    This is a generic filter evaluation function that works with any filter
    and SSM combination.
    
    Parameters
    ----------
    filter_obj : object
        Filter instance with predict() and update() methods.
    ssm : object
        State-space model (used for getting observation indices, etc.).
    true_states : tf.Tensor
        Ground truth states of shape (T, state_dim).
    measurements : tf.Tensor
        Observations of shape (T, meas_dim) or ragged for varying obs.
    controls : tf.Tensor, optional
        Control inputs of shape (T, control_dim). Default None (no control).
    observation_available : tf.Tensor, optional
        Boolean tensor of shape (T,) indicating which time steps have obs.
        Default None (all steps have observations).
    sensor_positions : tf.Tensor, optional
        Sensor positions for acoustic SSMs.
    landmarks : tf.Tensor, optional
        Landmark positions for range-bearing SSMs.
        
    Returns
    -------
    estimates : tf.Tensor
        Filter state estimates of shape (T, state_dim).
    rmse : float
        Root mean square error over the trajectory.
        
    Examples
    --------
    >>> estimates, rmse = run_filter_on_trajectory(ekf, ssm, states, measurements)
    """
    T = tf.shape(true_states)[0]
    state_dim = tf.shape(true_states)[1]
    estimates_list = []
    
    # Default: all observations available
    if observation_available is None:
        observation_available = tf.ones(T, dtype=tf.bool)
    
    # Check filter type for method signatures
    has_landmarks = hasattr(filter_obj, 'update') and landmarks is not None
    has_sensors = sensor_positions is not None
    
    for t in range(T):
        # Prediction step
        if controls is not None:
            control = controls[t]
            if hasattr(filter_obj, 'predict'):
                filter_obj.predict(control)
        else:
            if hasattr(filter_obj, 'predict'):
                try:
                    filter_obj.predict()
                except TypeError:
                    # Some filters require control input
                    filter_obj.predict(tf.zeros([2], dtype=tf.float32))
        
        # Update step (if observation available)
        if observation_available[t]:
            meas = measurements[t]
            
            try:
                if has_landmarks:
                    filter_obj.update(meas, landmarks)
                elif has_sensors:
                    # For acoustic SSMs, get observation indices
                    if hasattr(ssm, 'get_obs_indices_for_sensor_positions'):
                        obs_indices = ssm.get_obs_indices_for_sensor_positions(sensor_positions)
                        filter_obj.update(meas, obs_indices=obs_indices)
                    else:
                        filter_obj.update(meas)
                else:
                    filter_obj.update(meas)
            except Exception:
                # If update fails, skip this step
                pass
        
        # Store estimate
        if hasattr(filter_obj, 'state'):
            state = filter_obj.state
        elif hasattr(filter_obj, 'x_hat'):
            state = filter_obj.x_hat
        elif hasattr(filter_obj, 'm'):
            state = filter_obj.m
        else:
            state = tf.zeros([state_dim], dtype=tf.float32)
        
        # Ensure state is 1D
        state = tf.reshape(state, [-1])
        estimates_list.append(state)
    
    estimates = tf.stack(estimates_list, axis=0)
    rmse = compute_rmse(estimates, true_states)
    
    return estimates, rmse


def run_filter_with_timing(
    filter_obj,
    ssm,
    true_states: tf.Tensor,
    measurements: tf.Tensor,
    **kwargs
) -> FilterResult:
    """
    Run filter on trajectory with timing and error handling.
    
    Parameters
    ----------
    filter_obj : object
        Filter instance.
    ssm : object
        State-space model.
    true_states : tf.Tensor
        Ground truth states.
    measurements : tf.Tensor
        Observations.
    **kwargs
        Additional arguments passed to run_filter_on_trajectory.
        
    Returns
    -------
    FilterResult
        Dataclass containing estimates, RMSE, timing, and success status.
    """
    filter_name = type(filter_obj).__name__
    
    start_time = time.perf_counter()
    try:
        estimates, rmse = run_filter_on_trajectory(
            filter_obj, ssm, true_states, measurements, **kwargs
        )
        execution_time = time.perf_counter() - start_time
        
        return FilterResult(
            filter_name=filter_name,
            estimates=estimates,
            rmse=rmse,
            execution_time=execution_time,
            success=True
        )
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        return FilterResult(
            filter_name=filter_name,
            estimates=tf.zeros_like(true_states),
            rmse=float('inf'),
            execution_time=execution_time,
            success=False,
            error_message=str(e)
        )


def generate_generic_trajectory(
    ssm,
    n_steps: int,
    x0: Optional[tf.Tensor] = None,
    controls: Optional[tf.Tensor] = None,
    observation_rate: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Generate a trajectory from any SSM.
    
    This generic function works with different SSM types by checking
    available methods and attributes.
    
    Parameters
    ----------
    ssm : object
        State-space model with motion_model and measurement_model methods.
    n_steps : int
        Number of time steps.
    x0 : tf.Tensor, optional
        Initial state. If None, samples from SSM if available.
    controls : tf.Tensor, optional
        Control inputs of shape (n_steps, control_dim).
    observation_rate : float, optional
        Fraction of time steps with observations (0 to 1). Default 1.0.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    states : tf.Tensor
        True states of shape (n_steps, state_dim).
    measurements : tf.Tensor
        Observations of shape (n_steps, meas_dim).
    controls_out : tf.Tensor
        Control inputs used, shape (n_steps, control_dim).
    obs_available : tf.Tensor
        Boolean array indicating observation availability.
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    # Get initial state
    if x0 is None:
        if hasattr(ssm, 'sample_initial_state'):
            x0 = ssm.sample_initial_state(num_samples=1)
            if len(x0.shape) > 1:
                x0 = x0[0]
        else:
            x0 = tf.zeros([ssm.state_dim], dtype=tf.float32)
    
    x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
    x0 = tf.reshape(x0, [-1])
    state_dim = x0.shape[0]
    
    # Generate controls if not provided
    if controls is None:
        if hasattr(ssm, 'control_dim') and ssm.control_dim > 0:
            controls = tf.zeros([n_steps, ssm.control_dim], dtype=tf.float32)
        else:
            controls = tf.zeros([n_steps, 2], dtype=tf.float32)
    
    # Observation availability
    obs_available = tf.random.uniform([n_steps]) < observation_rate
    obs_available = tf.cast(obs_available, tf.bool)
    
    # Generate trajectory
    states_list = [x0]
    x = x0
    
    for t in range(1, n_steps):
        control = controls[t]
        
        # Propagate through motion model
        x_batch = tf.expand_dims(x, 0)
        control_batch = tf.expand_dims(control, 0)
        
        try:
            if hasattr(ssm, 'motion_model'):
                x_next = ssm.motion_model(x_batch, control_batch)[0]
            else:
                x_next = x_batch[0]
        except TypeError:
            # Some SSMs don't need control
            x_next = ssm.motion_model(x_batch)[0]
        
        # Add process noise
        if hasattr(ssm, 'Q'):
            try:
                L_Q = tf.linalg.cholesky(ssm.Q)
                noise = tf.random.normal([state_dim], dtype=tf.float32)
                x_next = x_next + noise @ tf.transpose(L_Q)
            except Exception:
                pass
        
        x = x_next
        states_list.append(x)
    
    states = tf.stack(states_list, axis=0)
    
    # Generate measurements
    measurements_list = []
    for t in range(n_steps):
        x_t = tf.expand_dims(states[t], 0)
        
        try:
            if hasattr(ssm, 'landmarks'):
                meas = ssm.measurement_model(x_t, ssm.landmarks)[0]
            elif hasattr(ssm, 'sensor_positions'):
                meas = ssm.measurement_model(x_t)[0]
            else:
                meas = ssm.measurement_model(x_t)[0]
        except Exception:
            meas = tf.zeros([ssm.meas_dim if hasattr(ssm, 'meas_dim') else 1], dtype=tf.float32)
        
        # Add measurement noise
        if hasattr(ssm, 'R'):
            R = ssm.R
            meas_dim = tf.shape(meas)[0]
            if len(R.shape) == 0 or R.shape[0] != meas_dim:
                R = tf.eye(meas_dim, dtype=tf.float32) * 0.1
            try:
                L_R = tf.linalg.cholesky(R)
                noise = tf.random.normal([meas_dim], dtype=tf.float32)
                meas = meas + noise @ tf.transpose(L_R)
            except Exception:
                pass
        
        meas = tf.reshape(meas, [-1])
        measurements_list.append(meas)
    
    # Pad measurements to same length
    max_meas_dim = max(m.shape[0] for m in measurements_list)
    measurements_padded = []
    for m in measurements_list:
        if m.shape[0] < max_meas_dim:
            m = tf.pad(m, [[0, max_meas_dim - m.shape[0]]])
        measurements_padded.append(m)
    
    measurements = tf.stack(measurements_padded, axis=0)
    
    return states, measurements, controls, obs_available


def aggregate_results(
    results: List[FilterResult],
    group_by: str = 'filter_name'
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple filter runs.
    
    Parameters
    ----------
    results : list of FilterResult
        List of filter results from multiple runs.
    group_by : str
        Field to group results by. Default 'filter_name'.
        
    Returns
    -------
    dict
        Dictionary mapping group keys to aggregated statistics.
    """
    from collections import defaultdict
    import math
    
    grouped = defaultdict(list)
    for r in results:
        key = getattr(r, group_by, 'unknown')
        grouped[key].append(r)
    
    aggregated = {}
    for key, group in grouped.items():
        rmses = [r.rmse for r in group if r.success and math.isfinite(r.rmse)]
        times = [r.execution_time for r in group if r.success]
        success_count = sum(1 for r in group if r.success)
        
        if rmses:
            mean_rmse = sum(rmses) / len(rmses)
            std_rmse = (sum((x - mean_rmse)**2 for x in rmses) / len(rmses)) ** 0.5
        else:
            mean_rmse = float('nan')
            std_rmse = float('nan')
        
        if times:
            mean_time = sum(times) / len(times)
        else:
            mean_time = float('nan')
        
        aggregated[key] = {
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_time': mean_time,
            'success_rate': success_count / len(group) if group else 0.0,
            'n_runs': len(group),
        }
    
    return aggregated


def print_results_table(
    aggregated: Dict[str, Dict[str, float]],
    title: str = "Filter Comparison Results"
) -> None:
    """
    Print aggregated results as a formatted table.
    
    Parameters
    ----------
    aggregated : dict
        Aggregated results from aggregate_results().
    title : str
        Table title.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"{'Filter':<15} {'RMSE':>12} {'Std':>10} {'Time (s)':>10} {'Success':>10}")
    print(f"{'-' * 60}")
    
    # Sort by mean RMSE
    sorted_filters = sorted(aggregated.items(), key=lambda x: x[1].get('mean_rmse', float('inf')))
    
    for filter_name, stats in sorted_filters:
        rmse = stats.get('mean_rmse', float('nan'))
        std = stats.get('std_rmse', float('nan'))
        time_s = stats.get('mean_time', float('nan'))
        success = stats.get('success_rate', 0.0) * 100
        
        rmse_str = f"{rmse:.4f}" if rmse == rmse else "N/A"
        std_str = f"{std:.4f}" if std == std else "N/A"
        time_str = f"{time_s:.3f}" if time_s == time_s else "N/A"
        
        print(f"{filter_name:<15} {rmse_str:>12} {std_str:>10} {time_str:>10} {success:>9.0f}%")
    
    print(f"{'=' * 60}\n")
