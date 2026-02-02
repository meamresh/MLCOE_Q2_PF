"""
Data generators for state-space model experiments.

This module provides functions for:
- Loading SSMs from YAML configuration
- Generating synthetic trajectories
- Creating test data for filter evaluation

"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import tensorflow as tf

from src.models.ssm_lgssm import LGSSM
from src.models.ssm_range_bearing import RangeBearingSSM


def _parse_yaml_simple(config_path: str) -> dict:
    """
    Simple YAML parser using basic string parsing.
    Only handles the specific format used in this project.
    """
    with open(config_path, "r") as f:
        content = f.read()
    
    cfg = {}
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        if ':' in line and not line.startswith(' '):
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == "name":
                cfg[key] = value
            elif key in ["seed", "N", "dt"]:
                cfg[key] = int(value) if '.' not in value else float(value)
            elif key in ["scenario"]:
                # Simple string value
                cfg[key] = value
            elif key == "dimensions":
                cfg[key] = {}
                i += 1
                while i < len(lines):
                    orig_line = lines[i]
                    # Check if line is indented (starts with spaces or tabs)
                    if orig_line and (orig_line[0] == ' ' or orig_line[0] == '\t'):
                        dim_line = orig_line.strip()
                        if not dim_line or dim_line.startswith('#'):
                            i += 1
                            continue
                        if ':' in dim_line:
                            dim_key, dim_value = dim_line.split(':', 1)
                            # Extract the number before any comment
                            dim_value_clean = dim_value.strip().split()[0] if dim_value.strip().split() else "0"
                            cfg[key][dim_key.strip()] = int(dim_value_clean)
                        i += 1
                    else:
                        # Not indented, we've reached the end of this section
                        break
                continue
            elif key == "params":
                cfg[key] = {}
                i += 1
                while i < len(lines):
                    orig_line = lines[i]
                    # Check if line is indented (starts with spaces or tabs)
                    if orig_line and (orig_line[0] == ' ' or orig_line[0] == '\t'):
                        param_line = orig_line.strip()
                        if not param_line or param_line.startswith('#'):
                            i += 1
                            continue
                        if ':' in param_line:
                            param_key, param_value = param_line.split(':', 1)
                            param_value = param_value.strip()
                            # Remove comments
                            if '#' in param_value:
                                param_value = param_value.split('#')[0].strip()
                            if param_value.startswith('['):
                                # List value
                                cfg[key][param_key.strip()] = [float(x.strip()) for x in param_value[1:-1].split(',')]
                            else:
                                cfg[key][param_key.strip()] = float(param_value)
                        i += 1
                    else:
                        # Not indented, we've reached the end of this section
                        break
                continue
            elif key in ["A", "B_raw", "C", "D", "landmarks"]:
                cfg[key] = []
                i += 1
                while i < len(lines):
                    row_line = lines[i].strip()
                    if not row_line or row_line.startswith('#'):
                        i += 1
                        continue
                    if row_line.startswith('- ['):
                        row_str = row_line[2:].strip()
                        if row_str.startswith('[') and row_str.endswith(']'):
                            row = [float(x.strip()) for x in row_str[1:-1].split(',')]
                            cfg[key].append(row)
                    elif not row_line.startswith('-') and ':' in row_line:
                        # Next top-level key, stop parsing this matrix
                        break
                    i += 1
                continue
            elif key in ["Q_diag", "R_diag", "P0_diag", "m0"]:
                # Simple list-valued numeric fields
                if value.startswith('[') and value.endswith(']'):
                    cfg[key] = [float(x.strip()) for x in value[1:-1].split(',')]
                else:
                    # Fallback to single float
                    cfg[key] = float(value) if value else 0.0
                i += 1
                continue
        i += 1
    
    return cfg

def generate_lgssm_from_yaml(config_path: str):
    """
    Load an LGSSM from YAML and generate synthetic data.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.

    Returns
    -------
    model : LGSSM
        The instantiated LGSSM.
    X : tf.Tensor
        True states, shape (N, nx).
    Y : tf.Tensor
        Observations, shape (N, ny).
    data_dict : dict
        Dictionary with time, true states, and observations as tensors.
    """
    cfg = _parse_yaml_simple(config_path)

    model = LGSSM.from_config(cfg)
    N = int(cfg["N"])
    seed = int(cfg.get("seed", 0))
    X, Y = model.sample(N=N, seed=seed)

    # Convert to numpy for easier indexing (then back to tensor if needed)
    X_np = X.numpy() if hasattr(X, 'numpy') else X
    Y_np = Y.numpy() if hasattr(Y, 'numpy') else Y
    
    data_dict = {
        "t": tf.range(N, dtype=tf.float32),
        "x_true": tf.constant(X_np[:, 0], dtype=tf.float32),
        "vx_true": tf.constant(X_np[:, 1], dtype=tf.float32),
        "y_true": tf.constant(X_np[:, 2], dtype=tf.float32),
        "vy_true": tf.constant(X_np[:, 3], dtype=tf.float32),
        "x_obs": tf.constant(Y_np[:, 0], dtype=tf.float32),
        "y_obs": tf.constant(Y_np[:, 1], dtype=tf.float32),
    }
    return model, X, Y, data_dict


def generate_range_bearing_from_yaml(config_path: str):
    """
    Load a RangeBearingSSM from YAML configuration.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.

    Returns
    -------
    model : RangeBearingSSM
        The instantiated nonlinear state-space model.
    landmarks : tf.Tensor
        Landmark positions, shape (M, 2).
    m0 : tf.Tensor
        Initial state mean, shape (3,).
    P0 : tf.Tensor
        Initial covariance matrix, shape (3, 3).
    meta : dict
        Dictionary with additional config fields such as 'scenario' and 'seed'.
    """
    cfg = _parse_yaml_simple(config_path)

    dt = float(cfg.get("dt", 0.1))

    # Process and measurement noise covariances
    q_diag = cfg.get("Q_diag", [0.01, 0.01, 0.01])
    r_diag = cfg.get("R_diag", [0.05, 0.05])

    Q = tf.linalg.diag(tf.constant(q_diag, dtype=tf.float32))
    R = tf.linalg.diag(tf.constant(r_diag, dtype=tf.float32))

    model = RangeBearingSSM(dt=dt, process_noise=Q, meas_noise=R)

    # Landmarks
    landmarks = tf.constant(cfg.get("landmarks", []), dtype=tf.float32)

    # Initial state and covariance
    m0 = tf.constant(cfg.get("m0", [0.0, 0.0, 0.0]), dtype=tf.float32)
    P0_diag = cfg.get("P0_diag", [0.5, 0.5, 0.5])
    P0 = tf.linalg.diag(tf.constant(P0_diag, dtype=tf.float32))

    meta = {
        "name": cfg.get("name", "range_bearing_ssm"),
        "scenario": cfg.get("scenario", "moderate"),
        "seed": int(cfg.get("seed", 0)),
    }

    return model, landmarks, m0, P0, meta


# ============================================================================
# Generic Trajectory Generation
# ============================================================================

def generate_trajectory(
    ssm,
    n_steps: int,
    x0: Optional[tf.Tensor] = None,
    controls: Optional[tf.Tensor] = None,
    observation_rate: float = 1.0,
    measurement_correlation: float = 0.0,
    landmarks: Optional[tf.Tensor] = None,
    seed: Optional[int] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Generate a trajectory from any state-space model.
    
    This generic function works with different SSM types (LGSSM, RangeBearing,
    MultiTargetAcoustic, Lorenz96) by checking available methods and attributes.
    
    Parameters
    ----------
    ssm : object
        State-space model with motion_model and measurement_model methods.
    n_steps : int
        Number of time steps.
    x0 : tf.Tensor, optional
        Initial state of shape (state_dim,). If None, samples from SSM.
    controls : tf.Tensor, optional
        Control inputs of shape (n_steps, control_dim). Default zeros.
    observation_rate : float, optional
        Fraction of time steps with observations (0 to 1). Default 1.0.
    measurement_correlation : float, optional
        Correlation coefficient between measurement dimensions. Default 0.0.
    landmarks : tf.Tensor, optional
        Landmark positions for range-bearing SSMs.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    states : tf.Tensor
        True states of shape (n_steps, state_dim).
    measurements : tf.Tensor
        Observations of shape (n_steps, meas_dim).
    controls_out : tf.Tensor
        Control inputs used.
    obs_available : tf.Tensor
        Boolean tensor indicating observation availability.
        
    Examples
    --------
    >>> ssm = RangeBearingSSM(dt=0.1)
    >>> states, meas, ctrl, obs = generate_trajectory(ssm, 100)
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    # Get state dimension
    state_dim = ssm.state_dim if hasattr(ssm, 'state_dim') else ssm.n
    
    # Get initial state
    if x0 is None:
        if hasattr(ssm, 'sample_initial_state'):
            x0 = ssm.sample_initial_state(num_samples=1)
            if len(x0.shape) > 1:
                x0 = x0[0]
        elif hasattr(ssm, 'm0'):
            x0 = ssm.m0
        else:
            x0 = tf.zeros([state_dim], dtype=tf.float32)
    
    x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
    x0 = tf.reshape(x0, [-1])
    
    # Get landmarks for range-bearing SSMs
    if landmarks is None and hasattr(ssm, 'landmarks'):
        landmarks = ssm.landmarks
    
    # Generate controls if not provided
    if controls is None:
        if hasattr(ssm, 'control_dim') and ssm.control_dim > 0:
            controls = tf.zeros([n_steps, ssm.control_dim], dtype=tf.float32)
        else:
            controls = tf.zeros([n_steps, 2], dtype=tf.float32)
    
    # Observation availability mask
    if observation_rate >= 1.0:
        obs_available = tf.ones([n_steps], dtype=tf.bool)
    else:
        obs_available = tf.random.uniform([n_steps]) < observation_rate
    
    # Get process noise covariance
    Q = ssm.Q if hasattr(ssm, 'Q') else tf.eye(state_dim, dtype=tf.float32) * 0.01
    try:
        L_Q = tf.linalg.cholesky(Q)
    except Exception:
        Q_reg = Q + 1e-6 * tf.eye(state_dim, dtype=Q.dtype)
        L_Q = tf.linalg.cholesky(Q_reg)
    
    # Generate state trajectory
    states_list = [x0]
    x = x0
    
    for t in range(1, n_steps):
        control = controls[t]
        x_batch = tf.expand_dims(x, 0)
        control_batch = tf.expand_dims(control, 0)
        
        # Propagate through motion model
        try:
            if hasattr(ssm, 'motion_model'):
                x_next = ssm.motion_model(x_batch, control_batch)[0]
            else:
                x_next = x
        except TypeError:
            # Some SSMs (e.g., Lorenz96) don't use control
            try:
                x_next = ssm.motion_model(x_batch)[0]
            except Exception:
                if hasattr(ssm, 'step'):
                    x_next = ssm.step(x)
                else:
                    x_next = x
        
        # Add process noise
        noise = tf.random.normal([state_dim], dtype=tf.float32)
        x_next = x_next + noise @ tf.transpose(L_Q)
        
        x = x_next
        states_list.append(x)
    
    states = tf.stack(states_list, axis=0)
    
    # Generate measurements
    measurements_list = []
    meas_dim = None
    
    for t in range(n_steps):
        x_t = tf.expand_dims(states[t], 0)
        
        try:
            if landmarks is not None:
                meas = ssm.measurement_model(x_t, landmarks)[0]
            elif hasattr(ssm, 'sensor_positions'):
                meas = ssm.measurement_model(x_t)[0]
            else:
                meas = ssm.measurement_model(x_t)[0]
        except Exception:
            if meas_dim is None:
                meas_dim = ssm.meas_dim if hasattr(ssm, 'meas_dim') else ssm.m if hasattr(ssm, 'm') else 1
            meas = tf.zeros([meas_dim], dtype=tf.float32)
        
        meas = tf.reshape(meas, [-1])
        
        if meas_dim is None:
            meas_dim = meas.shape[0]
        
        measurements_list.append(meas)
    
    # Get measurement noise covariance
    R = ssm.R if hasattr(ssm, 'R') else tf.eye(meas_dim, dtype=tf.float32) * 0.1
    if len(R.shape) == 0:
        R = tf.eye(meas_dim, dtype=tf.float32) * R
    elif R.shape[0] != meas_dim:
        if landmarks is not None and hasattr(ssm, 'full_measurement_cov'):
            num_landmarks = tf.shape(landmarks)[0]
            R = ssm.full_measurement_cov(num_landmarks)
        else:
            R = tf.eye(meas_dim, dtype=tf.float32) * 0.1
    
    # Apply correlation if specified
    if measurement_correlation != 0.0:
        R_base = R
        # Build correlation structure
        indices = tf.cast(tf.range(meas_dim), tf.float32)
        i_grid, j_grid = tf.meshgrid(indices, indices, indexing='ij')
        corr_matrix = measurement_correlation ** tf.abs(i_grid - j_grid)
        # Scale by R diagonal
        R_diag = tf.sqrt(tf.linalg.diag_part(R_base))
        R = corr_matrix * tf.outer(R_diag, R_diag)
    
    # Add measurement noise
    try:
        L_R = tf.linalg.cholesky(R)
    except Exception:
        R_reg = R + 1e-6 * tf.eye(meas_dim, dtype=R.dtype)
        L_R = tf.linalg.cholesky(R_reg)
    
    # Stack measurements and add noise
    measurements = tf.stack(measurements_list, axis=0)
    meas_noise = tf.random.normal([n_steps, meas_dim], dtype=tf.float32)
    measurements = measurements + meas_noise @ tf.transpose(L_R)
    
    return states, measurements, controls, obs_available


def generate_range_bearing_trajectory(
    ssm,
    n_steps: int,
    landmarks: tf.Tensor,
    x0: Optional[tf.Tensor] = None,
    velocity: float = 1.0,
    turn_rate: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Generate a range-bearing navigation trajectory.
    
    Parameters
    ----------
    ssm : RangeBearingSSM
        Range-bearing state-space model.
    n_steps : int
        Number of time steps.
    landmarks : tf.Tensor
        Landmark positions of shape (n_landmarks, 2).
    x0 : tf.Tensor, optional
        Initial state [x, y, theta]. Default [0, 0, 0].
    velocity : float
        Constant velocity. Default 1.0.
    turn_rate : float
        Constant turn rate. Default 0.0.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    states : tf.Tensor
        True states of shape (n_steps, 3).
    measurements : tf.Tensor
        Range-bearing observations of shape (n_steps, 2*n_landmarks).
    controls : tf.Tensor
        Control inputs [v, omega].
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    if x0 is None:
        x0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    
    # Constant velocity/turn rate controls
    controls = tf.tile(
        tf.constant([[velocity, turn_rate]], dtype=tf.float32),
        [n_steps, 1]
    )
    
    states, measurements, _, _ = generate_trajectory(
        ssm, n_steps, x0=x0, controls=controls, landmarks=landmarks, seed=seed
    )
    
    return states, measurements, controls


def generate_acoustic_trajectory(
    ssm,
    n_steps: int,
    x0: Optional[tf.Tensor] = None,
    seed: Optional[int] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate a multi-target acoustic tracking trajectory.
    
    Parameters
    ----------
    ssm : MultiTargetAcousticSSM
        Multi-target acoustic state-space model.
    n_steps : int
        Number of time steps.
    x0 : tf.Tensor, optional
        Initial state. If None, samples from SSM.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    states : tf.Tensor
        True states of shape (n_steps, state_dim).
    measurements : tf.Tensor
        Acoustic power observations of shape (n_steps, n_sensors).
    """
    states, measurements, _, _ = generate_trajectory(
        ssm, n_steps, x0=x0, seed=seed
    )
    return states, measurements
