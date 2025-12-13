
# src/data/generators.py
from __future__ import annotations

import tensorflow as tf
from src.models.ssm_lgssm import LGSSM

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
            elif key in ["A", "B_raw", "C", "D"]:
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
