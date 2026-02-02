
# src/experiments/exp_part1_lgssm_kf.py
from __future__ import annotations

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from src.data.generators import generate_lgssm_from_yaml
from src.filters.kalman import KalmanFilter
from src.metrics.accuracy import compute_rmse

def run_experiment(config_path: str, out_dir: str) -> dict:
    """
    Run LGSSM + Kalman Filter experiment and save results.

    Parameters
    ----------
    config_path : str
        Path to the LGSSM YAML configuration.
    out_dir : str
        Output directory for results.

    Returns
    -------
    summary : dict
        Metrics and artifact paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    model, X_true, Y_obs, data_dict = generate_lgssm_from_yaml(config_path)

    # Reshape m0 to (n, 1) if needed
    try:
        if model.m0.shape.ndims == 1:
            x0_reshaped = tf.reshape(model.m0, [-1, 1])
        else:
            x0_reshaped = model.m0
    except (AttributeError, ValueError):
        x0_rank = tf.rank(model.m0)
        x0_reshaped = tf.cond(tf.equal(x0_rank, 1),
                             lambda: tf.reshape(model.m0, [-1, 1]),
                             lambda: model.m0)
    kf = KalmanFilter(
        A=model.A, C=model.C, x0=x0_reshaped, P0=model.P0,
        Q=model.Q, R=model.R
    )
    out = kf.filter(Y_obs)  # Y_obs: (N, ny)

    xf = tf.squeeze(out["x_filt"], axis=-1)  # (N, nx)
    
    # Extract position and velocity components
    xf_pos = tf.stack([xf[:, 0], xf[:, 2]], axis=1)
    X_true_pos = tf.stack([X_true[:, 0], X_true[:, 2]], axis=1)
    xf_vel = tf.stack([xf[:, 1], xf[:, 3]], axis=1)
    X_true_vel = tf.stack([X_true[:, 1], X_true[:, 3]], axis=1)

    rmse_pos = compute_rmse(xf_pos, X_true_pos)
    rmse_vel = compute_rmse(xf_vel, X_true_vel)

    # Convert to numpy for plotting and CSV
    N = int(tf.shape(xf)[0])
    t_vals = data_dict["t"].numpy()
    x_true_vals = data_dict["x_true"].numpy()
    vx_true_vals = data_dict["vx_true"].numpy()
    y_true_vals = data_dict["y_true"].numpy()
    vy_true_vals = data_dict["vy_true"].numpy()
    x_obs_vals = data_dict["x_obs"].numpy()
    y_obs_vals = data_dict["y_obs"].numpy()
    xf_vals = xf.numpy()

    # Save results as CSV
    csv_path = os.path.join(out_dir, "lgssm_kf_results.csv")
    with open(csv_path, 'w') as f:
        f.write("t,x_true,vx_true,y_true,vy_true,x_obs,y_obs,x_filt,vx_filt,y_filt,vy_filt\n")
        for i in range(N):
            f.write(f"{t_vals[i]},{x_true_vals[i]},{vx_true_vals[i]},{y_true_vals[i]},{vy_true_vals[i]},"
                   f"{x_obs_vals[i]},{y_obs_vals[i]},{xf_vals[i,0]},{xf_vals[i,1]},{xf_vals[i,2]},{xf_vals[i,3]}\n")

    # Generate plots
    plt.figure(figsize=(12, 8))
    # X position
    plt.subplot(2, 2, 1)
    plt.plot(t_vals, x_true_vals, label="True x", c="k")
    plt.plot(t_vals, x_obs_vals, label="Obs x", c="gray", alpha=0.5)
    plt.plot(t_vals, xf_vals[:, 0], label="KF x", c="tab:blue")
    plt.title("Position x"); plt.legend()

    # Y position
    plt.subplot(2, 2, 2)
    plt.plot(t_vals, y_true_vals, label="True y", c="k")
    plt.plot(t_vals, y_obs_vals, label="Obs y", c="gray", alpha=0.5)
    plt.plot(t_vals, xf_vals[:, 2], label="KF y", c="tab:blue")
    plt.title("Position y"); plt.legend()

    # vx
    plt.subplot(2, 2, 3)
    plt.plot(t_vals, vx_true_vals, label="True vx", c="k")
    plt.plot(t_vals, xf_vals[:, 1], label="KF vx", c="tab:blue")
    plt.title("Velocity x"); plt.legend()

    # vy
    plt.subplot(2, 2, 4)
    plt.plot(t_vals, vy_true_vals, label="True vy", c="k")
    plt.plot(t_vals, xf_vals[:, 3], label="KF vy", c="tab:blue")
    plt.title("Velocity y"); plt.legend()

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "lgssm_kf_fig.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return {"rmse_pos": rmse_pos, "rmse_vel": rmse_vel,
            "csv_path": csv_path, "fig_path": fig_path}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ssm_linear.yaml")
    p.add_argument("--out_dir", type=str, default="reports/1_LinearGaussianSSM/figures")
    args = p.parse_args()
    summary = run_experiment(args.config, args.out_dir)
    print(summary)
