"""
Estimate PMCMCNonlinearSSM states with PFPF-LEDH.

This experiment:
- simulates data from the Kitagawa nonlinear growth model
- runs `PFPFLEDHFilter` for sequential state estimation
- reports RMSE and saves plots/results to reports/
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.filters.pfpf_filter import PFPFLEDHFilter
from src.models.ssm_katigawa import PMCMCNonlinearSSM


class KatigawaPMCMC2DAugmentedSSM:
    """
    Augment the 1D Katigawa/PMCMC SSM to 2D so PFPFLEDHFilter can run.

    - State is [x, dummy]
    - Dynamics only model x; dummy follows identity dynamics
    - Measurement depends only on x: y = x^2 / 20 + w
    """

    def __init__(self, base_ssm: PMCMCNonlinearSSM, dummy_process_var: float = 1e-3) -> None:
        self.base_ssm = base_ssm
        self.state_dim = 2
        self.meas_per_landmark = 1

        q = tf.cast(base_ssm.Q[0, 0], tf.float32)
        r = tf.cast(base_ssm.R[0, 0], tf.float32)
        self.Q = tf.stack(
            [
                tf.stack([q, tf.constant(0.0, dtype=tf.float32)], axis=0),
                tf.stack([tf.constant(0.0, dtype=tf.float32), tf.constant(dummy_process_var, dtype=tf.float32)], axis=0),
            ],
            axis=0,
        )
        self.R = tf.reshape(r, [1, 1])

        self.initial_var = float(base_ssm.initial_var)

    def full_measurement_cov(self, num_landmarks: int | tf.Tensor = 1) -> tf.Tensor:
        del num_landmarks
        return self.R

    def motion_model(self, state: tf.Tensor, control: tf.Tensor) -> tf.Tensor:
        state = tf.cast(state, tf.float32)
        control = tf.cast(control, tf.float32)

        # state: (batch, 2), control: (batch, control_dim)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        if len(control.shape) == 1:
            control = control[tf.newaxis, :]

        x = state[:, 0]
        dummy = state[:, 1]
        t = control[:, 0]

        x_next = 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * tf.cos(1.2 * t)
        # Dummy follows identity dynamics; carries uncertainty but doesn't affect y.
        dummy_next = dummy

        return tf.stack([x_next, dummy_next], axis=1)

    def measurement_model(self, state: tf.Tensor, landmarks: tf.Tensor = None) -> tf.Tensor:
        del landmarks
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]

        x = state[:, 0]
        y = (x**2) / 20.0
        return y[:, tf.newaxis]

    def motion_jacobian(self, state: tf.Tensor, control: tf.Tensor) -> tf.Tensor:
        del control
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]

        x = state[:, 0]
        dx_dx = 0.5 + 25.0 * (1.0 - x**2) / tf.square(1.0 + x**2)
        batch = tf.shape(state)[0]

        # Jacobian is [[dx_dx, 0],[0, 1]] for each batch element.
        zeros = tf.zeros_like(dx_dx)
        ones = tf.ones_like(dx_dx)
        row0 = tf.stack([dx_dx, zeros], axis=1)   # (batch, 2)
        row1 = tf.stack([zeros, ones], axis=1)   # (batch, 2)
        J = tf.stack([row0, row1], axis=1)       # (batch, 2, 2)
        return J

    def measurement_jacobian(self, state: tf.Tensor, landmarks: tf.Tensor = None) -> tf.Tensor:
        del landmarks
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]

        x = state[:, 0]
        dy_dx = x / 10.0
        batch = tf.shape(state)[0]

        # H = [dy/dx, dy/ddummy] = [dy_dx, 0]
        H = tf.concat([dy_dx[:, tf.newaxis], tf.zeros([batch, 1], dtype=tf.float32)], axis=1)
        return H[:, tf.newaxis, :]  # (batch, meas_dim=1, state_dim=2)


def simulate_kitagawa(
    ssm: PMCMCNonlinearSSM,
    T: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Simulate latent states and observations from PMCMCNonlinearSSM dynamics."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    q_std = tf.sqrt(tf.cast(ssm.Q[0, 0], tf.float32))
    r_std = tf.sqrt(tf.cast(ssm.R[0, 0], tf.float32))

    x_curr = tf.random.normal([1], stddev=tf.sqrt(tf.cast(ssm.initial_var, tf.float32)))
    x_traj = []
    y_traj = []

    for t in range(T):
        t_float = tf.cast(t + 1, tf.float32)
        x_next = (
            0.5 * x_curr
            + 25.0 * x_curr / (1.0 + x_curr**2)
            + 8.0 * tf.cos(1.2 * t_float)
            + tf.random.normal([1], stddev=q_std)
        )
        y_t = (x_next**2) / 20.0 + tf.random.normal([1], stddev=r_std)
        x_traj.append(x_next[0])
        y_traj.append(y_t[0])
        x_curr = x_next

    return tf.stack(x_traj), tf.stack(y_traj)


def run_experiment(
    T: int = 80,
    num_particles: int = 300,
    n_lambda: int = 29,
    seed: int = 42,
) -> dict:
    ssm = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0, initial_var=10.0)
    x_true, y_obs = simulate_kitagawa(ssm=ssm, T=T, seed=seed)

    # PFPFLEDHFilter's LEDH code assumes state_dim >= 2 (it uses [:, :2] and
    # builds a 2D max-velocity vector). So we wrap the 1D model to 2D.
    ssm2 = KatigawaPMCMC2DAugmentedSSM(ssm)
    initial_state = tf.constant([0.0, 0.0], dtype=tf.float32)
    initial_cov = tf.constant(
        [[ssm.initial_var, 0.0], [0.0, 1e-3]],
        dtype=tf.float32,
    )

    pfpf = PFPFLEDHFilter(
        ssm=ssm2,
        initial_state=initial_state,
        initial_cov=initial_cov,
        num_particles=num_particles,
        n_lambda=n_lambda,
        filter_type="ekf",
        show_progress=True,
    )

    # LEDH code expects landmarks for shape bookkeeping; model ignores landmarks.
    dummy_landmarks = tf.constant([0.0], dtype=tf.float32)

    estimates = []
    ess_vals = []

    for t in range(T):
        control_t = tf.constant([float(t + 1)], dtype=tf.float32)  # time as control
        pfpf.predict(control_t)
        pfpf.update(tf.reshape(y_obs[t], [1]), dummy_landmarks)
        # Original scalar state is x = state[0]
        estimates.append(pfpf.state[0])
        ess_vals.append(float(pfpf.ess_before_resample.numpy()))

    x_hat = tf.stack(estimates)
    rmse = float(tf.sqrt(tf.reduce_mean((x_hat - x_true) ** 2)).numpy())

    return {
        "x_true": x_true.numpy(),
        "y_obs": y_obs.numpy(),
        "x_hat": x_hat.numpy(),
        "ess": np.array(ess_vals, dtype=np.float32),
        "rmse": rmse,
        "config": {
            "T": T,
            "num_particles": num_particles,
            "n_lambda": n_lambda,
            "seed": seed,
        },
    }


def save_outputs(results: dict) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "reports" / "6_BonusQ1_HMC_Invertible_Flows" / "PFPF_LEDH"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_true = results["x_true"]
    x_hat = results["x_hat"]
    y_obs = results["y_obs"]
    ess = results["ess"]
    rmse = results["rmse"]

    # Combined diagnostics: state + measurement fit + ESS
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax_state, ax_meas, ax_ess = axes

    ax_state.plot(x_true, label="True state", linewidth=2)
    ax_state.plot(x_hat, label="PFPF-LEDH estimate", linewidth=2, linestyle="--")
    ax_state.set_title(f"PFPF-LEDH on Kitagawa model (RMSE={rmse:.4f})")
    ax_state.set_ylabel("State value")
    ax_state.grid(True, alpha=0.3)
    ax_state.legend(loc="best")

    ax_meas.plot(y_obs, label="Observations", alpha=0.7)
    ax_meas.plot((x_hat**2) / 20.0, label="Predicted measurement from estimate", alpha=0.9)
    ax_meas.set_ylabel("Measurement value")
    ax_meas.set_title("Observation consistency")
    ax_meas.grid(True, alpha=0.3)
    ax_meas.legend(loc="best")

    ax_ess.plot(ess, color="tab:purple")
    ax_ess.set_xlabel("Time step")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size before resampling")
    ax_ess.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "pfpf_ledh_diagnostics_combined.png", dpi=180)
    plt.close(fig)

    # Trajectory plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_true, label="True state", linewidth=2)
    ax.plot(x_hat, label="PFPF-LEDH estimate", linewidth=2)
    ax.set_title(f"PFPF-LEDH on Kitagawa model (RMSE={rmse:.4f})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("State value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "state_trajectory.png", dpi=180)
    plt.close(fig)

    # Observation + estimate plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_obs, label="Observations", alpha=0.7)
    ax.plot((x_hat**2) / 20.0, label="Predicted measurement from estimate", alpha=0.9)
    ax.set_title("Observation consistency")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Measurement value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "measurement_fit.png", dpi=180)
    plt.close(fig)

    # ESS plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ess, color="tab:purple")
    ax.set_title("Effective Sample Size before resampling")
    ax.set_xlabel("Time step")
    ax.set_ylabel("ESS")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ess_trace.png", dpi=180)
    plt.close(fig)

    summary_path = out_dir / "results.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("PFPF-LEDH on PMCMCNonlinearSSM (Kitagawa)\n")
        f.write("=" * 60 + "\n")
        for k, v in results["config"].items():
            f.write(f"{k}: {v}\n")
        f.write(f"rmse: {rmse:.6f}\n")
        f.write(f"mean_ess: {float(np.mean(ess)):.4f}\n")
        f.write(f"min_ess: {float(np.min(ess)):.4f}\n")

    np.savez(
        out_dir / "timeseries.npz",
        x_true=x_true,
        y_obs=y_obs,
        x_hat=x_hat,
        ess=ess,
    )

    print(f"Saved outputs to: {out_dir}")
    print(f"RMSE: {rmse:.6f}")


def main():
    results = run_experiment(T=80, num_particles=300, n_lambda=29, seed=42)
    save_outputs(results)


if __name__ == "__main__":
    main()

