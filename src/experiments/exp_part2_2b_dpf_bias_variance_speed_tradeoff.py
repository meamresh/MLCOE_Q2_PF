"""
Reproduce Table 1 from Corenflos et al. (ICML 2021) as in filterflow.

This script matches the filterflow implementation:
- Uses same parameters as simple_linear_comparison.py
- T=150, N=25, batch_size=100
- epsilon values: 0.25, 0.5, 0.75
- Compares PF vs DPF with different epsilon values
- Includes soft-resampling (mixture with uniform) baseline with different alpha values
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from pathlib import Path
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from src.filters.dpf import (
    BootstrapModel,
    DifferentiableParticleFilter,
    StandardParticleFilter,
    SoftResamplingParticleFilter,
)
from src.models.ssm_lgssm import LGSSM

tfd = tfp.distributions


class LGSSMWrapper:
    """
    Minimal wrapper to expose an LGSSM as a BootstrapModel-compatible interface.

    Provides:
      - sample_initial(N, y1) -> x1, log_w1
      - step(t, x_prev, y_t)  -> x_t, log_w_t
    where log_w_t is the observation log-likelihood log p(y_t | x_t).
    """

    def __init__(self, lgssm: LGSSM):
        self.lgssm = lgssm
        self.Q = lgssm.Q
        self.R = lgssm.R
        self.nx = lgssm.nx
        self.ny = lgssm.ny

    def sample_initial(self, N: int, y1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample x1 ~ N(m0, P0) and return log p(y1 | x1)."""
        P0_chol = tf.linalg.cholesky(self.lgssm.P0)
        eps = tf.random.normal((N, self.lgssm.nx), dtype=y1.dtype)
        x1 = self.lgssm.m0 + tf.matmul(eps, P0_chol, transpose_b=True)
        log_w1 = self.log_obs_density(x1, y1)
        return x1, log_w1

    def step(self, t: int, x_prev: tf.Tensor, y_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Linear Gaussian step:
            x_t = A x_{t-1} + B v_t,   v_t ~ N(0, I)
            y_t | x_t ~ N(C x_t, R)
        """
        batch_size = tf.shape(x_prev)[0]
        v_t = tf.random.normal((batch_size, self.lgssm.nv), dtype=x_prev.dtype)
        x_t = tf.matmul(x_prev, self.lgssm.A, transpose_b=True) + tf.matmul(
            v_t, self.lgssm.B, transpose_b=True
        )
        log_w_t = self.log_obs_density(x_t, y_t)
        return x_t, log_w_t

    def log_obs_density(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Compute log p(y | x) for linear Gaussian observations y = Cx + D w."""
        mean_y = tf.matmul(x, self.lgssm.C, transpose_b=True)  # (batch, ny)
        diff = tf.expand_dims(y, axis=0) - mean_y              # (batch, ny)
        R = self.lgssm.R
        inv_R = tf.linalg.inv(R)
        logdet_R = tf.linalg.logdet(R)
        quad = tf.reduce_sum(diff * tf.matmul(diff, inv_R), axis=-1)
        d = tf.cast(self.lgssm.ny, x.dtype)
        return -0.5 * (quad + logdet_R + d * tf.math.log(2.0 * math.pi))


def build_lgssm_for_theta(theta: tf.Tensor) -> LGSSM:
    """
    Build a 2D LGSSM matching the Corenflos/filterflow linear benchmark:
        x_t = diag(theta) x_{t-1} + v_t,   v_t ~ N(0, I_2)
        y_t = x_t + w_t,                   w_t ~ N(0, 0.1 I_2)
    """
    theta = tf.convert_to_tensor(theta, dtype=tf.float32)
    A = tf.linalg.diag(theta)                        # (2, 2)
    B = tf.eye(2, dtype=tf.float32)                  # so Q = I_2
    C = tf.eye(2, dtype=tf.float32)
    D = tf.sqrt(0.1) * tf.eye(2, dtype=tf.float32)   # so R = 0.1 I_2
    m0 = tf.zeros(2, dtype=tf.float32)
    P0 = tf.eye(2, dtype=tf.float32)
    Q = tf.matmul(B, B, transpose_b=True)
    R = tf.matmul(D, D, transpose_b=True)
    return LGSSM(
        A=A,
        B=B,
        C=C,
        D=D,
        m0=m0,
        P0=P0,
        Q=Q,
        R=R,
        nx=2,
        ny=2,
        nv=2,
        nw=2,
    )


def build_bootstrap_model_from_theta(theta: tf.Tensor) -> BootstrapModel:
    """Helper to build a BootstrapModel using LGSSM + wrapper for a given theta."""
    lgssm = build_lgssm_for_theta(theta)
    wrapped = LGSSMWrapper(lgssm)
    return BootstrapModel(sample_initial=wrapped.sample_initial, transition=wrapped.step)


def simulate_data(theta: tf.Tensor, T: int = 150, seed: int = 111) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simulate data from linear Gaussian SSM using NumPy RNG (matches filterflow's get_data)."""
    rng = np.random.RandomState(seed=seed)

    theta_np = theta.numpy()
    A = np.diag(theta_np)
    process_cov = np.eye(2, dtype=np.float32)  # Matches filterflow
    obs_cov = 0.1 * np.eye(2, dtype=np.float32)
    chol_process = np.linalg.cholesky(process_cov)
    chol_obs = np.linalg.cholesky(obs_cov)

    x_list = []
    y_list = []

    x0 = rng.randn(2).astype(np.float32)
    x_list.append(x0)
    y0 = x0 + (chol_obs @ rng.randn(2)).astype(np.float32)
    y_list.append(y0)

    for _ in range(1, T):
        mean = A @ x_list[-1]
        x_t = mean + (chol_process @ rng.randn(2)).astype(np.float32)
        x_list.append(x_t)
        y_t = x_t + (chol_obs @ rng.randn(2)).astype(np.float32)
        y_list.append(y_t)

    x = tf.constant(np.stack(x_list, axis=0), dtype=tf.float32)  # (T, 2)
    y = tf.constant(np.stack(y_list, axis=0), dtype=tf.float32)  # (T, 2)
    return x, y


def kalman_loglikelihood(theta: tf.Tensor, y: tf.Tensor) -> float:
    """
    Kalman filter for exact log-likelihood.
    Matches filterflow's kf_loglikelihood.
    """
    A = tf.linalg.diag(theta)
    Q = tf.eye(2, dtype=theta.dtype)  # Matches filterflow: transition_covariance = I₂
    R = 0.1 * tf.eye(2, dtype=theta.dtype)

    T_seq = y.shape[0]
    m = tf.zeros(2, dtype=theta.dtype)
    P = tf.eye(2, dtype=theta.dtype)

    loglik = tf.constant(0.0, dtype=theta.dtype)
    for t in range(T_seq):
        if t > 0:
            m = tf.matmul(tf.expand_dims(m, axis=0), A, transpose_b=True)[0]
            P = tf.matmul(tf.matmul(A, P), A, transpose_b=True) + Q

        S = P + R
        K = tf.matmul(P, tf.linalg.inv(S))
        innovation = y[t] - m
        mvn = tfd.MultivariateNormalTriL(
            loc=tf.zeros(2, dtype=theta.dtype),
            scale_tril=tf.linalg.cholesky(S),
        )
        loglik = loglik + mvn.log_prob(innovation)
        m = m + tf.squeeze(tf.matmul(K, tf.expand_dims(innovation, axis=1)), axis=1)
        P = (tf.eye(2, dtype=theta.dtype) - K) @ P

    return float(loglik)


@tf.function
def get_elbo(pf, y, theta_eval):
    """Compute ELBO for given theta, matching filterflow's get_elbos."""
    model = build_bootstrap_model_from_theta(theta_eval)
    pf.model = model
    loglik, _ = pf(y)
    return loglik / tf.cast(tf.shape(y)[0], dtype=loglik.dtype)


def main(
    T: int = 150,
    n_particles: int = 25,
    batch_size: int = 100,
    epsilon_vals: tuple = (0.25, 0.5, 0.75),
    alpha_vals: tuple = (0.25, 0.5, 0.75),
    theta_values: tuple = (0.25, 0.5, 0.75),
    data_seed: int = 111,
    filter_seed: int = 555,
):
    """
    Main experiment matching filterflow's simple_linear_comparison.py.
    """
    print("=" * 70)
    print("Reproducing Corenflos et al. Table 1: Linear Gaussian SSM")
    print("=" * 70)
    print(f"T={T}, N={n_particles}, batch_size={batch_size}")
    print(f"theta_values: {theta_values}")
    print(f"epsilon_values: {epsilon_vals}")
    print(f"alpha_values (soft-resampling): {alpha_vals}")
    print("=" * 70)

    theta_true = tf.constant([0.5, 0.5], dtype=tf.float32)
    theta_eval_list = [tf.constant([t, t], dtype=tf.float32) for t in theta_values]

    results: dict[str, dict[str, tuple[float, float]]] = {}

    # Generate data once (as in filterflow)
    _, y_data = simulate_data(theta_true, T=T, seed=data_seed)

    for theta_eval in tqdm(theta_eval_list, desc="Theta values"):
        theta_key = f"{theta_eval.numpy()[0]:.2f}"
        results[theta_key] = {}

        # Compute true log-likelihood
        l_true = kalman_loglikelihood(theta_eval, y_data) / T

        # Standard PF
        elbos_pf = []
        for b in tqdm(range(batch_size), desc=f"PF (theta={theta_key})", leave=False):
            tf.random.set_seed(filter_seed + b)
            model_pf = build_bootstrap_model_from_theta(theta_eval)
            pf = StandardParticleFilter(model_pf, num_particles=n_particles, resample_threshold=0.5)
            loglik, _ = pf(y_data)
            elbo = float(loglik.numpy()) / T
            elbos_pf.append(elbo)

        elbos_pf_arr = np.array(elbos_pf)
        mean_pf = np.mean(elbos_pf_arr)
        std_pf = np.std(elbos_pf_arr, ddof=1)
        bias_pf = mean_pf - l_true
        results[theta_key]["PF"] = (bias_pf, std_pf)

        # DPF with different epsilons
        for eps in epsilon_vals:
            elbos_dpf = []
            for b in tqdm(range(batch_size), desc=f"DPF eps={eps} (theta={theta_key})", leave=False):
                tf.random.set_seed(filter_seed + b)
                model_dpf = build_bootstrap_model_from_theta(theta_eval)
                dpf = DifferentiableParticleFilter(
                    model_dpf,
                    num_particles=n_particles,
                    epsilon=eps,
                    sinkhorn_iters=30,  # 30 is sufficient for convergence and faster
                    resample_threshold=0.5,
                )
                loglik, _ = dpf(y_data)
                elbo = float(loglik.numpy()) / T
                elbos_dpf.append(elbo)

            elbos_dpf_arr = np.array(elbos_dpf)
            mean_dpf = np.mean(elbos_dpf_arr)
            std_dpf = np.std(elbos_dpf_arr, ddof=1)
            bias_dpf = mean_dpf - l_true
            results[theta_key][f"DPF_eps={eps}"] = (bias_dpf, std_dpf)

        # Soft-resampling PF (mixture with uniform) for each alpha
        for alpha in alpha_vals:
            elbos_soft = []
            for b in tqdm(
                range(batch_size),
                desc=f"Soft alpha={alpha} (theta={theta_key})",
                leave=False,
            ):
                tf.random.set_seed(filter_seed + b)
                model_soft = build_bootstrap_model_from_theta(theta_eval)
                pf_soft = SoftResamplingParticleFilter(
                    model_soft,
                    num_particles=n_particles,
                    alpha=alpha,
                    resample_threshold=0.5,
                )
                loglik, _ = pf_soft(y_data)
                elbo = float(loglik.numpy()) / T
                elbos_soft.append(elbo)

            elbos_soft_arr = np.array(elbos_soft)
            mean_soft = np.mean(elbos_soft_arr)
            std_soft = np.std(elbos_soft_arr, ddof=1)
            bias_soft = mean_soft - l_true
            results[theta_key][f"Soft_alpha={alpha}"] = (bias_soft, std_soft)

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS: Mean & Std of (1/T)(\\hat l(θ;U) - l(θ))")
    print("=" * 70)
    print(f"{'Method':<20s} {'theta=0.25':>15s} {'theta=0.50':>15s} {'theta=0.75':>15s}")
    print("-" * 70)

    methods = (
        ["PF"]
        + [f"DPF_eps={eps}" for eps in epsilon_vals]
        + [f"Soft_alpha={a}" for a in alpha_vals]
    )
    for method in methods:
        row = f"{method:<20s}"
        for theta_key in ["0.25", "0.50", "0.75"]:
            if method in results[theta_key]:
                bias, std = results[theta_key][method]
                row += f"  {bias:+.3f} ± {std:.3f}"
            else:
                row += "  " + " " * 15
        print(row)

    print("=" * 70)
    print("\nPaper Table 1 (for reference, from Corenflos et al.):")
    print("PF:        -1.13 ± 0.20  |  -0.93 ± 0.18  |  -1.05 ± 0.17")
    print("DPF(0.25): -1.14 ± 0.20  |  -0.94 ± 0.18  |  -1.07 ± 0.19")
    print("DPF(0.5):  -1.14 ± 0.20  |  -0.94 ± 0.18  |  -1.08 ± 0.18")
    print("DPF(0.75): -1.14 ± 0.20  |  -0.94 ± 0.18  |  -1.08 ± 0.18")

    return results


def run_tradeoff_experiment(
    T: int = 150,
    n_particles: int = 25,
    batch_size: int = 50,  # Slightly smaller batch for faster grid search
    epsilon_vals: list[float] = [0.1, 0.25, 0.5, 1.0],
    iter_vals: list[int] = [10, 20, 30, 50],
    seed: int = 111,
) -> list[dict]:
    """
    Tune the regularization parameter (epsilon) and Sinkhorn iterations (iters)
    for bias-variance-speed trade-off.
    """
    print("=" * 80)
    print("DPF Trade-off Analysis: Bias vs Variance vs Speed")
    print("=" * 80)
    print(f"T={T}, N={n_particles}, batch_size={batch_size}")
    print("=" * 80)

    theta_true = tf.constant([0.5, 0.5], dtype=tf.float32)
    _, y_data = simulate_data(theta_true, T=T, seed=seed)

    l_true = kalman_loglikelihood(theta_true, y_data) / T

    results: list[dict] = []

    # Build the bootstrap model once (same theta for all epsilon/iters)
    model_true = build_bootstrap_model_from_theta(theta_true)

    for epsilon in epsilon_vals:
        for n_iters in iter_vals:
            elbos: list[float] = []

            start_time = time.time()

            # Create the filter once per hyperparameter pair
            dpf = DifferentiableParticleFilter(
                model_true,
                num_particles=n_particles,
                epsilon=epsilon,
                sinkhorn_iters=n_iters,
                resample_threshold=0.5,
            )

            for b in tqdm(range(batch_size), desc=f"eps={epsilon:.2f}, iters={n_iters}"):
                tf.random.set_seed(seed + b)
                loglik, _ = dpf(y_data)
                elbo = float(loglik.numpy()) / T
                elbos.append(elbo)

            total_time = time.time() - start_time
            avg_time_ms = (total_time / batch_size) * 1000.0

            elbos_arr = np.array(elbos, dtype=np.float64)
            mean_elbo = float(np.mean(elbos_arr))
            std_elbo = float(np.std(elbos_arr, ddof=1))
            bias = mean_elbo - float(l_true)

            results.append(
                {
                    "epsilon": float(epsilon),
                    "iters": int(n_iters),
                    "bias": float(bias),
                    "std": float(std_elbo),
                    "time_ms": float(avg_time_ms),
                }
            )

            print(
                f"eps={epsilon:.2f}, iters={n_iters:2d} | "
                f"Bias: {bias:+.4f} | Std: {std_elbo:.4f} | Time: {avg_time_ms:6.1f}ms"
            )

    print("\n" + "=" * 80)
    print(f"{'Epsilon':<10} {'Iters':<10} {'Bias':<15} {'Std (Var)':<15} {'Time (ms)':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['epsilon']:<10.2f} {r['iters']:<10d} {r['bias']:<15.4f} {r['std']:<15.4f} {r['time_ms']:<10.1f}")
    print("=" * 80)

    _plot_tradeoff_results(results, out_dir=None)
    return results


def _plot_tradeoff_results(results: list[dict], out_dir=None) -> None:
    """
    Create professional trade-off plots for the bias/variance/speed sweep.
    Saves PNGs into the bias_variance_speed report directory.
    """
    if out_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = repo_root / "reports" / "5_Differential_PF_OT_Resampling" / "bias_variance_speed"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epsilon_vals = sorted(set(float(r["epsilon"]) for r in results))
    iter_vals = sorted(set(int(r["iters"]) for r in results))

    Ne, Ni = len(epsilon_vals), len(iter_vals)

    lookup = {(float(r["epsilon"]), int(r["iters"])): r for r in results}

    def grid(key: str) -> np.ndarray:
        return np.array(
            [[lookup[(float(e), int(it))][key] for it in iter_vals] for e in epsilon_vals],
            dtype=np.float64,
        )

    bias_g = grid("bias")
    std_g = grid("std")
    time_g = grid("time_ms")

    eps_lbl = [f"{e:g}" for e in epsilon_vals]
    iter_lbl = [str(i) for i in iter_vals]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_bias = axes[0, 0]
    ax_std = axes[0, 1]
    ax_time = axes[1, 0]
    ax_pareto = axes[1, 1]

    def _heat(ax, z: np.ndarray, cmap: str, title: str, fmt: str, show_cbar: bool = True):
        im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(Ni))
        ax.set_xticklabels(iter_lbl)
        ax.set_yticks(range(Ne))
        ax.set_yticklabels(eps_lbl)
        ax.set_xlabel("iters")
        ax.set_ylabel("epsilon")
        ax.set_title(title)
        ax.grid(False)

        for i in range(Ne):
            for j in range(Ni):
                ax.text(
                    j,
                    i,
                    fmt.format(z[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

        if show_cbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _heat(
        ax=ax_bias,
        z=bias_g,
        cmap="RdBu",
        title="Bias heatmap (mean ELBO - true log-likelihood)",
        fmt="{:.3f}",
    )
    _heat(
        ax=ax_std,
        z=std_g,
        cmap="Blues_r",
        title="Std heatmap (variance estimate)",
        fmt="{:.3f}",
    )
    _heat(
        ax=ax_time,
        z=time_g,
        cmap="YlOrRd",
        title="Runtime heatmap (ms per run)",
        fmt="{:.1f}",
    )

    # Pareto: bias vs std, size ∝ runtime, color ∝ epsilon.
    time_min = float(np.min([r["time_ms"] for r in results]))
    time_max = float(np.max([r["time_ms"] for r in results]))
    denom = (time_max - time_min) if (time_max - time_min) > 1e-12 else 1.0

    norm = plt.Normalize(vmin=min(epsilon_vals), vmax=max(epsilon_vals))
    cmap = plt.get_cmap("viridis")

    for e in epsilon_vals:
        sub = [r for r in results if float(r["epsilon"]) == float(e)]
        x = [r["bias"] for r in sub]
        y = [r["std"] for r in sub]
        # Larger marker sizes for readability on saved PNGs.
        s = [((r["time_ms"] - time_min) / denom) * 80.0 + 25.0 for r in sub]
        color = cmap(norm(e))
        ax_pareto.scatter(
            x,
            y,
            s=s,
            c=[color] * len(sub),
            alpha=0.9,
            edgecolors="white",
            linewidths=1.0,
            label=f"epsilon={e:g}",
        )

    ax_pareto.set_xlabel("Bias")
    ax_pareto.set_ylabel("Std")
    ax_pareto.set_title("Pareto (size ∝ runtime)")
    ax_pareto.grid(True, alpha=0.25)
    ax_pareto.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "dpf_tradeoff_grid.png", dpi=200)
    plt.close(fig)




class _Tee(io.TextIOBase):
    """Write to multiple streams (console + file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def _run_and_log(fn, *, log_path: Path, **kwargs):
    """Run `fn(**kwargs)` while teeing stdout into `log_path`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as f:
        tee = _Tee(original_stdout, f)
        with contextlib.redirect_stdout(tee):
            return fn(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table1",
        action="store_true",
        help="Reproduce Corenflos et al. Table 1 (PF vs DPF vs soft-resampling).",
    )
    parser.add_argument(
        "--bias_variance",
        action="store_true",
        help="Run bias-variance-speed trade-off grid search for DPF.",
    )

    args = parser.parse_args()

    # Set seeds for reproducibility (matching filterflow defaults)
    tf.random.set_seed(555)
    np.random.seed(111)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "reports" / "5_Differential_PF_OT_Resampling" / "bias_variance_speed"
    corenflos_path = out_dir / "corenflos_table1.txt"
    tradeoff_path = out_dir / "bias_variance_speed.txt"

    if args.table1:
        _run_and_log(
            main,
            log_path=corenflos_path,
            T=150,
            n_particles=25,
            batch_size=100,
            epsilon_vals=(0.25, 0.5, 0.75),
            alpha_vals=(0.25, 0.5, 0.75),
            theta_values=(0.25, 0.5, 0.75),
            data_seed=111,
            filter_seed=555,
        )

    if args.bias_variance:
        _run_and_log(
            run_tradeoff_experiment,
            log_path=tradeoff_path,
            T=150,
            n_particles=25,
            batch_size=50,
            epsilon_vals=[0.1, 0.25, 0.5, 1.0],
            iter_vals=[10, 20, 30, 50],
            seed=111,
        )

    if not args.table1 and not args.bias_variance:
        parser.print_help()

#python -m src.experiments.exp_part2_2b_dpf_bias_variance_speed_tradeoff --bias_variance