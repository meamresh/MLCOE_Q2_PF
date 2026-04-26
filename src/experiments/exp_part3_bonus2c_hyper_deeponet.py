"""
Experiment: Hyper-DeepONet vs DeepONet vs mGradNet vs Sinkhorn
==============================================================

Compares four resampling approaches:
  (a) Sinkhorn LEDH          -- iterative Sinkhorn (baseline)
  (b) mGradNet Neural OT     -- DeepSet + FiLM conditioning
  (c) DeepONet Monotone OT   -- branch-trunk, amplitude only
  (d) Hyper-DeepONet          -- branch-trunk, amplitude + phase

The Hyper-DeepONet adds branch-produced bias shifts delta_k to the trunk
activations, recovering FiLM's phase modulation within the DeepONet
framework.  This should close the 14x MSE gap between DeepONet and
mGradNet while preserving the PSD Jacobian guarantee.

Usage
-----
    python -m src.experiments.exp_part3_bonus2c_hyper_deeponet
    python -m src.experiments.exp_part3_bonus2c_hyper_deeponet --quick
    python -m src.experiments.exp_part3_bonus2c_hyper_deeponet --n_basis 512
"""

from __future__ import annotations

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

try:
    from tqdm import tqdm, trange
except ImportError:
    tqdm = lambda x, **kwargs: x
    trange = range

from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.hyper_deeponet_ot import HyperDeepONetMonotoneOT
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.neural_ot_resampling import (
    NeuralOTTrainer,
    _generate_filter_data,
    TrainingConfig,
)
from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.filters.bonus.differentiable_ledh_neural_ot import DifferentiableLEDHNeuralOT
from src.models.ssm_katigawa import PMCMCNonlinearSSM

_EPS = 1e-6


def generate_data(T, sv2, sw2, seed=42):
    tf.random.set_seed(seed)
    x_arr = [0.0]
    y_arr = []
    for t in range(1, T + 1):
        x_prev = x_arr[-1]
        x_t = (0.5 * x_prev
               + 25.0 * x_prev / (1.0 + x_prev ** 2)
               + 8.0 * np.cos(1.2 * t)
               + np.random.RandomState(seed + t).randn() * np.sqrt(sv2))
        y_t = x_t ** 2 / 20.0 + np.random.RandomState(seed + 1000 + t).randn() * np.sqrt(sw2)
        x_arr.append(x_t)
        y_arr.append(y_t)
    return tf.constant(x_arr, dtype=tf.float32), tf.constant(y_arr, dtype=tf.float32)


def _train_neural_model(
    model_class,
    model_name: str,
    y_obs: tf.Tensor,
    *,
    num_particles: int,
    n_basis: int,
    d_branch: int,
    sinkhorn_epsilon: float,
    epochs: int,
    batch_size: int = 512,
    lr: float = 2e-3,
    n_theta_samples: int = 100,
    n_seeds_per_theta: int = 2,
    verbose: bool = True,
) -> Tuple[tf.keras.Model, dict]:
    """Generic training loop for any model with the DeepONet interface."""

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Training {model_name}")
        print(f"  N={num_particles}  n_basis={n_basis}  d_branch={d_branch}  "
              f"eps={sinkhorn_epsilon}")
        print(f"{'=' * 60}")

    cfg = TrainingConfig(
        num_particles=num_particles,
        state_dim=1,
        sinkhorn_epsilon=sinkhorn_epsilon,
        n_theta_samples=n_theta_samples,
        n_seeds_per_theta=n_seeds_per_theta,
    )

    if verbose:
        print("  Generating training data ...")
    data = _generate_filter_data(cfg, y_obs)
    M = data["particles_norm"].shape[0]
    if verbose:
        print(f"  Generated {M} training examples")

    model = model_class(
        state_dim=1,
        n_basis=n_basis,
        d_branch=d_branch,
        d_trunk=d_branch,
        n_scalar_ctx=6,
    )

    ds = tf.data.Dataset.from_tensor_slices((
        data["particles_norm"],
        data["weights"],
        data["targets_norm"],
        data["context_scalars"],
    )).shuffle(min(M, 10000)).batch(batch_size).prefetch(2)

    optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function(jit_compile=True)
    def train_step(p_batch, w_batch, c_batch, t_batch):
        with tf.GradientTape() as tape:
            pred = model(p_batch, w_batch, c_batch)
            loss = tf.reduce_mean((pred - t_batch) ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else g
                 for g in grads]
        optimiser.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_loss = float("inf")
    final_loss = float("inf")
    loss_history = []

    pbar = trange(epochs, desc=f"  {model_name}") if verbose else range(epochs)

    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for p_batch, w_batch, t_batch, c_batch in ds:
            loss = train_step(p_batch, w_batch, c_batch, t_batch)
            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        final_loss = avg_loss
        loss_history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and hasattr(pbar, "set_postfix"):
            pbar.set_postfix(loss=f"{avg_loss:.6f}", best=f"{best_loss:.6f}")

    if verbose:
        print(f"  Done.  Best MSE = {best_loss:.6f}\n")

    return model, {"final_loss": final_loss, "best_loss": best_loss,
                   "epochs": epochs, "training_examples": M,
                   "loss_history": loss_history}


def pearson_r(x, y):
    mx, my = np.mean(x), np.mean(y)
    return np.sum((x - mx) * (y - my)) / (
        np.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2)) + 1e-20)


def run_comparison(
    y_obs, true_sv2, true_sw2, N, out_dir,
    *, n_basis, d_branch, sinkhorn_epsilon, epochs, n_eval_thetas,
    n_theta_samples,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train mGradNet ----
    print("\n  [1/4] Training mGradNet ...")
    trainer = NeuralOTTrainer(
        state_dim=1, num_particles=N,
        n_ridges=n_basis, d_set=64, d_scalar=64,
    )
    mgradnet_model, mgradnet_diag = trainer.train(
        y_obs, sinkhorn_epsilon=sinkhorn_epsilon,
        epochs=epochs, data_mode="filter",
        n_theta_samples=n_theta_samples, n_seeds_per_theta=2,
        batch_size=512, lr=2e-3, verbose=True,
    )

    # ---- Train DeepONet ----
    print("\n  [2/4] Training DeepONet ...")
    deeponet_model, deeponet_diag = _train_neural_model(
        DeepONetMonotoneOT, "DeepONet",
        y_obs, num_particles=N, n_basis=n_basis,
        d_branch=d_branch, sinkhorn_epsilon=sinkhorn_epsilon,
        epochs=epochs, n_theta_samples=n_theta_samples,
    )

    # ---- Train Hyper-DeepONet ----
    print("\n  [3/4] Training Hyper-DeepONet ...")
    hyper_model, hyper_diag = _train_neural_model(
        HyperDeepONetMonotoneOT, "Hyper-DeepONet",
        y_obs, num_particles=N, n_basis=n_basis,
        d_branch=d_branch, sinkhorn_epsilon=sinkhorn_epsilon,
        epochs=epochs, n_theta_samples=n_theta_samples,
    )

    # ---- Build filters ----
    filt_sink = DifferentiableLEDHLogLikelihood(
        num_particles=N, n_lambda=5,
        sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iters=30,
        grad_window=5, jit_compile=True,
    )

    def _make_neural_filt(model):
        return DifferentiableLEDHNeuralOT(
            neural_ot_model=model, num_particles=N,
            n_lambda=5, sinkhorn_epsilon=sinkhorn_epsilon,
            grad_window=5, jit_compile=False,
        )

    filt_mg = _make_neural_filt(mgradnet_model)
    filt_do = _make_neural_filt(deeponet_model)
    filt_hy = _make_neural_filt(hyper_model)

    # ---- Evaluate ----
    print("\n  [4/4] Evaluating ...")
    rng = np.random.RandomState(42)
    log_sv2_range = (np.log(0.5), np.log(50.0))
    log_sw2_range = (np.log(0.1), np.log(10.0))

    ll_s, ll_m, ll_d, ll_h = [], [], [], []
    gc_m, gc_d, gc_h = [], [], []
    gr_m, gr_d, gr_h = [], [], []
    t_s, t_m, t_d, t_h = [], [], [], []

    def _eval_filter(filt, theta_vals, seed, use_theta=False):
        tf.random.set_seed(seed)
        theta_var = tf.Variable(theta_vals, dtype=tf.float32)
        t0 = time.perf_counter()
        with tf.GradientTape() as tape:
            ssm = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(theta_var[0]),
                sigma_w_sq=tf.exp(theta_var[1]),
            )
            if use_theta:
                ll = filt(ssm, y_obs, theta=theta_var)
            else:
                ll = filt(ssm, y_obs)
        grad = tape.gradient(ll, theta_var)
        elapsed = time.perf_counter() - t0
        if grad is None:
            grad = tf.zeros_like(theta_var)
        return float(ll), grad.numpy(), elapsed

    for i in trange(n_eval_thetas, desc="  Eval"):
        theta_vals = [rng.uniform(*log_sv2_range), rng.uniform(*log_sw2_range)]
        seed = 1000 + i

        ls, gs, ts = _eval_filter(filt_sink, theta_vals, seed)
        lm, gm, tm = _eval_filter(filt_mg, theta_vals, seed, use_theta=True)
        ld, gd, td = _eval_filter(filt_do, theta_vals, seed, use_theta=True)
        lh, gh, th = _eval_filter(filt_hy, theta_vals, seed, use_theta=True)

        ll_s.append(ls); ll_m.append(lm); ll_d.append(ld); ll_h.append(lh)
        t_s.append(ts); t_m.append(tm); t_d.append(td); t_h.append(th)

        def _grad_metrics(g_ref, g_test, cos_list, ratio_list):
            if np.all(np.isfinite(g_ref)) and np.all(np.isfinite(g_test)):
                cos = float(np.dot(g_ref, g_test) / (
                    np.linalg.norm(g_ref) * np.linalg.norm(g_test) + 1e-20))
                ratio = float(np.linalg.norm(g_test) / (np.linalg.norm(g_ref) + 1e-20))
                cos_list.append(cos)
                ratio_list.append(ratio)

        _grad_metrics(gs, gm, gc_m, gr_m)
        _grad_metrics(gs, gd, gc_d, gr_d)
        _grad_metrics(gs, gh, gc_h, gr_h)

    # ---- Print & Save ----
    ll_s_np = np.array(ll_s)

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("  Hyper-DeepONet vs DeepONet vs mGradNet vs Sinkhorn")
    lines.append("=" * 90)

    lines.append("\n--- A. Log-Likelihood Accuracy ---")
    lines.append(f"  {'Metric':<35s} {'mGradNet':>12s} {'DeepONet':>12s} {'HyperDeepONet':>14s}")
    lines.append("  " + "-" * 75)
    for name, arr in [("Pearson r vs Sinkhorn",
                       [pearson_r(ll_s_np, np.array(x)) for x in [ll_m, ll_d, ll_h]]),
                      ("Mean absolute error",
                       [np.mean(np.abs(ll_s_np - np.array(x))) for x in [ll_m, ll_d, ll_h]]),
                      ("Mean relative error",
                       [np.mean(np.abs((ll_s_np - np.array(x)) / (np.abs(ll_s_np) + 1e-20)))
                        for x in [ll_m, ll_d, ll_h]])]:
        lines.append(f"  {name:<35s} {arr[0]:>12.4f} {arr[1]:>12.4f} {arr[2]:>14.4f}")

    lines.append("\n--- B. Gradient Agreement ---")
    lines.append(f"  {'Metric':<35s} {'mGradNet':>12s} {'DeepONet':>12s} {'HyperDeepONet':>14s}")
    lines.append("  " + "-" * 75)
    for label, arrs in [
        ("Cosine similarity (mean+-std)",
         [(np.mean(x), np.std(x)) for x in [gc_m, gc_d, gc_h]]),
    ]:
        strs = [f"{m:>5.3f}+-{s:.3f}" for m, s in arrs]
        lines.append(f"  {label:<35s} {strs[0]:>12s} {strs[1]:>12s} {strs[2]:>14s}")
    lines.append(f"  {'Norm ratio (neural/sinkhorn)':<35s} "
          f"{np.mean(gr_m):>12.4f} {np.mean(gr_d):>12.4f} {np.mean(gr_h):>14.4f}")

    lines.append("\n--- C. Runtime (forward + backward, per eval) ---")
    lines.append(f"  {'Method':<20s} {'Mean (s)':>10s} {'Std (s)':>10s}")
    lines.append("  " + "-" * 42)
    for name, arr in [("Sinkhorn", t_s), ("mGradNet", t_m),
                      ("DeepONet", t_d), ("Hyper-DeepONet", t_h)]:
        lines.append(f"  {name:<20s} {np.mean(arr):>10.4f} {np.std(arr):>10.4f}")

    lines.append("\n--- D. Training Loss ---")
    lines.append(f"  {'Model':<20s} {'Best MSE':>12s}")
    lines.append("  " + "-" * 34)
    lines.append(f"  {'mGradNet':<20s} {mgradnet_diag.best_loss:>12.6f}")
    lines.append(f"  {'DeepONet':<20s} {deeponet_diag['best_loss']:>12.6f}")
    lines.append(f"  {'Hyper-DeepONet':<20s} {hyper_diag['best_loss']:>12.6f}")

    lines.append("\n--- E. Parameter Count ---")
    for name, mdl in [("mGradNet", mgradnet_model),
                      ("DeepONet", deeponet_model),
                      ("Hyper-DeepONet", hyper_model)]:
        n = sum(int(tf.reduce_prod(v.shape)) for v in mdl.trainable_variables)
        lines.append(f"  {name:<20s} {n:>10,d}")

    lines.append("\n" + "=" * 90)
    
    out_str = "\n".join(lines)
    print(out_str)
    
    out_file = out_dir / "hyper_deeponet_results.txt"
    with open(out_file, "w") as f:
        f.write(out_str)
    print(f"\nSaved results to {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n_particles", type=int, default=50)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--n_basis", type=int, default=128)
    p.add_argument("--d_branch", type=int, default=128)
    p.add_argument("--sinkhorn_epsilon", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=500)
    args = p.parse_args()

    tf.random.set_seed(42)
    T, N, eps, epochs = args.T, args.n_particles, args.sinkhorn_epsilon, args.epochs
    n_eval = 30

    if args.quick:
        T, N, epochs, n_eval = 20, 30, 200, 10

    print("=" * 70)
    print("  Hyper-DeepONet vs DeepONet vs mGradNet vs Sinkhorn")
    print(f"  T={T}  N={N}  n_basis={args.n_basis}  eps={eps}  quick={args.quick}")
    print("=" * 70)

    true_sv2, true_sw2 = 10.0, 1.0
    _, y_obs = generate_data(T, true_sv2, true_sw2, seed=42)

    run_comparison(
        y_obs, true_sv2, true_sw2, N,
        Path("reports/7_BonusQ2_NeuralOT/DeepONet"),
        n_basis=args.n_basis, d_branch=args.d_branch,
        sinkhorn_epsilon=eps, epochs=epochs,
        n_eval_thetas=n_eval, n_theta_samples=100,
    )

    print("\n  Experiment complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
