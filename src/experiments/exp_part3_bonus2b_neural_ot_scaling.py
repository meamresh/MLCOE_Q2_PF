"""
Experiment: Neural OT vs Sinkhorn — Asymptotic Runtime & Accuracy Scaling
========================================================================

Demonstrates:
  1. The theoretical scaling crossover of Sinkhorn O(N^2) vs Neural OT O(N).
  2. The Test Mean Squared Error (Accuracy) of an amortised neural map 
     as particle dimensions increase.

At lower values (N=50), Neural network training and forward passes are 
relatively tight, maintaining low MSE. As N scales, we observe how well 
a fixed-capacity DeepSet Encoder handles massive point-cloud distributions 
compared to exact iterative Sinkhorn.
"""

from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.filters.dpf.resampling import det_resample
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet


def generate_scaling_data(N: int, n_examples: int, sinkhorn_eps: float):
    # DUMMY dataset for scaling accuracy estimation.
    # Uniform weights, normal particles.
    tf.random.set_seed(1234)
    x = tf.random.normal([n_examples, N, 1])
    w_raw = tf.random.uniform([n_examples, N])
    w = w_raw / tf.reduce_sum(w_raw, axis=1, keepdims=True)
    ctx = tf.zeros([n_examples, 6])
    
    # Compute true Sinkhorn targets
    targets = []
    print(f"    Generating {n_examples} Sinkhorn targets for N={N} ...")
    
    @tf.function(jit_compile=False)
    def sink_map(xx, ww):
        p_out, _ = det_resample(xx, ww, epsilon=sinkhorn_eps, n_iters=30)
        return p_out

    for i in trange(n_examples):
        targets.append(sink_map(x[i], w[i]))
    
    targets = tf.stack(targets)
    return x, w, ctx, targets


def benchmark_scaling() -> int:
    out_dir = Path("reports/7_BonusQ2_NeuralOT/scaling/")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Experiment: Asymptotic Scaling & Accuracy (Neural OT vs Sinkhorn)")
    print("=" * 80)

    # Config
    N_list = [50, 100, 250, 500]
    n_iters = 50    # repeats for timing
    n_warmup = 10   # warmup calls
    sinkhorn_eps = 2.0
    
    n_train = 1000  # Quick dataset sizes for training reliable nets
    n_test = 200
    epochs = 150

    results_sink_fwd = []
    results_sink_bwd = []
    results_neural_fwd = []
    results_neural_bwd = []
    results_mse = []

    print("\n  Running Benchmark over varied N ...\n")
    print(f"  {'N Particles':<12} | {'Sinkhorn FWD':<15} | {'Neural FWD':<15} | {'Test MSE':<15}")
    print(f"  {'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")

    for N in N_list:
        print(f"\n  --- Testing N={N} ---")
        x_all, w_all, ctx_all, targets_all = generate_scaling_data(N, n_train + n_test, sinkhorn_eps)
        
        x_tr, w_tr, c_tr, t_tr = x_all[:n_train], w_all[:n_train], ctx_all[:n_train], targets_all[:n_train]
        x_te, w_te, c_te, t_te = x_all[n_train:], w_all[n_train:], ctx_all[n_train:], targets_all[n_train:]

        model = ConditionalMGradNet(state_dim=1, n_ridges=128, d_set=64, d_scalar=64)
        optimiser = tf.keras.optimizers.Adam(learning_rate=2e-3)
        
        # Training
        ds = tf.data.Dataset.from_tensor_slices((x_tr, w_tr, c_tr, t_tr)).batch(128)

        @tf.function(jit_compile=True)
        def _train_step(p_batch, w_batch, c_batch, t_batch):
            with tf.GradientTape() as tape:
                pred = model(p_batch, w_batch, c_batch)
                loss = tf.reduce_mean((pred - t_batch) ** 2)
            grads = tape.gradient(loss, model.trainable_variables)
            # Safe clipping
            grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
            optimiser.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        print(f"    Training Neural OT for {epochs} epochs ...")
        
        for ep in range(epochs):
            for p_batch, w_batch, c_batch, t_batch in ds:
                _train_step(p_batch, w_batch, c_batch, t_batch)

        # Accuracy Evaluation
        test_pred = model(x_te, w_te, c_te)
        test_mse = float(tf.reduce_mean((test_pred - t_te)**2).numpy())
        results_mse.append(test_mse)
        print(f"    Test MSE vs Sinkhorn: {test_mse:.6f}")

        # Timing Setup
        @tf.function(jit_compile=False)
        def run_sinkhorn_fwd(xx, ww):
            p_out, _ = det_resample(xx, ww, epsilon=sinkhorn_eps, n_iters=30)
            return p_out
            
        @tf.function(jit_compile=False)
        def run_sinkhorn_bwd(xx, ww):
            with tf.GradientTape() as tape:
                tape.watch(xx)
                p_out, _ = det_resample(xx, ww, epsilon=sinkhorn_eps, n_iters=30)
                loss = tf.reduce_mean(p_out)
            return tape.gradient(loss, xx)

        @tf.function(jit_compile=False)
        def run_neural_fwd(xx, ww, cc):
            return model(xx, ww, cc)
            
        @tf.function(jit_compile=False)
        def run_neural_bwd(xx, ww, cc):
            with tf.GradientTape() as tape:
                tape.watch(xx)
                p_out = model(xx, ww, cc)
                loss = tf.reduce_mean(p_out)
            return tape.gradient(loss, xx)

        x_time = x_te[0]
        w_time = w_te[0]
        c_time = c_te[0]

        # Warmups
        for _ in range(n_warmup):
            run_sinkhorn_fwd(x_time, w_time)
            run_sinkhorn_bwd(x_time, w_time)
            run_neural_fwd(x_time, w_time, c_time)
            run_neural_bwd(x_time, w_time, c_time)
            
        print("    Benchmarking ...")
        t0 = time.time()
        for _ in range(n_iters):
            run_sinkhorn_fwd(x_time, w_time)
        sink_fwd_t = ((time.time() - t0) / n_iters) * 1000.0

        t0 = time.time()
        for _ in range(n_iters):
            run_sinkhorn_bwd(x_time, w_time)
        sink_bwd_t = ((time.time() - t0) / n_iters) * 1000.0

        t0 = time.time()
        for _ in range(n_iters):
            run_neural_fwd(x_time, w_time, c_time)
        nn_fwd_t = ((time.time() - t0) / n_iters) * 1000.0

        t0 = time.time()
        for _ in range(n_iters):
            run_neural_bwd(x_time, w_time, c_time)
        nn_bwd_t = ((time.time() - t0) / n_iters) * 1000.0

        results_sink_fwd.append(sink_fwd_t)
        results_sink_bwd.append(sink_bwd_t)
        results_neural_fwd.append(nn_fwd_t)
        results_neural_bwd.append(nn_bwd_t)

        print(f"  SUMMARY N={N:<6} | {sink_fwd_t:<15.2f} | {nn_fwd_t:<15.2f} | {test_mse:<15.6f}")

    # ---- Plotting ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Forward Plot
    ax = axes[0]
    ax.plot(N_list, results_sink_fwd, marker="o", lw=2, color="#DD8452", label="Sinkhorn O(N^2)")
    ax.plot(N_list, results_neural_fwd, marker="s", lw=2, color="#4C72B0", label="Neural OT O(N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Execution Time (ms) - log scale")
    ax.set_title("Forward Pass: Resampling Runtime", fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # 2. Backward Plot
    ax = axes[1]
    ax.plot(N_list, results_sink_bwd, marker="o", lw=2, color="#DD8452", label="Sinkhorn O(N^2)")
    ax.plot(N_list, results_neural_bwd, marker="s", lw=2, color="#4C72B0", label="Neural OT O(N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Execution Time (ms) - log scale")
    ax.set_title("Backward Pass: Tensor Gradient Runtime", fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # 3. Accuracy Plot
    ax = axes[2]
    ax.plot(N_list, results_mse, marker="D", lw=2, color="#55A868", label="Neural OT Test MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Mean Squared Error (vs Sinkhorn)")
    ax.set_title("Reliability: Test Accuracy vs Scaling", fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    plt.tight_layout()
    plot_path = out_dir / "scaling_with_accuracy.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 80)
    print(f"  Benchmark complete. Plot saved to: {plot_path}")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    raise SystemExit(benchmark_scaling())
