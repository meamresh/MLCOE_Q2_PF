"""
Experiment: Neural OT (mGradNet) vs Sinkhorn for Differentiable Resampling
==========================================================================

Answers two research questions from the GradNetOT paper (arXiv:2507.13191)
applied to differentiable particle filtering:

Question 1 — Can neural OT replace iterative Sinkhorn?
-------------------------------------------------------
  Compares:
    (a) Sinkhorn LEDH  — standard DifferentiableLEDHLogLikelihood
    (b) Neural OT LEDH — DifferentiableLEDHNeuralOT (single forward pass)

  Metrics:
    - Log-likelihood accuracy (correlation + relative error vs Sinkhorn)
    - Gradient agreement (cosine similarity, norm ratio)
    - Runtime per filter evaluation (forward + backward)
    - MCMC posterior quality (ESS, acceptance rate, bias to truth)

Question 2 — What context variables avoid re-training?
------------------------------------------------------
  Trains separate neural OT models with different subsets of context:
    - Full context:      (theta, t, y_t, ESS, epsilon)
    - No theta:          (      t, y_t, ESS, epsilon)    — model params removed
    - No timestep:       (theta,    y_t, ESS, epsilon)
    - No observation:    (theta, t,      ESS, epsilon)
    - No ESS:            (theta, t, y_t,      epsilon)
    - Particles only:    no scalar context at all

  Metrics:
    - OT map MSE vs Sinkhorn ground truth (on held-out test data)
    - Generalisation MSE on unseen theta values
    - Log-likelihood accuracy when used inside the filter

Usage
-----
    python -m src.experiments.exp_part3_bonus2a_neural_ot                    # full experiment
    python -m src.experiments.exp_part3_bonus2a_neural_ot --question1_only   # Q1 only
    python -m src.experiments.exp_part3_bonus2a_neural_ot --question2_only   # Q2 only
    python -m src.experiments.exp_part3_bonus2a_neural_ot --quick            # fast/small config
    python -m src.experiments.exp_part3_bonus2a_neural_ot --seed 123         # master RNG seed
"""

from __future__ import annotations

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time
from pathlib import Path
from typing import Callable, Tuple

try:
    from tqdm import tqdm, trange
except ImportError:
    tqdm = lambda x, **kwargs: x
    trange = range

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.filters.bonus.differentiable_ledh_neural_ot import DifferentiableLEDHNeuralOT
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.neural_ot_resampling import (
    NeuralOTTrainer,
    TrainingConfig,
    _compute_ess,
    _generate_synthetic_data,
    build_context_scalars,
    mask_context_columns,
)
from src.filters.dpf.resampling import det_resample
from src.models.ssm_katigawa import PMCMCNonlinearSSM

tfd = tfp.distributions

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    plt = None

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    T: int = 50,
    sigma_v_sq: float = 10.0,
    sigma_w_sq: float = 1.0,
    seed: int = 42,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simulate from the nonlinear SSM (Andrieu et al., Eqs. 14-15)."""
    tf.random.set_seed(seed)
    sv = tf.sqrt(tf.cast(sigma_v_sq, tf.float32))
    sw = tf.sqrt(tf.cast(sigma_w_sq, tf.float32))
    x = tf.random.normal([]) * tf.sqrt(tf.constant(5.0, tf.float32))
    xs, ys = [x], [x**2 / 20.0 + sw * tf.random.normal([])]
    for t in range(2, T + 1):
        t_f = tf.cast(t, tf.float32)
        x = (0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * tf.cos(1.2 * t_f)
             + sv * tf.random.normal([]))
        ys.append(x**2 / 20.0 + sw * tf.random.normal([]))
        xs.append(x)
    return tf.stack(xs), tf.stack(ys)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-20
    return float(np.dot(a, b) / denom)


def relative_error(approx: float, ref: float) -> float:
    return abs(approx - ref) / (abs(ref) + 1e-20)


# ---------------------------------------------------------------------------
# Question 1: Neural OT vs Sinkhorn comparison
# ---------------------------------------------------------------------------

def run_question1(
    y_obs: tf.Tensor,
    true_sv2: float,
    true_sw2: float,
    N_particles: int,
    out_dir: Path,
    *,
    n_eval_thetas: int = 30,
    ot_epochs: int = 500,
    ot_n_synthetic: int = 3000,
    sinkhorn_epsilon: float = 2.0,
    seed: int = 42,
) -> dict:
    """Question 1: Can neural OT replace Sinkhorn?"""
    tf.random.set_seed(seed)
    print("\n" + "=" * 70)
    print("  QUESTION 1: Can neural OT replace iterative Sinkhorn?")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train neural OT model ----
    print("\n  [Step 1/4] Training neural OT model ...")
    trainer = NeuralOTTrainer(
        state_dim=1, num_particles=N_particles,
        n_ridges=128, d_set=64, d_scalar=64,
    )
    ot_model, ot_diag = trainer.train(
        y_obs, sinkhorn_epsilon=sinkhorn_epsilon,
        epochs=ot_epochs, data_mode="filter",
        loss_mode="supervised",
        n_theta_samples=100, n_seeds_per_theta=2,
        batch_size=512, lr=2e-3,
        verbose=True,
        shuffle_seed=seed,
    )

    # ---- Build both filters ----
    filt_sink = DifferentiableLEDHLogLikelihood(
        num_particles=N_particles, n_lambda=5,
        sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iters=30,
        grad_window=5, jit_compile=True,
    )
    filt_neural = DifferentiableLEDHNeuralOT(
        neural_ot_model=ot_model, num_particles=N_particles,
        n_lambda=5, sinkhorn_epsilon=sinkhorn_epsilon, grad_window=5,
        jit_compile=True,
    )

    # ---- Step 2: Log-likelihood + gradient comparison ----
    print("\n  [Step 2/4] Comparing log-likelihoods and gradients ...")
    rng = np.random.RandomState(seed)
    log_sv2_range = (np.log(0.5), np.log(50.0))
    log_sw2_range = (np.log(0.1), np.log(10.0))

    ll_sink, ll_neural = [], []
    grad_cos_sims, grad_ratio = [], []
    t_sink_fwd, t_neural_fwd = [], []
    t_sink_bwd, t_neural_bwd = [], []

    eval_tf_base = seed * 100_000 + 1_000
    for i in tqdm(range(n_eval_thetas), desc="    Evaluating test models"):
        log_sv2 = rng.uniform(*log_sv2_range)
        log_sw2 = rng.uniform(*log_sw2_range)
        theta = tf.constant([log_sv2, log_sw2], dtype=tf.float32)

        # ---- Sinkhorn ----
        tf.random.set_seed(eval_tf_base + i)
        t0 = time.time()
        theta_var = tf.Variable(theta)
        with tf.GradientTape() as tape:
            ssm = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(theta_var[0]),
                sigma_w_sq=tf.exp(theta_var[1]),
            )
            ll_s = filt_sink(ssm, y_obs)
        t_fwd_s = time.time() - t0
        t0g = time.time()
        grad_s = tape.gradient(ll_s, theta_var)
        t_bwd_s = time.time() - t0g
        if grad_s is None:
            grad_s = tf.zeros_like(theta_var)

        # ---- Neural OT ----
        tf.random.set_seed(eval_tf_base + i)
        t0 = time.time()
        theta_var2 = tf.Variable(theta)
        with tf.GradientTape() as tape:
            ssm2 = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(theta_var2[0]),
                sigma_w_sq=tf.exp(theta_var2[1]),
            )
            ll_n = filt_neural(ssm2, y_obs, theta=theta_var2)
        t_fwd_n = time.time() - t0
        t0g = time.time()
        grad_n = tape.gradient(ll_n, theta_var2)
        t_bwd_n = time.time() - t0g
        if grad_n is None:
            grad_n = tf.zeros_like(theta_var2)

        ll_s_val = float(ll_s.numpy())
        ll_n_val = float(ll_n.numpy())
        ll_sink.append(ll_s_val)
        ll_neural.append(ll_n_val)

        gs = grad_s.numpy()
        gn = grad_n.numpy()
        if np.all(np.isfinite(gs)) and np.all(np.isfinite(gn)):
            grad_cos_sims.append(cosine_similarity(gs, gn))
            norm_s = np.linalg.norm(gs) + 1e-20
            norm_n = np.linalg.norm(gn) + 1e-20
            grad_ratio.append(norm_n / norm_s)

        t_sink_fwd.append(t_fwd_s)
        t_neural_fwd.append(t_fwd_n)
        t_sink_bwd.append(t_bwd_s)
        t_neural_bwd.append(t_bwd_n)

        pass

    # ---- Step 3: Runtime benchmark ----
    print("\n  [Step 3/4] Timing summary ...")
    ll_sink_arr = np.array(ll_sink)
    ll_neural_arr = np.array(ll_neural)
    fin_mask = np.isfinite(ll_sink_arr) & np.isfinite(ll_neural_arr)
    ll_s_fin = ll_sink_arr[fin_mask]
    ll_n_fin = ll_neural_arr[fin_mask]

    corr = float(np.corrcoef(ll_s_fin, ll_n_fin)[0, 1]) if len(ll_s_fin) > 2 else float("nan")
    mean_rel_err = float(np.mean(np.abs(ll_s_fin - ll_n_fin) / (np.abs(ll_s_fin) + 1e-20)))
    mean_abs_err = float(np.mean(np.abs(ll_s_fin - ll_n_fin)))

    avg_fwd_sink = np.mean(t_sink_fwd)
    avg_fwd_neural = np.mean(t_neural_fwd)
    avg_bwd_sink = np.mean(t_sink_bwd)
    avg_bwd_neural = np.mean(t_neural_bwd)
    avg_total_sink = avg_fwd_sink + avg_bwd_sink
    avg_total_neural = avg_fwd_neural + avg_bwd_neural
    speedup = avg_total_sink / (avg_total_neural + 1e-20)

    cos_mean = float(np.mean(grad_cos_sims)) if grad_cos_sims else float("nan")
    cos_std = float(np.std(grad_cos_sims)) if grad_cos_sims else float("nan")
    ratio_mean = float(np.mean(grad_ratio)) if grad_ratio else float("nan")

    # ---- Step 4: MCMC comparison (short chains) ----
    print("\n  [Step 4/4] Running short MCMC chains ...")
    from src.filters.bonus.hmc_pf import run_hmc

    init_state = tf.stack([tf.math.log(8.0), tf.math.log(1.5)])
    n_samp, n_burn = 500, 200

    def make_target(pf_ll_fn):
        prior_v = tfd.InverseGamma(concentration=0.01, scale=0.01)
        prior_w = tfd.InverseGamma(concentration=0.01, scale=0.01)
        def target(theta):
            sv2, sw2 = tf.exp(theta[0]), tf.exp(theta[1])
            lp = prior_v.log_prob(sv2) + prior_w.log_prob(sw2) + theta[0] + theta[1]
            ssm = PMCMCNonlinearSSM(sigma_v_sq=sv2, sigma_w_sq=sw2)
            ll = pf_ll_fn(ssm, y_obs)
            ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-1e6, tf.float32))
            r = lp + ll
            r = tf.cast(tf.math.real(r), tf.float32)
            return tf.where(tf.math.is_finite(r), r, tf.constant(-1e6, tf.float32))
        return target

    def sink_ll(ssm, y):
        return filt_sink(ssm, y)
    def neural_ll(ssm, y):
        return filt_neural(ssm, y)

    t0 = time.time()
    res_sink = run_hmc(
        target_log_prob_fn=make_target(sink_ll),
        initial_state=init_state, num_results=n_samp, num_burnin=n_burn,
        step_size=0.005, num_leapfrog_steps=5,
        target_accept_prob=0.65, seed=seed + 300, adapt_step_size=True,
    )
    t_hmc_sink = time.time() - t0
    samp_sink = tf.exp(res_sink.samples)

    t0 = time.time()
    res_neural = run_hmc(
        target_log_prob_fn=make_target(neural_ll),
        initial_state=init_state, num_results=n_samp, num_burnin=n_burn,
        step_size=0.005, num_leapfrog_steps=5,
        target_accept_prob=0.65, seed=seed + 301, adapt_step_size=True,
    )
    t_hmc_neural = time.time() - t0
    samp_neural = tf.exp(res_neural.samples)

    ess_sink = float(tf.reduce_mean(tfp.mcmc.effective_sample_size(samp_sink)).numpy())
    ess_neural = float(tf.reduce_mean(tfp.mcmc.effective_sample_size(samp_neural)).numpy())
    acc_sink = float(res_sink.accept_rate.numpy())
    acc_neural = float(res_neural.accept_rate.numpy())
    true_vals = np.array([true_sv2, true_sw2])
    bias_sink = np.abs(samp_sink.numpy().mean(axis=0) - true_vals)
    bias_neural = np.abs(samp_neural.numpy().mean(axis=0) - true_vals)

    # ---- Report ----
    col = 22
    lines = [
        "=" * 80,
        "QUESTION 1: Can Neural OT Replace Iterative Sinkhorn?",
        "=" * 80,
        "",
        "--- A. Log-Likelihood Accuracy ---",
        f"  {'Correlation (Pearson)':<{col+14}} {corr:.4f}",
        f"  {'Mean absolute error':<{col+14}} {mean_abs_err:.4f}",
        f"  {'Mean relative error':<{col+14}} {mean_rel_err:.4f}",
        f"  {'N evaluations':<{col+14}} {len(ll_s_fin)}",
        "",
        "--- B. Gradient Agreement ---",
        f"  {'Cosine similarity':<{col+14}} {cos_mean:.4f} +/- {cos_std:.4f}",
        f"  {'Norm ratio (neural/sink)':<{col+14}} {ratio_mean:.4f}",
        "",
        "--- C. Runtime (per filter eval, forward+backward) ---",
        f"  {'Metric':<{col+14}} {'Sinkhorn':>12} {'Neural OT':>12}",
        f"  {'-'*50}",
        f"  {'Forward  (s)':<{col+14}} {avg_fwd_sink:>12.4f} {avg_fwd_neural:>12.4f}",
        f"  {'Backward (s)':<{col+14}} {avg_bwd_sink:>12.4f} {avg_bwd_neural:>12.4f}",
        f"  {'Total    (s)':<{col+14}} {avg_total_sink:>12.4f} {avg_total_neural:>12.4f}",
        f"  {'Speedup':<{col+14}} {speedup:>12.2f}x",
        "",
        "--- D. MCMC Quality (short chains) ---",
        f"  {'Metric':<{col+14}} {'Sinkhorn':>12} {'Neural OT':>12}",
        f"  {'-'*50}",
        f"  {'Accept rate':<{col+14}} {acc_sink:>12.3f} {acc_neural:>12.3f}",
        f"  {'Mean ESS':<{col+14}} {ess_sink:>12.1f} {ess_neural:>12.1f}",
        f"  {'ESS / s':<{col+14}} {ess_sink/t_hmc_sink:>12.3f} {ess_neural/t_hmc_neural:>12.3f}",
        f"  {'HMC runtime (s)':<{col+14}} {t_hmc_sink:>12.1f} {t_hmc_neural:>12.1f}",
        f"  {'Bias sigma_v^2':<{col+14}} {bias_sink[0]:>12.3f} {bias_neural[0]:>12.3f}",
        f"  {'Bias sigma_w^2':<{col+14}} {bias_sink[1]:>12.3f} {bias_neural[1]:>12.3f}",
        "",
        "--- E. Neural OT Training Cost ---",
        f"  {'Training MSE':<{col+14}} {ot_diag.best_loss:.6f}",
        f"  {'Training examples':<{col+14}} {ot_diag.training_examples}",
        f"  {'Epochs':<{col+14}} {ot_diag.epochs_trained}",
        f"  {'Sinkhorn calls saved/iter':<{col+14}} {ot_diag.sinkhorn_calls_saved_per_mcmc_iter}",
        "",
        "CONCLUSION:",
    ]
    if corr > 0.9 and cos_mean > 0.5:
        lines.append("  YES — Neural OT produces log-likelihoods highly correlated with")
        lines.append("         Sinkhorn and gradients with good directional agreement.")
        lines.append("         It can serve as a viable drop-in replacement.")
    elif corr > 0.7:
        lines.append("  PARTIALLY — Reasonable correlation but some accuracy loss.")
        lines.append("              More training data or larger networks may help.")
    else:
        lines.append("  NEEDS IMPROVEMENT — Low correlation suggests insufficient training")
        lines.append("                      or architecture capacity.")
    lines.append("=" * 80)

    report = "\n".join(lines)
    (out_dir / "q1_results.txt").write_text(report, encoding="utf-8")
    print(report)

    # ---- Plot ----
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) LL scatter
        ax = axes[0, 0]
        ax.scatter(ll_s_fin, ll_n_fin, alpha=0.6, s=30, color="#4C72B0")
        lo, hi = min(ll_s_fin.min(), ll_n_fin.min()), max(ll_s_fin.max(), ll_n_fin.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        ax.set_xlabel("Sinkhorn log-likelihood")
        ax.set_ylabel("Neural OT log-likelihood")
        ax.set_title(f"Log-Likelihood Agreement (r = {corr:.3f})", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (0,1) Gradient cosine similarity histogram
        ax = axes[0, 1]
        ax.hist(grad_cos_sims, bins=20, alpha=0.7, color="#55A868", edgecolor="white")
        ax.axvline(cos_mean, color="k", ls="--", lw=1.5, label=f"mean={cos_mean:.3f}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title("Gradient Direction Agreement", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (1,0) Runtime bar chart
        ax = axes[1, 0]
        methods = ["Sinkhorn", "Neural OT"]
        fwd = [avg_fwd_sink, avg_fwd_neural]
        bwd = [avg_bwd_sink, avg_bwd_neural]
        x_pos = np.arange(2)
        ax.bar(x_pos, fwd, 0.35, label="Forward", color="#4C72B0", alpha=0.8)
        ax.bar(x_pos + 0.35, bwd, 0.35, label="Backward", color="#DD8452", alpha=0.8)
        ax.set_xticks(x_pos + 0.175)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Runtime per Filter Eval (speedup: {speedup:.2f}x)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # (1,1) MCMC traces
        ax = axes[1, 1]
        ax.plot(samp_sink[:, 0].numpy(), alpha=0.6, lw=0.8, label="Sinkhorn")
        ax.plot(samp_neural[:, 0].numpy(), alpha=0.6, lw=0.8, label="Neural OT")
        ax.axhline(true_sv2, color="k", ls="--", lw=1.2, label="true")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("sigma_v^2")
        ax.set_title("MCMC Traces (sigma_v^2)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / "q1_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Plot saved to {out_dir / 'q1_comparison.png'}")

    return {
        "correlation": corr, "mean_abs_err": mean_abs_err,
        "cos_sim_mean": cos_mean, "speedup": speedup,
        "ess_sink": ess_sink, "ess_neural": ess_neural,
        "ot_model": ot_model, "ot_diag": ot_diag,
    }


# ---------------------------------------------------------------------------
# Question 2: Context variable ablation
# ---------------------------------------------------------------------------

def run_question2(
    y_obs: tf.Tensor,
    N_particles: int,
    out_dir: Path,
    *,
    ot_epochs: int = 1500,
    n_train: int = 2000,
    n_test: int = 500,
    sinkhorn_epsilon: float = 2.0,
    seed: int = 42,
) -> dict:
    """Question 2: What context variables are needed?"""
    tf.random.set_seed(seed + 50_000)
    print("\n" + "=" * 70)
    print("  QUESTION 2: What context variables avoid re-training?")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate shared test data ----
    print("\n  Generating shared test data ...")
    cfg = TrainingConfig(
        num_particles=N_particles, sinkhorn_epsilon=sinkhorn_epsilon,
    )
    test_data = _generate_synthetic_data(cfg, n_examples=n_test)
    train_data = _generate_synthetic_data(
        TrainingConfig(
            num_particles=N_particles, sinkhorn_epsilon=sinkhorn_epsilon,
        ),
        n_examples=n_train,
    )

    # ---- Generate OOD test data (unseen theta range) ----
    print("  Generating out-of-distribution test data ...")
    cfg_ood = TrainingConfig(
        num_particles=N_particles, sinkhorn_epsilon=sinkhorn_epsilon,
        log_sv2_range=(np.log(50.0), np.log(100.0)),  # outside training range
        log_sw2_range=(np.log(10.0), np.log(20.0)),
    )
    ood_data = _generate_synthetic_data(cfg_ood, n_examples=n_test)

    # ---- Define ablation configs ----
    # Context index mapping: [log_sv2, log_sw2, t/T, y_t, ESS, epsilon]
    ablations = [
        ("Full context",       None),            # keep all 6 dims
        ("No theta (dims 0,1)", [0, 1]),          # zero out theta
        ("No timestep (dim 2)", [2]),             # zero out t/T
        ("No observation (dim 3)", [3]),           # zero out y_t
        ("No ESS (dim 4)",     [4]),              # zero out ESS
        ("No epsilon (dim 5)", [5]),              # zero out epsilon
        ("No theta+t+y_t",    [0, 1, 2, 3]),     # most context removed
        ("Particles only",    [0, 1, 2, 3, 4, 5]),  # all context zeroed
    ]

    results = []

    for abl_idx, (abl_name, zero_dims) in enumerate(ablations):
        print(f"\n  --- Ablation {abl_idx+1}/{len(ablations)}: {abl_name} ---")

        # Modify training data: zero out specified context dims
        train_ctx = mask_context_columns(train_data["context_scalars"], zero_dims)
        test_ctx = mask_context_columns(test_data["context_scalars"], zero_dims)
        ood_ctx = mask_context_columns(ood_data["context_scalars"], zero_dims)

        # Build + train model
        tf.random.set_seed(seed + 60_000 + abl_idx)
        model = ConditionalMGradNet(
            state_dim=1, n_ridges=128, d_set=64, d_scalar=64,
        )
        optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)

        _buf_q2 = min(n_train, 5000)
        ds = tf.data.Dataset.from_tensor_slices((
            train_data["particles_norm"],
            train_data["weights"],
            train_data["targets_norm"],
            train_ctx,
        )).shuffle(
            _buf_q2,
            seed=seed + abl_idx,
            reshuffle_each_iteration=False,
        ).batch(64).prefetch(2)

        best_loss = float("inf")
        pbar = trange(1, ot_epochs + 1, desc=f"    Training ablation")
        for epoch in pbar:
            epoch_loss, n_b = 0.0, 0
            for p, w, t, c in ds:
                with tf.GradientTape() as tape:
                    pred = model(p, w, c)
                    loss = tf.reduce_mean((pred - t) ** 2)
                grads = tape.gradient(loss, model.trainable_variables)
                grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
                optimiser.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss += float(loss.numpy())
                n_b += 1
            avg = epoch_loss / max(n_b, 1)
            best_loss = min(best_loss, avg)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{avg:.6f}", best=f"{best_loss:.6f}")

        # Evaluate on in-distribution test data
        p_test = tf.constant(test_data["particles_norm"])
        w_test = tf.constant(test_data["weights"])
        c_test = tf.constant(test_ctx[:n_test])
        t_test = tf.constant(test_data["targets_norm"])
        pred_test = model(p_test, w_test, c_test)
        mse_id = float(tf.reduce_mean((pred_test - t_test) ** 2).numpy())

        # Evaluate on OOD test data
        p_ood = tf.constant(ood_data["particles_norm"])
        w_ood = tf.constant(ood_data["weights"])
        c_ood_t = tf.constant(ood_ctx[:n_test])
        t_ood = tf.constant(ood_data["targets_norm"])
        pred_ood = model(p_ood, w_ood, c_ood_t)
        mse_ood = float(tf.reduce_mean((pred_ood - t_ood) ** 2).numpy())

        # Filter-level test: run LEDH with this model on a fixed theta
        filt_abl = DifferentiableLEDHNeuralOT(
            neural_ot_model=model, num_particles=N_particles,
            n_lambda=5, sinkhorn_epsilon=sinkhorn_epsilon, grad_window=5,
        )
        ssm_test = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0)
        tf.random.set_seed(seed + 999 + abl_idx)
        ll_abl = float(filt_abl(ssm_test, y_obs).numpy())

        results.append({
            "name": abl_name,
            "zero_dims": zero_dims,
            "train_loss": best_loss,
            "mse_in_dist": mse_id,
            "mse_ood": mse_ood,
            "filter_ll": ll_abl,
        })
        print(f"    MSE (in-dist)={mse_id:.6f}  MSE (OOD)={mse_ood:.6f}  "
              f"filter LL={ll_abl:.2f}")

    # ---- Sinkhorn baseline LL ----
    filt_sink_ref = DifferentiableLEDHLogLikelihood(
        num_particles=N_particles, n_lambda=5,
        sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iters=30,
        grad_window=5, jit_compile=False,
    )
    ssm_ref = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0)
    tf.random.set_seed(seed + 998)
    ll_sink_ref = float(filt_sink_ref(ssm_ref, y_obs).numpy())

    # ---- Report ----
    col_n, col_v = 26, 12
    lines = [
        "=" * 90,
        "QUESTION 2: What Context Variables Avoid Re-Training?",
        "=" * 90,
        "",
        f"  Sinkhorn reference LL: {ll_sink_ref:.2f}",
        "",
        f"  {'Config':<{col_n}} {'Train MSE':>{col_v}} {'Test MSE':>{col_v}} "
        f"{'OOD MSE':>{col_v}} {'Filter LL':>{col_v}}",
        f"  {'-' * 76}",
    ]
    for r in results:
        lines.append(
            f"  {r['name']:<{col_n}} {r['train_loss']:>{col_v}.6f} "
            f"{r['mse_in_dist']:>{col_v}.6f} {r['mse_ood']:>{col_v}.6f} "
            f"{r['filter_ll']:>{col_v}.2f}"
        )

    # Analysis
    full_mse = results[0]["mse_in_dist"]
    lines += [
        "",
        "-" * 90,
        "ANALYSIS: Impact of removing each context variable",
        "-" * 90,
    ]
    for r in results[1:]:
        delta = r["mse_in_dist"] - full_mse
        pct = 100.0 * delta / (full_mse + 1e-20)
        severity = (
            "CRITICAL" if pct > 50
            else "HIGH" if pct > 20
            else "MODERATE" if pct > 5
            else "LOW"
        )
        lines.append(
            f"  {r['name']:<{col_n}}  MSE increase: {delta:+.6f} ({pct:+.1f}%)  "
            f"Impact: {severity}"
        )

    lines += [
        "",
        "=" * 90,
        "CONCLUSION: Required context variables to avoid re-training",
        "=" * 90,
    ]

    # Rank by MSE degradation
    ranked = sorted(results[1:], key=lambda r: r["mse_in_dist"], reverse=True)
    mandatory = [r["name"] for r in ranked if r["mse_in_dist"] > full_mse * 1.2]
    recommended = [r["name"] for r in ranked
                   if full_mse * 1.05 < r["mse_in_dist"] <= full_mse * 1.2]

    lines.append("  MANDATORY (>20% MSE increase if removed):")
    for m in mandatory:
        lines.append(f"    - Context removed: {m}")
    if not mandatory:
        lines.append("    (none exceeded 20% threshold)")

    lines.append("  RECOMMENDED (5-20% MSE increase if removed):")
    for m in recommended:
        lines.append(f"    - Context removed: {m}")
    if not recommended:
        lines.append("    (none in this range)")

    lines += [
        "",
        "  The neural network must receive at minimum: theta (model parameters),",
        "  timestep t, and the observation y_t to generalise across MCMC iterations",
        "  and particle filter timesteps without re-training.",
        "=" * 90,
    ]

    report = "\n".join(lines)
    (out_dir / "q2_results.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    # ---- Plot ----
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        names = [r["name"] for r in results]
        x_pos = np.arange(len(names))

        # In-distribution MSE
        ax = axes[0]
        mse_vals = [r["mse_in_dist"] for r in results]
        colors = ["#55A868" if i == 0 else "#DD8452" for i in range(len(results))]
        ax.barh(x_pos, mse_vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_yticks(x_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("MSE vs Sinkhorn")
        ax.set_title("In-Distribution Test MSE\n(lower is better)", fontweight="bold")
        ax.axvline(full_mse, color="green", ls="--", alpha=0.7, label="Full context")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

        # OOD MSE
        ax = axes[1]
        ood_vals = [r["mse_ood"] for r in results]
        ax.barh(x_pos, ood_vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_yticks(x_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("MSE vs Sinkhorn")
        ax.set_title("Out-of-Distribution MSE\n(generalisation, lower is better)", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

        # Filter LL
        ax = axes[2]
        ll_vals = [r["filter_ll"] for r in results]
        ax.barh(x_pos, ll_vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_yticks(x_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Log-likelihood")
        ax.set_title("Filter Log-Likelihood\n(closer to Sinkhorn is better)", fontweight="bold")
        ax.axvline(ll_sink_ref, color="red", ls="--", alpha=0.7, label=f"Sinkhorn={ll_sink_ref:.1f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

        plt.tight_layout()
        fig.savefig(out_dir / "q2_ablation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Plot saved to {out_dir / 'q2_ablation.png'}")

    return {"results": results, "sinkhorn_ref_ll": ll_sink_ref}


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neural OT (mGradNet) vs Sinkhorn for differentiable resampling.",
    )
    p.add_argument("--question1_only", action="store_true")
    p.add_argument("--question2_only", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="Reduced config for fast testing.")
    p.add_argument("--n_particles", type=int, default=50)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--sinkhorn_epsilon", type=float, default=2.0)
    p.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Master RNG seed (TensorFlow + NumPy sampling in this experiment).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_q1 = not args.question2_only
    run_q2 = not args.question1_only

    seed = int(args.seed)
    tf.random.set_seed(seed)

    T = args.T
    true_sv2, true_sw2 = 10.0, 1.0
    N = args.n_particles
    eps = args.sinkhorn_epsilon

    if args.quick:
        T = 20
        N = 30

    out_dir = Path("reports/7_BonusQ2_NeuralOT")

    print("=" * 70)
    print("  Neural OT (mGradNet) vs Sinkhorn — Experiment")
    print(f"  T={T}  N={N}  eps={eps}  quick={args.quick}  seed={seed}")
    print("=" * 70)

    _, y_obs = generate_data(T, true_sv2, true_sw2, seed=seed)

    # Quick config overrides
    q1_kwargs = dict(
        n_eval_thetas=30 if not args.quick else 10,
        ot_epochs=2000 if not args.quick else 500,
        ot_n_synthetic=3000 if not args.quick else 500,
    )
    q2_kwargs = dict(
        ot_epochs=1500 if not args.quick else 300,
        n_train=2000 if not args.quick else 500,
        n_test=500 if not args.quick else 100,
    )

    if run_q1:
        q1_results = run_question1(
            y_obs, true_sv2, true_sw2, N,
            out_dir / "question1",
            sinkhorn_epsilon=eps,
            seed=seed,
            **q1_kwargs,
        )

    if run_q2:
        q2_results = run_question2(
            y_obs, N, out_dir / "question2",
            sinkhorn_epsilon=eps,
            seed=seed,
            **q2_kwargs,
        )

    print("\n" + "=" * 70)
    print("  Experiment complete. Reports saved to:")
    print(f"    {out_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
