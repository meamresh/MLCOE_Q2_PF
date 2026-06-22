"""
Phase 2 experiment driver: train ConditionalMGradNet on SVSSM data.

Steps:
  1. Build a small (mu, phi, sigma_eta) grid covering the priors used by
     the SVSSM HMC pipeline.
  2. Generate (particles, weights, ctx, sinkhorn_target) tuples by
     running the SVSSM filter at each grid point.
  3. Train an mGradNet (n_scalar_ctx=7) with plateau-based early stop.
  4. Validate: run the 6-check sanity test from Phase 1 with the trained
     model. Expect distinct-theta outputs to remain distinct AND the
     filter LL gap vs Sinkhorn to shrink dramatically.
  5. Persist: save weights + history + summary to
     reports/.../section2_phase2/.

Defaults are scoped for a ~5-15 min run; pass --max_epochs / --grid_size /
--seeds_per_theta to scale up.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm import (
    DifferentiableLEDHNeuralOTSVSSM,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    DifferentiableLEDHLogLikelihoodSVSSM,
)
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.extra_bonus.svssm_neural_ot_training import (
    SVSSMNeuralOTTrainer,
    generate_svssm_training_data,
)


def build_theta_grid(n_mu: int, n_phi: int, n_sigma: int) -> list[tuple]:
    """Regular grid centred on the truth used in §7."""
    # Priors used in §7 production runs:
    #   mu ~ N(0, 1)              -> grid +/- 1.5 sigma
    #   phi_raw ~ N(2, 0.5)       -> grid +/- 1 sigma in unconstrained
    #   log_sigma_eta_sq ~ N(-2,1)-> grid +/- 1.5 sigma
    mu_vals = np.linspace(-1.5, 1.5, n_mu)
    phi_raw_vals = np.linspace(1.5, 2.5, n_phi)
    phi_vals = np.tanh(phi_raw_vals)
    log_sig_vals = np.linspace(-3.5, -0.5, n_sigma)
    sigma_eta_vals = np.exp(0.5 * log_sig_vals)  # sqrt(exp(.))

    grid = []
    for mu in mu_vals:
        for phi in phi_vals:
            for sig in sigma_eta_vals:
                grid.append((float(mu), float(phi), float(sig)))
    return grid


def run_sanity_check(model, T=20, N=64):
    """Mirror of the 6-check Phase-1 sanity test with the trained model.

    Returns a dict with the key metrics."""
    print("\n[sanity] running post-training sanity test")

    def gen_obs(seed=42):
        tf.random.set_seed(seed)
        sigma_eta = tf.constant(0.3, tf.float32)
        h = tf.constant(0.0, tf.float32)
        ys = []
        mu, phi = 0.0, 0.95
        for _ in range(T):
            h = mu + phi * (h - mu) + sigma_eta * tf.random.normal([])
            ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
        return tf.stack(ys)

    y_obs = gen_obs()

    ll_nn = DifferentiableLEDHNeuralOTSVSSM(
        neural_ot_model=model, num_particles=N, n_lambda=10,
        sinkhorn_epsilon=1.0, grad_window=4, jit_compile=False,
        integrator="exp",
    )
    ll_base = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N, n_lambda=10, sinkhorn_epsilon=1.0,
        sinkhorn_iters=10, grad_window=4, jit_compile=True,
        integrator="exp",
    )

    tf.random.set_seed(123)
    v_nn = float(ll_nn(
        tf.constant(0.0, tf.float32), tf.constant(0.95, tf.float32),
        tf.constant(0.09, tf.float32), y_obs,
    ).numpy())
    tf.random.set_seed(123)
    v_base = float(ll_base(
        tf.constant(0.0, tf.float32), tf.constant(0.95, tf.float32),
        tf.constant(0.09, tf.float32), y_obs,
    ).numpy())
    print(f"  Sinkhorn baseline    log p = {v_base:.4f}")
    print(f"  Neural OT (trained)  log p = {v_nn:.4f}")
    print(f"  |delta|              = {abs(v_base - v_nn):.4f}")

    # gradient flow
    params = tf.constant([0.0, 0.95, 0.09], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(params)
        v = ll_nn(params[0], params[1], params[2], y_obs)
    g = tape.gradient(v, params).numpy()
    finite_g = bool(np.all(np.isfinite(g)))
    print(f"  Gradient w.r.t. (mu,phi,sigma2) = {g}  "
          f"({'finite' if finite_g else 'NOT finite'})")

    return {
        "log_p_sinkhorn_baseline": v_base,
        "log_p_neural_ot_trained": v_nn,
        "abs_delta_vs_baseline": abs(v_base - v_nn),
        "gradient_finite": finite_g,
        "gradient": g.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10,
                   help="Sinkhorn iterations (for training targets).")
    p.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    # Theta grid
    p.add_argument("--n_mu", type=int, default=3)
    p.add_argument("--n_phi", type=int, default=3)
    p.add_argument("--n_sigma", type=int, default=3)
    p.add_argument("--seeds_per_theta", type=int, default=3)
    # Model
    p.add_argument("--n_ridges", type=int, default=64)
    # Trainer
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase2")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase 2 trainer] TF {tf.__version__}")
    print(f"  grid: {args.n_mu} x {args.n_phi} x {args.n_sigma} = "
          f"{args.n_mu * args.n_phi * args.n_sigma} theta points "
          f"x {args.seeds_per_theta} seeds/theta = "
          f"{args.n_mu * args.n_phi * args.n_sigma * args.seeds_per_theta} series, "
          f"each T={args.T} -> "
          f"{args.n_mu * args.n_phi * args.n_sigma * args.seeds_per_theta * args.T} samples")
    print(f"  model: ConditionalMGradNet(n_ridges={args.n_ridges}, n_scalar_ctx=7)")
    print(f"  trainer: lr={args.learning_rate}  bs={args.batch_size}  "
          f"max_ep={args.max_epochs}  patience={args.patience}  val={args.val_frac}\n")

    # ---- Data generation ----
    print(f"[data-gen] building theta grid")
    theta_grid = build_theta_grid(args.n_mu, args.n_phi, args.n_sigma)
    print(f"  example points (first 3): {theta_grid[:3]}")
    print(f"  example points (last 3):  {theta_grid[-3:]}")
    print(f"[data-gen] generating training data")
    t0 = time.perf_counter()
    ds = generate_svssm_training_data(
        theta_grid=theta_grid, T=args.T, N=args.N,
        n_lambda=args.n_lambda,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.K,
        integrator="exp",
        seeds_per_theta=args.seeds_per_theta,
        base_seed=args.seed,
        verbose=True,
    )
    t_data = time.perf_counter() - t0
    print(f"[data-gen] done. {len(ds)} samples in {t_data:.1f}s\n")

    # ---- Train/val split ----
    train_ds, val_ds = ds.split_train_val(val_frac=args.val_frac, seed=args.seed)
    print(f"[split] train={len(train_ds)}  val={len(val_ds)}\n")

    # ---- Model build ----
    model = ConditionalMGradNet(
        state_dim=1, n_ridges=args.n_ridges, n_scalar_ctx=7,
    )
    # Build by calling with one sample
    _ = model(
        tf.constant(train_ds.particles_norm[0], tf.float32),
        tf.constant(train_ds.weights[0], tf.float32),
        tf.constant(train_ds.ctx[0], tf.float32),
    )
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"[model] mGradNet built, {n_params:,} trainable params\n")

    # ---- Train ----
    trainer = SVSSMNeuralOTTrainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(f"[train] starting")
    ckpt_path = out_dir / "mgradnet.weights.h5"
    history = trainer.train(
        train_ds=train_ds, val_ds=val_ds,
        checkpoint_path=ckpt_path, verbose=True,
    )
    print(f"[train] done. wall {history.elapsed_s:.1f}s, "
          f"best val {history.best_val_loss:.5f} at epoch {history.best_epoch + 1}\n")

    # ---- Post-training sanity ----
    sanity = run_sanity_check(model, T=args.T, N=args.N)

    # ---- Persist ----
    summary = {
        "tf": tf.__version__,
        "config": vars(args),
        "n_train_samples": int(len(train_ds)),
        "n_val_samples": int(len(val_ds)),
        "n_trainable_params": int(n_params),
        "data_gen_s": float(t_data),
        "train_curves": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
        },
        "best_val_loss": float(history.best_val_loss),
        "best_epoch": int(history.best_epoch),
        "converged_at_epoch": int(history.converged_at_epoch),
        "elapsed_train_s": float(history.elapsed_s),
        "sanity": sanity,
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {out_dir / 'phase2_summary.json'}")

    # ---- Verdict line ----
    print("\n" + "=" * 70)
    print("PHASE 2 VERDICT")
    print("=" * 70)
    print(f"  trainable params:    {n_params:,}")
    print(f"  train samples:       {len(train_ds)}")
    print(f"  val samples:         {len(val_ds)}")
    print(f"  best val MSE:        {history.best_val_loss:.5f}  "
          f"(epoch {history.best_epoch + 1})")
    print(f"  converged at epoch:  {history.converged_at_epoch}")
    print(f"  data gen wall:       {t_data:.1f}s")
    print(f"  train wall:          {history.elapsed_s:.1f}s")
    print(f"  Sinkhorn baseline log p: {sanity['log_p_sinkhorn_baseline']:.4f}")
    print(f"  Trained NN-OT log p:    {sanity['log_p_neural_ot_trained']:.4f}")
    print(f"  |delta|:               {sanity['abs_delta_vs_baseline']:.4f}")
    print(f"  gradient finite:       {sanity['gradient_finite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
