"""
Phase 9: train any architecture at any N with supervised MSE.

Mirrors phase2_train_svssm_neural_ot.py / phase4_loss_modes.py but
parameterises both architecture and particle count, so we can produce
a training-matched checkpoint at the deployment N.

Used in Phase 8 follow-up: Phase 8 showed the Phase-4 N=64 checkpoint
generalises poorly to larger N. Phase 9 retrains at the target N
and verifies the speed + recovery story holds when training-matched.

Outputs:
  reports/.../section2_phase9/{arch}_N{N}.weights.h5
  reports/.../section2_phase9/{arch}_N{N}_summary.json
  reports/.../section2_phase9/{arch}_N{N}_loss_curve.png
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus import (
    DifferentiableLEDHLogLikelihoodSVSSM,
    DifferentiableLEDHNeuralOTSVSSM,
    SVSSMNeuralOTTrainer,
    generate_svssm_training_data,
)
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.hyper_deeponet_ot import HyperDeepONetMonotoneOT


_ARCH_BUILDERS = {
    "mgradnet":       (ConditionalMGradNet,        "n_ridges"),
    "deeponet":       (DeepONetMonotoneOT,         "n_basis"),
    "hyper_deeponet": (HyperDeepONetMonotoneOT,    "n_basis"),
}


def build_theta_grid(n_mu: int, n_phi: int, n_sigma: int) -> list[tuple]:
    mu_vals = np.linspace(-1.5, 1.5, n_mu)
    phi_raw_vals = np.linspace(1.5, 2.5, n_phi)
    phi_vals = np.tanh(phi_raw_vals)
    log_sig_vals = np.linspace(-3.5, -0.5, n_sigma)
    sigma_eta_vals = np.exp(0.5 * log_sig_vals)
    return [(float(m), float(p), float(s))
            for m in mu_vals for p in phi_vals for s in sigma_eta_vals]


def time_callable(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tf.Tensor):
            _ = out.numpy()
        ts.append(time.perf_counter() - t0)
    return float(stats.median(ts))


def sanity_metrics(model, T, N, n_lambda, sinkhorn_epsilon):
    """Compare trained-NN-OT filter to Sinkhorn baseline at the training N."""
    tf.random.set_seed(42)
    sigma_eta = tf.constant(0.3, tf.float32)
    h = tf.constant(0.0, tf.float32)
    ys = []
    for _ in range(T):
        h = 0.0 + 0.95 * (h - 0.0) + sigma_eta * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    y_obs = tf.stack(ys)

    ll_nn = DifferentiableLEDHNeuralOTSVSSM(
        neural_ot_model=model, num_particles=N, n_lambda=n_lambda,
        sinkhorn_epsilon=sinkhorn_epsilon, grad_window=4,
        jit_compile=False, integrator="exp",
    )
    ll_base = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N, n_lambda=n_lambda, sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=10, grad_window=4, jit_compile=True, integrator="exp",
    )

    mu_t = tf.constant(0.0, tf.float32)
    phi_t = tf.constant(0.95, tf.float32)
    sig_t = tf.constant(0.09, tf.float32)

    tf.random.set_seed(123)
    v_nn = float(ll_nn(mu_t, phi_t, sig_t, y_obs).numpy())
    tf.random.set_seed(123)
    v_base = float(ll_base(mu_t, phi_t, sig_t, y_obs).numpy())
    return {
        "log_p_neural_ot": v_nn,
        "log_p_sinkhorn_baseline": v_base,
        "abs_delta_vs_baseline": abs(v_nn - v_base),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="deeponet",
                   choices=list(_ARCH_BUILDERS.keys()))
    p.add_argument("--N", type=int, default=256,
                   help="Particle count at training (= deployment N).")
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    p.add_argument("--n_mu", type=int, default=3)
    p.add_argument("--n_phi", type=int, default=3)
    p.add_argument("--n_sigma", type=int, default=3)
    p.add_argument("--seeds_per_theta", type=int, default=3)
    p.add_argument("--n_ridges", type=int, default=64,
                   help="Network width (n_ridges for mGradNet, n_basis otherwise)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase9")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase9] TF {tf.__version__}")
    print(f"  arch={args.arch}  N={args.N}  T={args.T}  width={args.n_ridges}")
    print(f"  grid: {args.n_mu} x {args.n_phi} x {args.n_sigma}  "
          f"seeds/theta: {args.seeds_per_theta}  n_lambda={args.n_lambda}")
    print(f"  training: lr={args.learning_rate} bs={args.batch_size} "
          f"max_ep={args.max_epochs} patience={args.patience}")
    print(f"  out_dir: {out_dir}\n")

    # ---- Data generation ----
    print("[data-gen] building theta grid + generating training data")
    theta_grid = build_theta_grid(args.n_mu, args.n_phi, args.n_sigma)
    t0 = time.perf_counter()
    ds = generate_svssm_training_data(
        theta_grid=theta_grid, T=args.T, N=args.N, n_lambda=args.n_lambda,
        sinkhorn_epsilon=args.sinkhorn_epsilon, sinkhorn_iters=args.K,
        integrator="exp", seeds_per_theta=args.seeds_per_theta,
        base_seed=args.seed, verbose=True,
    )
    t_data = time.perf_counter() - t0
    train_ds, val_ds = ds.split_train_val(val_frac=args.val_frac, seed=args.seed)
    print(f"[data-gen] {len(ds)} samples in {t_data:.1f}s "
          f"(train {len(train_ds)} / val {len(val_ds)})\n")

    # ---- Build model ----
    ModelClass, width_kwarg = _ARCH_BUILDERS[args.arch]
    tf.random.set_seed(args.seed + 1)
    model = ModelClass(state_dim=1, n_scalar_ctx=7, **{width_kwarg: args.n_ridges})
    # warm-up to create variables
    _ = model(tf.constant(train_ds.particles_norm[0], tf.float32),
              tf.constant(train_ds.weights[0], tf.float32),
              tf.constant(train_ds.ctx[0], tf.float32))
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"[build] {args.arch}: {n_params:,} trainable parameters\n")

    # ---- Train ----
    trainer = SVSSMNeuralOTTrainer(
        model=model, learning_rate=args.learning_rate,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, loss_mode="supervised",
    )
    ckpt_name = f"{args.arch}_N{args.N}.weights.h5"
    ckpt_path = out_dir / ckpt_name
    history = trainer.train(train_ds=train_ds, val_ds=val_ds,
                             checkpoint_path=ckpt_path, verbose=True)
    print(f"\n[train] saved checkpoint: {ckpt_path}")

    # ---- Sanity: filter LL vs Sinkhorn at training-N ----
    sanity = sanity_metrics(model, T=args.T, N=args.N,
                             n_lambda=args.n_lambda,
                             sinkhorn_epsilon=args.sinkhorn_epsilon)
    print(f"\n[sanity] at test theta (mu=0, phi=0.95, sigma_eta=0.3), N={args.N}:")
    print(f"  Sinkhorn baseline log p = {sanity['log_p_sinkhorn_baseline']:.4f}")
    print(f"  NN-OT log p             = {sanity['log_p_neural_ot']:.4f}")
    print(f"  |Δ|                     = {sanity['abs_delta_vs_baseline']:.4f}")

    # ---- Persist summary ----
    summary = {
        "tf": tf.__version__,
        "config": vars(args),
        "n_train_samples": int(len(train_ds)),
        "n_val_samples": int(len(val_ds)),
        "data_gen_s": float(t_data),
        "n_params": n_params,
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "val_mse_vs_sinkhorn": history.val_mse_vs_sinkhorn,
        "best_val_loss": history.best_val_loss,
        "best_val_mse_vs_sinkhorn": history.best_val_mse_vs_sinkhorn,
        "best_epoch": history.best_epoch,
        "converged_at_epoch": history.converged_at_epoch,
        "elapsed_s": history.elapsed_s,
        "checkpoint": str(ckpt_path),
        "sanity": sanity,
    }
    summary_path = out_dir / f"{args.arch}_N{args.N}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {summary_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        ax.plot(history.train_loss, label="train")
        ax.plot(history.val_loss, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"{args.arch} @ N={args.N}: best val MSE = {history.best_val_loss:.5f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.arch}_N{args.N}_loss_curve.png", dpi=120,
                     bbox_inches="tight")
        plt.close(fig)
        print(f"       wrote {out_dir / f'{args.arch}_N{args.N}_loss_curve.png'}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
