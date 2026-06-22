"""
Section 2 Phase 4: loss-mode comparison.

Trains three identical mGradNet-M networks on the same SVSSM dataset
with three different objectives:

  - supervised         (MSE against Sinkhorn targets; Phase 2/3 baseline)
  - monge_ampere       (self-supervised log|det J| = log(N w) residual)
  - mixed              ((1 - ma_weight) MSE + ma_weight MA)

Reports for each mode:
  - trainable parameter count
  - best val loss (in the configured objective)
  - best val MSE vs Sinkhorn   (apples-to-apples cross-mode metric)
  - convergence epoch
  - training wall time
  - filter-level log p at the test theta vs Sinkhorn baseline
  - warm forward + gradient wall time inside the filter

Outputs:
  reports/.../section2_phase4/
    - {supervised,monge_ampere,mixed}.weights.h5
    - phase4_summary.json
    - phase4_loss_curves.png
    - phase4_report.txt
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
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, tf.Tensor):
                    _ = o.numpy()
        ts.append(time.perf_counter() - t0)
    return float(stats.median(ts))


def sanity_metrics(model, T=20, N=64, n_lambda=10, sinkhorn_epsilon=1.0):
    """Run trained model in the SVSSM filter and compare to Sinkhorn baseline."""
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

    def fwd():
        tf.random.set_seed(123)
        return ll_nn(mu_t, phi_t, sig_t, y_obs)

    def fwd_grad():
        tf.random.set_seed(123)
        p = tf.constant([0.0, 0.95, 0.09], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(p)
            v = ll_nn(p[0], p[1], p[2], y_obs)
        return tape.gradient(v, p)

    _ = fwd(); _ = fwd_grad()
    fwd_ms = time_callable(fwd, 5) * 1000
    fg_ms = time_callable(fwd_grad, 5) * 1000

    return {
        "log_p_neural_ot": v_nn,
        "log_p_sinkhorn_baseline": v_base,
        "abs_delta_vs_baseline": abs(v_nn - v_base),
        "warm_forward_ms": fwd_ms,
        "warm_forward_grad_ms": fg_ms,
    }


def train_one_loss_mode(mode, ma_weight, model, train_ds, val_ds, args, out_dir):
    """Train one loss mode and return diagnostics + sanity dict."""
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    label = mode if mode != "mixed" else f"mixed(ma_w={ma_weight:.2f})"
    print(f"\n{'=' * 60}\n[{label}] params: {n_params:,}\n{'=' * 60}")
    trainer = SVSSMNeuralOTTrainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        loss_mode=mode,
        ma_weight=ma_weight,
    )
    ckpt = out_dir / f"{mode}.weights.h5"
    history = trainer.train(train_ds=train_ds, val_ds=val_ds,
                             checkpoint_path=ckpt, verbose=True)
    sanity = sanity_metrics(model, T=args.T, N=args.N,
                             n_lambda=args.n_lambda,
                             sinkhorn_epsilon=args.sinkhorn_epsilon)
    return {
        "name": mode,
        "label": label,
        "ma_weight": ma_weight if mode == "mixed" else None,
        "n_params": n_params,
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "val_mse_vs_sinkhorn": history.val_mse_vs_sinkhorn,
        "best_val_loss": history.best_val_loss,
        "best_val_mse_vs_sinkhorn": history.best_val_mse_vs_sinkhorn,
        "best_epoch": history.best_epoch,
        "converged_at_epoch": history.converged_at_epoch,
        "elapsed_s": history.elapsed_s,
        "checkpoint": str(ckpt),
        "sanity": sanity,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    p.add_argument("--n_mu", type=int, default=3)
    p.add_argument("--n_phi", type=int, default=3)
    p.add_argument("--n_sigma", type=int, default=3)
    p.add_argument("--seeds_per_theta", type=int, default=3)
    p.add_argument("--n_ridges", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--ma_weight", type=float, default=0.5,
                   help="MA weight for mixed mode (0.5 = equal MSE+MA)")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase4")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase 4] TF {tf.__version__}")
    print(f"  grid: {args.n_mu} x {args.n_phi} x {args.n_sigma}  "
          f"seeds/theta: {args.seeds_per_theta}  T={args.T} N={args.N}")
    print(f"  shared training config: lr={args.learning_rate} bs={args.batch_size} "
          f"max_ep={args.max_epochs} patience={args.patience}")
    print(f"  shared architecture: mGradNet n_ridges={args.n_ridges}")
    print(f"  mixed-mode ma_weight={args.ma_weight}\n")

    # ---- Data generation (once, shared by all three modes) ----
    print("[data-gen] building theta grid + generating training data once")
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

    # ---- Build three independent mGradNet instances (identical seeds) ----
    # Each mode trains a fresh model — apples-to-apples comparison.
    def build_fresh_mgradnet(seed_offset: int) -> ConditionalMGradNet:
        tf.random.set_seed(args.seed + seed_offset)
        m = ConditionalMGradNet(state_dim=1, n_ridges=args.n_ridges, n_scalar_ctx=7)
        # warm-up the model so all weights are created
        _ = m(tf.constant(train_ds.particles_norm[0], tf.float32),
              tf.constant(train_ds.weights[0], tf.float32),
              tf.constant(train_ds.ctx[0], tf.float32))
        return m

    modes_to_run = [
        ("supervised", 0.0, 1),
        ("monge_ampere", 0.0, 1),  # same seed -> identical init
        ("mixed", args.ma_weight, 1),
    ]

    results = []
    for mode, ma_w, seed_offset in modes_to_run:
        try:
            model = build_fresh_mgradnet(seed_offset)
            r = train_one_loss_mode(mode, ma_w, model, train_ds, val_ds,
                                     args, out_dir)
            results.append(r)
        except Exception as e:
            print(f"[{mode}] FAILED: {e!r}")
            import traceback
            traceback.print_exc()
            results.append({"name": mode, "error": repr(e)})

    # ---- Comparison table ----
    print("\n\n" + "=" * 100)
    print(f"{'PHASE 4 — LOSS-MODE COMPARISON':<100}")
    print("=" * 100)
    hdr = (f"{'mode':<16}{'params':>9}{'best val':>11}{'best vMSE':>11}"
           f"{'best ep':>9}{'conv ep':>9}{'train s':>9}{'fwd ms':>9}"
           f"{'fwd+g ms':>10}{'|d| lp':>9}")
    print(hdr)
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<16}FAILED: {r['error'][:80]}")
            continue
        s = r['sanity']
        print(f"{r['name']:<16}{r['n_params']:>9,}"
              f"{r['best_val_loss']:>11.5f}"
              f"{r['best_val_mse_vs_sinkhorn']:>11.5f}"
              f"{r['best_epoch'] + 1:>9}"
              f"{r['converged_at_epoch']:>9}"
              f"{r['elapsed_s']:>9.1f}"
              f"{s['warm_forward_ms']:>9.1f}"
              f"{s['warm_forward_grad_ms']:>10.1f}"
              f"{s['abs_delta_vs_baseline']:>9.4f}")
    print("=" * 100)
    if results and 'sanity' in results[0]:
        print(f"  Sinkhorn baseline log p (test theta): "
              f"{results[0]['sanity']['log_p_sinkhorn_baseline']:.4f}")

    # ---- Persist ----
    summary = {
        "tf": tf.__version__,
        "config": vars(args),
        "n_train_samples": int(len(train_ds)),
        "n_val_samples": int(len(val_ds)),
        "data_gen_s": float(t_data),
        "results": results,
    }
    (out_dir / "phase4_summary.json").write_text(json.dumps(summary, indent=2))

    text_lines = [
        "=" * 100,
        "Section 2 Phase 4: loss-mode comparison on mGradNet",
        "=" * 100,
        f"TF {tf.__version__}",
        f"Data: {len(train_ds)} train / {len(val_ds)} val samples "
        f"(grid {args.n_mu}x{args.n_phi}x{args.n_sigma} thetas, "
        f"{args.seeds_per_theta} seeds/theta, T={args.T}, N={args.N})",
        f"Training: lr={args.learning_rate} bs={args.batch_size} "
        f"max_ep={args.max_epochs} patience={args.patience} "
        f"n_ridges={args.n_ridges} ma_weight={args.ma_weight}",
        f"Test theta for sanity LL: (mu=0, phi=0.95, sigma_eta=0.3)",
        "",
        hdr,
        "-" * 100,
    ]
    for r in results:
        if "error" in r:
            text_lines.append(f"{r['name']:<16}FAILED: {r['error'][:80]}")
            continue
        s = r['sanity']
        text_lines.append(
            f"{r['name']:<16}{r['n_params']:>9,}"
            f"{r['best_val_loss']:>11.5f}"
            f"{r['best_val_mse_vs_sinkhorn']:>11.5f}"
            f"{r['best_epoch'] + 1:>9}{r['converged_at_epoch']:>9}"
            f"{r['elapsed_s']:>9.1f}{s['warm_forward_ms']:>9.1f}"
            f"{s['warm_forward_grad_ms']:>10.1f}{s['abs_delta_vs_baseline']:>9.4f}"
        )
    text_lines.append("=" * 100)
    if results and 'sanity' in results[0]:
        text_lines.append(
            f"Sinkhorn baseline log p (test theta): "
            f"{results[0]['sanity']['log_p_sinkhorn_baseline']:.4f}"
        )
    (out_dir / "phase4_report.txt").write_text("\n".join(text_lines))

    # ---- Loss-curve plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for r in results:
            if "error" in r:
                continue
            axes[0].plot(r['train_loss'], label=r['label'])
            axes[1].plot(r['val_loss'], label=r['label'])
            axes[2].plot(r['val_mse_vs_sinkhorn'], label=r['label'])
        axes[0].set_title("Train loss (configured objective)")
        axes[1].set_title("Val loss (configured objective)")
        axes[2].set_title("Val MSE vs Sinkhorn (cross-mode metric)")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "phase4_loss_curves.png", dpi=120,
                     bbox_inches="tight")
        plt.close(fig)
        print(f"\n[plot] wrote {out_dir / 'phase4_loss_curves.png'}")
    except ImportError:
        print("\n[plot] matplotlib not available; skipping loss curves")

    print(f"\n[done] wrote {out_dir / 'phase4_summary.json'}")
    print(f"       wrote {out_dir / 'phase4_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
