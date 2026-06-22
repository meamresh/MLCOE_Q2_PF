"""
N-sweep probe: how do Sinkhorn (XLA) and NN-OT (XLA) filters scale with N?

For each particle count in --n_grid, time:
  - warm forward call
  - warm forward+gradient call
for both pipelines on the same trained mGradNet checkpoint.

Sinkhorn's per-resample work is O(N^2 * K); the NN forward is O(N).
At some N the curves should cross. This probe measures where.

Outputs:
  reports/.../section2_phase8/probe_n_sweep_summary.json
  reports/.../section2_phase8/probe_n_sweep_wall_vs_N.png   (log-log)

Caveat: the trained network was trained at N=64. At larger N the
set-encoder is OOD (it uses mean/std summaries, which are N-invariant
in expectation, but the empirical fluctuation shrinks at larger N).
This probe measures speed only; an OT-quality follow-up is needed
before claiming NN-OT is usable at high N for HMC.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    DifferentiableLEDHLogLikelihoodSVSSM,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_jit import (
    DifferentiableLEDHNeuralOTSVSSMJIT,
)
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.hyper_deeponet_ot import HyperDeepONetMonotoneOT


_ARCH_BUILDERS = {
    "mgradnet":       (ConditionalMGradNet,       "n_ridges"),
    "deeponet":       (DeepONetMonotoneOT,        "n_basis"),
    "hyper_deeponet": (HyperDeepONetMonotoneOT,   "n_basis"),
}

_DEFAULT_CKPTS = {
    "mgradnet":       "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase4/supervised.weights.h5",
    "deeponet":       "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase3/deeponet.weights.h5",
    "hyper_deeponet": "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase3/hyper_deeponet.weights.h5",
}


def build_model(N: int, arch: str, ckpt: Path, n_ridges: int = 64):
    ModelClass, width_kwarg = _ARCH_BUILDERS[arch]
    tf.random.set_seed(0)
    m = ModelClass(state_dim=1, n_scalar_ctx=7, **{width_kwarg: n_ridges})
    dummy_p = tf.zeros([N], tf.float32)
    dummy_w = tf.fill([N], 1.0 / float(N))
    dummy_c = tf.zeros([7], tf.float32)
    _ = m(dummy_p, dummy_w, dummy_c)
    m.load_weights(str(ckpt))
    return m


def gen_y_obs(T: int, seed: int = 42) -> tf.Tensor:
    tf.random.set_seed(seed)
    h = tf.constant(0.0, tf.float32)
    sigma_eta = tf.constant(0.3, tf.float32)
    ys = []
    for _ in range(T):
        h = 0.0 + 0.95 * (h - 0.0) + sigma_eta * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def time_call(fn, reps: int) -> float:
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


def probe(name: str, ll, mu_t, phi_t, sig_t, y_obs, reps: int):
    print(f"  [{name}] warming...", flush=True)

    def fwd():
        tf.random.set_seed(123)
        return ll(mu_t, phi_t, sig_t, y_obs)

    def fwd_grad():
        tf.random.set_seed(123)
        p = tf.constant([0.0, 0.95, 0.09], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(p)
            v = ll(p[0], p[1], p[2], y_obs)
        return tape.gradient(v, p)

    try:
        t_warm = time.perf_counter()
        v0 = fwd()
        g0 = fwd_grad()
        warm_s = time.perf_counter() - t_warm
        fwd_ms = time_call(fwd, reps) * 1000
        fg_ms = time_call(fwd_grad, reps) * 1000
        print(f"  [{name}] warm={warm_s:6.1f}s  fwd={fwd_ms:7.1f} ms  "
              f"fwd+grad={fg_ms:7.1f} ms  log p={float(v0):.4f}", flush=True)
        return {
            "ok": True, "warm_s": warm_s,
            "fwd_ms": fwd_ms, "fwd_grad_ms": fg_ms,
            "log_p": float(v0),
            "grad": g0.numpy().tolist() if g0 is not None else None,
        }
    except Exception as e:
        print(f"  [{name}] FAILED: {e!r}")
        traceback.print_exc()
        return {"ok": False, "error": repr(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_grid", type=int, nargs="+",
                   default=[64, 128, 256, 512])
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--n_ridges", type=int, default=64)
    p.add_argument("--K", type=int, default=10, help="Sinkhorn iterations")
    p.add_argument("--reps", type=int, default=5,
                   help="Timing reps after warm-up.")
    p.add_argument("--arch", type=str, default="deeponet",
                   choices=list(_ARCH_BUILDERS.keys()))
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override the default checkpoint for --arch.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Default: reports/.../section2_phase8_<arch>")
    args = p.parse_args()

    ckpt = Path(args.checkpoint or _DEFAULT_CKPTS[args.arch])
    out_dir = Path(args.out_dir or
                    f"reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                    f"section2_phase8_{args.arch}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[n-sweep] TF {tf.__version__}")
    print(f"  arch={args.arch}  T={args.T}  n_grid={args.n_grid}  K={args.K}  reps={args.reps}")
    print(f"  checkpoint: {ckpt}")
    print(f"  out_dir: {out_dir}\n")

    y_obs = gen_y_obs(args.T)
    mu_t = tf.constant(0.0, tf.float32)
    phi_t = tf.constant(0.95, tf.float32)
    sig_t = tf.constant(0.09, tf.float32)

    rows = []
    for N in args.n_grid:
        print(f"\n=== N = {N} ===")

        # --- Sinkhorn (XLA, jit_compile=True is the production default) ---
        try:
            ll_sk = DifferentiableLEDHLogLikelihoodSVSSM(
                num_particles=N, n_lambda=args.n_lambda,
                sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
                grad_window=4, jit_compile=True, integrator="exp",
            )
            r_sk = probe("sinkhorn-xla", ll_sk, mu_t, phi_t, sig_t, y_obs, args.reps)
        except Exception as e:
            print(f"  [sinkhorn-xla] FAILED: {e!r}")
            traceback.print_exc()
            r_sk = {"ok": False, "error": repr(e)}

        # --- NN-OT (XLA) ---
        try:
            model = build_model(N=N, arch=args.arch, ckpt=ckpt,
                                 n_ridges=args.n_ridges)
            ll_nn = DifferentiableLEDHNeuralOTSVSSMJIT(
                neural_ot_model=model, num_particles=N, n_lambda=args.n_lambda,
                sinkhorn_epsilon=1.0, grad_window=4, integrator="exp",
                graph_mode="xla",
            )
            r_nn = probe("nnot-xla", ll_nn, mu_t, phi_t, sig_t, y_obs, args.reps)
        except Exception as e:
            print(f"  [nnot-xla] FAILED: {e!r}")
            traceback.print_exc()
            r_nn = {"ok": False, "error": repr(e)}

        rows.append({"N": N, "sinkhorn_xla": r_sk, "nnot_xla": r_nn})

        # Free graphs between N's so XLA cache doesn't grow unbounded.
        tf.keras.backend.clear_session()

    # ---- Persist + summarise ----
    summary = {
        "tf": tf.__version__,
        "config": vars(args),
        "arch": args.arch,
        "checkpoint": str(ckpt),
        "rows": rows,
    }
    (out_dir / "probe_n_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n\n" + "=" * 100)
    print(f"{'N':>8}{'SK fwd ms':>14}{'SK f+g ms':>14}{'NN fwd ms':>14}"
          f"{'NN f+g ms':>14}{'speedup fwd':>14}{'speedup f+g':>14}")
    print("-" * 100)
    for r in rows:
        N = r["N"]
        sk = r["sinkhorn_xla"]
        nn = r["nnot_xla"]
        if not (sk.get("ok") and nn.get("ok")):
            print(f"{N:>8}  FAILED (sk_ok={sk.get('ok')}, nn_ok={nn.get('ok')})")
            continue
        sx_f = sk["fwd_ms"] / nn["fwd_ms"]
        sx_g = sk["fwd_grad_ms"] / nn["fwd_grad_ms"]
        print(f"{N:>8}{sk['fwd_ms']:>14.1f}{sk['fwd_grad_ms']:>14.1f}"
              f"{nn['fwd_ms']:>14.1f}{nn['fwd_grad_ms']:>14.1f}"
              f"{sx_f:>13.2f}x{sx_g:>13.2f}x")
    print("=" * 100)

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Ns = [r["N"] for r in rows]
        sk_fwd = [r["sinkhorn_xla"]["fwd_ms"] if r["sinkhorn_xla"].get("ok") else np.nan for r in rows]
        sk_fg  = [r["sinkhorn_xla"]["fwd_grad_ms"] if r["sinkhorn_xla"].get("ok") else np.nan for r in rows]
        nn_fwd = [r["nnot_xla"]["fwd_ms"] if r["nnot_xla"].get("ok") else np.nan for r in rows]
        nn_fg  = [r["nnot_xla"]["fwd_grad_ms"] if r["nnot_xla"].get("ok") else np.nan for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(Ns, sk_fwd, "o-", label="Sinkhorn XLA", color="C0")
        axes[0].plot(Ns, nn_fwd, "s-", label="NN-OT XLA", color="C1")
        axes[0].set_title(f"Warm forward time vs N  (T={args.T})")
        axes[1].plot(Ns, sk_fg, "o-", label="Sinkhorn XLA", color="C0")
        axes[1].plot(Ns, nn_fg, "s-", label="NN-OT XLA", color="C1")
        axes[1].set_title(f"Warm forward+grad time vs N  (T={args.T})")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("N (particles)")
            ax.set_ylabel("wall (ms, median of warm calls)")
            ax.grid(alpha=0.3, which="both")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "probe_n_sweep_wall_vs_N.png", dpi=120,
                     bbox_inches="tight")
        plt.close(fig)
        print(f"\n[plot] wrote {out_dir / 'probe_n_sweep_wall_vs_N.png'}")
    except ImportError:
        print("\n[plot] matplotlib not available; skipping plot")

    print(f"\n[done] wrote {out_dir / 'probe_n_sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
