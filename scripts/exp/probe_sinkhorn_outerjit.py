"""
Probe: does wrapping the Sinkhorn filter's outer loop in
@tf.function(jit_compile=True) help or hurt?

Background: Phase 1 of the project ran the Sinkhorn outer loop in
plain Python with a JIT'd per-timestep body. Phase 6 found that the
NN-OT filter could be fully JIT'd after fixing 3 `.numpy()` calls.
The Sinkhorn filter does not have those calls — it's pure TF inside.
So in principle the whole outer loop could be wrapped in a single
XLA-compiled function. This probe tests whether that is:

  (a) numerically correct (log p agrees with the default),
  (b) faster on forward and forward+grad,
  (c) compile-time-feasible at T = 20 and T = 50.

Three configs timed on the same y_obs / theta:
  - default        : Python outer loop + JIT'd per-step body (production)
  - outer-jit      : sandbox class with outer_jit=True
  - eager-fallback : Python outer + eager per-step body (sanity)

Outputs:
  reports/.../section1_outerjit_probe/summary_T{T}.json
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
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_outerjit import (
    DifferentiableLEDHLogLikelihoodSVSSMOuterJIT,
)


def gen_y_obs(T: int, mu: float = 0.0, phi: float = 0.95,
              sigma_eta: float = 0.3, seed: int = 42) -> tf.Tensor:
    tf.random.set_seed(seed)
    h = tf.constant(0.0, tf.float32)
    s = tf.constant(sigma_eta, tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + s * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def time_call(fn, reps: int = 7) -> float:
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
    print(f"\n[{name}] warming...", flush=True)

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
        grad_arr = g0.numpy() if g0 is not None else None
        print(f"[{name}] warm={warm_s:6.1f}s  fwd={fwd_ms:7.1f} ms  "
              f"fwd+grad={fg_ms:7.1f} ms  log p={float(v0):.4f}", flush=True)
        if grad_arr is not None:
            print(f"[{name}]   grad = {grad_arr}", flush=True)
        return {
            "ok": True,
            "warm_s": warm_s,
            "fwd_ms": fwd_ms,
            "fwd_grad_ms": fg_ms,
            "log_p": float(v0),
            "grad": grad_arr.tolist() if grad_arr is not None else None,
        }
    except Exception as e:
        print(f"[{name}] FAILED: {e!r}")
        traceback.print_exc()
        return {"ok": False, "error": repr(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section1_outerjit_probe")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[outerjit-probe] TF {tf.__version__}")
    print(f"  T={args.T}  N={args.N}  K={args.K}  reps={args.reps}")
    print(f"  out_dir={out_dir}\n")

    y_obs = gen_y_obs(args.T)
    mu_t = tf.constant(0.0, tf.float32)
    phi_t = tf.constant(0.95, tf.float32)
    sig_t = tf.constant(0.09, tf.float32)

    results = {}

    # ---- 1) production default (per-step body JIT, Python outer loop) ----
    ll_default = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=args.N, n_lambda=args.n_lambda,
        sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
        grad_window=4, jit_compile=True, integrator="exp",
    )
    results["default_perstep_jit"] = probe(
        "default (perstep-JIT, Python outer)",
        ll_default, mu_t, phi_t, sig_t, y_obs, args.reps,
    )

    # ---- 2) outer-loop JIT (this experiment) ----
    ll_outer = DifferentiableLEDHLogLikelihoodSVSSMOuterJIT(
        num_particles=args.N, n_lambda=args.n_lambda,
        sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
        grad_window=4, jit_compile=True, integrator="exp",
        outer_jit=True,
    )
    results["outer_jit"] = probe(
        "outer-JIT (whole _run_1d_xla wrapped)",
        ll_outer, mu_t, phi_t, sig_t, y_obs, args.reps,
    )

    # ---- 3) eager fallback (sanity / numerical reference) ----
    ll_eager = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=args.N, n_lambda=args.n_lambda,
        sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
        grad_window=4, jit_compile=False, integrator="exp",
    )
    results["eager_perstep"] = probe(
        "eager (no JIT anywhere)",
        ll_eager, mu_t, phi_t, sig_t, y_obs, args.reps,
    )

    # ---- Summary ----
    print("\n" + "=" * 90)
    print(f"{'config':<40}{'warm s':>10}{'fwd ms':>10}{'fwd+grad ms':>14}{'log p':>12}")
    print("-" * 90)
    base_fg = None
    for key, r in results.items():
        if not r.get("ok"):
            print(f"{key:<40}  FAILED: {r.get('error', 'unknown')[:50]}")
            continue
        print(f"{key:<40}{r['warm_s']:>10.1f}{r['fwd_ms']:>10.1f}"
              f"{r['fwd_grad_ms']:>14.1f}{r['log_p']:>12.4f}")
        if key == "default_perstep_jit":
            base_fg = r['fwd_grad_ms']
    print("=" * 90)

    if base_fg is not None:
        print(f"\nForward+grad speedup vs default:")
        for key, r in results.items():
            if r.get("ok") and key != "default_perstep_jit":
                print(f"  {key:<40} {base_fg / r['fwd_grad_ms']:.2f}x")

    # ---- Numerical agreement check ----
    print("\nLog-p numerical agreement (all should match within MC noise):")
    for key, r in results.items():
        if r.get("ok"):
            print(f"  {key:<40} log p = {r['log_p']:.6f}")

    # ---- Persist ----
    out_path = out_dir / f"summary_T{args.T}_N{args.N}.json"
    summary = {
        "tf": tf.__version__,
        "T": args.T, "N": args.N, "K": args.K, "reps": args.reps,
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
