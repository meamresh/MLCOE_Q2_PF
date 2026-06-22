"""
Speed probe for the JIT-experiment NN-OT SVSSM filter.

Times three configurations on the same trained mGradNet checkpoint:
  1. eager   — current production filter
  2. graph   — @tf.function wrapper, no XLA
  3. xla     — @tf.function(jit_compile=True)

For each, measures:
  - one warm forward call
  - one warm forward+gradient call (the cost HMC actually pays)

Reports the speedup of graph and xla relative to eager.

Usage:
    PYTHONPATH=. python scripts/exp/probe_nnot_jit.py
"""

from __future__ import annotations

import statistics as stats
import time
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm import (
    DifferentiableLEDHNeuralOTSVSSM,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_jit import (
    DifferentiableLEDHNeuralOTSVSSMJIT,
)
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet


CKPT = Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
            "section2_phase4/supervised.weights.h5")


def build_model(N: int = 64, n_ridges: int = 64) -> ConditionalMGradNet:
    tf.random.set_seed(0)
    m = ConditionalMGradNet(state_dim=1, n_ridges=n_ridges, n_scalar_ctx=7)
    dummy_p = tf.zeros([N], tf.float32)
    dummy_w = tf.fill([N], 1.0 / float(N))
    dummy_c = tf.zeros([7], tf.float32)
    _ = m(dummy_p, dummy_w, dummy_c)
    m.load_weights(str(CKPT))
    return m


def gen_y_obs(T: int = 20, seed: int = 42) -> tf.Tensor:
    tf.random.set_seed(seed)
    h = tf.constant(0.0, tf.float32)
    sigma_eta = tf.constant(0.3, tf.float32)
    ys = []
    for _ in range(T):
        h = 0.0 + 0.95 * (h - 0.0) + sigma_eta * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def time_call(fn, reps: int = 7) -> float:
    """Return median wall (s)."""
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


def probe_one(name: str, ll, mu_t, phi_t, sig_t, y_obs):
    """Time forward + forward-and-gradient for one filter instance."""
    print(f"\n[{name}] warming up…", flush=True)

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
        # warm-up (compiles)
        v0 = fwd()
        g0 = fwd_grad()
        fwd_ms = time_call(fwd, 7) * 1000
        fg_ms = time_call(fwd_grad, 7) * 1000
        print(f"[{name}]  forward       : {fwd_ms:8.1f} ms (log p = {float(v0):.4f})")
        print(f"[{name}]  forward+grad  : {fg_ms:8.1f} ms (grad = {g0.numpy()})")
        return {
            "name": name, "ok": True,
            "fwd_ms": fwd_ms, "fwd_grad_ms": fg_ms,
            "log_p": float(v0),
            "grad": g0.numpy().tolist(),
        }
    except Exception as e:
        print(f"[{name}] FAILED: {e!r}")
        traceback.print_exc()
        return {"name": name, "ok": False, "error": repr(e)}


def main():
    print(f"TF {tf.__version__}")
    print(f"checkpoint: {CKPT}")
    print(f"running on: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

    T, N = 20, 64
    y_obs = gen_y_obs(T)
    mu_t = tf.constant(0.0, tf.float32)
    phi_t = tf.constant(0.95, tf.float32)
    sig_t = tf.constant(0.09, tf.float32)

    results = []

    # ---- 1) Current eager production filter ----
    m1 = build_model(N=N)
    ll1 = DifferentiableLEDHNeuralOTSVSSM(
        neural_ot_model=m1, num_particles=N, n_lambda=10,
        sinkhorn_epsilon=1.0, grad_window=4,
        jit_compile=False, integrator="exp",
    )
    results.append(probe_one("eager (production)", ll1, mu_t, phi_t, sig_t, y_obs))

    # ---- 2) tf.function (graph), no XLA ----
    m2 = build_model(N=N)
    ll2 = DifferentiableLEDHNeuralOTSVSSMJIT(
        neural_ot_model=m2, num_particles=N, n_lambda=10,
        sinkhorn_epsilon=1.0, grad_window=4,
        integrator="exp", graph_mode="graph",
    )
    results.append(probe_one("graph (tf.function)", ll2, mu_t, phi_t, sig_t, y_obs))

    # ---- 3) tf.function(jit_compile=True) ----
    m3 = build_model(N=N)
    ll3 = DifferentiableLEDHNeuralOTSVSSMJIT(
        neural_ot_model=m3, num_particles=N, n_lambda=10,
        sinkhorn_epsilon=1.0, grad_window=4,
        integrator="exp", graph_mode="xla",
    )
    results.append(probe_one("xla (jit_compile)", ll3, mu_t, phi_t, sig_t, y_obs))

    # ---- Summary ----
    print("\n" + "=" * 80)
    print(f"{'config':<25}{'fwd ms':>12}{'fwd+grad ms':>16}{'speedup fwd':>14}{'speedup fg':>14}")
    print("-" * 80)
    base_fwd = next((r['fwd_ms'] for r in results if r['name'].startswith('eager') and r['ok']), None)
    base_fg = next((r['fwd_grad_ms'] for r in results if r['name'].startswith('eager') and r['ok']), None)
    for r in results:
        if not r['ok']:
            print(f"{r['name']:<25} FAILED: {r['error'][:60]}")
            continue
        sx_f = base_fwd / r['fwd_ms'] if base_fwd else 1.0
        sx_g = base_fg / r['fwd_grad_ms'] if base_fg else 1.0
        print(f"{r['name']:<25}{r['fwd_ms']:>12.1f}{r['fwd_grad_ms']:>16.1f}"
              f"{sx_f:>13.2f}x{sx_g:>13.2f}x")
    print("=" * 80)

    # log p agreement check (sanity)
    print("\nlog p agreement (should be identical):")
    for r in results:
        if r['ok']:
            print(f"  {r['name']:<25} log p = {r['log_p']:.6f}")

    # Persist
    import json
    out = Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase4/nnot_jit_probe.json")
    out.write_text(json.dumps({
        "tf": tf.__version__, "T": T, "N": N,
        "ckpt": str(CKPT),
        "results": results,
    }, indent=2))
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
