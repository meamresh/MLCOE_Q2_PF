"""
Verify that LEDH-PF-PF-OT inside HMC does NOT retrace after warmup.

Strategy:
  - Snapshot `experimental_get_tracing_count()` on every JIT'd kernel used by
    KitagawaPFPFLEDHLogLikelihood and the Sinkhorn module.
  - Warm up once (forward and forward+grad).
  - Run M evaluations with M distinct theta values of the same shape/dtype,
    mimicking HMC's varying-proposal pattern. Snapshot the counts again.
  - Report the delta. Pass criterion: every kernel's trace count is unchanged
    after warmup.

If any kernel retraces, this script identifies which one — that's the bug.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus import differentiable_pfpf_ledh as dpfl
from src.filters.dpf import sinkhorn as sk
from src.filters.bonus.differentiable_pfpf_ledh import KitagawaPFPFLEDHLogLikelihood


JIT_KERNELS = [
    ("_ledh_flow_jit", dpfl._ledh_flow_jit),
    ("_weight_increment_jit", dpfl._weight_increment_jit),
    ("_kitagawa_predict_1d_jit", dpfl._kitagawa_predict_1d_jit),
    ("_kitagawa_ledh_flow_1d_jit", dpfl._kitagawa_ledh_flow_1d_jit),
    ("_kitagawa_weight_increment_1d_jit", dpfl._kitagawa_weight_increment_1d_jit),
    ("sinkhorn_potentials", sk.sinkhorn_potentials),
]


def snapshot_trace_counts() -> dict[str, int]:
    return {name: int(fn.experimental_get_tracing_count()) for name, fn in JIT_KERNELS}


def diff_counts(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: after[k] - before[k] for k in before}


def generate_kitagawa(T: int, sigma_v_sq: float, sigma_w_sq: float, seed: int) -> tf.Tensor:
    tf.random.set_seed(seed)
    sv = tf.sqrt(tf.cast(sigma_v_sq, tf.float32))
    sw = tf.sqrt(tf.cast(sigma_w_sq, tf.float32))
    x = tf.random.normal([]) * tf.sqrt(tf.constant(5.0, tf.float32))
    ys = [x**2 / 20.0 + sw * tf.random.normal([])]
    for t in range(2, T + 1):
        t_f = tf.cast(t, tf.float32)
        x = (
            0.5 * x
            + 25.0 * x / (1.0 + x**2)
            + 8.0 * tf.cos(1.2 * t_f)
            + sv * tf.random.normal([])
        )
        ys.append(x**2 / 20.0 + sw * tf.random.normal([]))
    return tf.stack(ys)


def run_forward_evals(ll, y_obs, thetas):
    for theta in thetas:
        tf.random.set_seed(123)
        val = ll(tf.exp(theta[0]), tf.exp(theta[1]), y_obs)
        _ = val.numpy()


def run_grad_evals(ll, y_obs, thetas):
    for theta in thetas:
        tf.random.set_seed(123)
        theta_t = tf.identity(theta)
        with tf.GradientTape() as tape:
            tape.watch(theta_t)
            val = ll(tf.exp(theta_t[0]), tf.exp(theta_t[1]), y_obs)
        grad = tape.gradient(val, theta_t)
        _ = val.numpy()
        _ = grad.numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=5)
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    p.add_argument("--n_eval", type=int, default=10,
                   help="Number of post-warmup theta evaluations.")
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1")
    args = p.parse_args()

    print("[verify-no-retracing] TF", tf.__version__)
    print(f"  Config: T={args.T} N={args.N} n_lambda={args.n_lambda} "
          f"sinkhorn_iters={args.sinkhorn_iters} eps={args.sinkhorn_epsilon} "
          f"n_eval={args.n_eval}")

    y_obs = generate_kitagawa(args.T, sigma_v_sq=10.0, sigma_w_sq=1.0, seed=42)
    ll = KitagawaPFPFLEDHLogLikelihood(
        num_particles=args.N,
        n_lambda=args.n_lambda,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        resample_threshold=0.5,
        clip_weight_terms=False,
    )

    rng = np.random.default_rng(0)
    thetas = [
        tf.constant(rng.normal(loc=[np.log(10.0), np.log(1.0)], scale=0.05),
                    dtype=tf.float32)
        for _ in range(args.n_eval)
    ]
    theta_warm = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)

    # === Forward-only path ===
    c0 = snapshot_trace_counts()
    print(f"\n[forward] pre-warmup trace counts: {c0}")
    t0 = time.perf_counter()
    run_forward_evals(ll, y_obs, [theta_warm])
    t_warm = time.perf_counter() - t0
    c1 = snapshot_trace_counts()
    print(f"[forward] post-warmup trace counts ({t_warm:.2f}s): {c1}")
    print(f"[forward] warmup delta:           {diff_counts(c0, c1)}")

    t0 = time.perf_counter()
    run_forward_evals(ll, y_obs, thetas)
    t_eval = time.perf_counter() - t0
    c2 = snapshot_trace_counts()
    fwd_delta = diff_counts(c1, c2)
    print(f"[forward] after {args.n_eval} distinct-theta evals ({t_eval:.2f}s)")
    print(f"[forward] eval-phase delta:       {fwd_delta}")
    fwd_pass = all(v == 0 for v in fwd_delta.values())
    print(f"[forward] {'PASS' if fwd_pass else 'FAIL'}: "
          f"{'zero retraces' if fwd_pass else 'retracing detected'}")

    # === Forward + gradient path (HMC-realistic) ===
    c3 = snapshot_trace_counts()
    t0 = time.perf_counter()
    run_grad_evals(ll, y_obs, [theta_warm])
    t_warm_g = time.perf_counter() - t0
    c4 = snapshot_trace_counts()
    print(f"\n[forward+grad] post-warmup trace counts ({t_warm_g:.2f}s): {c4}")
    print(f"[forward+grad] warmup delta:      {diff_counts(c3, c4)}")

    t0 = time.perf_counter()
    run_grad_evals(ll, y_obs, thetas)
    t_eval_g = time.perf_counter() - t0
    c5 = snapshot_trace_counts()
    grad_delta = diff_counts(c4, c5)
    print(f"[forward+grad] after {args.n_eval} distinct-theta evals ({t_eval_g:.2f}s)")
    print(f"[forward+grad] eval-phase delta:  {grad_delta}")
    grad_pass = all(v == 0 for v in grad_delta.values())
    print(f"[forward+grad] {'PASS' if grad_pass else 'FAIL'}: "
          f"{'zero retraces' if grad_pass else 'retracing detected'}")

    # === Final verdict ===
    print("\n" + "=" * 60)
    overall = fwd_pass and grad_pass
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    out = {
        "tf": tf.__version__,
        "config": vars(args),
        "forward": {
            "pre_warmup": c0, "post_warmup": c1, "post_eval": c2,
            "warmup_delta": diff_counts(c0, c1),
            "eval_delta": fwd_delta,
            "warmup_s": t_warm, "eval_s": t_eval,
            "pass": fwd_pass,
        },
        "forward_grad": {
            "pre_warmup": c3, "post_warmup": c4, "post_eval": c5,
            "warmup_delta": diff_counts(c3, c4),
            "eval_delta": grad_delta,
            "warmup_s": t_warm_g, "eval_s": t_eval_g,
            "pass": grad_pass,
        },
        "overall_pass": overall,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retracing_verification.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_dir / 'retracing_verification.json'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
