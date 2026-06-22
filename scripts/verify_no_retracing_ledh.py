"""
Verify Section 1 properties for the LEDH HMC target.

We use DifferentiableLEDHLogLikelihood (not PFPF) as the HMC target because
PFPF's importance-ratio term destabilizes gradients (10% NaN rate, ll CV=2.2;
see reports/.../gradient_stability_report.txt). This script proves that the
JIT + no-retrace guarantees from Section 1 transfer to LEDH-OT:

  - per-step kernel trace count: 1 after warmup, stays at 1 across 10 evals
  - outer __call__ accepts a fresh PMCMCNonlinearSSM per call (HMC-realistic);
    confirm this does NOT cause retracing of the per-step XLA kernel
  - forward + gradient stable, no NaN
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.filters.dpf import sinkhorn as sk
from src.models.ssm_katigawa import PMCMCNonlinearSSM


def gen_data(T, sigma_v_sq=10.0, sigma_w_sq=1.0, seed=42):
    tf.random.set_seed(seed)
    sv = tf.sqrt(tf.cast(sigma_v_sq, tf.float32))
    sw = tf.sqrt(tf.cast(sigma_w_sq, tf.float32))
    x = tf.random.normal([]) * tf.sqrt(tf.constant(5.0, tf.float32))
    ys = [x ** 2 / 20.0 + sw * tf.random.normal([])]
    for t in range(2, T + 1):
        t_f = tf.cast(t, tf.float32)
        x = 0.5 * x + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_f) + sv * tf.random.normal([])
        ys.append(x ** 2 / 20.0 + sw * tf.random.normal([]))
    return tf.stack(ys)


def time_fn(fn, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tf.Tensor):
            _ = out.numpy()
        elif isinstance(out, (list, tuple)):
            for o in out:
                if isinstance(o, tf.Tensor):
                    _ = o.numpy()
        ts.append(time.perf_counter() - t0)
    return float(stats.median(ts)), float(stats.pstdev(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=20)
    p.add_argument("--sinkhorn_iters", type=int, default=10)
    p.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    p.add_argument("--grad_window", type=int, default=4)
    p.add_argument("--n_eval", type=int, default=10)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1")
    args = p.parse_args()

    print(f"[verify-no-retracing-ledh] TF {tf.__version__}")
    print(f"  Config: T={args.T} N={args.N} n_lambda={args.n_lambda} "
          f"sinkhorn_iters={args.sinkhorn_iters} grad_window={args.grad_window}")

    y_obs = gen_data(args.T)
    ll = DifferentiableLEDHLogLikelihood(
        num_particles=args.N,
        n_lambda=args.n_lambda,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        resample_threshold=0.5,
        grad_window=args.grad_window,
        jit_compile=True,
    )

    # The per-step kernel that should NOT retrace. It's a bound method of the
    # LL instance, so we read trace count off the descriptor.
    kernels = [
        ("_timestep_1d_xla (bound)", ll._timestep_1d_xla),
        ("sinkhorn_potentials", sk.sinkhorn_potentials),
    ]

    def trace_counts():
        return {n: int(f.experimental_get_tracing_count()) for n, f in kernels}

    rng = np.random.default_rng(0)
    thetas = [
        tf.constant(rng.normal(loc=[np.log(10.0), np.log(1.0)], scale=0.05), dtype=tf.float32)
        for _ in range(args.n_eval)
    ]

    def call_with_theta(theta):
        """Mimic HMC: fresh SSM object per proposal, theta as tensor."""
        ssm = PMCMCNonlinearSSM(
            sigma_v_sq=tf.exp(theta[0]),
            sigma_w_sq=tf.exp(theta[1]),
            initial_var=5.0,
        )
        return ll(ssm, y_obs)

    def call_with_grad(theta):
        theta_t = tf.identity(theta)
        with tf.GradientTape() as tape:
            tape.watch(theta_t)
            ssm = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(theta_t[0]),
                sigma_w_sq=tf.exp(theta_t[1]),
                initial_var=5.0,
            )
            v = ll(ssm, y_obs)
        return v, tape.gradient(v, theta_t)

    theta_warm = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)

    # === Forward-only path ===
    c0 = trace_counts()
    print(f"\n[forward] pre-warmup trace counts: {c0}")

    t0 = time.perf_counter()
    tf.random.set_seed(123)
    val_warm = float(call_with_theta(theta_warm).numpy())
    t_warm = time.perf_counter() - t0

    c1 = trace_counts()
    print(f"[forward] post-warmup: {c1}  (warmup {t_warm:.2f}s, log p={val_warm:.4f})")

    t0 = time.perf_counter()
    vals = []
    for th in thetas:
        tf.random.set_seed(123)
        vals.append(float(call_with_theta(th).numpy()))
    t_eval = time.perf_counter() - t0
    c2 = trace_counts()
    fwd_delta = {k: c2[k] - c1[k] for k in c1}
    print(f"[forward] {args.n_eval} distinct-theta evals + fresh SSMs ({t_eval:.2f}s)")
    print(f"           log p range: [{min(vals):.3f}, {max(vals):.3f}], all finite: {all(np.isfinite(vals))}")
    print(f"           per-step kernel deltas: {fwd_delta}")
    fwd_pass = all(v == 0 for v in fwd_delta.values()) and all(np.isfinite(vals))
    print(f"           {'PASS' if fwd_pass else 'FAIL'}")

    # === Forward + gradient path ===
    t0 = time.perf_counter()
    tf.random.set_seed(123)
    v0, g0 = call_with_grad(theta_warm)
    _ = v0.numpy(); _ = g0.numpy()
    t_warm_g = time.perf_counter() - t0
    c4 = trace_counts()
    print(f"\n[fwd+grad] post-warmup: {c4}  (warmup {t_warm_g:.2f}s, grad={g0.numpy()})")

    t0 = time.perf_counter()
    grads = []
    finite_grads = 0
    for th in thetas:
        tf.random.set_seed(123)
        _, g = call_with_grad(th)
        g_np = g.numpy()
        grads.append(g_np)
        if np.all(np.isfinite(g_np)):
            finite_grads += 1
    t_eval_g = time.perf_counter() - t0
    c5 = trace_counts()
    grad_delta = {k: c5[k] - c4[k] for k in c4}
    grads_arr = np.asarray(grads)
    print(f"[fwd+grad] {args.n_eval} distinct-theta evals ({t_eval_g:.2f}s)")
    print(f"           finite grad rate: {finite_grads}/{args.n_eval}")
    print(f"           grad norm range: [{np.linalg.norm(grads_arr, axis=1).min():.3e}, "
          f"{np.linalg.norm(grads_arr, axis=1).max():.3e}]")
    print(f"           per-step kernel deltas: {grad_delta}")
    grad_pass = all(v == 0 for v in grad_delta.values()) and finite_grads == args.n_eval
    print(f"           {'PASS' if grad_pass else 'FAIL'}")

    # === Timing ===
    print("\n[timing]")

    def fwd():
        tf.random.set_seed(123)
        return call_with_theta(theta_warm)

    def fg():
        tf.random.set_seed(123)
        return call_with_grad(theta_warm)

    fwd_med, _ = time_fn(fwd, 5)
    fg_med, _ = time_fn(fg, 5)
    print(f"  warm forward      = {fwd_med * 1000:7.2f} ms")
    print(f"  warm forward+grad = {fg_med * 1000:7.2f} ms")
    print(f"  backward/forward  = {fg_med / fwd_med:5.2f}x")

    overall_pass = fwd_pass and grad_pass
    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)

    out = {
        "tf": tf.__version__,
        "config": vars(args),
        "forward": {
            "pre_warmup": c0, "post_warmup": c1, "post_eval": c2,
            "eval_delta": fwd_delta, "vals_min": float(min(vals)), "vals_max": float(max(vals)),
            "warmup_s": t_warm, "eval_s": t_eval, "pass": fwd_pass,
        },
        "fwd_grad": {
            "post_warmup": c4, "post_eval": c5, "eval_delta": grad_delta,
            "finite_grad_count": int(finite_grads), "n_eval": int(args.n_eval),
            "grad_norm_min": float(np.linalg.norm(grads_arr, axis=1).min()),
            "grad_norm_max": float(np.linalg.norm(grads_arr, axis=1).max()),
            "warmup_s": t_warm_g, "eval_s": t_eval_g, "pass": grad_pass,
        },
        "timing": {
            "warm_forward_s_median": fwd_med,
            "warm_forward_plus_grad_s_median": fg_med,
            "backward_over_forward": fg_med / fwd_med,
        },
        "overall_pass": overall_pass,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retracing_verification_ledh.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_dir / 'retracing_verification_ledh.json'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
