"""
A/B compare the new full-T jit_compile path against the original Python-loop
eager path inside KitagawaPFPFLEDHLogLikelihood.

Reports:
  - log-likelihood value equivalence (same tf.random.set_seed -> same value, ideally)
  - gradient equivalence
  - warm forward time
  - warm forward+grad time
  - retracing count delta after 10 distinct-theta evals (must be 0)
"""

from __future__ import annotations

import statistics as stats
import time

import numpy as np
import tensorflow as tf

from src.filters.bonus import differentiable_pfpf_ledh as dpfl
from src.filters.bonus.differentiable_pfpf_ledh import KitagawaPFPFLEDHLogLikelihood
from src.filters.dpf import sinkhorn as sk


JIT_KERNELS = [
    ("_ledh_flow_jit", dpfl._ledh_flow_jit),
    ("_weight_increment_jit", dpfl._weight_increment_jit),
    ("_kitagawa_predict_1d_jit", dpfl._kitagawa_predict_1d_jit),
    ("_kitagawa_ledh_flow_1d_jit", dpfl._kitagawa_ledh_flow_1d_jit),
    ("_kitagawa_weight_increment_1d_jit", dpfl._kitagawa_weight_increment_1d_jit),
    ("sinkhorn_potentials", sk.sinkhorn_potentials),
]


def trace_counts():
    return {n: int(f.experimental_get_tracing_count()) for n, f in JIT_KERNELS}


def gen_data(T=20, sigma_v_sq=10.0, sigma_w_sq=1.0, seed=42):
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
    T = 20
    N = 64
    n_lambda = 5
    sk_iters = 20
    eps = 1.0

    y_obs = gen_data(T=T)
    ll = KitagawaPFPFLEDHLogLikelihood(
        num_particles=N,
        n_lambda=n_lambda,
        sinkhorn_epsilon=eps,
        sinkhorn_iters=sk_iters,
        resample_threshold=0.5,
        clip_weight_terms=False,
    )
    theta = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)

    # === Value equivalence with fixed seed ===
    # Note: identical results across the two paths are not guaranteed because
    # tf.random.normal inside jit_compile vs eager may sample in a different
    # order. We require both to be in the same plausible range, not bit-identical.
    print("[value equivalence — eager first, then JIT]")
    tf.random.set_seed(7)
    val_eager_a = float(ll._call_eager(tf.exp(theta[0]), tf.exp(theta[1]), y_obs).numpy())
    tf.random.set_seed(7)
    val_jit_a = float(ll(tf.exp(theta[0]), tf.exp(theta[1]), y_obs).numpy())
    print(f"  eager (1st): log p = {val_eager_a:.6f}")
    print(f"  JIT   (2nd): log p = {val_jit_a:.6f}")

    print("\n[value equivalence — JIT first, then eager]")
    tf.random.set_seed(7)
    val_jit_b = float(ll(tf.exp(theta[0]), tf.exp(theta[1]), y_obs).numpy())
    tf.random.set_seed(7)
    val_eager_b = float(ll._call_eager(tf.exp(theta[0]), tf.exp(theta[1]), y_obs).numpy())
    print(f"  JIT   (1st): log p = {val_jit_b:.6f}")
    print(f"  eager (2nd): log p = {val_eager_b:.6f}")

    val_jit = val_jit_a
    val_eager = val_eager_a

    # === Gradient equivalence ===
    print("\n[gradient equivalence]")
    tf.random.set_seed(11)
    th = tf.identity(theta)
    with tf.GradientTape() as tape:
        tape.watch(th)
        v = ll(tf.exp(th[0]), tf.exp(th[1]), y_obs)
    g_jit = tape.gradient(v, th).numpy()

    tf.random.set_seed(11)
    th = tf.identity(theta)
    with tf.GradientTape() as tape:
        tape.watch(th)
        v = ll._call_eager(tf.exp(th[0]), tf.exp(th[1]), y_obs)
    g_eager = tape.gradient(v, th).numpy()
    print(f"  JIT   grad = {g_jit}")
    print(f"  eager grad = {g_eager}")
    print(f"  delta L2 = {float(np.linalg.norm(g_jit - g_eager)):.4g}")

    # === Retracing check on JIT path ===
    print("\n[retracing on JIT path]")
    rng = np.random.default_rng(0)
    thetas = [
        tf.constant(rng.normal(loc=[np.log(10.0), np.log(1.0)], scale=0.05), dtype=tf.float32)
        for _ in range(10)
    ]
    # one warmup
    _ = ll(tf.exp(theta[0]), tf.exp(theta[1]), y_obs).numpy()
    c0 = trace_counts()
    # also track outer _loglik_jit
    outer_count_0 = int(ll._loglik_jit.experimental_get_tracing_count())
    for th_v in thetas:
        _ = ll(tf.exp(th_v[0]), tf.exp(th_v[1]), y_obs).numpy()
    c1 = trace_counts()
    outer_count_1 = int(ll._loglik_jit.experimental_get_tracing_count())
    delta = {k: c1[k] - c0[k] for k in c0}
    print(f"  per-step kernel deltas: {delta}")
    print(f"  outer _loglik_jit trace count: {outer_count_0} -> {outer_count_1}")
    overall_no_retrace = all(v == 0 for v in delta.values()) and outer_count_0 == outer_count_1
    print(f"  {'PASS' if overall_no_retrace else 'FAIL'}: "
          f"{'zero retraces' if overall_no_retrace else 'RETRACING DETECTED'}")

    # === Timing ===
    print("\n[timing]")

    def fwd_jit():
        tf.random.set_seed(123)
        return ll(tf.exp(theta[0]), tf.exp(theta[1]), y_obs)

    def fwd_eager():
        tf.random.set_seed(123)
        return ll._call_eager(tf.exp(theta[0]), tf.exp(theta[1]), y_obs)

    def fg_jit():
        tf.random.set_seed(123)
        th = tf.identity(theta)
        with tf.GradientTape() as tape:
            tape.watch(th)
            v = ll(tf.exp(th[0]), tf.exp(th[1]), y_obs)
        return v, tape.gradient(v, th)

    def fg_eager():
        tf.random.set_seed(123)
        th = tf.identity(theta)
        with tf.GradientTape() as tape:
            tape.watch(th)
            v = ll._call_eager(tf.exp(th[0]), tf.exp(th[1]), y_obs)
        return v, tape.gradient(v, th)

    # warm both paths
    _ = fwd_jit(); _ = fwd_eager(); _ = fg_jit(); _ = fg_eager()

    j_fwd, j_fwd_s = time_fn(fwd_jit, 5)
    e_fwd, e_fwd_s = time_fn(fwd_eager, 5)
    j_fg, j_fg_s = time_fn(fg_jit, 5)
    e_fg, e_fg_s = time_fn(fg_eager, 5)
    print(f"  warm forward      JIT={j_fwd*1000:7.2f}ms  eager={e_fwd*1000:7.2f}ms  "
          f"speedup={e_fwd/j_fwd:5.2f}x")
    print(f"  warm forward+grad JIT={j_fg*1000:7.2f}ms  eager={e_fg*1000:7.2f}ms  "
          f"speedup={e_fg/j_fg:5.2f}x")

    return 0 if overall_no_retrace else 1


if __name__ == "__main__":
    raise SystemExit(main())
