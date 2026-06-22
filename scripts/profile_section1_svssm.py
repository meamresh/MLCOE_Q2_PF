"""
Section-1 (JIT / efficiency) profiling for the CURRENT extra_bonus SVSSM
filter ``DifferentiableLEDHLogLikelihoodSVSSM`` -- the actual HMC target of
this project. Mirrors scripts/profile_section1.py (which profiled the older
generic ``differentiable_pfpf_ledh``/``differentiable_ledh`` modules) so the
interview answer's Section-1 numbers describe the module that is really used.

Produces, in one self-documenting JSON:
  - retracing: per-step XLA kernel trace-count delta over 10 distinct theta,
  - cold vs warm forward and forward+grad timing (the JIT speedup),
  - full-filter N-scaling (fwd, fwd+grad) at N in {64,128,256,512},
  - standalone entropic-Sinkhorn O(N^2 K) fwd/backward N-scaling (shared OT),
  - one HMC step wall (bootstrap + warm one_step).

Config is embedded in the output for self-documentation.
"""

from __future__ import annotations

import json
import statistics as stats
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    DifferentiableLEDHLogLikelihoodSVSSM,
)
from src.filters.dpf.sinkhorn import sinkhorn_potentials


def gen_data(T, mu=0.0, phi=0.95, sigma_eta=0.3, seed=42):
    tf.random.set_seed(seed)
    se = tf.constant(sigma_eta, tf.float32)
    h = tf.constant(float(mu), tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + se * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def time_call(fn, repeats=5):
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


def make_ll(N, n_lambda, K):
    return DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N, n_lambda=n_lambda, sinkhorn_epsilon=1.0,
        sinkhorn_iters=K, grad_window=4, jit_compile=True, integrator="exp",
        init_type="stationary",
    )


def fwd_fn(ll, mu, phi, s2, y):
    def f():
        tf.random.set_seed(123)
        return ll(mu, phi, s2, y)
    return f


def fwd_grad_fn(ll, mu, phi, s2, y):
    def f():
        tf.random.set_seed(123)
        p = tf.identity(tf.stack([mu, phi, s2]))
        with tf.GradientTape() as tape:
            tape.watch(p)
            v = ll(p[0], p[1], p[2], y)
        return v, tape.gradient(v, p)
    return f


def main():
    T, N, n_lambda, K = 20, 64, 10, 10
    repeats = 5
    sizes = [64, 128, 256, 512]
    out_dir = Path(
        "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1_svssm"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    y = gen_data(T)
    mu = tf.constant(0.0, tf.float32)
    phi = tf.constant(0.95, tf.float32)
    s2 = tf.constant(0.09, tf.float32)
    print(f"[profile-svssm] TF {tf.__version__} TFP {tfp.__version__}  "
          f"module=DifferentiableLEDHLogLikelihoodSVSSM (extra_bonus)")
    print(f"  T={T} N={N} n_lambda={n_lambda} K={K} integrator=exp  device=CPU")

    # ---- cold + warm timing at N=64 ----
    ll = make_ll(N, n_lambda, K)
    f_fwd = fwd_fn(ll, mu, phi, s2, y)
    f_fg = fwd_grad_fn(ll, mu, phi, s2, y)

    t0 = time.perf_counter()
    _ = f_fwd().numpy()
    cold_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    v, g = f_fg()
    _ = v.numpy(); _ = g.numpy()
    cold_fg = time.perf_counter() - t0
    warm_fwd, _ = time_call(f_fwd, repeats)
    warm_fg, _ = time_call(f_fg, repeats)
    print(f"  cold fwd {cold_fwd:.3f}s -> warm {warm_fwd*1000:.1f}ms  "
          f"({cold_fwd/warm_fwd:.0f}x)")
    print(f"  cold fwd+grad {cold_fg:.3f}s -> warm {warm_fg*1000:.1f}ms  "
          f"({cold_fg/warm_fg:.0f}x)  backward/forward {warm_fg/warm_fwd:.1f}x")

    # ---- retracing over 10 distinct theta ----
    rng = np.random.default_rng(0)
    thetas = [
        tf.constant([rng.normal(0, 0.1),
                     np.tanh(rng.normal(2, 0.2)),
                     np.exp(rng.normal(-2, 0.3))], tf.float32)
        for _ in range(10)
    ]
    c1 = int(ll._timestep_1d_xla.experimental_get_tracing_count())
    for th in thetas:
        tf.random.set_seed(123)
        _ = ll(th[0], th[1], th[2], y).numpy()
    c2 = int(ll._timestep_1d_xla.experimental_get_tracing_count())
    retrace_delta = c2 - c1
    print(f"  retracing: trace delta over 10 evals = {retrace_delta} "
          f"({'PASS' if retrace_delta == 0 else 'FAIL'})")

    # ---- full-filter N-scaling (fresh filter per N) ----
    nscale = []
    for n in sizes:
        lln = make_ll(n, n_lambda, K)
        ff = fwd_fn(lln, mu, phi, s2, y)
        fg = fwd_grad_fn(lln, mu, phi, s2, y)
        _ = ff().numpy()
        vv, gg = fg(); _ = vv.numpy(); _ = gg.numpy()  # warm
        wf, _ = time_call(ff, repeats)
        wg, _ = time_call(fg, repeats)
        nscale.append({"N": n, "fwd_ms": wf * 1000, "fwd_grad_ms": wg * 1000,
                       "grad_over_fwd": wg / wf})
        print(f"  filter N={n}: fwd {wf*1000:.1f}ms  fwd+grad {wg*1000:.1f}ms  "
              f"({wg/wf:.1f}x)")

    # ---- standalone entropic-Sinkhorn O(N^2 K) N-scaling (shared OT module) ----
    sink = []
    for n in sizes:
        a = tf.fill([n], 1.0 / n)
        b = tf.fill([n], 1.0 / n)
        xs = tf.random.normal([n, 1], seed=0)
        ys = tf.random.normal([n, 1], seed=1)

        def s_fwd():
            return sinkhorn_potentials(a, b, xs, ys, 1.0, K)

        def s_fg():
            with tf.GradientTape() as tape:
                tape.watch(xs)
                f, gpot = sinkhorn_potentials(a, b, xs, ys, 1.0, K)
                loss = tf.reduce_sum(f) + tf.reduce_sum(gpot)
            return loss, tape.gradient(loss, xs)

        _ = s_fwd()
        _ = s_fg()  # warm
        sf, _ = time_call(s_fwd, repeats)
        sg, _ = time_call(s_fg, repeats)
        sink.append({"N": n, "fwd_ms": sf * 1000, "fwd_grad_ms": sg * 1000,
                     "grad_over_fwd": sg / sf})
        print(f"  sinkhorn N={n}: fwd {sf*1000:.2f}ms  fwd+grad {sg*1000:.2f}ms")

    # ---- one HMC step ----
    def target(raw):
        tf.random.set_seed(2026)
        vv = ll(raw[0], tf.tanh(raw[1]), tf.exp(raw[2]), y)
        vv = tf.cast(tf.math.real(vv), tf.float32)
        return tf.where(tf.math.is_finite(vv), vv, tf.constant(-1e6, tf.float32))

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=tf.constant(0.05, tf.float32),
        num_leapfrog_steps=5,
    )
    init = tf.constant([0.0, 2.0, -2.0], tf.float32)
    t0 = time.perf_counter()
    kr = kernel.bootstrap_results(init)
    boot = time.perf_counter() - t0
    t0 = time.perf_counter()
    state, kr = kernel.one_step(init, kr, seed=[1, 2])
    _ = state.numpy()
    first = time.perf_counter() - t0
    steps = []
    for i in range(3):
        t0 = time.perf_counter()
        state, kr = kernel.one_step(state, kr, seed=[i + 3, i + 4])
        _ = state.numpy()
        steps.append(time.perf_counter() - t0)
    warm_step = float(stats.median(steps))
    print(f"  HMC step: bootstrap {boot:.2f}s  first {first:.2f}s  "
          f"warm {warm_step:.2f}s")

    result = {
        "module": "DifferentiableLEDHLogLikelihoodSVSSM (src/filters/bonus/extra_bonus)",
        "env": {"tf": tf.__version__, "tfp": tfp.__version__, "device": "CPU"},
        "config": {"T": T, "N": N, "n_lambda": n_lambda, "sinkhorn_iters": K,
                   "integrator": "exp", "repeats": repeats},
        "timing_N64": {
            "cold_fwd_s": cold_fwd, "warm_fwd_ms": warm_fwd * 1000,
            "cold_fwd_grad_s": cold_fg, "warm_fwd_grad_ms": warm_fg * 1000,
            "jit_speedup_fwd": cold_fwd / warm_fwd,
            "jit_speedup_fwd_grad": cold_fg / warm_fg,
            "backward_over_forward": warm_fg / warm_fwd,
        },
        "retracing": {"trace_delta_over_10_evals": retrace_delta,
                      "pass": retrace_delta == 0},
        "filter_n_scaling": nscale,
        "sinkhorn_n_scaling": sink,
        "hmc_step": {"bootstrap_s": boot, "first_step_s": first,
                     "warm_step_s": warm_step},
    }
    (out_dir / "profile_svssm.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_dir / 'profile_svssm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
