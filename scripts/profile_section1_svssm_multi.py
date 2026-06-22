"""
Section-1 (JIT / efficiency) profiling for the multivariate SVSSM filter
``DifferentiableLEDHLogLikelihoodSVSSMmulti`` (diagonal-Phi XLA path, the
one the d=2 HMC sweep uses). Companion to scripts/profile_section1_svssm.py
(univariate); same metrics, d=2:

  - retracing trace-count delta over 10 distinct theta,
  - cold vs warm forward and forward+grad (the JIT speedup),
  - full-filter N-scaling (fwd, fwd+grad) at N in {64,128,256,512},
  - standalone entropic-Sinkhorn O(N^2 K) fwd/backward at dim=d,
  - one HMC step (bootstrap + warm one_step) over the 3d-vector.

Config is embedded in the output JSON for self-documentation.
"""

from __future__ import annotations

import json
import statistics as stats
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
)
from src.filters.dpf.sinkhorn import sinkhorn_potentials


def gen_data_multi(T, mu, phi, sigma_eta, seed=42):
    """Diagonal-Phi SVSSM: y_{t,i} = exp(h_{t,i}/2) eps, h AR(1) per component."""
    tf.random.set_seed(seed)
    d = len(mu)
    mu = tf.constant(mu, tf.float32)
    phi = tf.constant(phi, tf.float32)
    se = tf.constant(sigma_eta, tf.float32)
    h = tf.identity(mu)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + se * tf.random.normal([d])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([d]))
    return tf.stack(ys)  # (T, d)


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


def make_ll(d, N, n_lambda, K):
    return DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=d, num_particles=N, n_lambda=n_lambda, sinkhorn_epsilon=1.0,
        sinkhorn_iters=K, grad_window=4, jit_compile=True, init_type="stationary",
    )


def fwd_fn(ll, mu, phi, s2, y):
    def f():
        tf.random.set_seed(123)
        return ll(mu, phi, s2, y)
    return f


def fwd_grad_fn(ll, mu, phi, s2, y):
    def f():
        tf.random.set_seed(123)
        p = tf.identity(tf.stack([mu, phi, s2]))  # (3, d)
        with tf.GradientTape() as tape:
            tape.watch(p)
            v = ll(p[0], p[1], p[2], y)
        return v, tape.gradient(v, p)
    return f


def main():
    d, T, N, n_lambda, K = 2, 20, 64, 10, 10
    repeats = 5
    sizes = [64, 128, 256, 512]
    out_dir = Path(
        "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1_svssm_multi"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_v, phi_v, se_v = [1.0, -1.0], [0.95, 0.80], [0.312, 0.6]
    y = gen_data_multi(T, mu_v, phi_v, se_v)
    mu = tf.constant(mu_v, tf.float32)
    phi = tf.constant(phi_v, tf.float32)
    s2 = tf.constant([s * s for s in se_v], tf.float32)
    print(f"[profile-svssm-multi] TF {tf.__version__} TFP {tfp.__version__}  "
          f"module=DifferentiableLEDHLogLikelihoodSVSSMmulti (extra_bonus)")
    print(f"  d={d} T={T} N={N} n_lambda={n_lambda} K={K}  device=CPU")

    # ---- cold + warm timing at N=64 ----
    ll = make_ll(d, N, n_lambda, K)
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
        (tf.constant(rng.normal(0, 0.1, d), tf.float32),
         tf.constant(np.tanh(rng.normal(2, 0.2, d)), tf.float32),
         tf.constant(np.exp(rng.normal(-2, 0.3, d)), tf.float32))
        for _ in range(10)
    ]
    kernel = ll._jit_step  # built lazily on the warm call above
    c1 = int(kernel.experimental_get_tracing_count())
    for (m, p, s) in thetas:
        tf.random.set_seed(123)
        _ = ll(m, p, s, y).numpy()
    c2 = int(kernel.experimental_get_tracing_count())
    retrace_delta = c2 - c1
    print(f"  retracing: _jit_step trace delta over 10 evals = {retrace_delta} "
          f"({'PASS' if retrace_delta == 0 else 'FAIL'})")

    # ---- full-filter N-scaling (fresh filter per N) ----
    nscale = []
    for n in sizes:
        lln = make_ll(d, n, n_lambda, K)
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

    # ---- standalone entropic-Sinkhorn O(N^2 K) N-scaling at dim=d ----
    sink = []
    for n in sizes:
        a = tf.fill([n], 1.0 / n)
        b = tf.fill([n], 1.0 / n)
        xs = tf.random.normal([n, d], seed=0)
        ys = tf.random.normal([n, d], seed=1)

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
        sink.append({"N": n, "fwd_ms": sf * 1000, "fwd_grad_ms": sg * 1000})
        print(f"  sinkhorn(d={d}) N={n}: fwd {sf*1000:.2f}ms  "
              f"fwd+grad {sg*1000:.2f}ms")

    # ---- one HMC step over the 3d-vector ----
    def target(raw):
        tf.random.set_seed(2026)
        vv = ll(raw[0:d], tf.tanh(raw[d:2*d]), tf.exp(raw[2*d:3*d]), y)
        vv = tf.cast(tf.math.real(vv), tf.float32)
        return tf.where(tf.math.is_finite(vv), vv, tf.constant(-1e6, tf.float32))

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=tf.constant(0.02, tf.float32),
        num_leapfrog_steps=5,
    )
    init = tf.constant([0.0, 0.0, 1.5, 1.0, -2.0, -2.0], tf.float32)
    t0 = time.perf_counter()
    kr = hmc.bootstrap_results(init)
    boot = time.perf_counter() - t0
    t0 = time.perf_counter()
    state, kr = hmc.one_step(init, kr, seed=[1, 2])
    _ = state.numpy()
    first = time.perf_counter() - t0
    steps = []
    for i in range(3):
        t0 = time.perf_counter()
        state, kr = hmc.one_step(state, kr, seed=[i + 3, i + 4])
        _ = state.numpy()
        steps.append(time.perf_counter() - t0)
    warm_step = float(stats.median(steps))
    print(f"  HMC step: bootstrap {boot:.2f}s  first {first:.2f}s  "
          f"warm {warm_step:.2f}s")

    result = {
        "module": "DifferentiableLEDHLogLikelihoodSVSSMmulti (src/filters/bonus/extra_bonus)",
        "env": {"tf": tf.__version__, "tfp": tfp.__version__, "device": "CPU"},
        "config": {"d": d, "T": T, "N": N, "n_lambda": n_lambda,
                   "sinkhorn_iters": K, "repeats": repeats,
                   "truth": {"mu": mu_v, "phi": phi_v, "sigma_eta": se_v}},
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
    (out_dir / "profile_svssm_multi.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_dir / 'profile_svssm_multi.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
