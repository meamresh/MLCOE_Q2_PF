"""
Full Section 1 verification for DifferentiableLEDHLogLikelihoodSVSSM.

Mirrors verify_no_retracing_ledh.py + verify_no_retracing_in_hmc.py for the
SVSSM-specific filter. Covers what the original sanity test did NOT:

  (a) standalone retracing across 10 distinct-theta forward evals,
  (b) standalone retracing across 10 distinct-theta forward+grad evals,
  (c) retracing inside an actual TFP HMC chain (12 steps),
  (d) all of the above for EACH init_type in {stationary, fixed_mu, diffuse}
      to verify the new init_type / init_mean plumbing did not break
      trace caching.

Pass criterion: per-step XLA kernels' tracing count delta = 0 after warmup,
across all evaluations and all init types.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import DifferentiableLEDHLogLikelihoodSVSSM
from src.filters.dpf import sinkhorn as sk


def gen_data(T, mu=0.0, phi=0.95, sigma_eta=0.3, seed=42):
    tf.random.set_seed(seed)
    sigma_eta_t = tf.constant(sigma_eta, tf.float32)
    h = tf.constant(float(mu), tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta_t * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def make_ll(init_type: str, N=64, n_lambda=10, K=10) -> DifferentiableLEDHLogLikelihoodSVSSM:
    return DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N, n_lambda=n_lambda,
        sinkhorn_epsilon=1.0, sinkhorn_iters=K,
        grad_window=4, jit_compile=True, integrator="exp",
        init_type=init_type,
    )


def trace_counts(ll):
    return {
        "_timestep_1d_xla":      int(ll._timestep_1d_xla.experimental_get_tracing_count()),
        "sinkhorn_potentials":   int(sk.sinkhorn_potentials.experimental_get_tracing_count()),
    }


def diff(a, b):
    return {k: b[k] - a[k] for k in a}


def check_for_init(init_type: str, y_obs, n_eval=10, run_hmc=True):
    print(f"\n=== init_type = {init_type} ===")
    ll = make_ll(init_type)

    rng = np.random.default_rng(0)
    thetas = [
        tf.constant(
            [rng.normal(loc=0.0, scale=0.1),
             np.tanh(rng.normal(loc=2.0, scale=0.2)),
             np.exp(rng.normal(loc=-2.0, scale=0.3))],
            dtype=tf.float32,
        )
        for _ in range(n_eval)
    ]
    theta_warm = tf.constant([0.0, 0.95, 0.09], dtype=tf.float32)

    # ---- (a) Forward-only retracing ----
    c0 = trace_counts(ll)
    tf.random.set_seed(123)
    _ = ll(theta_warm[0], theta_warm[1], theta_warm[2], y_obs).numpy()
    c1 = trace_counts(ll)
    for th in thetas:
        tf.random.set_seed(123)
        _ = ll(th[0], th[1], th[2], y_obs).numpy()
    c2 = trace_counts(ll)
    fwd_pass = all(v == 0 for v in diff(c1, c2).values())
    print(f"  [forward] warmup={c1}  after {n_eval}-eval delta={diff(c1, c2)}  "
          f"{'PASS' if fwd_pass else 'FAIL'}")

    # ---- (b) Forward+grad retracing ----
    def grad_call(theta):
        tf.random.set_seed(123)
        th = tf.identity(theta)
        with tf.GradientTape() as tape:
            tape.watch(th)
            v = ll(th[0], th[1], th[2], y_obs)
        g = tape.gradient(v, th)
        return v, g

    c3 = trace_counts(ll)
    v, g = grad_call(theta_warm)
    _ = v.numpy(); _ = g.numpy()
    c4 = trace_counts(ll)
    finite_grads = 0
    for th in thetas:
        v, g = grad_call(th)
        _ = v.numpy()
        g_np = g.numpy()
        if np.all(np.isfinite(g_np)):
            finite_grads += 1
    c5 = trace_counts(ll)
    grad_pass = all(v == 0 for v in diff(c4, c5).values())
    print(f"  [fwd+grad] warmup={c4}  after {n_eval}-eval delta={diff(c4, c5)}  "
          f"finite grad: {finite_grads}/{n_eval}  "
          f"{'PASS' if grad_pass else 'FAIL'}")

    # ---- (c) HMC-realistic retracing (12 chain steps) ----
    hmc_pass = True
    if run_hmc:
        def target(theta_raw):
            tf.random.set_seed(2026)
            mu = theta_raw[0]
            phi = tf.tanh(theta_raw[1])
            sigma_eta_sq = tf.exp(theta_raw[2])
            v = ll(mu, phi, sigma_eta_sq, y_obs)
            v = tf.cast(tf.math.real(v), tf.float32)
            return tf.where(tf.math.is_finite(v), v, tf.constant(-1e6, tf.float32))

        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target,
            step_size=tf.constant(0.05, tf.float32),
            num_leapfrog_steps=5,
        )
        init = tf.constant([0.0, 2.0, -2.0], dtype=tf.float32)

        c6 = trace_counts(ll)
        kr = kernel.bootstrap_results(init)
        state = init
        for i in range(12):
            state, kr = kernel.one_step(state, kr, seed=[i + 1, i + 9])
            _ = state.numpy()
        c7 = trace_counts(ll)
        hmc_delta = diff(c6, c7)
        hmc_pass = all(v == 0 for v in hmc_delta.values())
        print(f"  [12 HMC steps] delta={hmc_delta}  "
              f"{'PASS' if hmc_pass else 'FAIL'}")

    return {
        "init_type": init_type,
        "fwd_pass": fwd_pass, "grad_pass": grad_pass, "hmc_pass": hmc_pass,
        "fwd_delta": diff(c1, c2), "grad_delta": diff(c4, c5),
        "finite_grads": finite_grads, "n_eval": n_eval,
    }


def main():
    T = 20
    y_obs = gen_data(T=T)
    print(f"[verify-no-retracing-svssm] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  T={T}  y range: [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")

    results = []
    t0 = time.perf_counter()
    for init_type in ("stationary", "fixed_mu", "diffuse"):
        r = check_for_init(init_type, y_obs, n_eval=10, run_hmc=True)
        results.append(r)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*70}")
    print("OVERALL SVSSM RETRACING + JIT VERIFICATION")
    print(f"{'='*70}")
    all_pass = True
    for r in results:
        cells = [
            f"{r['init_type']:<12s}",
            f"fwd={'PASS' if r['fwd_pass'] else 'FAIL'}",
            f"grad={'PASS' if r['grad_pass'] else 'FAIL'}",
            f"hmc={'PASS' if r['hmc_pass'] else 'FAIL'}",
            f"finite_grad={r['finite_grads']}/{r['n_eval']}",
        ]
        verdict = all([r['fwd_pass'], r['grad_pass'], r['hmc_pass']])
        all_pass = all_pass and verdict
        print(f"  {'  '.join(cells)}   {'OVERALL: PASS' if verdict else 'OVERALL: FAIL'}")
    print(f"\nTotal wall: {elapsed:.1f}s   FINAL: {'PASS' if all_pass else 'FAIL'}")

    out_dir = Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retracing_verification_svssm.json").write_text(json.dumps({
        "tf": tf.__version__, "tfp": tfp.__version__,
        "T": T, "elapsed_s": elapsed,
        "results": results, "all_pass": bool(all_pass),
    }, indent=2))
    print(f"Wrote {out_dir / 'retracing_verification_svssm.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
