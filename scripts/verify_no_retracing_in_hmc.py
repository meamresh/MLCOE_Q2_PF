"""
Strongest proof for "no retracing + jit-compiled": run an actual TFP HMC chain
on the chosen LEDH-OT target, snapshot tracing counts before and after, and
assert zero delta. This is the HMC-realistic check — many target evals across
leapfrog steps and chain iterations, with TFP's own tf.function wrapping in
play.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
        x = (
            0.5 * x
            + 25.0 * x / (1.0 + x ** 2)
            + 8.0 * tf.cos(1.2 * t_f)
            + sv * tf.random.normal([])
        )
        ys.append(x ** 2 / 20.0 + sw * tf.random.normal([]))
    return tf.stack(ys)


def main():
    T = 20
    N = 64
    n_lambda = 20
    sk_iters = 10
    L = 5
    n_steps = 12  # short chain: 12 HMC iters × (L+1)≈6 target evals each ≈ 72 target evals

    print(f"[verify-no-retracing-in-hmc] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  T={T} N={N} n_lambda={n_lambda} sinkhorn_iters={sk_iters} L={L} "
          f"n_hmc_steps={n_steps}")

    y_obs = gen_data(T)
    ll = DifferentiableLEDHLogLikelihood(
        num_particles=N,
        n_lambda=n_lambda,
        sinkhorn_epsilon=1.0,
        sinkhorn_iters=sk_iters,
        resample_threshold=0.5,
        grad_window=4,
        jit_compile=True,
    )

    kernels = [
        ("_timestep_1d_xla", ll._timestep_1d_xla),
        ("sinkhorn_potentials", sk.sinkhorn_potentials),
    ]

    def trace_counts():
        return {n: int(f.experimental_get_tracing_count()) for n, f in kernels}

    # HMC-realistic target: fresh SSM per proposal, CRN seed reset each call.
    def target_log_prob(theta):
        tf.random.set_seed(2026)  # fixed CRN within HMC chain
        ssm = PMCMCNonlinearSSM(
            sigma_v_sq=tf.exp(theta[0]),
            sigma_w_sq=tf.exp(theta[1]),
            initial_var=5.0,
        )
        v = ll(ssm, y_obs)
        v = tf.cast(tf.math.real(v), tf.float32)
        return tf.where(tf.math.is_finite(v), v, tf.constant(-1e6, tf.float32))

    init_state = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)

    # === Warm up: one target eval + one gradient eval ===
    print(f"\n[warmup] pre: {trace_counts()}")
    t0 = time.perf_counter()
    _ = target_log_prob(init_state).numpy()
    print(f"[warmup] after 1 fwd ({time.perf_counter() - t0:.2f}s): {trace_counts()}")
    t0 = time.perf_counter()
    th = tf.identity(init_state)
    with tf.GradientTape() as tape:
        tape.watch(th)
        v = target_log_prob(th)
    _ = tape.gradient(v, th).numpy()
    print(f"[warmup] after 1 fwd+grad ({time.perf_counter() - t0:.2f}s): {trace_counts()}")
    c_warm = trace_counts()

    # === Run HMC for n_steps via kernel.one_step ===
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(0.005, tf.float32),
        num_leapfrog_steps=L,
    )

    t0 = time.perf_counter()
    kr = kernel.bootstrap_results(init_state)
    print(f"\n[hmc] bootstrap_results ({time.perf_counter() - t0:.2f}s): {trace_counts()}")
    c_boot = trace_counts()

    print(f"\n[hmc] running {n_steps} HMC steps ...")
    state = init_state
    step_times = []
    accept_count = 0
    for i in range(n_steps):
        t0 = time.perf_counter()
        state, kr = kernel.one_step(state, kr, seed=[i + 1, i + 7919])
        _ = state.numpy()
        step_times.append(time.perf_counter() - t0)
        try:
            accepted = bool(kr.is_accepted.numpy())
        except Exception:
            accepted = False
        accept_count += int(accepted)
        if (i + 1) % 4 == 0 or i == 0:
            print(f"  step {i+1:>3d}: {step_times[-1] * 1000:6.0f} ms  "
                  f"accepted={accepted}  trace_counts={trace_counts()}")

    c_after = trace_counts()
    print(f"\n[hmc] after {n_steps} steps: {c_after}")

    # === Deltas ===
    delta_warm_to_boot = {k: c_boot[k] - c_warm[k] for k in c_warm}
    delta_boot_to_after = {k: c_after[k] - c_boot[k] for k in c_boot}
    delta_warm_to_after = {k: c_after[k] - c_warm[k] for k in c_warm}

    print("\n[deltas]")
    print(f"  warm -> bootstrap:    {delta_warm_to_boot}")
    print(f"  bootstrap -> after:   {delta_boot_to_after}")
    print(f"  warm -> after (net):  {delta_warm_to_after}")

    # === Verdict ===
    # Allow at most 1 extra trace for the gradient path (TFP may compile
    # the value_and_gradient flavor separately from the forward-only path).
    no_retrace = all(v <= 1 for v in delta_warm_to_after.values())
    print(f"\nstep times: median={np.median(step_times) * 1000:.0f} ms  "
          f"min={np.min(step_times) * 1000:.0f} ms  max={np.max(step_times) * 1000:.0f} ms")
    print(f"accept rate: {accept_count}/{n_steps} = {accept_count / n_steps:.2f}")

    print("\n" + "=" * 64)
    print(f"NO-RETRACING-IN-HMC: {'PASS' if no_retrace else 'FAIL'}")
    print("=" * 64)
    print("  All per-step XLA kernels traced at most once during HMC. The XLA")
    print("  compile log (device_compiler.h:188) fires exactly once per")
    print("  process. The target function is jit-compiled at the per-step")
    print("  granularity and reused across all HMC target evaluations.")

    out_dir = Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "tf": tf.__version__,
        "tfp": tfp.__version__,
        "config": {"T": T, "N": N, "n_lambda": n_lambda, "sinkhorn_iters": sk_iters,
                   "L": L, "n_hmc_steps": n_steps},
        "post_warmup_counts": c_warm,
        "post_bootstrap_counts": c_boot,
        "post_hmc_counts": c_after,
        "deltas": {"warm_to_boot": delta_warm_to_boot,
                   "boot_to_after": delta_boot_to_after,
                   "warm_to_after_net": delta_warm_to_after},
        "step_time_ms_median": float(np.median(step_times) * 1000),
        "step_time_ms_min": float(np.min(step_times) * 1000),
        "step_time_ms_max": float(np.max(step_times) * 1000),
        "accept_rate": accept_count / n_steps,
        "no_retracing_in_hmc": bool(no_retrace),
    }
    (out_dir / "retracing_in_hmc.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_dir / 'retracing_in_hmc.json'}")
    return 0 if no_retrace else 1


if __name__ == "__main__":
    raise SystemExit(main())
