"""
Section 1 profiling: baseline timings for the LEDH-PF-PF-OT + HMC pipeline.

Measures, on the locked-in 1-D Kitagawa SSM:
  - cold vs warm forward log-likelihood time (Python T-loop + JIT'd per-step body)
  - forward + gradient time (HMC's per-leapfrog-step cost shape)
  - one full TFP HMC step (target eval + leapfrog + accept/reject)
  - retracing event count over a short sequence of evaluations
  - standalone Sinkhorn forward and forward+backward at varying N

Output: prints a table and writes JSON to
  reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.differentiable_pfpf_ledh import KitagawaPFPFLEDHLogLikelihood
from src.filters.dpf.sinkhorn import sinkhorn_potentials


def generate_kitagawa(T: int, sigma_v_sq: float, sigma_w_sq: float, seed: int) -> tf.Tensor:
    """Kitagawa/Andrieu nonlinear SSM observations only (we don't need latent x)."""
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


def time_call(fn, repeats: int = 5) -> tuple[float, float]:
    """Median wall-time and stdev (seconds) of `fn()` over `repeats` runs."""
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


def profile_filter(
    *,
    T: int,
    N: int,
    n_lambda: int,
    sinkhorn_iters: int,
    sinkhorn_epsilon: float,
    repeats: int,
) -> dict:
    y_obs = generate_kitagawa(T, sigma_v_sq=10.0, sigma_w_sq=1.0, seed=42)
    ll = KitagawaPFPFLEDHLogLikelihood(
        num_particles=N,
        n_lambda=n_lambda,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=sinkhorn_iters,
        resample_threshold=0.5,
        clip_weight_terms=False,
    )

    theta = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)

    def forward():
        tf.random.set_seed(123)
        return ll(tf.exp(theta[0]), tf.exp(theta[1]), y_obs)

    def forward_grad():
        tf.random.set_seed(123)
        theta_t = tf.identity(theta)
        with tf.GradientTape() as tape:
            tape.watch(theta_t)
            val = ll(tf.exp(theta_t[0]), tf.exp(theta_t[1]), y_obs)
        g = tape.gradient(val, theta_t)
        return val, g

    t_cold_fwd = time.perf_counter()
    _ = forward().numpy()
    t_cold_fwd = time.perf_counter() - t_cold_fwd

    t_cold_grad = time.perf_counter()
    _, _ = forward_grad()
    t_cold_grad = time.perf_counter() - t_cold_grad

    warm_fwd, warm_fwd_std = time_call(forward, repeats=repeats)
    warm_grad, warm_grad_std = time_call(forward_grad, repeats=repeats)

    return {
        "T": T,
        "N": N,
        "n_lambda": n_lambda,
        "sinkhorn_iters": sinkhorn_iters,
        "sinkhorn_epsilon": sinkhorn_epsilon,
        "cold_forward_s": t_cold_fwd,
        "cold_forward_plus_grad_s": t_cold_grad,
        "warm_forward_s_median": warm_fwd,
        "warm_forward_s_pstdev": warm_fwd_std,
        "warm_forward_plus_grad_s_median": warm_grad,
        "warm_forward_plus_grad_s_pstdev": warm_grad_std,
    }


def profile_hmc_step(
    *,
    T: int,
    N: int,
    n_lambda: int,
    sinkhorn_iters: int,
    sinkhorn_epsilon: float,
    num_leapfrog: int,
) -> dict:
    """Time bootstrap + one TFP HMC step (= num_leapfrog target evals + grads + MH)."""
    y_obs = generate_kitagawa(T, sigma_v_sq=10.0, sigma_w_sq=1.0, seed=42)
    ll = KitagawaPFPFLEDHLogLikelihood(
        num_particles=N,
        n_lambda=n_lambda,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=sinkhorn_iters,
        resample_threshold=0.5,
        clip_weight_terms=False,
    )

    def target_log_prob(q):
        tf.random.set_seed(123)
        val = ll(tf.exp(q[0]), tf.exp(q[1]), y_obs)
        val = tf.cast(tf.math.real(val), tf.float32)
        return tf.where(tf.math.is_finite(val), val, tf.constant(-1e6, tf.float32))

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(0.005, tf.float32),
        num_leapfrog_steps=int(num_leapfrog),
    )
    state = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)

    t0 = time.perf_counter()
    kr = kernel.bootstrap_results(state)
    t_bootstrap = time.perf_counter() - t0

    t0 = time.perf_counter()
    state2, kr2 = kernel.one_step(state, kr, seed=[1, 2])
    _ = state2.numpy()
    t_first_step = time.perf_counter() - t0

    step_times = []
    cur_state, cur_kr = state2, kr2
    for i in range(3):
        t0 = time.perf_counter()
        cur_state, cur_kr = kernel.one_step(cur_state, cur_kr, seed=[3 + i, 7 + i])
        _ = cur_state.numpy()
        step_times.append(time.perf_counter() - t0)

    return {
        "T": T,
        "N": N,
        "num_leapfrog": num_leapfrog,
        "bootstrap_s": t_bootstrap,
        "first_step_s": t_first_step,
        "warm_step_s_median": float(stats.median(step_times)),
        "warm_step_s_pstdev": float(stats.pstdev(step_times)),
    }


def profile_sinkhorn(*, sizes: list[int], dim: int, n_iters: int, eps: float, repeats: int) -> list[dict]:
    rows = []
    for N in sizes:
        a = tf.fill([N], 1.0 / N)
        b = tf.fill([N], 1.0 / N)
        x = tf.random.normal([N, dim], seed=0)
        y = tf.random.normal([N, dim], seed=1)

        def fwd():
            f, g = sinkhorn_potentials(a, b, x, y, eps, n_iters)
            return f, g

        def fwd_grad():
            with tf.GradientTape() as tape:
                tape.watch(x)
                f, g = sinkhorn_potentials(a, b, x, y, eps, n_iters)
                loss = tf.reduce_sum(f) + tf.reduce_sum(g)
            return tape.gradient(loss, x)

        _ = fwd()
        _ = fwd_grad()

        fwd_med, fwd_std = time_call(fwd, repeats=repeats)
        bwd_med, bwd_std = time_call(fwd_grad, repeats=repeats)

        rows.append({
            "N": N,
            "dim": dim,
            "sinkhorn_iters": n_iters,
            "epsilon": eps,
            "fwd_s_median": fwd_med,
            "fwd_s_pstdev": fwd_std,
            "fwd_plus_grad_s_median": bwd_med,
            "fwd_plus_grad_s_pstdev": bwd_std,
            "backward_overhead_ratio": bwd_med / fwd_med if fwd_med > 0 else float("nan"),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=5)
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    p.add_argument("--num_leapfrog", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--sinkhorn_sizes", type=str, default="64,128,256,512")
    p.add_argument("--sinkhorn_dim", type=int, default=1)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1")
    args = p.parse_args()

    print("[Section 1 profile]")
    print(f"  TF {tf.__version__}, TFP {tfp.__version__}, devices: {[d.name for d in tf.config.list_logical_devices()]}")
    print(f"  XLA global JIT: {tf.config.optimizer.get_jit()}")
    print(f"  Config: T={args.T} N={args.N} n_lambda={args.n_lambda} "
          f"sinkhorn_iters={args.sinkhorn_iters} eps={args.sinkhorn_epsilon} "
          f"L={args.num_leapfrog} repeats={args.repeats}")

    filt = profile_filter(
        T=args.T, N=args.N, n_lambda=args.n_lambda,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        repeats=args.repeats,
    )
    print("\n[filter timings]")
    for k, v in filt.items():
        print(f"  {k:>36s} = {v if not isinstance(v, float) else f'{v:.4f}'}")

    hmc = profile_hmc_step(
        T=args.T, N=args.N, n_lambda=args.n_lambda,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        num_leapfrog=args.num_leapfrog,
    )
    print("\n[HMC step timings]")
    for k, v in hmc.items():
        print(f"  {k:>36s} = {v if not isinstance(v, float) else f'{v:.4f}'}")

    sizes = [int(s) for s in args.sinkhorn_sizes.split(",")]
    sk = profile_sinkhorn(
        sizes=sizes, dim=args.sinkhorn_dim,
        n_iters=args.sinkhorn_iters,
        eps=args.sinkhorn_epsilon,
        repeats=args.repeats,
    )
    print("\n[Sinkhorn standalone (forward / forward+grad)]")
    print(f"  {'N':>5s} {'fwd_med_s':>12s} {'bwd_med_s':>12s} {'ratio':>8s}")
    for row in sk:
        print(f"  {row['N']:>5d} {row['fwd_s_median']:>12.5f} "
              f"{row['fwd_plus_grad_s_median']:>12.5f} {row['backward_overhead_ratio']:>8.2f}x")

    out = {
        "env": {
            "tf": tf.__version__,
            "tfp": tfp.__version__,
            "devices": [d.name for d in tf.config.list_logical_devices()],
            "xla_global_jit": tf.config.optimizer.get_jit(),
        },
        "config": vars(args),
        "filter": filt,
        "hmc_step": hmc,
        "sinkhorn": sk,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
