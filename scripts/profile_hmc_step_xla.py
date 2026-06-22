"""
TF-profiler + cProfile capture of HMC steps for op-level efficiency analysis.

This complements the wall-clock and component-share profiling in
``scripts/profile_section1.py`` etc. by producing:

  (a) a TensorBoard-viewable XPlane trace of a few HMC steps (open with
      ``tensorboard --logdir reports/.../profile_section1/tf_profile`` or load
      the ``trace.json.gz`` in ``chrome://tracing``);

  (b) a cProfile breakdown of the Python-side cost (where dispatch, TFP
      kernel logic, eager-outer-loop overhead show up);

  (c) a printed top-N op summary parsed from the XPlane.

Defaults: 2 warmup HMC steps (not profiled), then 3 captured HMC steps on the
exp-integrator LEDH target.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.models.ssm_katigawa import PMCMCNonlinearSSM


def gen_data(T: int, sigma_v_sq: float = 10.0, sigma_w_sq: float = 1.0,
             seed: int = 42) -> tf.Tensor:
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


def build_kernel_and_target(T: int, N: int, n_lambda: int, K: int, L: int,
                            step_size: float, integrator: str):
    y_obs = gen_data(T)
    ll = DifferentiableLEDHLogLikelihood(
        num_particles=N,
        n_lambda=n_lambda,
        sinkhorn_epsilon=1.0,
        sinkhorn_iters=K,
        resample_threshold=0.5,
        grad_window=4,
        jit_compile=True,
        integrator=integrator,
    )

    def target_log_prob(theta):
        tf.random.set_seed(2026)  # fixed CRN
        ssm = PMCMCNonlinearSSM(
            sigma_v_sq=tf.exp(theta[0]),
            sigma_w_sq=tf.exp(theta[1]),
            initial_var=5.0,
        )
        v = ll(ssm, y_obs)
        v = tf.cast(tf.math.real(v), tf.float32)
        return tf.where(tf.math.is_finite(v), v, tf.constant(-1e6, tf.float32))

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(step_size, tf.float32),
        num_leapfrog_steps=int(L),
    )
    return kernel, target_log_prob


def parse_xplane_top_ops(logdir: Path, top_n: int = 20) -> list[dict]:
    """Best-effort: parse the latest XSpace under logdir and aggregate op times.

    Returns empty list if parsing fails (e.g. protobuf schema mismatch). The
    raw .xplane.pb is always usable in TensorBoard regardless of parse success.
    """
    try:
        from tensorflow.core.profiler.protobuf import xplane_pb2
    except Exception:
        return []
    pb_files = sorted(logdir.glob("**/*.xplane.pb"))
    if not pb_files:
        return []
    space = xplane_pb2.XSpace()
    try:
        space.ParseFromString(pb_files[-1].read_bytes())
    except Exception:
        return []

    # Aggregate per-event total duration across all planes/lines.
    totals_ns: dict[str, int] = {}
    counts: dict[str, int] = {}
    for plane in space.planes:
        # Op name dictionary (event metadata id -> name)
        meta_name = {mid: m.name for mid, m in plane.event_metadata.items()}
        for line in plane.lines:
            for ev in line.events:
                name = meta_name.get(ev.metadata_id, f"meta_{ev.metadata_id}")
                totals_ns[name] = totals_ns.get(name, 0) + int(ev.duration_ps // 1000)
                counts[name] = counts.get(name, 0) + 1
    if not totals_ns:
        return []
    total_us = sum(totals_ns.values()) / 1000.0
    rows = sorted(totals_ns.items(), key=lambda kv: -kv[1])[:top_n]
    out = []
    for name, dur_ns in rows:
        dur_ms = dur_ns / 1e6
        out.append({
            "op": name,
            "calls": counts[name],
            "total_ms": float(dur_ms),
            "pct": float(100.0 * dur_ms / max(total_us / 1000.0, 1e-12)),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--integrator", type=str, default="exp")
    p.add_argument("--step_size", type=float, default=0.005)
    p.add_argument("--warmup_steps", type=int, default=2)
    p.add_argument("--profile_steps", type=int, default=3)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    tb_dir = out_dir / "tf_profile"
    tb_dir.mkdir(parents=True, exist_ok=True)

    print(f"[profile-hmc-step-xla] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  T={args.T} N={args.N} n_lambda={args.n_lambda} K={args.K} L={args.L} "
          f"integrator={args.integrator}")
    print(f"  warmup={args.warmup_steps} steps  profiled={args.profile_steps} steps")
    print(f"  logdir={tb_dir}")

    kernel, target = build_kernel_and_target(
        T=args.T, N=args.N, n_lambda=args.n_lambda, K=args.K, L=args.L,
        step_size=args.step_size, integrator=args.integrator,
    )
    init_state = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)

    # === Warmup (not profiled): pay XLA cold-compile cost outside the trace ===
    t0 = time.perf_counter()
    kr = kernel.bootstrap_results(init_state)
    state = init_state
    for i in range(args.warmup_steps):
        state, kr = kernel.one_step(state, kr, seed=[i + 1, i + 7919])
        _ = state.numpy()
    t_warmup = time.perf_counter() - t0
    print(f"\n[warmup] {args.warmup_steps} steps + bootstrap took {t_warmup:.2f} s "
          f"(includes XLA compile)")

    # === Profile capture window ===
    options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=2,
        python_tracer_level=1,
        device_tracer_level=1,
    )
    profiled_step_times: list[float] = []
    profiled_accepts: list[bool] = []

    # cProfile wraps the same window for Python-side breakdown
    cp = cProfile.Profile()
    tf.profiler.experimental.start(str(tb_dir), options=options)
    cp.enable()
    try:
        for i in range(args.profile_steps):
            t0 = time.perf_counter()
            state, kr = kernel.one_step(
                state, kr, seed=[100 + i, 200 + i]
            )
            _ = state.numpy()
            profiled_step_times.append(time.perf_counter() - t0)
            try:
                profiled_accepts.append(bool(kr.is_accepted.numpy()))
            except Exception:
                profiled_accepts.append(False)
    finally:
        cp.disable()
        tf.profiler.experimental.stop()

    print(f"\n[profile] captured {args.profile_steps} HMC steps")
    for i, t in enumerate(profiled_step_times):
        print(f"  step {i+1}: {t * 1000:7.1f} ms  accepted={profiled_accepts[i]}")
    median_step = float(np.median(profiled_step_times) * 1000)
    print(f"  median step time: {median_step:.1f} ms")

    # === XPlane top-ops ===
    print("\n[tf.profiler] top-20 ops by aggregated duration (XPlane)")
    top_ops = parse_xplane_top_ops(tb_dir, top_n=20)
    if top_ops:
        print(f"  {'op':<48s} {'calls':>7s} {'total_ms':>10s} {'%':>6s}")
        for r in top_ops:
            print(f"  {r['op'][:48]:<48s} {r['calls']:>7d} "
                  f"{r['total_ms']:>10.2f} {r['pct']:>5.1f}%")
    else:
        print("  (XPlane parse skipped or unavailable; open in TensorBoard for op view)")

    # === cProfile top-30 Python-side ===
    print("\n[cProfile] top-30 by cumulative time")
    s = io.StringIO()
    ps = pstats.Stats(cp, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    cprofile_text = s.getvalue()
    print(cprofile_text)

    # === Persist ===
    summary = {
        "tf": tf.__version__,
        "tfp": tfp.__version__,
        "config": vars(args),
        "warmup_seconds": t_warmup,
        "profiled_step_times_ms": [float(x * 1000) for x in profiled_step_times],
        "median_step_ms": median_step,
        "tensorboard_logdir": str(tb_dir),
        "top_ops_xplane": top_ops,
    }
    (out_dir / "profile_hmc_step_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "profile_hmc_step_cprofile.txt").write_text(cprofile_text)
    print(f"\nWrote {out_dir / 'profile_hmc_step_summary.json'}")
    print(f"Wrote {out_dir / 'profile_hmc_step_cprofile.txt'}")
    print(f"View TF trace: tensorboard --logdir {tb_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
