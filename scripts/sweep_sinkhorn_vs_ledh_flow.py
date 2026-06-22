"""
Sweep particle count N to locate the cost crossover between the LEDH-flow
backward and the Sinkhorn backward inside DifferentiableLEDHLogLikelihood.

For each (N, n_lambda) we measure
  - standalone sinkhorn_potentials forward and forward+grad at (N, K, 1-D)
  - full LEDH forward and forward+grad at (T, N, n_lambda, K)
and decompose
  Sinkhorn-attributed  = T * standalone_time            (unconditional resample)
  LEDH-flow + rest     = full_time - Sinkhorn-attributed

LEDH does an unconditional OT resample at every timestep, so the per-call
Sinkhorn cost is T * standalone_time. The remainder is dominated by the LEDH
pseudo-time flow (n_lambda sequential clipped substeps per t) plus a small
non-Sinkhorn overhead inside _ot_resample_1d (normalize / transport apply /
de-normalize) and weight bookkeeping.

Outputs:
  reports/.../profile_section1/sweep_sinkhorn_vs_ledh_flow.json
  reports/.../profile_section1/sweep_sinkhorn_vs_ledh_flow.csv
  console table per n_lambda + crossover N
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.filters.dpf.sinkhorn import sinkhorn_potentials
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


def median_time(fn, reps: int) -> float:
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


def measure_sinkhorn_at(N: int, K: int, dim: int, eps: float,
                        reps: int) -> tuple[float, float]:
    """Return (forward_s, forward+grad_s) medians for standalone Sinkhorn."""
    a = tf.fill([N], 1.0 / N)
    b = tf.fill([N], 1.0 / N)
    x = tf.random.normal([N, dim], seed=0)
    y = tf.random.normal([N, dim], seed=1)

    def fwd():
        f, g = sinkhorn_potentials(a, b, x, y, eps, K)
        return f, g

    def fwd_grad():
        with tf.GradientTape() as tape:
            tape.watch(x)
            f, g = sinkhorn_potentials(a, b, x, y, eps, K)
            loss = tf.reduce_sum(f) + tf.reduce_sum(g)
        return tape.gradient(loss, x)

    # warmup compile at this shape
    _ = fwd()
    _ = fwd_grad()
    return median_time(fwd, reps), median_time(fwd_grad, reps)


def measure_full_ledh(T: int, N: int, n_lambda: int, K: int, eps: float,
                      grad_window: int, reps: int) -> tuple[float, float]:
    """Return (forward_s, forward+grad_s) medians for full LEDH log-likelihood."""
    y_obs = gen_data(T)
    ll = DifferentiableLEDHLogLikelihood(
        num_particles=N,
        n_lambda=n_lambda,
        sinkhorn_epsilon=eps,
        sinkhorn_iters=K,
        resample_threshold=0.5,
        grad_window=grad_window,
        jit_compile=True,
    )
    theta = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)

    def fwd():
        tf.random.set_seed(123)
        ssm = PMCMCNonlinearSSM(
            sigma_v_sq=tf.exp(theta[0]),
            sigma_w_sq=tf.exp(theta[1]),
            initial_var=5.0,
        )
        return ll(ssm, y_obs)

    def fwd_grad():
        tf.random.set_seed(123)
        th = tf.identity(theta)
        with tf.GradientTape() as tape:
            tape.watch(th)
            ssm = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(th[0]),
                sigma_w_sq=tf.exp(th[1]),
                initial_var=5.0,
            )
            v = ll(ssm, y_obs)
        return tape.gradient(v, th)

    # warmup compile at this shape (cold can be long)
    _ = fwd().numpy()
    _ = fwd_grad().numpy()
    return median_time(fwd, reps), median_time(fwd_grad, reps)


def run_sweep(N_values: list[int], n_lambda: int, T: int, K: int, eps: float,
              grad_window: int, sk_reps: int, ll_reps_small: int,
              ll_reps_large: int, large_N: int) -> list[dict]:
    """Run the (N) sweep at one fixed n_lambda and return per-row stats."""
    rows = []
    print(f"\n=== Sweep at n_lambda={n_lambda} ===")
    print(f"{'N':>5s} {'sk_f':>9s} {'sk_b':>9s} {'L_f':>9s} {'L_b':>9s} "
          f"{'sk_fT':>9s} {'sk_bT':>9s} {'L_flow_f':>10s} {'L_flow_b':>10s} "
          f"{'%sk_b':>7s} {'%L_b':>7s}")
    print("-" * 110)
    for N in N_values:
        sk_f, sk_b = measure_sinkhorn_at(N=N, K=K, dim=1, eps=eps, reps=sk_reps)
        ll_reps = ll_reps_large if N >= large_N else ll_reps_small
        full_f, full_b = measure_full_ledh(
            T=T, N=N, n_lambda=n_lambda, K=K, eps=eps,
            grad_window=grad_window, reps=ll_reps,
        )
        # Per-call Sinkhorn attribution: T calls (unconditional resampling).
        sk_f_T = T * sk_f
        sk_b_T = T * sk_b
        flow_f = max(full_f - sk_f_T, 0.0)
        flow_b = max(full_b - sk_b_T, 0.0)
        pct_sk_b = 100.0 * sk_b_T / full_b if full_b > 0 else float("nan")
        pct_flow_b = 100.0 * flow_b / full_b if full_b > 0 else float("nan")

        rows.append({
            "N": int(N),
            "n_lambda": int(n_lambda),
            "T": int(T),
            "K": int(K),
            "sinkhorn_fwd_s": sk_f,
            "sinkhorn_fwd_grad_s": sk_b,
            "ll_fwd_s": full_f,
            "ll_fwd_grad_s": full_b,
            "sinkhorn_fwd_total_s": sk_f_T,
            "sinkhorn_fwd_grad_total_s": sk_b_T,
            "ledh_flow_fwd_s": flow_f,
            "ledh_flow_fwd_grad_s": flow_b,
            "pct_sinkhorn_of_fwd_grad": pct_sk_b,
            "pct_ledh_flow_of_fwd_grad": pct_flow_b,
            "ll_reps": int(ll_reps),
        })
        print(f"{N:>5d} {sk_f*1000:>9.3f} {sk_b*1000:>9.3f} "
              f"{full_f*1000:>9.2f} {full_b*1000:>9.2f} "
              f"{sk_f_T*1000:>9.2f} {sk_b_T*1000:>9.2f} "
              f"{flow_f*1000:>10.2f} {flow_b*1000:>10.2f} "
              f"{pct_sk_b:>6.1f}% {pct_flow_b:>6.1f}%")
    return rows


def find_crossover(rows: list[dict]) -> dict:
    """Find the smallest N at which Sinkhorn fwd+grad share >= LEDH flow share."""
    crossover_N = None
    interp_N = None
    for i, r in enumerate(rows):
        if r["pct_sinkhorn_of_fwd_grad"] >= r["pct_ledh_flow_of_fwd_grad"]:
            crossover_N = int(r["N"])
            # linear interpolation in log2(N) between adjacent rows
            if i > 0:
                prev = rows[i - 1]
                d0 = prev["pct_sinkhorn_of_fwd_grad"] - prev["pct_ledh_flow_of_fwd_grad"]
                d1 = r["pct_sinkhorn_of_fwd_grad"] - r["pct_ledh_flow_of_fwd_grad"]
                if d1 != d0:
                    frac = (0.0 - d0) / (d1 - d0)
                    interp_N = float(2 ** (np.log2(prev["N"]) + frac * (np.log2(r["N"]) - np.log2(prev["N"]))))
            break
    return {"crossover_N_first_geq": crossover_N, "interpolated_crossover_N": interp_N}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--grad_window", type=int, default=4)
    p.add_argument("--N_values", type=str, default="32,64,128,256,512")
    p.add_argument("--n_lambda_values", type=str, default="20,10")
    p.add_argument("--sk_reps", type=int, default=8)
    p.add_argument("--ll_reps_small", type=int, default=5,
                   help="LL reps for N < large_N")
    p.add_argument("--ll_reps_large", type=int, default=3,
                   help="LL reps for N >= large_N")
    p.add_argument("--large_N", type=int, default=256)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1")
    args = p.parse_args()

    N_values = [int(x) for x in args.N_values.split(",")]
    n_lambda_values = [int(x) for x in args.n_lambda_values.split(",")]

    print(f"[sweep] TF {tf.__version__}")
    print(f"  T={args.T} K={args.K} eps={args.eps} grad_window={args.grad_window}")
    print(f"  N_values={N_values}  n_lambda_values={n_lambda_values}")
    print(f"  sk_reps={args.sk_reps} ll_reps={args.ll_reps_small}/{args.ll_reps_large} "
          f"(small/large; large_N>={args.large_N})")
    print("  Columns: sk_f / sk_b = standalone Sinkhorn fwd / fwd+grad (one call, ms)")
    print("           L_f / L_b   = full LEDH fwd / fwd+grad (ms)")
    print(f"           sk_fT/sk_bT = T*{args.T} standalone-Sinkhorn time")
    print("           L_flow_f / L_flow_b = full LL minus Sinkhorn attribution")
    print("           %sk_b / %L_b = share of forward+grad time")

    t_start = time.perf_counter()
    all_rows: list[dict] = []
    per_lambda: dict[str, dict] = {}
    for nl in n_lambda_values:
        rows = run_sweep(
            N_values=N_values, n_lambda=nl, T=args.T, K=args.K, eps=args.eps,
            grad_window=args.grad_window, sk_reps=args.sk_reps,
            ll_reps_small=args.ll_reps_small, ll_reps_large=args.ll_reps_large,
            large_N=args.large_N,
        )
        all_rows.extend(rows)
        xover = find_crossover(rows)
        per_lambda[f"n_lambda={nl}"] = {"rows": rows, "crossover": xover}
        print(f"\n[crossover @ n_lambda={nl}] first N where Sinkhorn% >= LEDH-flow%: "
              f"{xover['crossover_N_first_geq']}; "
              f"interpolated: {xover['interpolated_crossover_N']}")
    print(f"\nTotal sweep wall time: {time.perf_counter() - t_start:.1f} s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = {
        "tf": tf.__version__,
        "config": vars(args),
        "results": per_lambda,
    }
    (out_dir / "sweep_sinkhorn_vs_ledh_flow.json").write_text(json.dumps(out_json, indent=2))
    csv_path = out_dir / "sweep_sinkhorn_vs_ledh_flow.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {out_dir / 'sweep_sinkhorn_vs_ledh_flow.json'}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
