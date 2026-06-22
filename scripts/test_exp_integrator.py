"""
Test the exponential integrator for the LEDH flow against the existing
forward-Euler path. Reports:

  - CRN-paired log-likelihood and gradient comparison across configs:
      * baseline       Euler, n_lambda=20
      * reference      Euler, n_lambda=40 (slow but more accurate)
      * proposed-{5,10,20}  Exp,   n_lambda in {5, 10, 20}
  - Per-config wall time (warm forward and warm forward+grad)
  - Pass criterion: for "exp, n_lambda>=5" the median |Delta log p| across
    seeds is within Monte-Carlo PF noise of the Euler baseline (we use the
    Euler@20 vs Euler@40 spread as the MC-noise envelope).

Outputs:
  reports/.../profile_section1/exp_integrator_test.json
  reports/.../profile_section1/exp_integrator_test.csv
  reports/.../profile_section1/exp_integrator_report.txt
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


def median(times):
    return float(stats.median(times)) if times else float("nan")


def time_callable(fn, reps: int) -> float:
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
    return median(ts)


def make_ll(integrator: str, n_lambda: int, N: int, K: int, eps: float,
            grad_window: int) -> DifferentiableLEDHLogLikelihood:
    return DifferentiableLEDHLogLikelihood(
        num_particles=N,
        n_lambda=n_lambda,
        sinkhorn_epsilon=eps,
        sinkhorn_iters=K,
        resample_threshold=0.5,
        grad_window=grad_window,
        jit_compile=True,
        integrator=integrator,
    )


def eval_one(ll, theta: tf.Tensor, y_obs: tf.Tensor, seed: int,
             with_grad: bool) -> tuple[float, np.ndarray | None]:
    tf.random.set_seed(seed)
    if with_grad:
        theta_t = tf.identity(theta)
        with tf.GradientTape() as tape:
            tape.watch(theta_t)
            ssm = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(theta_t[0]),
                sigma_w_sq=tf.exp(theta_t[1]),
                initial_var=5.0,
            )
            v = ll(ssm, y_obs)
        g = tape.gradient(v, theta_t)
        return float(v.numpy()), g.numpy().astype(np.float64)
    else:
        ssm = PMCMCNonlinearSSM(
            sigma_v_sq=tf.exp(theta[0]),
            sigma_w_sq=tf.exp(theta[1]),
            initial_var=5.0,
        )
        return float(ll(ssm, y_obs).numpy()), None


def run_seeds(ll, theta, y_obs, seeds: list[int]) -> dict:
    """Per-seed log p and gradient over fixed CRN seeds."""
    lps, grads = [], []
    for s in seeds:
        lp, g = eval_one(ll, theta, y_obs, s, with_grad=True)
        lps.append(lp)
        grads.append(g)
    lps = np.asarray(lps, dtype=np.float64)
    grads = np.asarray(grads, dtype=np.float64)
    finite_lp = int(np.sum(np.isfinite(lps)))
    finite_g = int(np.sum(np.all(np.isfinite(grads), axis=1)))
    return {
        "loglik_mean": float(np.nanmean(lps)),
        "loglik_std": float(np.nanstd(lps, ddof=1)) if lps.size > 1 else float("nan"),
        "loglik_min": float(np.nanmin(lps)),
        "loglik_max": float(np.nanmax(lps)),
        "grad_norm_mean": float(np.nanmean(np.linalg.norm(grads, axis=1))),
        "grad_norm_std": float(np.nanstd(np.linalg.norm(grads, axis=1), ddof=1))
            if grads.shape[0] > 1 else float("nan"),
        "finite_loglik": finite_lp,
        "finite_grad": finite_g,
        "n_seeds": len(seeds),
        "_lps": lps,
        "_grads": grads,
    }


def benchmark_warm(ll, theta, y_obs, reps: int) -> dict:
    def fwd():
        tf.random.set_seed(123)
        ssm = PMCMCNonlinearSSM(
            sigma_v_sq=tf.exp(theta[0]),
            sigma_w_sq=tf.exp(theta[1]),
            initial_var=5.0,
        )
        return ll(ssm, y_obs)

    def fg():
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

    _ = fwd(); _ = fg()  # warm compile
    return {
        "warm_forward_s": time_callable(fwd, reps),
        "warm_forward_grad_s": time_callable(fg, reps),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--grad_window", type=int, default=4)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/profile_section1")
    args = p.parse_args()

    seeds = list(range(101, 101 + args.n_seeds))
    theta = tf.constant([np.log(10.0), np.log(1.0)], dtype=tf.float32)
    y_obs = gen_data(args.T)

    configs = [
        ("euler_n20",   "euler", 20),
        ("euler_n40",   "euler", 40),  # reference
        ("exp_n5",      "exp",   5),
        ("exp_n10",     "exp",   10),
        ("exp_n20",     "exp",   20),
    ]

    print("[exp-integrator test] TF", tf.__version__)
    print(f"  T={args.T} N={args.N} K={args.K} n_seeds={args.n_seeds} reps={args.reps}")
    print(f"  seeds={seeds[:3]}...{seeds[-1]}")

    results: dict[str, dict] = {}
    for label, integrator, nl in configs:
        print(f"\n[{label}] integrator={integrator} n_lambda={nl}")
        t0 = time.perf_counter()
        ll = make_ll(integrator, nl, args.N, args.K, args.eps, args.grad_window)
        seeds_res = run_seeds(ll, theta, y_obs, seeds)
        bench = benchmark_warm(ll, theta, y_obs, args.reps)
        elapsed = time.perf_counter() - t0
        print(f"  log p: mean={seeds_res['loglik_mean']:.4f}  "
              f"std={seeds_res['loglik_std']:.4f}  "
              f"[{seeds_res['loglik_min']:.2f}, {seeds_res['loglik_max']:.2f}]   "
              f"finite={seeds_res['finite_loglik']}/{args.n_seeds}")
        print(f"  ||grad||: mean={seeds_res['grad_norm_mean']:.3e}  "
              f"std={seeds_res['grad_norm_std']:.3e}  "
              f"finite={seeds_res['finite_grad']}/{args.n_seeds}")
        print(f"  warm fwd     = {bench['warm_forward_s']*1000:7.2f} ms")
        print(f"  warm fwd+grad= {bench['warm_forward_grad_s']*1000:7.2f} ms")
        print(f"  total config wall time: {elapsed:.1f}s")
        results[label] = {**seeds_res, **bench, "integrator": integrator, "n_lambda": nl}

    # === Equivalence analysis ===
    # CRN-paired deltas vs the Euler@n_lambda=20 baseline.
    base = results["euler_n20"]
    base_lps = base["_lps"]
    base_grads = base["_grads"]
    # MC-noise envelope: Euler@20 vs Euler@40 (same RNG seeds, deeper Euler refinement)
    ref_delta = np.abs(results["euler_n40"]["_lps"] - base_lps)
    mc_envelope = float(np.median(ref_delta))
    print("\n=== Equivalence vs Euler@n_lambda=20 (per CRN seed, MC envelope from Euler@20 vs Euler@40) ===")
    print(f"  MC-noise envelope (median |Euler@40 - Euler@20|) = {mc_envelope:.4f} log-units")
    print(f"  {'config':<14} {'med|Δlogp|':>14} {'max|Δlogp|':>14} "
          f"{'med|Δgrad|/||g||':>18} {'speedup (f+g)':>14}")
    print("  " + "-" * 80)

    equivalence: dict[str, dict] = {}
    for label, _, _ in configs:
        if label == "euler_n20":
            continue
        rl = results[label]
        d_lp = np.abs(rl["_lps"] - base_lps)
        d_g = np.linalg.norm(rl["_grads"] - base_grads, axis=1)
        g_norms = np.maximum(np.linalg.norm(base_grads, axis=1), 1e-12)
        d_g_rel = d_g / g_norms
        med_dlp = float(np.median(d_lp))
        max_dlp = float(np.max(d_lp))
        med_dgrel = float(np.median(d_g_rel))
        speedup_fg = base["warm_forward_grad_s"] / rl["warm_forward_grad_s"]
        equivalence[label] = {
            "med_abs_delta_loglik": med_dlp,
            "max_abs_delta_loglik": max_dlp,
            "med_rel_delta_grad": med_dgrel,
            "speedup_warm_fwd_grad": float(speedup_fg),
            "within_mc_envelope": bool(med_dlp <= 2 * mc_envelope),  # 2x MC envelope = lenient
        }
        marker = "[OK]" if equivalence[label]["within_mc_envelope"] else "[CHECK]"
        print(f"  {label:<14} {med_dlp:>14.4f} {max_dlp:>14.4f} "
              f"{med_dgrel:>18.3e} {speedup_fg:>13.2f}x {marker}")

    # === Persist ===
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, d in results.items():
        d2 = {k: v for k, v in d.items() if not k.startswith("_")}
        d2["label"] = label
        rows.append(d2)
    with (out_dir / "exp_integrator_test.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        wr.writeheader(); wr.writerows(rows)

    out = {
        "tf": tf.__version__,
        "config": vars(args),
        "mc_noise_envelope": mc_envelope,
        "results": {
            label: {k: v for k, v in r.items() if not k.startswith("_")}
            for label, r in results.items()
        },
        "equivalence_vs_euler20": equivalence,
    }
    (out_dir / "exp_integrator_test.json").write_text(json.dumps(out, indent=2))

    report = [
        "=" * 80,
        "Exp-integrator vs Euler-integrator equivalence + benchmark",
        "=" * 80,
        f"T={args.T} N={args.N} K={args.K} grad_window={args.grad_window} "
        f"n_seeds={args.n_seeds} reps={args.reps}",
        "",
        f"MC-noise envelope (median |Euler@40 - Euler@20| over seeds): {mc_envelope:.4f}",
        "",
        f"{'config':<14} {'logp_mean':>10} {'logp_std':>10} {'fwd_ms':>9} {'fwd+g_ms':>10} {'med|Δlp|':>10} {'spd(f+g)':>10}",
        "-" * 80,
    ]
    for label, _, _ in configs:
        r = results[label]
        if label == "euler_n20":
            dlp = "-"
            spd = "1.00x"
        else:
            dlp = f"{equivalence[label]['med_abs_delta_loglik']:.3f}"
            spd = f"{equivalence[label]['speedup_warm_fwd_grad']:.2f}x"
        report.append(
            f"{label:<14} {r['loglik_mean']:>10.3f} {r['loglik_std']:>10.3f} "
            f"{r['warm_forward_s']*1000:>9.2f} {r['warm_forward_grad_s']*1000:>10.2f} "
            f"{dlp:>10} {spd:>10}"
        )
    report.append("=" * 80)
    (out_dir / "exp_integrator_report.txt").write_text("\n".join(report))

    print(f"\nWrote:\n  {out_dir/'exp_integrator_test.json'}\n  {out_dir/'exp_integrator_test.csv'}\n  {out_dir/'exp_integrator_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
