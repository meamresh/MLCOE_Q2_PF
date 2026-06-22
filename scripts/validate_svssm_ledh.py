"""
Deep validation of DifferentiableLEDHLogLikelihoodSVSSM before committing to HMC.

Four checks:

  1. Variance reduction with N: log p std at truth should decrease as
     particle count N grows (CLT for the PF estimator).
  2. Concentration with T: the per-observation log-likelihood gap
     (log p(truth) - log p(perturbed)) / T should be stable across T;
     and the absolute gap should grow ~linearly with T (likelihood-ratio
     test power).
  3. Likelihood-surface 1D slices: sweep each parameter around truth and
     confirm the mean log p has a maximum (or at least clear monotonicity)
     at or near truth. Saves a 3-panel plot.
  4. Gradient direction at perturbed theta: the negative-log-likelihood
     gradient should point back toward truth. Cosine similarity should be
     positive on average.

Outputs:
  reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_validation/
    - validation_report.txt
    - validation_results.json
    - likelihood_slices.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import DifferentiableLEDHLogLikelihoodSVSSM

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    plt = None
    _HAVE_MPL = False


def gen_svssm(T: int, mu: float, phi: float, sigma_eta: float, seed: int):
    tf.random.set_seed(seed)
    sigma_eta_t = tf.constant(sigma_eta, tf.float32)
    h = tf.constant(float(mu), tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta_t * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def loglik(ll, mu, phi, sigma_eta_sq, y_obs, seed):
    tf.random.set_seed(seed)
    return float(ll(
        tf.constant(mu, tf.float32),
        tf.constant(phi, tf.float32),
        tf.constant(sigma_eta_sq, tf.float32),
        y_obs,
    ).numpy())


def loglik_with_grad(ll, mu, phi, sigma_eta_sq, y_obs, seed):
    tf.random.set_seed(seed)
    p = tf.constant([mu, phi, sigma_eta_sq], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(p)
        v = ll(p[0], p[1], p[2], y_obs)
    g = tape.gradient(v, p)
    return float(v.numpy()), g.numpy().astype(np.float64)


def make_ll(N: int, n_lambda: int = 10, K: int = 10) -> DifferentiableLEDHLogLikelihoodSVSSM:
    return DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N,
        n_lambda=n_lambda,
        sinkhorn_epsilon=1.0,
        sinkhorn_iters=K,
        grad_window=4,
        jit_compile=True,
        integrator="exp",
    )


# ---------------------------------------------------------------------------
# Check 1: variance reduction with N
# ---------------------------------------------------------------------------

def check_variance_vs_N(mu, phi, sigma_eta, T: int, N_values: list[int],
                        n_seeds: int) -> dict:
    print(f"\n[1] Variance reduction with N at truth, T={T}, n_seeds={n_seeds}")
    print(f"    {'N':>5s} {'mean log p':>14s} {'std':>10s} {'1/sqrt(N) ratio':>18s}")
    print(f"    {'-'*5} {'-'*14} {'-'*10} {'-'*18}")
    y_obs = gen_svssm(T, mu, phi, sigma_eta, seed=42)
    rows = []
    ref_std = None
    for N in N_values:
        ll = make_ll(N=N)
        vals = [loglik(ll, mu, phi, sigma_eta ** 2, y_obs, seed=100 + i)
                for i in range(n_seeds)]
        arr = np.asarray(vals)
        m, s = float(arr.mean()), float(arr.std(ddof=1))
        if ref_std is None:
            ref_std = s
        # Expected ratio under O(1/sqrt(N)) scaling vs the smallest N row:
        expected_ratio = math.sqrt(N_values[0]) / math.sqrt(N)
        observed_ratio = s / ref_std if ref_std > 0 else float("nan")
        print(f"    {N:>5d} {m:>14.4f} {s:>10.4f} "
              f"  obs={observed_ratio:.3f}  exp={expected_ratio:.3f}")
        rows.append({"N": int(N), "mean": m, "std": s,
                     "observed_ratio_vs_smallest": observed_ratio,
                     "expected_ratio_1over_sqrt_N": expected_ratio})
    # PASS criterion: std at largest N is less than at smallest N
    std_decreases = rows[-1]["std"] < rows[0]["std"]
    print(f"    {'PASS' if std_decreases else 'FAIL'}: "
          f"std at N={N_values[-1]} ({rows[-1]['std']:.3f}) < N={N_values[0]} ({rows[0]['std']:.3f})")
    return {"rows": rows, "pass": bool(std_decreases)}


# ---------------------------------------------------------------------------
# Check 2: concentration with T
# ---------------------------------------------------------------------------

def check_concentration_vs_T(mu, phi, sigma_eta, T_values: list[int],
                              N: int, n_seeds: int) -> dict:
    print(f"\n[2] Concentration with T at N={N}, n_seeds={n_seeds}")
    print(f"    Wrong-theta perturbation: phi=0.5 (vs true {phi})")
    print(f"    {'T':>5s} {'lp_truth':>10s} {'lp_wrong':>10s} {'gap':>10s} {'gap/T':>10s}")
    print(f"    {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    ll = make_ll(N=N)
    rows = []
    for T in T_values:
        y_obs = gen_svssm(T, mu, phi, sigma_eta, seed=42)
        lp_truth_vals = [loglik(ll, mu, phi, sigma_eta ** 2, y_obs, seed=200 + i)
                         for i in range(n_seeds)]
        lp_wrong_vals = [loglik(ll, mu, 0.5, sigma_eta ** 2, y_obs, seed=200 + i)
                         for i in range(n_seeds)]
        m_truth = float(np.mean(lp_truth_vals))
        m_wrong = float(np.mean(lp_wrong_vals))
        gap = m_truth - m_wrong
        rows.append({"T": int(T), "lp_truth_mean": m_truth, "lp_wrong_mean": m_wrong,
                     "gap": gap, "gap_per_T": gap / T})
        print(f"    {T:>5d} {m_truth:>10.3f} {m_wrong:>10.3f} {gap:>10.3f} {gap/T:>10.5f}")
    # PASS criterion: gap grows monotonically with T (or at least gap at largest T > smallest T)
    gap_grows = rows[-1]["gap"] > rows[0]["gap"]
    print(f"    {'PASS' if gap_grows else 'FAIL'}: "
          f"gap at T={T_values[-1]} ({rows[-1]['gap']:.2f}) > T={T_values[0]} ({rows[0]['gap']:.2f})")
    return {"rows": rows, "pass": bool(gap_grows)}


# ---------------------------------------------------------------------------
# Check 3: likelihood-surface 1D slices
# ---------------------------------------------------------------------------

def check_likelihood_surface(mu, phi, sigma_eta, T: int, N: int,
                              n_grid: int, n_seeds: int, out_dir: Path) -> dict:
    print(f"\n[3] Likelihood-surface 1D slices at T={T}, N={N}, "
          f"n_grid={n_grid}, n_seeds={n_seeds}")
    y_obs = gen_svssm(T, mu, phi, sigma_eta, seed=42)
    ll = make_ll(N=N)

    # Grids around truth (parameter-specific spreads)
    grids = {
        "mu":        np.linspace(mu - 1.5, mu + 1.5, n_grid),
        "phi":       np.linspace(max(-0.99, phi - 0.5), min(0.99, phi + 0.04), n_grid),
        "sigma_eta": np.linspace(max(0.05, sigma_eta - 0.25), sigma_eta + 0.35, n_grid),
    }
    truth = {"mu": mu, "phi": phi, "sigma_eta": sigma_eta}

    surface = {}
    print(f"    {'param':<10s} {'grid value':>10s} {'mean lp':>12s} {'std lp':>10s}")
    print(f"    {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    for pname, grid in grids.items():
        rows = []
        for v in grid:
            vals = []
            for i in range(n_seeds):
                m = mu if pname != "mu" else float(v)
                p = phi if pname != "phi" else float(v)
                s = sigma_eta if pname != "sigma_eta" else float(v)
                vals.append(loglik(ll, m, p, s ** 2, y_obs, seed=300 + i))
            arr = np.asarray(vals)
            rows.append({"value": float(v), "mean": float(arr.mean()),
                         "std": float(arr.std(ddof=1))})
            print(f"    {pname:<10s} {v:>10.3f} {arr.mean():>12.3f} {arr.std(ddof=1):>10.3f}")
        surface[pname] = rows

    # PASS criterion: argmax of mean lp is within +/- 1 grid step of truth
    # (lenient — PF noise + quasi-likelihood bias mean we don't expect exact recovery)
    pass_per_param = {}
    for pname, rows in surface.items():
        values = np.asarray([r["value"] for r in rows])
        means = np.asarray([r["mean"] for r in rows])
        argmax = int(np.argmax(means))
        argmax_val = values[argmax]
        truth_idx = int(np.argmin(np.abs(values - truth[pname])))
        within_1 = abs(argmax - truth_idx) <= 1
        pass_per_param[pname] = bool(within_1)
        print(f"    {pname}: argmax at {argmax_val:.3f} (idx {argmax}); "
              f"truth at idx {truth_idx} ({truth[pname]:.3f}); "
              f"{'OK' if within_1 else 'FAR'}")

    # Plot
    if _HAVE_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (pname, rows) in zip(axes, surface.items()):
            values = np.asarray([r["value"] for r in rows])
            means = np.asarray([r["mean"] for r in rows])
            stds = np.asarray([r["std"] for r in rows])
            ax.errorbar(values, means, yerr=stds, marker="o", capsize=3, lw=1.2,
                        ecolor="0.5")
            ax.axvline(truth[pname], color="red", linestyle="--", lw=1.2,
                       label=f"truth = {truth[pname]:.3f}")
            ax.set_xlabel(pname)
            ax.set_ylabel(r"mean $\log \hat{p}(z \mid \theta)$")
            ax.set_title(f"Likelihood slice over {pname}")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
        fig.suptitle(f"SVSSM LEDH: log-likelihood slices around truth "
                     f"(T={T}, N={N}, n_seeds={n_seeds})", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / "likelihood_slices.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    wrote {out_dir / 'likelihood_slices.png'}")

    overall = all(pass_per_param.values())
    print(f"    {'PASS' if overall else 'PARTIAL'}: "
          f"argmax within 1 grid step of truth on "
          f"{sum(pass_per_param.values())}/{len(pass_per_param)} parameters")
    return {"surface": surface, "pass_per_param": pass_per_param, "pass": bool(overall)}


# ---------------------------------------------------------------------------
# Check 4: gradient direction at perturbed theta
# ---------------------------------------------------------------------------

def check_gradient_direction(mu, phi, sigma_eta, T: int, N: int,
                              n_seeds: int) -> dict:
    print(f"\n[4] Gradient direction at perturbed theta (T={T}, N={N}, n_seeds={n_seeds})")
    y_obs = gen_svssm(T, mu, phi, sigma_eta, seed=42)
    ll = make_ll(N=N)
    truth = np.asarray([mu, phi, sigma_eta ** 2], dtype=np.float64)

    # Perturbations: each parameter ±10% of std of its grid
    perturbations = [
        ("mu-1.0      ", (mu - 1.0, phi, sigma_eta ** 2)),
        ("mu+1.0      ", (mu + 1.0, phi, sigma_eta ** 2)),
        ("phi-0.2     ", (mu, max(-0.99, phi - 0.2), sigma_eta ** 2)),
        ("phi+0.04    ", (mu, min(0.99, phi + 0.04), sigma_eta ** 2)),
        ("sigma2*4    ", (mu, phi, (sigma_eta * 2) ** 2)),
        ("sigma2/4    ", (mu, phi, (sigma_eta / 2) ** 2)),
    ]
    print(f"    {'perturb':<14s} {'mean cos':>10s} {'sign(grad)→truth':>18s}")
    print(f"    {'-'*14} {'-'*10} {'-'*18}")

    rows = []
    for label, theta in perturbations:
        theta_arr = np.asarray(theta, dtype=np.float64)
        direction_to_truth = truth - theta_arr
        if np.linalg.norm(direction_to_truth) < 1e-12:
            continue
        cosines = []
        sign_match = []
        for i in range(n_seeds):
            v, g = loglik_with_grad(ll, theta[0], theta[1], theta[2], y_obs, seed=400 + i)
            # Likelihood gradient points UPHILL. To go toward truth, gradient
            # should align with direction_to_truth (modulo step-size scaling).
            g64 = g.astype(np.float64)
            n_g = np.linalg.norm(g64)
            n_d = np.linalg.norm(direction_to_truth)
            if n_g > 1e-12:
                cosines.append(float(np.dot(g64, direction_to_truth) / (n_g * n_d)))
                # Per-component sign match
                signs = np.sign(g64) == np.sign(direction_to_truth)
                sign_match.append(int(np.sum(signs)))
        mean_cos = float(np.mean(cosines)) if cosines else float("nan")
        mean_signs = float(np.mean(sign_match)) if sign_match else float("nan")
        rows.append({"perturb": label.strip(), "mean_cos": mean_cos,
                     "mean_sign_match": mean_signs, "n_seeds": n_seeds})
        print(f"    {label} {mean_cos:>10.3f} {mean_signs:>17.2f}/3")
    # PASS criterion: mean cosine > 0 on majority of perturbations
    n_positive = sum(1 for r in rows if r["mean_cos"] > 0)
    overall = n_positive >= len(rows) // 2 + 1
    print(f"    {'PASS' if overall else 'PARTIAL'}: "
          f"{n_positive}/{len(rows)} perturbations have positive cosine to truth")
    return {"rows": rows, "n_positive": n_positive, "n_total": len(rows),
            "pass": bool(overall)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--phi", type=float, default=0.95)
    p.add_argument("--sigma_eta", type=float, default=0.3)
    p.add_argument("--T_main", type=int, default=100,
                   help="T for surface and gradient checks")
    p.add_argument("--N_variance", type=str, default="64,128,256,512")
    p.add_argument("--T_concentration", type=str, default="50,100,200")
    p.add_argument("--N_main", type=int, default=128)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--n_grid", type=int, default=9)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_validation")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[svssm-ledh validation] TF {tf.__version__}")
    print(f"  truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}")
    print(f"  out_dir={out_dir}")

    N_variance = [int(x) for x in args.N_variance.split(",")]
    T_concentration = [int(x) for x in args.T_concentration.split(",")]

    t0 = time.perf_counter()
    res1 = check_variance_vs_N(args.mu, args.phi, args.sigma_eta,
                                T=args.T_main, N_values=N_variance,
                                n_seeds=args.n_seeds)
    res2 = check_concentration_vs_T(args.mu, args.phi, args.sigma_eta,
                                     T_values=T_concentration, N=args.N_main,
                                     n_seeds=args.n_seeds)
    res3 = check_likelihood_surface(args.mu, args.phi, args.sigma_eta,
                                     T=args.T_main, N=args.N_main,
                                     n_grid=args.n_grid, n_seeds=args.n_seeds,
                                     out_dir=out_dir)
    res4 = check_gradient_direction(args.mu, args.phi, args.sigma_eta,
                                     T=args.T_main, N=args.N_main,
                                     n_seeds=args.n_seeds)
    elapsed = time.perf_counter() - t0

    overall = res1["pass"] and res2["pass"] and res3["pass"] and res4["pass"]
    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if overall else 'PARTIAL'}  (total time: {elapsed:.1f} s)")
    print(f"  [1] variance vs N   : {'PASS' if res1['pass'] else 'FAIL'}")
    print(f"  [2] concentration vs T: {'PASS' if res2['pass'] else 'FAIL'}")
    print(f"  [3] likelihood surface: {'PASS' if res3['pass'] else 'PARTIAL'}")
    print(f"  [4] gradient direction: {'PASS' if res4['pass'] else 'PARTIAL'}")
    print("=" * 70)

    summary = {
        "tf": tf.__version__,
        "config": vars(args),
        "elapsed_s": elapsed,
        "checks": {
            "variance_vs_N": res1,
            "concentration_vs_T": res2,
            "likelihood_surface": {
                "pass": res3["pass"],
                "pass_per_param": res3["pass_per_param"],
                "surface": {k: v for k, v in res3["surface"].items()},
            },
            "gradient_direction": res4,
        },
        "overall_pass": bool(overall),
    }
    (out_dir / "validation_results.json").write_text(json.dumps(summary, indent=2))

    # Text report
    report = [
        "=" * 80,
        "SVSSM-LEDH Filter Validation Report",
        "=" * 80,
        f"Date: TF {tf.__version__}",
        f"True parameters: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}",
        f"Main config: T={args.T_main}, N={args.N_main}, n_seeds={args.n_seeds}",
        f"Total runtime: {elapsed:.1f} s",
        "",
        "[1] Variance reduction with N",
        "    " + ", ".join(f"N={r['N']}: std={r['std']:.4f}" for r in res1["rows"]),
        f"    Result: {'PASS' if res1['pass'] else 'FAIL'}",
        "",
        "[2] Concentration with T (gap = log p(truth) - log p(phi=0.5))",
        "    " + ", ".join(f"T={r['T']}: gap={r['gap']:.2f}" for r in res2["rows"]),
        f"    Result: {'PASS' if res2['pass'] else 'FAIL'}",
        "",
        "[3] Likelihood-surface argmax vs truth (per parameter)",
    ]
    for p_name, ok in res3["pass_per_param"].items():
        report.append(f"    {p_name}: {'argmax within 1 grid step of truth' if ok else 'argmax NOT near truth'}")
    report.append(f"    Result: {'PASS' if res3['pass'] else 'PARTIAL'}")
    report.append("")
    report.append("[4] Gradient direction at perturbed theta")
    for r in res4["rows"]:
        report.append(f"    {r['perturb']:>14s}: mean cos={r['mean_cos']:>6.3f}")
    report.append(f"    {res4['n_positive']}/{res4['n_total']} perturbations have "
                  f"positive cosine to truth-direction")
    report.append(f"    Result: {'PASS' if res4['pass'] else 'PARTIAL'}")
    report.append("")
    report.append(f"OVERALL: {'PASS' if overall else 'PARTIAL'}")
    report.append("=" * 80)
    (out_dir / "validation_report.txt").write_text("\n".join(report))

    print(f"\nWrote:\n  {out_dir / 'validation_report.txt'}\n  {out_dir / 'validation_results.json'}")
    if _HAVE_MPL:
        print(f"  {out_dir / 'likelihood_slices.png'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
