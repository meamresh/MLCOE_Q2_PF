"""
Ablation: do the three h_0 choices give meaningfully different SVSSM HMC posteriors?

Runs three HMC experiments at the same data, priors, seeds:
  - stationary: h_0 ~ N(mu, sigma_eta^2/(1-phi^2))   [tied to params]
  - fixed_mu:   h_0 ~ N(mu, sigma_eta^2)             [point mass at mu, +1-step noise]
  - diffuse:    h_0 ~ N(0, 100)                       [independent of params]

Two kernels (set with --use_windowed_adaptive):
  - default: vanilla HMC + dual-averaging step-size adaptation, identity mass.
  - windowed: Stan-style windowed adaptive HMC with a DENSE inverse-covariance
    mass matrix (the production kernel; reuses run_chain_windowed_proper from
    exp_hmc_svssm). Use --no-dense_mass for the diagonal variant.

Execution modes:
  - default (no --chain_id, no --aggregate): run ALL chains in-process,
    sequentially, then print the comparison table + write the JSON
    (backward-compatible behaviour).
  - --chain_id C: run ONLY chain C (across all three init types), saving raw
    samples to {out_dir}/chain_C/{init_type}.npy. Lets a launcher run the
    chains in parallel (one process per chain).
  - --aggregate: read every {out_dir}/chain_*/{init_type}.npy, stack across
    chains, compute split-Rhat + summaries, print the table, write the JSON.

Output: reports/.../h0_ablation[...]/h0_ablation_results.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import DifferentiableLEDHLogLikelihoodSVSSM
from scripts.exp.exp_hmc_svssm import run_chain_windowed_proper

tfd = tfp.distributions
tfm = tfp.mcmc

INIT_TYPES = ("stationary", "fixed_mu", "diffuse")


def gen_svssm(T, mu, phi, sigma_eta, seed):
    tf.random.set_seed(seed)
    sigma_eta_t = tf.constant(sigma_eta, tf.float32)
    h = tf.constant(float(mu), tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta_t * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def make_target(ll, y_obs, crn_seed, priors):
    p_mu = tfd.Normal(loc=tf.constant(priors["mu_loc"], tf.float32),
                      scale=tf.constant(priors["mu_scale"], tf.float32))
    p_pr = tfd.Normal(loc=tf.constant(priors["phi_raw_loc"], tf.float32),
                      scale=tf.constant(priors["phi_raw_scale"], tf.float32))
    p_ls = tfd.Normal(loc=tf.constant(priors["log_sig_loc"], tf.float32),
                      scale=tf.constant(priors["log_sig_scale"], tf.float32))

    def target(theta_raw):
        tf.random.set_seed(int(crn_seed))
        mu = theta_raw[0]
        phi_raw = theta_raw[1]
        log_sig = theta_raw[2]
        phi = tf.tanh(phi_raw)
        sigma_eta_sq = tf.exp(log_sig)
        lp = p_mu.log_prob(mu) + p_pr.log_prob(phi_raw) + p_ls.log_prob(log_sig)
        ll_val = ll(mu, phi, sigma_eta_sq, y_obs)
        ll_val = tf.cast(tf.math.real(ll_val), tf.float32)
        ll_val = tf.where(tf.math.is_finite(ll_val), ll_val,
                          tf.constant(-1e6, tf.float32))
        return lp + ll_val
    return target


def run_chain_dualavg(target, init_raw, num_burn, num_samp, L, step_size, seed):
    """Vanilla HMC + dual-averaging step-size adaptation, identity mass."""
    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=tf.constant(step_size, tf.float32),
        num_leapfrog_steps=int(L),
    )
    kernel = tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(num_burn),
        target_accept_prob=tf.constant(0.65, tf.float32),
    )

    def trace_fn(_, kr):
        inner = kr.inner_results if hasattr(kr, "inner_results") else kr
        acc = tf.cast(inner.is_accepted, tf.bool) if hasattr(inner, "is_accepted") \
            else tf.constant(False)
        return acc

    init = tf.constant(init_raw, dtype=tf.float32)
    samples, accepted = tfm.sample_chain(
        num_results=int(num_samp),
        num_burnin_steps=int(num_burn),
        current_state=init,
        kernel=kernel,
        trace_fn=trace_fn,
        seed=int(seed),
    )
    return samples.numpy(), accepted.numpy()


def run_one_chain(target, init_raw, args, seed):
    """Dispatch to the windowed dense-mass kernel or the dual-averaging one."""
    if args.use_windowed_adaptive:
        samp, acc, _lp, _step = run_chain_windowed_proper(
            target, init_raw, args.num_results, args.num_burnin,
            args.step_size, args.L, seed,
            target_accept_prob=args.target_accept_prob,
            progress_every=args.progress_every, dense_mass=args.dense_mass)
        return np.asarray(samp), np.asarray(acc)
    return run_chain_dualavg(target, init_raw, args.num_burnin,
                             args.num_results, args.L, args.step_size, seed)


def constrain(theta_raw):
    th = np.asarray(theta_raw)
    return np.stack([th[..., 0], np.tanh(th[..., 1]), np.exp(th[..., 2])], axis=-1)


def split_rhat(samples):
    chains, draws, dim = samples.shape
    if chains < 2:
        return np.full(dim, float("nan"))
    half = draws // 2
    s = np.concatenate([samples[:, :half], samples[:, half:2 * half]], axis=0)
    N = s.shape[1]
    means = s.mean(axis=1)
    vars_ = s.var(axis=1, ddof=1)
    B = N * means.var(axis=0, ddof=1)
    W = vars_.mean(axis=0)
    var_hat = ((N - 1) / N) * W + B / N
    return np.sqrt(var_hat / np.maximum(W, 1e-12))


def summarise(samples_constrained, truth):
    flat = samples_constrained.reshape(-1, 3)
    rhat = split_rhat(samples_constrained)
    truth_arr = np.asarray([truth["mu"], truth["phi"], truth["sigma_eta"] ** 2])
    rows = []
    for i, name in enumerate(["mu", "phi", "sigma_eta_sq"]):
        x = flat[:, i]
        q025, q50, q975 = np.percentile(x, [2.5, 50.0, 97.5])
        rows.append({
            "param": name, "truth": float(truth_arr[i]),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "median": float(q50), "q025": float(q025), "q975": float(q975),
            "rhat": float(rhat[i]),
            "covered": bool(q025 <= truth_arr[i] <= q975),
        })
    se = np.sqrt(np.maximum(flat[:, 2], 1e-12))
    rows.append({
        "param": "sigma_eta", "truth": truth["sigma_eta"],
        "mean": float(se.mean()), "std": float(se.std(ddof=1)),
        "median": float(np.percentile(se, 50.0)),
        "q025": float(np.percentile(se, 2.5)),
        "q975": float(np.percentile(se, 97.5)),
        "rhat": float("nan"),
        "covered": bool(np.percentile(se, 2.5) <= truth["sigma_eta"] <= np.percentile(se, 97.5)),
    })
    return rows


def print_table_and_write(results, args, priors, truth, total_wall):
    out_dir = Path(args.out_dir)
    print("=" * 100)
    kernel_name = ("windowed-adaptive + " + ("DENSE" if args.dense_mass else "diagonal")
                   + " mass") if args.use_windowed_adaptive else "dual-averaging + identity mass"
    print(f"h_0 ablation: comparison   [kernel: {kernel_name}]")
    print("=" * 100)
    print(f"{'param':<14s} {'metric':<10s}  {'stationary':>14s} {'fixed_mu':>14s} {'diffuse':>14s}  {'truth':>10s}")
    print("-" * 100)
    truth_arr = {"mu": args.mu, "phi": args.phi, "sigma_eta_sq": args.sigma_eta ** 2,
                 "sigma_eta": args.sigma_eta}
    for i, name in enumerate(["mu", "phi", "sigma_eta_sq", "sigma_eta"]):
        for metric in ["median", "mean", "rhat"]:
            row = [name if metric == "median" else "", metric]
            for init_type in INIT_TYPES:
                r = results[init_type]["rows"][i]
                val = r[metric]
                row.append(f"{val:+.4f}" if not np.isnan(val) else "n/a")
            row.append(f"{truth_arr[name]:+.4f}")
            print(f"{row[0]:<14s} {row[1]:<10s}  {row[2]:>14s} {row[3]:>14s} {row[4]:>14s}  {row[5]:>10s}")
        print()
    print("=" * 100)
    for init_type in INIT_TYPES:
        n_cov = sum(1 for r in results[init_type]["rows"] if r["covered"])
        print(f"  {init_type:<12s}: accept={results[init_type]['accept_rate_overall']:.3f}  "
              f"coverage={n_cov}/4")

    out_json = {
        "tf": tf.__version__, "tfp": tfp.__version__,
        "config": vars(args), "priors": priors, "truth": truth,
        "kernel": kernel_name,
        "results": {it: {"rows": v["rows"],
                          "accept_rate_overall": v["accept_rate_overall"],
                          "elapsed_per_chain_s": v.get("elapsed_per_chain_s", [])}
                    for it, v in results.items()},
        "total_wall_s": total_wall,
    }
    (out_dir / "h0_ablation_results.json").write_text(json.dumps(out_json, indent=2))
    print(f"\nWrote {out_dir / 'h0_ablation_results.json'}")


def make_init_raws(args):
    truth_raw = np.asarray([args.mu, np.arctanh(min(max(args.phi, -0.999), 0.999)),
                            np.log(args.sigma_eta ** 2)], dtype=np.float32)
    rng = np.random.default_rng(args.base_seed)
    return [truth_raw + args.dispersion * rng.standard_normal(3).astype(np.float32)
            for _ in range(args.num_chains)]


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--phi", type=float, default=0.95)
    p.add_argument("--sigma_eta", type=float, default=0.3)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--step_size", type=float, default=0.05)
    p.add_argument("--num_chains", type=int, default=2)
    p.add_argument("--num_burnin", type=int, default=200)
    p.add_argument("--num_results", type=int, default=200)
    p.add_argument("--dispersion", type=float, default=0.10)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--target_accept_prob", type=float, default=0.65)
    p.add_argument("--progress_every", type=int, default=200)
    # kernel selection
    p.add_argument("--use_windowed_adaptive", action="store_true",
                   help="Stan-style windowed adaptive HMC with dense mass "
                        "(the production kernel). Default: dual-averaging + identity mass.")
    p.add_argument("--dense_mass", action=argparse.BooleanOptionalAction, default=True,
                   help="With --use_windowed_adaptive: dense (full inverse-cov) mass "
                        "matrix [default]; --no-dense_mass for diagonal.")
    # parallel execution
    p.add_argument("--chain_id", type=int, default=None,
                   help="Run ONLY this chain (across all init types), save raw "
                        "samples to {out_dir}/chain_C/. For the parallel launcher.")
    p.add_argument("--aggregate", action="store_true",
                   help="Read {out_dir}/chain_*/{init}.npy, stack, summarise, write JSON.")
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/h0_ablation")
    return p.parse_args()


PRIORS = {
    "mu_loc": 0.0, "mu_scale": 5.0,
    "phi_raw_loc": 0.0, "phi_raw_scale": 2.0,
    "log_sig_loc": -2.0, "log_sig_scale": 2.0,
}


def run_single_chain_mode(args, truth, y_obs):
    """--chain_id C: run chain C for all init types, save raw npys, exit."""
    c = args.chain_id
    init_raws = make_init_raws(args)
    if c < 0 or c >= args.num_chains:
        raise SystemExit(f"chain_id {c} out of range [0,{args.num_chains})")
    init_raw = init_raws[c]
    seed = args.base_seed + 1009 * (c + 1)
    chain_dir = Path(args.out_dir) / f"chain_{c}"
    chain_dir.mkdir(parents=True, exist_ok=True)
    print(f"[h0 ablation | chain {c}] kernel="
          f"{'windowed/' + ('dense' if args.dense_mass else 'diag') if args.use_windowed_adaptive else 'dualavg'}"
          f"  seed={seed}  init_raw={init_raw.tolist()}")
    for init_type in INIT_TYPES:
        print(f"  --- init_type = {init_type} (chain {c}) ---")
        ll = DifferentiableLEDHLogLikelihoodSVSSM(
            num_particles=args.N, n_lambda=10, sinkhorn_epsilon=1.0,
            sinkhorn_iters=10, grad_window=4, jit_compile=True,
            integrator="exp", init_type=init_type,
        )
        target = make_target(ll, y_obs, crn_seed=args.base_seed, priors=PRIORS)
        t0 = time.perf_counter()
        samp, acc = run_one_chain(target, init_raw, args, seed)
        elapsed = time.perf_counter() - t0
        np.save(chain_dir / f"{init_type}.npy", samp)
        np.save(chain_dir / f"{init_type}_acc.npy", acc)
        print(f"    chain {c}: {elapsed:.1f}s  accept={acc.mean():.3f}  "
              f"-> {chain_dir / (init_type + '.npy')}")
    (chain_dir / "meta.json").write_text(json.dumps(
        {"chain_id": c, "seed": seed, "init_raw": init_raw.tolist()}, indent=2))
    print(f"[chain {c}] done.")


def aggregate_mode(args, truth, total_wall=0.0):
    """--aggregate: stack chain_*/ npys, summarise, print + write JSON."""
    out_dir = Path(args.out_dir)
    results = {}
    for init_type in INIT_TYPES:
        chain_samps, chain_accs = [], []
        for c in range(args.num_chains):
            f = out_dir / f"chain_{c}" / f"{init_type}.npy"
            if not f.exists():
                raise SystemExit(f"missing {f} — did all chains finish?")
            chain_samps.append(np.load(f))
            af = out_dir / f"chain_{c}" / f"{init_type}_acc.npy"
            if af.exists():
                chain_accs.append(np.load(af))
        samples_raw = np.stack(chain_samps, axis=0)          # (chains, draws, 3)
        samples_constrained = constrain(samples_raw)
        rows = summarise(samples_constrained, truth)
        results[init_type] = {
            "rows": rows,
            "accept_rate_overall": float(np.mean([a.mean() for a in chain_accs]))
                                   if chain_accs else float("nan"),
            "elapsed_per_chain_s": [],
        }
    print_table_and_write(results, args, PRIORS, truth, total_wall)


def run_inprocess_mode(args, truth, y_obs):
    """Default: all chains in-process, sequential (backward-compatible)."""
    init_raws = make_init_raws(args)
    results = {}
    t_total = time.perf_counter()
    for init_type in INIT_TYPES:
        print(f"--- init_type = {init_type} ---")
        ll = DifferentiableLEDHLogLikelihoodSVSSM(
            num_particles=args.N, n_lambda=10, sinkhorn_epsilon=1.0,
            sinkhorn_iters=10, grad_window=4, jit_compile=True,
            integrator="exp", init_type=init_type,
        )
        target = make_target(ll, y_obs, crn_seed=args.base_seed, priors=PRIORS)
        all_samp, all_acc, elapsed_per_chain = [], [], []
        for c in range(args.num_chains):
            seed = args.base_seed + 1009 * (c + 1)
            t0 = time.perf_counter()
            samp, acc = run_one_chain(target, init_raws[c], args, seed)
            elapsed = time.perf_counter() - t0
            print(f"  chain {c+1}: {elapsed:.1f}s, accept={acc.mean():.3f}")
            all_samp.append(samp); all_acc.append(acc); elapsed_per_chain.append(elapsed)
        samples_constrained = constrain(np.stack(all_samp, axis=0))
        rows = summarise(samples_constrained, truth)
        results[init_type] = {
            "rows": rows,
            "accept_rate_overall": float(np.mean([a.mean() for a in all_acc])),
            "elapsed_per_chain_s": [float(x) for x in elapsed_per_chain],
        }
        for r in rows:
            cov = "OK" if r["covered"] else "OUT"
            print(f"    {r['param']:<14s} truth={r['truth']:+.4f}  mean={r['mean']:+.4f}  "
                  f"median={r['median']:+.4f}  CI=[{r['q025']:+.3f}, {r['q975']:+.3f}]  "
                  f"Rhat={r['rhat']:.3f}  {cov}")
        print()
    total = time.perf_counter() - t_total
    print(f"Total wall: {total:.1f}s\n")
    print_table_and_write(results, args, PRIORS, truth, total)


def main() -> int:
    args = build_args()
    truth = {"mu": args.mu, "phi": args.phi, "sigma_eta": args.sigma_eta}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[h0 ablation] TF {tf.__version__}  TFP {tfp.__version__}")
    print(f"  truth: {truth}")
    print(f"  config: T={args.T} N={args.N} chains={args.num_chains} "
          f"burn={args.num_burnin} samp={args.num_results} L={args.L}")
    print(f"  kernel: {'windowed/' + ('dense' if args.dense_mass else 'diag') if args.use_windowed_adaptive else 'dualavg+identity'}")
    print(f"  priors: {PRIORS}  (WIDE defaults)")
    print(f"  out_dir: {out_dir}\n")

    if args.aggregate:
        aggregate_mode(args, truth)
        return 0

    y_obs = gen_svssm(args.T, args.mu, args.phi, args.sigma_eta, seed=args.data_seed)

    if args.chain_id is not None:
        run_single_chain_mode(args, truth, y_obs)
        return 0

    run_inprocess_mode(args, truth, y_obs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
