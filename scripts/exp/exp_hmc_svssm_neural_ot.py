"""
HMC parameter recovery for the canonical SVSSM with the *neural-OT*
filter swapped in for Sinkhorn resampling.

Mirrors scripts/exp/exp_hmc_svssm.py but uses
:class:`DifferentiableLEDHNeuralOTSVSSM` instead of the Sinkhorn-based
filter. The trained ``ConditionalMGradNet`` weights are loaded from
the Phase 4 supervised checkpoint by default.

The point of this script: validate that the neural operator preserves
HMC's ability to recover (mu, phi, sigma_eta) — i.e., that the
0.10-log-unit gap between NN-OT and Sinkhorn at filter eval does not
degrade posterior recovery.

Outputs land in their own directory, separate from the Sinkhorn run,
so the two posteriors can be compared side-by-side.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Reuse helpers from the Sinkhorn driver
from scripts.exp.exp_hmc_svssm import (
    gen_svssm, make_target_log_prob, unconstrain, constrain,
    run_chain, summarise_posterior,
)

from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm import (
    DifferentiableLEDHNeuralOTSVSSM,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_jit import (
    DifferentiableLEDHNeuralOTSVSSMJIT,
)
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet
from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.hyper_deeponet_ot import HyperDeepONetMonotoneOT


_ARCH_BUILDERS = {
    "mgradnet":       (ConditionalMGradNet,    "n_ridges"),
    "deeponet":       (DeepONetMonotoneOT,     "n_basis"),
    "hyper_deeponet": (HyperDeepONetMonotoneOT, "n_basis"),
}

_DEFAULT_CKPTS = {
    "mgradnet":       "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase4/supervised.weights.h5",
    "deeponet":       "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase3/deeponet.weights.h5",
    "hyper_deeponet": "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase3/hyper_deeponet.weights.h5",
}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    plt = None
    _HAVE_MPL = False


def _aggregate_nnot_chains(out_dir: Path):
    """Load all chains/chain_*.npz, stack into the same arrays as the
    sequential path, return (samples_raw, samples_constrained, accs, lps,
    all_step_sizes, elapsed). Mirrors _aggregate_from_chains in exp_hmc_svssm.py."""
    chains_dir = out_dir / "chains"
    files = sorted(chains_dir.glob("chain_*.npz"))
    if not files:
        raise FileNotFoundError(f"No chain_*.npz under {chains_dir}")
    print(f"  [aggregate] loading {len(files)} chain file(s) from {chains_dir}")
    samples_raw_list, samples_con_list, acc_list, lp_list, step_list = [], [], [], [], []
    elapsed_list = []
    for f in files:
        d = np.load(f)
        samples_raw_list.append(d["samples_raw"])
        samples_con_list.append(d["samples_constrained"])
        acc_list.append(d["accept"])
        lp_list.append(d["log_prob"])
        step_list.append(d["step_size"])
        elapsed_list.append(float(d["elapsed_s"]))
        print(f"    {f.name}: shape={d['samples_raw'].shape}, "
              f"accept={float(d['accept'].mean()):.3f}, "
              f"elapsed={float(d['elapsed_s']):.1f}s")
    return (
        np.stack(samples_raw_list, axis=0),
        np.stack(samples_con_list, axis=0),
        np.stack(acc_list, axis=0),
        np.stack(lp_list, axis=0),
        step_list,
        float(np.max(elapsed_list)),  # parallel wall ~ max chain time
    )


def build_filter_from_checkpoint(checkpoint_path: Path, *, num_particles: int,
                                  n_lambda: int, n_ridges: int,
                                  grad_window: int,
                                  arch: str = "mgradnet",
                                  sinkhorn_epsilon: float = 1.0,
                                  graph_mode: str = "eager",
                                  ):
    """Instantiate the trained NN, load weights, wrap in NN-OT filter.

    ``arch``: one of ``mgradnet`` / ``deeponet`` / ``hyper_deeponet``.
    ``graph_mode``:
        ``"eager"``  — production filter (DifferentiableLEDHNeuralOTSVSSM)
        ``"graph"``  — JIT-experiment filter wrapped in tf.function (no XLA)
        ``"xla"``    — JIT-experiment filter wrapped in tf.function(jit_compile=True)
    """
    if arch not in _ARCH_BUILDERS:
        raise ValueError(f"arch must be one of {list(_ARCH_BUILDERS)}, got {arch!r}")
    ModelClass, width_kwarg = _ARCH_BUILDERS[arch]
    tf.random.set_seed(0)
    model = ModelClass(state_dim=1, n_scalar_ctx=7, **{width_kwarg: n_ridges})
    dummy_p = tf.zeros([num_particles], tf.float32)
    dummy_w = tf.fill([num_particles], 1.0 / float(num_particles))
    dummy_c = tf.zeros([7], tf.float32)
    _ = model(dummy_p, dummy_w, dummy_c)
    model.load_weights(str(checkpoint_path))
    print(f"  arch = {arch}  ({ModelClass.__name__}, {width_kwarg}={n_ridges})")
    print(f"  loaded weights from {checkpoint_path}")
    print(f"  graph_mode = {graph_mode}")
    if graph_mode == "eager":
        ll = DifferentiableLEDHNeuralOTSVSSM(
            neural_ot_model=model, num_particles=num_particles,
            n_lambda=n_lambda, sinkhorn_epsilon=sinkhorn_epsilon,
            grad_window=grad_window, jit_compile=False, integrator="exp",
        )
    else:
        ll = DifferentiableLEDHNeuralOTSVSSMJIT(
            neural_ot_model=model, num_particles=num_particles,
            n_lambda=n_lambda, sinkhorn_epsilon=sinkhorn_epsilon,
            grad_window=grad_window, integrator="exp",
            graph_mode=graph_mode,
        )
    return ll


def _finalize_and_report(args, out_dir: Path,
                          samples_raw, samples_constrained,
                          accs, lps, all_step_sizes, elapsed,
                          ckpt_path: str | None = None,
                          grad_time: float | None = None) -> int:
    """Write the combined npz, txt report, JSON, and traceplot.

    Shared by sequential mode, single-chain mode's aggregation pass, and
    --aggregate mode.
    """
    truth = {"mu": args.mu, "phi": args.phi, "sigma_eta": args.sigma_eta}
    truth_arr = np.asarray([args.mu, args.phi, args.sigma_eta], dtype=np.float32)

    np.savez_compressed(
        out_dir / "svssm_hmc_neural_ot_samples.npz",
        samples_raw=samples_raw.astype(np.float32),
        samples_constrained=samples_constrained.astype(np.float32),
        accept=accs.astype(np.float32),
        log_prob=lps.astype(np.float32),
        step_size=np.stack(all_step_sizes, axis=0).astype(np.float32),
        truth=truth_arr,
    )

    rows = summarise_posterior(samples_constrained, truth)

    print("\n" + "=" * 100)
    print(f"NN-OT HMC parameter recovery (total wall: {elapsed:.1f}s)")
    print("=" * 100)
    print(f"{'param':<20s} {'truth':>10s} {'mean':>10s} {'std':>10s} {'median':>10s} "
          f"{'2.5%':>10s} {'97.5%':>10s} {'Rhat':>8s} {'ESS':>8s} {'bias%':>8s} cov")
    for r in rows:
        ok = "OK" if r["covered_95ci"] else "OUT"
        print(f"{r['param']:<20s} {r['truth']:>10.4f} {r['mean']:>10.4f} "
              f"{r['std']:>10.4f} {r['median']:>10.4f} {r['q025']:>10.4f} "
              f"{r['q975']:>10.4f} {r['rhat']:>8.3f} {r['ess_bulk']:>8.1f} "
              f"{r['bias_pct']:>+7.1f}% {ok:>3s}")
    per_chain_acc = [float(a.mean()) for a in accs]
    print(f"\nAccept rate: per-chain={per_chain_acc}; overall={float(accs.mean()):.3f}")
    print("=" * 100)

    ckpt_label = ckpt_path if ckpt_path is not None else (
        args.checkpoint_path or _DEFAULT_CKPTS.get(args.arch, "?")
    )
    text_rows = [
        "=" * 100,
        "SVSSM HMC Parameter Recovery with Neural-OT Resampling",
        "=" * 100,
        f"TF {tf.__version__}",
        f"Checkpoint: {ckpt_label}",
        f"Arch: {args.arch}",
        f"truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}",
        f"T={args.T} N={args.N} n_lambda={args.n_lambda} L={args.L} "
        f"step_size={args.step_size} dispersion={args.dispersion}",
        f"num_chains={args.num_chains} burnin={args.num_burnin} "
        f"results={args.num_results}",
        f"priors: mu~N({args.prior_mu_loc},{args.prior_mu_scale}^2), "
        f"phi_raw~N({args.prior_phi_raw_loc},{args.prior_phi_raw_scale}^2), "
        f"log_sigma_eta_sq~N({args.prior_log_sigma_eta_sq_loc},"
        f"{args.prior_log_sigma_eta_sq_scale}^2)",
        f"Wall: {elapsed:.1f}s "
        f"({'max-chain (parallel)' if args.aggregate or args.chain_id is not None else 'sum (sequential)'})",
        "",
        f"{'param':<20s} {'truth':>10s} {'mean':>10s} {'std':>10s} {'median':>10s} "
        f"{'2.5%':>10s} {'97.5%':>10s} {'Rhat':>8s} {'ESS':>8s} {'bias%':>8s}",
        "-" * 110,
    ]
    for r in rows:
        text_rows.append(
            f"{r['param']:<20s} {r['truth']:>10.4f} {r['mean']:>10.4f} "
            f"{r['std']:>10.4f} {r['median']:>10.4f} {r['q025']:>10.4f} "
            f"{r['q975']:>10.4f} {r['rhat']:>8.3f} {r['ess_bulk']:>8.1f} "
            f"{r['bias_pct']:>+7.1f}%"
        )
    text_rows += [
        "",
        f"Accept rate per chain: {per_chain_acc}",
        f"Overall accept rate:   {float(accs.mean()):.3f}",
        "=" * 100,
    ]
    (out_dir / "svssm_hmc_neural_ot_results.txt").write_text("\n".join(text_rows))

    summary_json = {
        "tf": tf.__version__,
        "config": vars(args),
        "truth": truth,
        "rows": [{k: v for k, v in r.items()} for r in rows],
        "accept_rate_per_chain": per_chain_acc,
        "accept_rate_overall": float(accs.mean()),
        "elapsed_s": elapsed,
    }
    if grad_time is not None:
        summary_json["per_step_grad_ms"] = grad_time * 1000
    (out_dir / "svssm_hmc_neural_ot_summary.json").write_text(
        json.dumps(summary_json, indent=2)
    )

    if _HAVE_MPL:
        names = ["mu", "phi", "sigma_eta_sq"]
        truth_arr2 = np.asarray([args.mu, args.phi, args.sigma_eta ** 2])
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for i, (ax, name) in enumerate(zip(axes, names)):
            for c in range(samples_constrained.shape[0]):
                ax.plot(samples_constrained[c, :, i], alpha=0.7,
                        label=f"chain {c+1}")
            ax.axhline(truth_arr2[i], color="red", linestyle="--", lw=1.2,
                       label="truth")
            ax.set_ylabel(name)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right")
        axes[-1].set_xlabel("iteration")
        fig.suptitle("NN-OT HMC trace")
        fig.tight_layout()
        fig.savefig(out_dir / "svssm_hmc_neural_ot_traces.png", dpi=120,
                     bbox_inches="tight")
        plt.close(fig)
        print(f"\n[plot] wrote {out_dir / 'svssm_hmc_neural_ot_traces.png'}")

    print(f"\n[done] wrote {out_dir / 'svssm_hmc_neural_ot_results.txt'}")
    print(f"       wrote {out_dir / 'svssm_hmc_neural_ot_samples.npz'}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20,
                   help="Observation length; smaller than Sinkhorn default "
                        "because the NN-OT filter is eager (slower per step).")
    p.add_argument("--N", type=int, default=64,
                   help="Particle count. Must match the training data N "
                        "for the trained model.")
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--n_ridges", type=int, default=64,
                   help="mGradNet width. Must match the trained checkpoint.")
    p.add_argument("--grad_window", type=int, default=4)
    p.add_argument("--graph_mode", type=str, default="eager",
                   choices=["eager", "graph", "xla"],
                   help="eager=production filter; graph/xla=JIT-experiment filter")
    p.add_argument("--arch", type=str, default="deeponet",
                   choices=list(_ARCH_BUILDERS.keys()),
                   help="Architecture of the trained NN-OT model. "
                        "Default 'deeponet' (the canonical neural operator; "
                        "best filter LL in Phase 3, best mu posterior recovery "
                        "in Phase 7).")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Override; defaults to the canonical checkpoint for --arch.")
    # HMC kernel
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--step_size", type=float, default=0.05)
    p.add_argument("--num_chains", type=int, default=2)
    p.add_argument("--num_burnin", type=int, default=50)
    p.add_argument("--num_results", type=int, default=100)
    p.add_argument("--dispersion", type=float, default=0.15)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--progress_every", type=int, default=10)
    # Ground truth
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--phi", type=float, default=0.95)
    p.add_argument("--sigma_eta", type=float, default=0.3)
    # Priors — tight, matching the previously-converged Sinkhorn config
    p.add_argument("--prior_mu_loc", type=float, default=0.0)
    p.add_argument("--prior_mu_scale", type=float, default=1.0)
    p.add_argument("--prior_phi_raw_loc", type=float, default=2.0)
    p.add_argument("--prior_phi_raw_scale", type=float, default=0.5)
    p.add_argument("--prior_log_sigma_eta_sq_loc", type=float, default=-2.0)
    p.add_argument("--prior_log_sigma_eta_sq_scale", type=float, default=1.0)
    # Output
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "svssm_hmc_neural_ot")
    # Per-chain process isolation (for parallel launchers)
    p.add_argument("--chain_id", type=int, default=None,
                   help="If set, run ONLY this chain index (0-based) and save to "
                        "<out_dir>/chains/chain_{id}.npz. Use with run_nnot_parallel.sh.")
    p.add_argument("--aggregate", action="store_true",
                   help="Skip running; load all <out_dir>/chains/chain_*.npz, "
                        "compute diagnostics, write combined report.")
    p.add_argument("--use_windowed_adaptive", action="store_true",
                   help="Use the v7 windowed-adaptive PreconditionedHMC kernel "
                        "(dense mass matrix by default). Routes to "
                        "run_chain_windowed_proper via run_chain. Matches the "
                        "kernel used in the Sinkhorn Phase 16 T-sweep "
                        "(scripts/exp/slurm.slurm).")
    p.add_argument("--diagonal_mass", action="store_true",
                   help="With --use_windowed_adaptive: use a DIAGONAL mass "
                        "matrix (per-dim variance only) instead of the default "
                        "DENSE mass matrix.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chains_dir = out_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hmc-neural-ot] TF {tf.__version__}")
    print(f"  truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}")
    print(f"  T={args.T} N={args.N} n_lambda={args.n_lambda}  L={args.L} step_size={args.step_size}")
    print(f"  chains={args.num_chains} burnin={args.num_burnin} results={args.num_results}")
    if args.aggregate:
        print(f"  MODE: aggregate (load chains/, compute diagnostics)")
    elif args.chain_id is not None:
        print(f"  MODE: single-chain  chain_id={args.chain_id} of {args.num_chains}")
    else:
        print(f"  MODE: sequential (all {args.num_chains} chains in this process)")

    # === Aggregate mode: skip the run; load per-chain files and produce combined report ===
    if args.aggregate:
        samples_raw, samples_constrained, accs, lps, all_step_sizes, elapsed = \
            _aggregate_nnot_chains(out_dir)
        return _finalize_and_report(
            args, out_dir, samples_raw, samples_constrained, accs, lps,
            all_step_sizes, elapsed,
        )

    # ---- Generate data (same seed as Sinkhorn driver default → identical y_obs) ----
    y_obs = gen_svssm(args.T, args.mu, args.phi, args.sigma_eta, seed=args.data_seed)
    print(f"  y_obs range: [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")

    # ---- Build the NN-OT filter ----
    ckpt_path = args.checkpoint_path or _DEFAULT_CKPTS[args.arch]
    ll = build_filter_from_checkpoint(
        Path(ckpt_path),
        num_particles=args.N, n_lambda=args.n_lambda,
        n_ridges=args.n_ridges, grad_window=args.grad_window,
        arch=args.arch,
        sinkhorn_epsilon=1.0,
        graph_mode=args.graph_mode,
    )

    # ---- Sanity: one forward + one gradient ----
    mu_t = tf.constant(args.mu, tf.float32)
    phi_t = tf.constant(args.phi, tf.float32)
    sig_t = tf.constant(args.sigma_eta ** 2, tf.float32)
    tf.random.set_seed(args.base_seed)
    v0 = float(ll(mu_t, phi_t, sig_t, y_obs).numpy())
    print(f"  sanity: log p at truth = {v0:.4f}")

    t_grad = time.perf_counter()
    p_raw_test = tf.constant([args.mu, np.arctanh(args.phi),
                               np.log(args.sigma_eta ** 2)], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(p_raw_test)
        v = ll(p_raw_test[0], tf.tanh(p_raw_test[1]), tf.exp(p_raw_test[2]),
                y_obs)
    g = tape.gradient(v, p_raw_test)
    grad_time = time.perf_counter() - t_grad
    print(f"  one forward+grad: {grad_time*1000:.1f} ms  grad={g.numpy()}")
    if not np.all(np.isfinite(g.numpy())):
        raise RuntimeError("Non-finite gradient at truth — aborting.")

    # ---- Set up target log-prob ----
    crn_seed = args.base_seed
    target_log_prob_fn = make_target_log_prob(
        ll, y_obs, crn_seed=crn_seed,
        prior_mu_loc=args.prior_mu_loc, prior_mu_scale=args.prior_mu_scale,
        prior_phi_raw_loc=args.prior_phi_raw_loc,
        prior_phi_raw_scale=args.prior_phi_raw_scale,
        prior_log_sigma_eta_sq_loc=args.prior_log_sigma_eta_sq_loc,
        prior_log_sigma_eta_sq_scale=args.prior_log_sigma_eta_sq_scale,
    )

    # ---- Initial states, dispersed ----
    truth_raw = unconstrain(args.mu, args.phi, args.sigma_eta ** 2)
    rng = np.random.default_rng(args.base_seed)
    init_raws = [truth_raw + args.dispersion * rng.standard_normal(3).astype(np.float32)
                 for _ in range(args.num_chains)]

    chain_seeds = [args.base_seed + 1009 * (c + 1) for c in range(args.num_chains)]

    # === Single-chain mode: run only chain_id and save to chains/chain_{id}.npz ===
    if args.chain_id is not None:
        c = int(args.chain_id)
        if c < 0 or c >= args.num_chains:
            raise ValueError(f"chain_id {c} out of range [0, {args.num_chains})")
        print(f"\n  [chain {c+1}/{args.num_chains}] init_raw={init_raws[c].tolist()}")
        print(f"    init_constrained={constrain(init_raws[c]).tolist()}")
        t0 = time.perf_counter()
        samples, acc, lp, step = run_chain(
            target_log_prob_fn=target_log_prob_fn,
            init_raw=init_raws[c],
            num_results=args.num_results,
            num_burnin=args.num_burnin,
            step_size=args.step_size,
            num_leapfrog=args.L,
            seed=chain_seeds[c],
            progress_every=args.progress_every,
            use_windowed_adaptive=args.use_windowed_adaptive,
            dense_mass=(not args.diagonal_mass),
        )
        elapsed_chain = time.perf_counter() - t0
        samples_con_one = np.stack([
            samples[:, 0],
            np.tanh(samples[:, 1]),
            np.exp(samples[:, 2]),
        ], axis=-1).astype(np.float32)
        chain_path = chains_dir / f"chain_{c}.npz"
        np.savez_compressed(
            chain_path,
            samples_raw=samples.astype(np.float32),
            samples_constrained=samples_con_one,
            accept=acc.astype(np.float32),
            log_prob=lp.astype(np.float32),
            step_size=step.astype(np.float32),
            truth=np.asarray([args.mu, args.phi, args.sigma_eta], dtype=np.float32),
            elapsed_s=np.asarray(elapsed_chain, dtype=np.float32),
            chain_id=np.asarray(c, dtype=np.int32),
        )
        print(f"    chain {c+1} done: {elapsed_chain:.1f}s, "
              f"accept rate={acc.mean():.3f}, final eps={float(step[-1]):.5f}")
        print(f"    saved {chain_path}")
        return 0

    # ---- Run chains sequentially ----
    all_samples = []
    all_accs = []
    all_lps = []
    all_step_sizes = []
    t_total = time.perf_counter()
    for c in range(args.num_chains):
        print(f"\n  [chain {c+1}/{args.num_chains}] init_raw={init_raws[c].tolist()}")
        print(f"    init_constrained={constrain(init_raws[c]).tolist()}")
        t0 = time.perf_counter()
        samples, acc, lp, step = run_chain(
            target_log_prob_fn=target_log_prob_fn,
            init_raw=init_raws[c],
            num_results=args.num_results,
            num_burnin=args.num_burnin,
            step_size=args.step_size,
            num_leapfrog=args.L,
            seed=chain_seeds[c],
            progress_every=args.progress_every,
            use_windowed_adaptive=args.use_windowed_adaptive,
            dense_mass=(not args.diagonal_mass),
        )
        elapsed_chain = time.perf_counter() - t0
        accept_rate = float(np.mean(acc))
        print(f"    chain {c+1} done in {elapsed_chain:.1f}s  "
              f"accept={accept_rate:.3f}  final_step_size={float(step[-1]):.4f}")
        all_samples.append(samples)
        all_accs.append(acc)
        all_lps.append(lp)
        all_step_sizes.append(step)

        # defensive per-chain save into chains/ in the same format as
        # single-chain mode (so --aggregate can read either kind).
        samples_con_one = np.stack([
            samples[:, 0],
            np.tanh(samples[:, 1]),
            np.exp(samples[:, 2]),
        ], axis=-1).astype(np.float32)
        np.savez_compressed(
            chains_dir / f"chain_{c}.npz",
            samples_raw=samples.astype(np.float32),
            samples_constrained=samples_con_one,
            accept=acc.astype(np.float32),
            log_prob=lp.astype(np.float32),
            step_size=step.astype(np.float32),
            truth=np.asarray([args.mu, args.phi, args.sigma_eta], dtype=np.float32),
            elapsed_s=np.asarray(elapsed_chain, dtype=np.float32),
            chain_id=np.asarray(c, dtype=np.int32),
        )

    elapsed = time.perf_counter() - t_total
    samples_raw = np.stack(all_samples, axis=0)
    samples_constrained = np.stack(
        [np.stack([
            samples_raw[c, :, 0],
            np.tanh(samples_raw[c, :, 1]),
            np.exp(samples_raw[c, :, 2]),
        ], axis=-1) for c in range(args.num_chains)],
        axis=0,
    )
    accs = np.stack(all_accs, axis=0)
    lps = np.stack(all_lps, axis=0)

    return _finalize_and_report(
        args, out_dir, samples_raw, samples_constrained,
        accs, lps, all_step_sizes, elapsed,
        ckpt_path=ckpt_path, grad_time=grad_time,
    )


if __name__ == "__main__":
    raise SystemExit(main())
