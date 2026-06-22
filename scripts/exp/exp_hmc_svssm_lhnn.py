"""
exp_hmc_svssm_lhnn.py
=====================
Latent-HNN-accelerated HMC for SVSSM parameter recovery — agreement check
against the vanilla-HMC baseline produced by ``exp_hmc_svssm.py``.

Pipeline:
  1. Generate the SAME T observations as ``exp_hmc_svssm.py`` (matching
     truth, data_seed and T) so the two posteriors target identical data.
  2. Build the SAME target log-posterior (priors, unconstrained
     parameterisation, CRN-frozen DifferentiableLEDH likelihood).
  3. Pilot phase: ``num_pilot_trajectories × pilot_steps_per_trajectory``
     leapfrog steps with real ∇ log π gradients, producing the
     (q, p, dq/dt, dp/dt) training set.
  4. Train the Latent HNN to minimise the Hamilton-residual loss
     (Dhulipala, Che & Shields 2022, Eq. 3.1).
  5. Sample: cheap L-HNN-gradient leapfrog with online error monitoring;
     occasional real-gradient fallback; value-only accept/reject.
  6. Optional comparison (``--baseline_npz``): per-marginal KS test vs
     the vanilla HMC posterior, plus side-by-side histogram plot.

Outputs (under ``--out_dir``):
  svssm_lhnn_results.txt          human-readable summary + grad accounting
  svssm_lhnn_samples.npz          (chains, draws, 3) constrained samples,
                                  raw samples, acceptance, log-probs,
                                  step sizes, real-grad counts
  svssm_lhnn_summary.json         machine-readable rows + config
  svssm_lhnn_traces.png           per-chain traceplots
  svssm_lhnn_marginals.png        posterior histograms with truth
  svssm_lhnn_vs_baseline.png      [if --baseline_npz] overlapping LHNN vs
                                  vanilla histograms; KS p in each title
  svssm_lhnn_comparison.json      [if --baseline_npz] per-param KS results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats as sp_stats

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    DifferentiableLEDHLogLikelihoodSVSSM,
)
from src.filters.bonus.lhnn_hmc_pf import (
    LHNNConfig,
    run_lhnn_hmc_multi_chain,
)
from scripts.exp.exp_hmc_svssm import (
    constrain,
    ess_bulk,
    gen_svssm,
    make_target_log_prob,
    split_rhat,
    summarise_posterior,
    unconstrain,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    plt = None
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# LHNN config + dispersed inits
# ---------------------------------------------------------------------------

def build_lhnn_config(args) -> LHNNConfig:
    return LHNNConfig(
        hidden_units=int(args.hidden_units),
        num_hidden=int(args.num_hidden),
        epochs=int(args.lhnn_epochs),
        lr=float(args.lhnn_lr),
        batch_size=int(args.lhnn_batch_size),
        num_pilot_trajectories=int(args.num_pilot_trajectories),
        pilot_steps_per_trajectory=int(args.pilot_steps_per_trajectory),
        error_threshold=float(args.error_threshold),
        cooldown_steps=int(args.cooldown_steps),
    )


def make_dispersed_inits(truth_raw: np.ndarray, num_chains: int,
                          dispersion: float, base_seed: int) -> tf.Tensor:
    rng = np.random.default_rng(base_seed)
    inits = np.stack([
        truth_raw + dispersion * rng.standard_normal(3).astype(np.float32)
        for _ in range(num_chains)
    ], axis=0)
    return tf.constant(inits, dtype=tf.float32)


# ---------------------------------------------------------------------------
# Comparison vs vanilla HMC baseline
# ---------------------------------------------------------------------------

def _ks_compare(x: np.ndarray, y: np.ndarray):
    ks_stat, ks_p = sp_stats.ks_2samp(x, y)
    return float(ks_stat), float(ks_p)


def compare_with_baseline(lhnn_constrained: np.ndarray,
                            baseline_npz_path: Path,
                            param_names: list[str]) -> tuple[list[dict], np.ndarray]:
    """Per-marginal KS + median/sd comparison vs vanilla HMC samples.

    Both arrays are flattened across (chains, draws) so KS sees one mixture
    sample per side. KS p > 0.05 ⇒ marginals are statistically
    indistinguishable.
    """
    base = np.load(baseline_npz_path)
    base_con = base["samples_constrained"]  # (chains, draws, 3)

    rows = []
    for i, name in enumerate(param_names):
        x = lhnn_constrained[..., i].ravel()
        y = base_con[..., i].ravel()
        ks_stat, ks_p = _ks_compare(x, y)
        m_x, m_y = float(np.median(x)), float(np.median(y))
        s_x, s_y = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
        rows.append({
            "param": name,
            "lhnn_median": m_x,
            "baseline_median": m_y,
            "median_abs_diff": abs(m_x - m_y),
            "lhnn_sd": s_x,
            "baseline_sd": s_y,
            "sd_ratio_lhnn_over_baseline": s_x / max(s_y, 1e-12),
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "ks_indistinguishable_at_p05": ks_p > 0.05,
        })
    return rows, base_con


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_traces(out_path: Path, samples_constrained: np.ndarray,
                truth_arr: np.ndarray, names: list[str], T: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, names)):
        for c in range(samples_constrained.shape[0]):
            ax.plot(samples_constrained[c, :, i], alpha=0.7,
                    label=f"chain {c+1}")
        ax.axhline(truth_arr[i], color="red", linestyle="--", lw=1.2,
                   label="truth")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best", ncol=samples_constrained.shape[0] + 1)
    axes[-1].set_xlabel("iteration (post burn-in)")
    fig.suptitle(f"SVSSM LHNN HMC traces (T={T})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_marginals(out_path: Path, samples_constrained: np.ndarray,
                     truth_arr: np.ndarray, names: list[str], T: int) -> None:
    flat = samples_constrained.reshape(-1, 3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.hist(flat[:, i], bins=40, alpha=0.7, color="darkorange",
                edgecolor="white")
        ax.axvline(truth_arr[i], color="red", linestyle="--", lw=1.2,
                   label=f"truth = {truth_arr[i]:.3f}")
        ax.axvline(flat[:, i].mean(), color="black", linestyle=":", lw=1.2,
                   label=f"post mean = {flat[:, i].mean():.3f}")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle(f"SVSSM LHNN HMC posterior marginals (T={T})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_vs_baseline(out_path: Path, lhnn_constrained: np.ndarray,
                       base_constrained: np.ndarray, truth_arr: np.ndarray,
                       comparison_rows: list[dict], names: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, (ax, name, row) in enumerate(zip(axes, names, comparison_rows)):
        lhnn_flat = lhnn_constrained[..., i].ravel()
        base_flat = base_constrained[..., i].ravel()
        lo = float(min(np.percentile(lhnn_flat, 0.5),
                        np.percentile(base_flat, 0.5)))
        hi = float(max(np.percentile(lhnn_flat, 99.5),
                        np.percentile(base_flat, 99.5)))
        bins = np.linspace(lo, hi, 50)
        ax.hist(base_flat, bins=bins, density=True, alpha=0.55,
                color="steelblue",
                label=f"vanilla HMC (sd={row['baseline_sd']:.3f})")
        ax.hist(lhnn_flat, bins=bins, density=True, alpha=0.55,
                color="darkorange",
                label=f"LHNN HMC (sd={row['lhnn_sd']:.3f})")
        ax.axvline(truth_arr[i], color="red", linestyle="--", lw=1.5,
                   label=f"truth={truth_arr[i]:.3f}")
        verdict = "AGREE" if row["ks_indistinguishable_at_p05"] else "DIFFER"
        ax.set_title(f"{name}  KS p={row['ks_p']:.3f}  [{verdict}]")
        ax.set_xlabel(name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("LHNN HMC vs vanilla HMC — posterior agreement check",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    # SVSSM truth / data
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--phi", type=float, default=0.95)
    p.add_argument("--sigma_eta", type=float, default=0.3)
    p.add_argument("--T", type=int, default=50)
    # Filter
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    # HMC
    p.add_argument("--L", type=int, default=5,
                   help="leapfrog steps per HMC iter (matches vanilla baseline)")
    p.add_argument("--step_size", type=float, default=0.05)
    p.add_argument("--num_chains", type=int, default=2)
    p.add_argument("--num_burnin", type=int, default=100)
    p.add_argument("--num_results", type=int, default=200)
    p.add_argument("--dispersion", type=float, default=0.15)
    p.add_argument("--target_accept_prob", type=float, default=0.45,
                   help="LHNN default (0.45) — stochastic-target optimum, "
                        "lower than vanilla HMC's 0.65")
    p.add_argument("--adapt_step_size", action="store_true",
                   help="Enable Nesterov dual-averaging during burn-in")
    # Seeds (default matches exp_hmc_svssm.py baseline)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    # Priors (default matches exp_hmc_svssm.py baseline run)
    p.add_argument("--prior_mu_loc", type=float, default=0.0)
    p.add_argument("--prior_mu_scale", type=float, default=5.0)
    p.add_argument("--prior_phi_raw_loc", type=float, default=2.0)
    p.add_argument("--prior_phi_raw_scale", type=float, default=0.5)
    p.add_argument("--prior_log_sigma_eta_sq_loc", type=float, default=-2.0)
    p.add_argument("--prior_log_sigma_eta_sq_scale", type=float, default=1.0)
    # L-HNN config
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--num_hidden", type=int, default=3)
    p.add_argument("--lhnn_epochs", type=int, default=1500)
    p.add_argument("--lhnn_lr", type=float, default=1e-3)
    p.add_argument("--lhnn_batch_size", type=int, default=64)
    p.add_argument("--num_pilot_trajectories", type=int, default=20)
    p.add_argument("--pilot_steps_per_trajectory", type=int, default=30)
    p.add_argument("--error_threshold", type=float, default=10.0)
    p.add_argument("--cooldown_steps", type=int, default=10)
    # Output / comparison
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc_lhnn")
    p.add_argument("--baseline_npz", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "svssm_hmc/svssm_hmc_samples.npz",
                   help="Vanilla HMC samples for the agreement check. Pass "
                        "empty string to skip the comparison.")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce LHNN internal verbosity")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[svssm-lhnn] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}")
    print(f"  T={args.T} N={args.N} n_lambda={args.n_lambda} K={args.K} "
          f"L={args.L} step_size={args.step_size}")
    print(f"  num_chains={args.num_chains} burnin={args.num_burnin} "
          f"results={args.num_results} dispersion={args.dispersion}")
    print(f"  LHNN: hidden_units={args.hidden_units}, num_hidden={args.num_hidden}, "
          f"epochs={args.lhnn_epochs}")
    print(f"  pilot: {args.num_pilot_trajectories} traj × "
          f"{args.pilot_steps_per_trajectory} steps")
    print(f"  out_dir={out_dir}")

    truth = {"mu": args.mu, "phi": args.phi, "sigma_eta": args.sigma_eta}

    # --- Data (must match exp_hmc_svssm.py with same data_seed) ---
    y_obs = gen_svssm(args.T, args.mu, args.phi, args.sigma_eta,
                      seed=args.data_seed)
    print(f"  y_obs range: [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")

    # --- Filter ---
    ll = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=args.N, n_lambda=args.n_lambda,
        sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
        grad_window=4, jit_compile=True, integrator="exp",
    )

    # --- Target log-prob (same factory as vanilla baseline) ---
    crn_seed = args.base_seed  # frozen across all evaluations (deterministic target)
    target_log_prob_fn = make_target_log_prob(
        ll, y_obs, crn_seed=crn_seed,
        prior_mu_loc=args.prior_mu_loc,
        prior_mu_scale=args.prior_mu_scale,
        prior_phi_raw_loc=args.prior_phi_raw_loc,
        prior_phi_raw_scale=args.prior_phi_raw_scale,
        prior_log_sigma_eta_sq_loc=args.prior_log_sigma_eta_sq_loc,
        prior_log_sigma_eta_sq_scale=args.prior_log_sigma_eta_sq_scale,
        no_likelihood=False,
    )
    print(f"  priors: "
          f"mu~N({args.prior_mu_loc},{args.prior_mu_scale}^2), "
          f"phi_raw~N({args.prior_phi_raw_loc},{args.prior_phi_raw_scale}^2), "
          f"log_sigma_eta_sq~N({args.prior_log_sigma_eta_sq_loc},"
          f"{args.prior_log_sigma_eta_sq_scale}^2)")

    # --- Dispersed initial states (chain seed identical to vanilla baseline) ---
    truth_raw = unconstrain(args.mu, args.phi, args.sigma_eta ** 2)
    initial_states = make_dispersed_inits(
        truth_raw, args.num_chains, args.dispersion, args.base_seed,
    )
    print(f"  initial_states_constrained:")
    for c in range(args.num_chains):
        print(f"    chain {c+1}: "
              f"{constrain(initial_states[c].numpy()).tolist()}")

    # --- Run L-HNN HMC ---
    cfg = build_lhnn_config(args)
    t_total = time.perf_counter()
    result, _trained = run_lhnn_hmc_multi_chain(
        target_log_prob_fn=target_log_prob_fn,
        initial_states=initial_states,
        num_results=args.num_results,
        num_burnin=args.num_burnin,
        step_size=args.step_size,
        num_leapfrog_steps=args.L,
        target_accept_prob=args.target_accept_prob,
        seed=args.base_seed,
        verbose=(not args.quiet),
        adapt_step_size=args.adapt_step_size,
        lhnn_config=cfg,
        share_crn_across_chains=True,
    )
    elapsed = time.perf_counter() - t_total

    # --- Extract per-chain arrays ---
    samples_raw = result.samples.numpy()                # (chains, draws, 3)
    samples_constrained = np.stack(
        [constrain(samples_raw[c]) for c in range(args.num_chains)], axis=0,
    )
    accs = result.is_accepted.numpy().astype(np.float32)  # (chains, draws)
    lps = result.target_log_probs.numpy()                  # (chains, draws)
    step_arr = result.step_sizes.numpy()                   # (chains, total)

    # Gradient-eval accounting (Dhulipala et al. Table 1)
    training_evals = sum(d.training_gradient_evals
                         for d in result.per_chain_diagnostics)
    sampling_evals = sum(d.sampling_real_gradient_evals
                         for d in result.per_chain_diagnostics)
    total_lhnn_evals = training_evals + sampling_evals
    total_iters = args.num_chains * (args.num_burnin + args.num_results)
    vanilla_evals = total_iters * (args.L + 1)
    saved = vanilla_evals - total_lhnn_evals
    pct_saved = 100.0 * saved / max(vanilla_evals, 1)

    # --- Posterior summary (reuse vanilla's helper) ---
    rows = summarise_posterior(samples_constrained, truth)
    per_chain_acc = [float(a.mean()) for a in accs]

    # --- Save samples + raw arrays ---
    np.savez_compressed(
        out_dir / "svssm_lhnn_samples.npz",
        samples_raw=samples_raw,
        samples_constrained=samples_constrained,
        accept=accs,
        log_prob=lps,
        step_size=step_arr,
        truth=np.asarray([args.mu, args.phi, args.sigma_eta]),
    )
    print(f"  [save] {out_dir / 'svssm_lhnn_samples.npz'}")

    # --- Print summary table ---
    print("\n" + "=" * 110)
    print(f"SVSSM LHNN HMC parameter recovery  (wall: {elapsed:.1f}s)")
    print("=" * 110)
    print(f"{'param':<22s} {'truth':>10s} {'mean':>10s} {'std':>10s} {'median':>10s} "
          f"{'2.5%':>10s} {'97.5%':>10s} {'Rhat':>8s} {'ESS':>8s} {'bias%':>8s} cov")
    for r in rows:
        ok = "OK" if r["covered_95ci"] else "OUT"
        print(f"{r['param']:<22s} {r['truth']:>10.4f} {r['mean']:>10.4f} "
              f"{r['std']:>10.4f} {r['median']:>10.4f} {r['q025']:>10.4f} "
              f"{r['q975']:>10.4f} {r['rhat']:>8.3f} {r['ess_bulk']:>8.1f} "
              f"{r['bias_pct']:>+7.1f}% {ok:>3s}")
    print(f"\nAccept rate: per-chain={per_chain_acc}; "
          f"overall={float(accs.mean()):.3f}")
    print("=" * 110)
    print("LHNN gradient accounting:")
    print(f"  Training (pilot)  : {training_evals:>8d} real ∇log π evals")
    print(f"  Sampling fallback : {sampling_evals:>8d} real ∇log π evals")
    print(f"  Total LHNN cost   : {total_lhnn_evals:>8d} real ∇log π evals")
    print(f"  Vanilla HMC cost  : {vanilla_evals:>8d} = "
          f"{args.num_chains}*(L+1)*({args.num_burnin}+{args.num_results})")
    print(f"  Saved             : {saved:>8d} evals ({pct_saved:.1f}%)")
    print("=" * 110)

    # --- Optional: KS agreement check vs vanilla baseline ---
    names = ["mu", "phi", "sigma_eta_sq"]
    truth_arr = np.asarray([args.mu, args.phi, args.sigma_eta ** 2])
    comparison_rows: list[dict] = []
    base_con: np.ndarray | None = None
    if args.baseline_npz:
        bp = Path(args.baseline_npz)
        if bp.exists():
            comparison_rows, base_con = compare_with_baseline(
                samples_constrained, bp, names,
            )
            print("\nAgreement vs vanilla HMC baseline (KS_p>0.05 ⇒ agree):")
            print(f"{'param':<16s} {'lhnn_med':>10s} {'base_med':>10s} "
                  f"{'|Δmed|':>10s} {'sd_ratio':>10s} {'KS_p':>10s}  verdict")
            for row in comparison_rows:
                v = "AGREE" if row["ks_indistinguishable_at_p05"] else "DIFFER"
                print(f"{row['param']:<16s} {row['lhnn_median']:>10.4f} "
                      f"{row['baseline_median']:>10.4f} "
                      f"{row['median_abs_diff']:>10.4f} "
                      f"{row['sd_ratio_lhnn_over_baseline']:>10.3f} "
                      f"{row['ks_p']:>10.4f}  {v}")

            with open(out_dir / "svssm_lhnn_comparison.json", "w") as f:
                json.dump({
                    "baseline_npz": str(bp),
                    "comparison": comparison_rows,
                }, f, indent=2)
            print(f"  [save] {out_dir / 'svssm_lhnn_comparison.json'}")
        else:
            print(f"\n[warn] baseline_npz not found at {bp} — skipping comparison")

    # --- Plots ---
    if _HAVE_MPL:
        plot_traces(out_dir / "svssm_lhnn_traces.png",
                    samples_constrained, truth_arr, names, args.T)
        plot_marginals(out_dir / "svssm_lhnn_marginals.png",
                        samples_constrained, truth_arr, names, args.T)
        print(f"  [save] {out_dir / 'svssm_lhnn_traces.png'}")
        print(f"  [save] {out_dir / 'svssm_lhnn_marginals.png'}")
        if comparison_rows and base_con is not None:
            plot_vs_baseline(out_dir / "svssm_lhnn_vs_baseline.png",
                              samples_constrained, base_con, truth_arr,
                              comparison_rows, names)
            print(f"  [save] {out_dir / 'svssm_lhnn_vs_baseline.png'}")

    # --- Text + JSON report ---
    text_lines = [
        "=" * 110,
        "SVSSM LHNN HMC Parameter Recovery Report",
        "=" * 110,
        f"TF {tf.__version__}, TFP {tfp.__version__}",
        f"truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}",
        f"T={args.T} N={args.N} n_lambda={args.n_lambda} K={args.K} L={args.L} "
        f"step_size={args.step_size} dispersion={args.dispersion}",
        f"num_chains={args.num_chains} burnin={args.num_burnin} "
        f"results={args.num_results}",
        f"Wall: {elapsed:.1f}s",
        "",
        f"LHNN config: hidden_units={args.hidden_units}, "
        f"num_hidden={args.num_hidden}, epochs={args.lhnn_epochs}, "
        f"pilot=({args.num_pilot_trajectories},"
        f"{args.pilot_steps_per_trajectory})",
        f"  error_threshold={args.error_threshold}, "
        f"cooldown_steps={args.cooldown_steps}, "
        f"target_accept={args.target_accept_prob}",
        "",
        f"{'param':<22s} {'truth':>10s} {'mean':>10s} {'std':>10s} "
        f"{'median':>10s} {'2.5%':>10s} {'97.5%':>10s} {'Rhat':>8s} "
        f"{'ESS':>8s} {'bias%':>8s}",
        "-" * 120,
    ]
    for r in rows:
        text_lines.append(
            f"{r['param']:<22s} {r['truth']:>10.4f} {r['mean']:>10.4f} "
            f"{r['std']:>10.4f} {r['median']:>10.4f} {r['q025']:>10.4f} "
            f"{r['q975']:>10.4f} {r['rhat']:>8.3f} {r['ess_bulk']:>8.1f} "
            f"{r['bias_pct']:>+7.1f}%"
        )
    text_lines += [
        "",
        f"Accept rate per chain: {per_chain_acc}",
        f"Overall accept rate:   {float(accs.mean()):.3f}",
        "",
        "Gradient-eval accounting:",
        f"  Training (pilot)   : {training_evals}",
        f"  Sampling fallback  : {sampling_evals}",
        f"  Total LHNN         : {total_lhnn_evals}",
        f"  Vanilla HMC equiv  : {vanilla_evals} "
        f"(= chains * (L+1) * (burnin+results))",
        f"  Saved              : {saved} ({pct_saved:.1f}%)",
        "",
    ]
    if comparison_rows:
        text_lines += [
            "Agreement vs vanilla HMC baseline (KS p>0.05 ⇒ marginals "
            "statistically indistinguishable):",
            f"{'param':<16s} {'lhnn_med':>10s} {'base_med':>10s} "
            f"{'|Δmed|':>10s} {'sd_ratio':>10s} {'KS_p':>10s}  verdict",
        ]
        for row in comparison_rows:
            v = "AGREE" if row["ks_indistinguishable_at_p05"] else "DIFFER"
            text_lines.append(
                f"{row['param']:<16s} {row['lhnn_median']:>10.4f} "
                f"{row['baseline_median']:>10.4f} "
                f"{row['median_abs_diff']:>10.4f} "
                f"{row['sd_ratio_lhnn_over_baseline']:>10.3f} "
                f"{row['ks_p']:>10.4f}  {v}"
            )
        text_lines += [""]
    text_lines += [
        "Notes:",
        " - Same data + target + priors as exp_hmc_svssm.py at this base_seed.",
        " - LHNN cost is dominated by the one-off pilot phase; sampling-",
        "   phase gradient evals are typically <5% of vanilla HMC's.",
        " - Posterior shape MUST match vanilla HMC (the target is identical).",
        "   Any disagreement is a kernel-quality issue (under-trained L-HNN,",
        "   too-loose error_threshold, or too-short chain).",
        "=" * 110,
    ]
    (out_dir / "svssm_lhnn_results.txt").write_text("\n".join(text_lines))
    print(f"  [save] {out_dir / 'svssm_lhnn_results.txt'}")

    (out_dir / "svssm_lhnn_summary.json").write_text(json.dumps({
        "tf": tf.__version__, "tfp": tfp.__version__,
        "config": vars(args), "truth": truth, "rows": rows,
        "accept_rate_per_chain": per_chain_acc,
        "accept_rate_overall": float(accs.mean()),
        "elapsed_s": elapsed,
        "training_gradient_evals": training_evals,
        "sampling_gradient_evals": sampling_evals,
        "total_lhnn_gradient_evals": total_lhnn_evals,
        "vanilla_hmc_equivalent_gradient_evals": vanilla_evals,
        "fraction_saved": float(saved) / max(vanilla_evals, 1),
        "comparison": comparison_rows,
    }, indent=2))
    print(f"  [save] {out_dir / 'svssm_lhnn_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
