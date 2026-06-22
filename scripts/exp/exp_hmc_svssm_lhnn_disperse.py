"""
exp_hmc_svssm_lhnn_disperse.py
==============================
L-HNN HMC for SVSSM with a DISPERSE-INIT pilot phase.

Tests the fix hypothesis from section3 §L-HNN: at wide-shifted priors
and T=200, the single-anchor pilot chains 60 trajectories from a tight
cylinder around truth, leaving large portions of the prior support
uncovered. Sampling chains drift into those untrained regions and the
surrogate gradient becomes a biased extrapolation. The result was
4-chain rank-Rhat 1.43-1.75 (vs vanilla v7's 1.008-1.028 PASS).

This script replaces the single-anchor pilot with K disperse anchors
sampled around truth_raw with a configurable dispersion. From each
anchor, we run num_traj_per_anchor trajectories of steps_per_trajectory
leapfrog steps, then concatenate all the (q, p, dq/dt, dp/dt) tuples
into a single L-HNN training set. Total grad-eval count is held
approximately constant against the baseline single-anchor pilot for an
apples-to-apples test of the coverage hypothesis.

Pipeline:
  1. Sample K disperse anchor states from N(truth_raw, dispersion).
  2. From each anchor, call generate_training_data(...).
  3. Concatenate all (q, p, dq, dp) tensors.
  4. Train a LatentHNN on the combined data.
  5. Call run_lhnn_hmc_multi_chain with the pretrained L-HNN.
  6. Save samples and (if --baseline_npz set) auto-compare.

Outputs (under --out_dir):
  svssm_lhnn_samples.npz
  svssm_lhnn_summary.json
  svssm_lhnn_results.txt
  svssm_lhnn_traces.png
  svssm_lhnn_marginals.png
  svssm_lhnn_vs_baseline.png       (if --baseline_npz set)
  svssm_lhnn_comparison.json       (if --baseline_npz set)
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
    LatentHNN,
    generate_training_data,
    train_lhnn,
    run_lhnn_hmc_multi_chain,
)
from scripts.exp.exp_hmc_svssm import (
    constrain,
    gen_svssm,
    make_target_log_prob,
    summarise_posterior,
    unconstrain,
)
from scripts.exp.exp_hmc_svssm_lhnn import (
    build_lhnn_config,
    compare_with_baseline,
    make_dispersed_inits,
    plot_marginals,
    plot_traces,
    plot_vs_baseline,
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
# Disperse-anchor pilot
# ---------------------------------------------------------------------------

def sample_disperse_anchors(truth_raw: np.ndarray, num_anchors: int,
                              dispersion: float, seed: int) -> list[np.ndarray]:
    """Sample num_anchors disperse anchor states around truth_raw.

    Each anchor = truth_raw + dispersion * N(0, I). dispersion=1.5 in
    unconstrained space corresponds to roughly:
      mu      anchor offset      [-3, +3]   (truth 0)
      phi_raw  anchor offset      [-3, +3]  → phi in [-0.995, +0.995]
      log s²  anchor offset      [-3, +3]   → sigma² in [0.005, 1.81] around truth
    Broad enough to cover the basins the original LHNN T=200 run got
    stuck in (mu basins at -0.9, +1.3, -0.5, +0.2).
    """
    rng = np.random.default_rng(seed)
    return [truth_raw + dispersion * rng.standard_normal(len(truth_raw)).astype(np.float32)
            for _ in range(num_anchors)]


def generate_pilot_disperse(target_log_prob_fn, anchors, traj_per_anchor,
                              steps_per_traj, step_size, base_seed,
                              verbose=True):
    """Generate L-HNN training data from a list of disperse anchor states.

    For each anchor, call generate_training_data and concatenate the
    resulting (q, p, dq, dp) tensors. Returns the combined dataset and
    total grad-eval count.
    """
    q_list, p_list, dq_list, dp_list = [], [], [], []
    total_evals = 0
    for k, anchor in enumerate(anchors):
        anchor_t = tf.constant(anchor, dtype=tf.float32)
        anchor_seed = int(base_seed) + 10_007 * (k + 1)
        if verbose:
            print(f"\n  === Pilot anchor {k+1}/{len(anchors)} "
                  f"@ raw={anchor.tolist()} ===", flush=True)
        q, p, dq, dp, n = generate_training_data(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=anchor_t,
            num_trajectories=int(traj_per_anchor),
            steps_per_trajectory=int(steps_per_traj),
            step_size=float(step_size),
            seed=anchor_seed,
            verbose=verbose,
        )
        q_list.append(q)
        p_list.append(p)
        dq_list.append(dq)
        dp_list.append(dp)
        total_evals += int(n)
    q_all = tf.concat(q_list, axis=0)
    p_all = tf.concat(p_list, axis=0)
    dq_all = tf.concat(dq_list, axis=0)
    dp_all = tf.concat(dp_list, axis=0)
    if verbose:
        print(f"\n  Combined training data: {int(q_all.shape[0])} points "
              f"from {len(anchors)} anchors  ({total_evals} grad evals)",
              flush=True)
    return q_all, p_all, dq_all, dp_all, total_evals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    # SVSSM truth + data
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--phi", type=float, default=0.95)
    p.add_argument("--sigma_eta", type=float, default=0.3)
    p.add_argument("--T", type=int, default=200)
    # Filter
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    # HMC
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--step_size", type=float, default=0.01)
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--num_burnin", type=int, default=200)
    p.add_argument("--num_results", type=int, default=2500)
    p.add_argument("--dispersion", type=float, default=0.05,
                   help="chain init dispersion (around truth_raw)")
    p.add_argument("--target_accept_prob", type=float, default=0.45)
    p.add_argument("--adapt_step_size", action="store_true", default=True)
    # Seeds (match vanilla T=200 sweep)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    # Priors (default = T=200 wide-shifted sweep)
    p.add_argument("--prior_mu_loc", type=float, default=2.0)
    p.add_argument("--prior_mu_scale", type=float, default=3.0)
    p.add_argument("--prior_phi_raw_loc", type=float, default=0.0)
    p.add_argument("--prior_phi_raw_scale", type=float, default=2.0)
    p.add_argument("--prior_log_sigma_eta_sq_loc", type=float, default=1.5)
    p.add_argument("--prior_log_sigma_eta_sq_scale", type=float, default=3.0)
    # L-HNN architecture + training
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--num_hidden", type=int, default=3)
    p.add_argument("--lhnn_epochs", type=int, default=2500)
    p.add_argument("--lhnn_lr", type=float, default=1e-3)
    p.add_argument("--lhnn_batch_size", type=int, default=64)
    p.add_argument("--error_threshold", type=float, default=10.0)
    p.add_argument("--cooldown_steps", type=int, default=10)
    # Disperse-anchor pilot config
    p.add_argument("--num_pilot_anchors", type=int, default=8,
                   help="K disperse anchor states sampled around truth_raw")
    p.add_argument("--anchor_dispersion", type=float, default=1.5,
                   help="sd of N(truth_raw, ·) used to sample anchors")
    p.add_argument("--pilot_traj_per_anchor", type=int, default=8)
    p.add_argument("--pilot_steps_per_trajectory", type=int, default=60)
    # Output
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "svssm_hmc_lhnn_T200_disperse")
    p.add_argument("--baseline_npz", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "new/svssm_hmc_sweep_wide_T200/svssm_hmc_samples.npz",
                   help="Vanilla T=200 samples for the agreement check.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[svssm-lhnn-disperse] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}")
    print(f"  T={args.T} N={args.N} n_lambda={args.n_lambda} K={args.K} "
          f"L={args.L} step_size={args.step_size}")
    print(f"  num_chains={args.num_chains} burnin={args.num_burnin} "
          f"results={args.num_results} dispersion={args.dispersion}")
    print(f"  DISPERSE PILOT: {args.num_pilot_anchors} anchors × "
          f"{args.pilot_traj_per_anchor} traj × "
          f"{args.pilot_steps_per_trajectory} steps "
          f"= {args.num_pilot_anchors * args.pilot_traj_per_anchor * (args.pilot_steps_per_trajectory + 1)} "
          f"grad evals total")
    print(f"  anchor_dispersion={args.anchor_dispersion} (in unconstrained space)")
    print(f"  LHNN: hidden_units={args.hidden_units}, "
          f"num_hidden={args.num_hidden}, epochs={args.lhnn_epochs}")
    print(f"  out_dir={out_dir}")

    truth = {"mu": args.mu, "phi": args.phi, "sigma_eta": args.sigma_eta}

    # --- Data + target (same as exp_hmc_svssm_lhnn.py) ---
    y_obs = gen_svssm(args.T, args.mu, args.phi, args.sigma_eta,
                      seed=args.data_seed)
    print(f"  y_obs range: [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")

    ll = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=args.N, n_lambda=args.n_lambda,
        sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
        grad_window=4, jit_compile=True, integrator="exp",
    )

    crn_seed = args.base_seed
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

    # --- Sample disperse pilot anchors around truth_raw ---
    truth_raw = unconstrain(args.mu, args.phi, args.sigma_eta ** 2)
    anchors = sample_disperse_anchors(
        truth_raw, args.num_pilot_anchors, args.anchor_dispersion,
        args.base_seed,
    )
    print(f"\n  Pilot anchors (unconstrained):")
    for k, a in enumerate(anchors):
        print(f"    {k+1}: {a.tolist()}  -> constrained "
              f"{constrain(a).tolist()}")

    # --- Stage 1: pilot data generation across all anchors ---
    print(f"\n[stage 1/4] Generating disperse-anchor pilot data...")
    t_pilot = time.perf_counter()
    q_data, p_data, dq_data, dp_data, training_evals = generate_pilot_disperse(
        target_log_prob_fn=target_log_prob_fn,
        anchors=anchors,
        traj_per_anchor=args.pilot_traj_per_anchor,
        steps_per_traj=args.pilot_steps_per_trajectory,
        step_size=args.step_size,
        base_seed=args.base_seed,
        verbose=(not args.quiet),
    )
    pilot_wall = time.perf_counter() - t_pilot
    print(f"  [stage 1 done] {training_evals} grad evals in {pilot_wall:.1f}s")

    # --- Stage 2: build + train LatentHNN on combined data ---
    print(f"\n[stage 2/4] Training LatentHNN on {int(q_data.shape[0])} "
          f"data points for {args.lhnn_epochs} epochs...")
    t_train = time.perf_counter()
    d = int(truth_raw.shape[0])
    lhnn = LatentHNN(d, args.hidden_units, args.num_hidden)
    _ = lhnn(tf.zeros([1, 2 * d]))  # trigger weight creation
    train_lhnn(
        lhnn=lhnn,
        q_data=q_data, p_data=p_data,
        dq_dt=dq_data, dp_dt=dp_data,
        epochs=args.lhnn_epochs,
        lr=args.lhnn_lr,
        batch_size=args.lhnn_batch_size,
        verbose=(not args.quiet),
    )
    train_wall = time.perf_counter() - t_train
    print(f"  [stage 2 done] training wall: {train_wall:.1f}s")

    # --- Stage 3: sample dispersed chain inits + run multi-chain HMC ---
    print(f"\n[stage 3/4] Running {args.num_chains}-chain HMC "
          f"with pretrained L-HNN...")
    initial_states = make_dispersed_inits(
        truth_raw, args.num_chains, args.dispersion, args.base_seed,
    )
    print(f"  initial_states_constrained:")
    for c in range(args.num_chains):
        print(f"    chain {c+1}: "
              f"{constrain(initial_states[c].numpy()).tolist()}")

    cfg = LHNNConfig(
        hidden_units=args.hidden_units, num_hidden=args.num_hidden,
        epochs=args.lhnn_epochs, lr=args.lhnn_lr,
        batch_size=args.lhnn_batch_size,
        num_pilot_trajectories=0,   # not used (pretrained)
        pilot_steps_per_trajectory=0,
        error_threshold=args.error_threshold,
        cooldown_steps=args.cooldown_steps,
    )

    t_sample = time.perf_counter()
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
        pretrained_lhnn=lhnn,
        share_crn_across_chains=True,
    )
    sample_wall = time.perf_counter() - t_sample
    total_wall = pilot_wall + train_wall + sample_wall

    # --- Extract per-chain arrays ---
    samples_raw = result.samples.numpy()
    samples_constrained = np.stack(
        [constrain(samples_raw[c]) for c in range(args.num_chains)], axis=0,
    )
    accs = result.is_accepted.numpy().astype(np.float32)
    lps = result.target_log_probs.numpy()
    step_arr = result.step_sizes.numpy()
    per_chain_acc = [float(a.mean()) for a in accs]

    # Gradient accounting
    sampling_evals = sum(d.sampling_real_gradient_evals
                          for d in result.per_chain_diagnostics)
    total_lhnn_evals = training_evals + sampling_evals
    total_iters = args.num_chains * (args.num_burnin + args.num_results)
    vanilla_evals = total_iters * (args.L + 1)
    saved = vanilla_evals - total_lhnn_evals
    pct_saved = 100.0 * saved / max(vanilla_evals, 1)

    rows = summarise_posterior(samples_constrained, truth)

    # --- Save samples ---
    np.savez_compressed(
        out_dir / "svssm_lhnn_samples.npz",
        samples_raw=samples_raw,
        samples_constrained=samples_constrained,
        accept=accs,
        log_prob=lps,
        step_size=step_arr,
        truth=np.asarray([args.mu, args.phi, args.sigma_eta]),
    )
    print(f"\n  [save] {out_dir / 'svssm_lhnn_samples.npz'}")

    # --- Print summary ---
    print("\n" + "=" * 110)
    print(f"SVSSM L-HNN HMC (DISPERSE pilot) — recovery  "
          f"(total wall: {total_wall:.1f}s)")
    print(f"  Pilot: {pilot_wall:.1f}s ({training_evals} grad evals across "
          f"{args.num_pilot_anchors} anchors)")
    print(f"  Train: {train_wall:.1f}s ({args.lhnn_epochs} epochs)")
    print(f"  Sample: {sample_wall:.1f}s ({args.num_chains} chains × "
          f"{args.num_burnin + args.num_results} steps)")
    print("=" * 110)
    print(f"{'param':<22s} {'truth':>10s} {'mean':>10s} {'std':>10s} "
          f"{'median':>10s} {'2.5%':>10s} {'97.5%':>10s} "
          f"{'Rhat':>8s} {'ESS':>8s} {'bias%':>8s} cov")
    for r in rows:
        ok = "OK" if r["covered_95ci"] else "OUT"
        print(f"{r['param']:<22s} {r['truth']:>10.4f} {r['mean']:>10.4f} "
              f"{r['std']:>10.4f} {r['median']:>10.4f} {r['q025']:>10.4f} "
              f"{r['q975']:>10.4f} {r['rhat']:>8.3f} {r['ess_bulk']:>8.1f} "
              f"{r['bias_pct']:>+7.1f}% {ok:>3s}")
    print(f"\nAccept per chain: {per_chain_acc}  overall={float(accs.mean()):.3f}")
    print("=" * 110)
    print(f"Gradient accounting:")
    print(f"  Pilot evals (disperse): {training_evals}")
    print(f"  Sampling fallback ev. : {sampling_evals}")
    print(f"  Total LHNN cost       : {total_lhnn_evals}")
    print(f"  Vanilla HMC cost      : {vanilla_evals} "
          f"= {args.num_chains}*(L+1)*({args.num_burnin}+{args.num_results})")
    print(f"  Saved                 : {saved} ({pct_saved:.1f}%)")
    print("=" * 110)

    # --- Stage 4: KS comparison vs vanilla baseline ---
    names = ["mu", "phi", "sigma_eta_sq"]
    truth_arr = np.asarray([args.mu, args.phi, args.sigma_eta ** 2])
    comparison_rows: list[dict] = []
    base_con: np.ndarray | None = None

    if args.baseline_npz:
        bp = Path(args.baseline_npz)
        if bp.exists():
            print(f"\n[stage 4/4] KS agreement vs vanilla baseline at {bp}")
            comparison_rows, base_con = compare_with_baseline(
                samples_constrained, bp, names,
            )
            print(f"{'param':<16s} {'lhnn_med':>10s} {'base_med':>10s} "
                  f"{'|Δmed|':>10s} {'sd_ratio':>10s} {'KS_p':>10s}  verdict")
            for row in comparison_rows:
                v = "AGREE" if row["ks_indistinguishable_at_p05"] else "DIFFER"
                print(f"{row['param']:<16s} {row['lhnn_median']:>10.4f} "
                      f"{row['baseline_median']:>10.4f} "
                      f"{row['median_abs_diff']:>10.4f} "
                      f"{row['sd_ratio_lhnn_over_baseline']:>10.3f} "
                      f"{row['ks_p']:>10.4f}  {v}")
            (out_dir / "svssm_lhnn_comparison.json").write_text(
                json.dumps({
                    "baseline_npz": str(bp),
                    "comparison": comparison_rows,
                }, indent=2)
            )
            print(f"  [save] {out_dir / 'svssm_lhnn_comparison.json'}")
        else:
            print(f"\n[warn] baseline_npz not found at {bp}")

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

    # --- Save text + JSON reports ---
    text_lines = [
        "=" * 110,
        "SVSSM L-HNN HMC (DISPERSE pilot) Recovery Report",
        "=" * 110,
        f"TF {tf.__version__}, TFP {tfp.__version__}",
        f"truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}",
        f"T={args.T} N={args.N} L={args.L} step_size={args.step_size}",
        f"num_chains={args.num_chains} burnin={args.num_burnin} "
        f"results={args.num_results}",
        f"DISPERSE PILOT: K={args.num_pilot_anchors} anchors, "
        f"dispersion={args.anchor_dispersion}, "
        f"traj_per_anchor={args.pilot_traj_per_anchor}, "
        f"steps={args.pilot_steps_per_trajectory}",
        f"Total wall: {total_wall:.1f}s "
        f"(pilot {pilot_wall:.1f}, train {train_wall:.1f}, "
        f"sample {sample_wall:.1f})",
        "",
        f"{'param':<22s} {'truth':>10s} {'mean':>10s} {'std':>10s} "
        f"{'median':>10s} {'2.5%':>10s} {'97.5%':>10s} "
        f"{'Rhat':>8s} {'ESS':>8s} {'bias%':>8s}",
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
        f"Accept per chain: {per_chain_acc}",
        f"Overall accept:   {float(accs.mean()):.3f}",
        "",
        "Gradient-eval accounting:",
        f"  Pilot evals (disperse): {training_evals}",
        f"  Sampling fallback ev. : {sampling_evals}",
        f"  Total LHNN cost       : {total_lhnn_evals}",
        f"  Vanilla HMC cost      : {vanilla_evals}",
        f"  Saved                 : {saved} ({pct_saved:.1f}%)",
        "",
    ]
    if comparison_rows:
        text_lines += [
            "Agreement vs vanilla T=200 baseline:",
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
    text_lines += ["=" * 110]
    (out_dir / "svssm_lhnn_results.txt").write_text(
        "\n".join(text_lines)
    )
    print(f"  [save] {out_dir / 'svssm_lhnn_results.txt'}")

    (out_dir / "svssm_lhnn_summary.json").write_text(json.dumps({
        "tf": tf.__version__, "tfp": tfp.__version__,
        "config": vars(args), "truth": truth, "rows": rows,
        "accept_rate_per_chain": per_chain_acc,
        "accept_rate_overall": float(accs.mean()),
        "elapsed_s": total_wall,
        "pilot_wall_s": pilot_wall,
        "training_wall_s": train_wall,
        "sampling_wall_s": sample_wall,
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
