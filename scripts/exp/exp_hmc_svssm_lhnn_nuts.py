"""
exp_hmc_svssm_lhnn_nuts.py
==========================
NUTS with L-HNN gradient + online error monitor + final-state MH
correction on the T=200 wide-shifted SVSSM target.

Tests the hypothesis from the L-HNN writeup: the T=200 bias we observed
(rank-Rhat 1.75 on mu, chains stuck on different basins) was caused by
pairing the L-HNN with fixed-L=5 leapfrog, which gave the H_theta error
monitor no chance to fire. With NUTS's adaptive tree-doubling
trajectories (up to 2^max_treedepth = 256 leapfrog steps), the monitor
should fire whenever the surrogate is biased, triggering real-gradient
fallback and steering chains away from fake basins.

Pipeline:
  1. Same data, target, priors as the original L-HNN T=200 run.
  2. Single-anchor pilot (same as the original L-HNN T=200 — we test
     the NUTS hypothesis with the LEAST favourable pilot).
  3. Train L-HNN on the pilot data.
  4. Sample with NUTS-with-LHNN: each iter does tree-doubling forward
     OR backward in time, terminates on U-turn OR error threshold
     breach. Final MH correction against REAL target ensures unbiased
     posterior even when the surrogate is wrong.
  5. Compare against vanilla T=200 + the existing biased L-HNN T=200.

Diagnostic counters of interest:
  * total_error_triggers per chain — non-zero means the monitor fired
    (i.e. the surrogate was caught being wrong somewhere).
  * total_real_grad_evals — pilot + cooldown + per-iter MH evaluations.
  * avg_tree_depth — how big the trajectories grew on average.

Outputs (under --out_dir):
  svssm_lhnn_samples.npz                  posterior samples + accept + step + depth
  svssm_lhnn_summary.json                 config + rows + nuts diagnostics
  svssm_lhnn_results.txt                  human-readable
  svssm_lhnn_traces.png, _marginals.png   plots
  svssm_lhnn_vs_baseline.png              [if --baseline_npz] vs vanilla
  svssm_lhnn_comparison.json              [if --baseline_npz]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    DifferentiableLEDHLogLikelihoodSVSSM,
)
from src.filters.bonus.lhnn_hmc_pf import (
    LatentHNN,
    generate_training_data,
    train_lhnn,
)
from src.filters.bonus.lhnn_nuts import run_lhnn_nuts_multi_chain
from scripts.exp.exp_hmc_svssm import (
    constrain,
    gen_svssm,
    make_target_log_prob,
    summarise_posterior,
    unconstrain,
)
from scripts.exp.exp_hmc_svssm_lhnn import (
    compare_with_baseline,
    make_dispersed_inits,
    plot_marginals,
    plot_traces,
    plot_vs_baseline,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


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
    # NUTS HMC
    p.add_argument("--step_size", type=float, default=0.01)
    p.add_argument("--max_treedepth", type=int, default=8,
                   help="2^max_treedepth caps trajectory length. "
                        "Default 8 → up to 256 leapfrog steps.")
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--num_burnin", type=int, default=200)
    p.add_argument("--num_results", type=int, default=2500)
    p.add_argument("--dispersion", type=float, default=0.05)
    p.add_argument("--target_accept_prob", type=float, default=0.65)
    p.add_argument("--adapt_step_size", action="store_true", default=True)
    # Seeds (match vanilla T=200 sweep)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    # Priors (T=200 wide-shifted sweep)
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
    # Pilot (single-anchor, matching original biased run)
    p.add_argument("--num_pilot_trajectories", type=int, default=60)
    p.add_argument("--pilot_steps_per_trajectory", type=int, default=60)
    # Error monitor
    p.add_argument("--error_threshold", type=float, default=10.0)
    p.add_argument("--cooldown_steps", type=int, default=10)
    # I/O
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "svssm_hmc_lhnn_T200_nuts")
    p.add_argument("--baseline_npz", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "new/svssm_hmc_sweep_wide_T200/svssm_hmc_samples.npz",
                   help="Vanilla T=200 samples for the KS comparison.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--progress_every", type=int, default=10)
    # Weight cache: re-use a previously trained L-HNN to skip pilot+train.
    # If the file at this path exists, load it and skip stages 1-2. If not,
    # train normally and save the trained weights here for next time.
    p.add_argument("--weights_cache", type=str, default="",
                   help="Path to .weights.h5 file (Keras format). If it "
                        "exists, load and skip pilot+training. If not, "
                        "train normally and save weights to this path. "
                        "Empty string disables caching (re-train every run).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[svssm-lhnn-nuts] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}")
    print(f"  T={args.T} N={args.N} step_size={args.step_size} "
          f"max_treedepth={args.max_treedepth}")
    print(f"  num_chains={args.num_chains} burnin={args.num_burnin} "
          f"results={args.num_results} dispersion={args.dispersion}")
    print(f"  LHNN: hidden_units={args.hidden_units}, "
          f"num_hidden={args.num_hidden}, epochs={args.lhnn_epochs}")
    print(f"  pilot: single-anchor × {args.num_pilot_trajectories} traj × "
          f"{args.pilot_steps_per_trajectory} steps "
          f"= {args.num_pilot_trajectories * (args.pilot_steps_per_trajectory + 1)} "
          f"pilot grad evals")
    print(f"  error_threshold={args.error_threshold} "
          f"cooldown_steps={args.cooldown_steps}")
    print(f"  out_dir={out_dir}")

    truth = {"mu": args.mu, "phi": args.phi, "sigma_eta": args.sigma_eta}

    # --- Data + target ---
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

    truth_raw = unconstrain(args.mu, args.phi, args.sigma_eta ** 2)
    d = int(truth_raw.shape[0])

    # --- Weight cache check ---
    cache_path = Path(args.weights_cache) if args.weights_cache else None
    use_cache = (cache_path is not None and cache_path.exists())

    if use_cache:
        print(f"\n[cache] loading L-HNN weights from {cache_path} "
              f"(skipping pilot + training)")
        lhnn = LatentHNN(d, args.hidden_units, args.num_hidden)
        _ = lhnn(tf.zeros([1, 2 * d]))
        lhnn.load_weights(str(cache_path))
        pilot_wall = 0.0
        train_wall = 0.0
        training_evals = 0
        print(f"  [cache] L-HNN loaded; pilot=0s training=0s")
    else:
        # --- Stage 1: single-anchor pilot ---
        print(f"\n[stage 1/4] Single-anchor pilot @ truth_raw={truth_raw.tolist()} ...")
        t_pilot = time.perf_counter()
        q_data, p_data, dq_data, dp_data, training_evals = generate_training_data(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=tf.constant(truth_raw, dtype=tf.float32),
            num_trajectories=int(args.num_pilot_trajectories),
            steps_per_trajectory=int(args.pilot_steps_per_trajectory),
            step_size=float(args.step_size),
            seed=int(args.base_seed),
            verbose=(not args.quiet),
        )
        pilot_wall = time.perf_counter() - t_pilot
        print(f"  [stage 1 done] {training_evals} grad evals in {pilot_wall:.1f}s")

        # --- Stage 2: train L-HNN ---
        print(f"\n[stage 2/4] Training LatentHNN on {int(q_data.shape[0])} "
              f"data points for {args.lhnn_epochs} epochs...")
        t_train = time.perf_counter()
        lhnn = LatentHNN(d, args.hidden_units, args.num_hidden)
        _ = lhnn(tf.zeros([1, 2 * d]))
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

        # Save trained weights to cache path if specified
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            lhnn.save_weights(str(cache_path))
            print(f"  [cache] saved L-HNN weights to {cache_path}")

    # --- Stage 3: NUTS-with-LHNN sampling ---
    print(f"\n[stage 3/4] Running {args.num_chains}-chain NUTS "
          f"with pretrained L-HNN (max_treedepth={args.max_treedepth})...")
    initial_states = make_dispersed_inits(
        truth_raw, args.num_chains, args.dispersion, args.base_seed,
    )
    print(f"  initial_states_constrained:")
    for c in range(args.num_chains):
        print(f"    chain {c+1}: "
              f"{constrain(initial_states[c].numpy()).tolist()}")

    t_sample = time.perf_counter()
    result = run_lhnn_nuts_multi_chain(
        target_log_prob_fn=target_log_prob_fn,
        initial_states=initial_states,
        lhnn=lhnn,
        num_results=args.num_results,
        num_burnin=args.num_burnin,
        step_size=args.step_size,
        max_treedepth=args.max_treedepth,
        error_threshold=args.error_threshold,
        cooldown_steps=args.cooldown_steps,
        adapt_step_size=args.adapt_step_size,
        target_accept_prob=args.target_accept_prob,
        seed=args.base_seed,
        verbose=(not args.quiet),
        share_crn_across_chains=True,
        progress_every=args.progress_every,
    )
    sample_wall = time.perf_counter() - t_sample
    total_wall = pilot_wall + train_wall + sample_wall

    # --- Extract arrays ---
    samples_raw = result.samples.numpy()
    samples_constrained = np.stack(
        [constrain(samples_raw[c]) for c in range(args.num_chains)], axis=0,
    )
    accs = result.is_accepted.numpy().astype(np.float32)
    lps = result.target_log_probs.numpy()
    step_arr = result.step_sizes.numpy()
    depth_arr = result.tree_depths.numpy()
    per_chain_acc = [float(a.mean()) for a in accs]

    # NUTS-specific diagnostics
    error_triggers_per_chain = [d.total_error_triggers
                                  for d in result.per_chain_diagnostics]
    real_grads_per_chain = [d.total_real_grad_evals
                              for d in result.per_chain_diagnostics]
    lhnn_steps_per_chain = [d.total_lhnn_leapfrog_evals
                              for d in result.per_chain_diagnostics]
    avg_depth_per_chain = [float(d.avg_tree_depth.numpy())
                             for d in result.per_chain_diagnostics]

    # Pilot evals + per-iter MH evals + cooldown fallbacks
    total_real_grad_evals = training_evals + sum(real_grads_per_chain)
    total_iters = args.num_chains * (args.num_burnin + args.num_results)
    # For "vanilla equivalent", count what a full L=5 vanilla HMC would have used
    vanilla_evals = total_iters * (5 + 1)
    saved = vanilla_evals - total_real_grad_evals
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
        tree_depth=depth_arr,
        truth=np.asarray([args.mu, args.phi, args.sigma_eta]),
    )
    print(f"\n  [save] {out_dir / 'svssm_lhnn_samples.npz'}")

    # --- Print summary ---
    print("\n" + "=" * 110)
    print(f"SVSSM L-HNN NUTS — recovery  (total wall: {total_wall:.1f}s)")
    print(f"  Pilot: {pilot_wall:.1f}s ({training_evals} grad evals)")
    print(f"  Train: {train_wall:.1f}s ({args.lhnn_epochs} epochs)")
    print(f"  Sample: {sample_wall:.1f}s ({args.num_chains} chains × "
          f"{args.num_burnin + args.num_results} NUTS iters)")
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
    print(f"NUTS-specific diagnostics:")
    print(f"  Error triggers per chain : {error_triggers_per_chain}")
    print(f"  Real grad evals per chain: {real_grads_per_chain}")
    print(f"  LHNN leapfrog steps/chain: {lhnn_steps_per_chain}")
    print(f"  Avg tree depth per chain : {[f'{x:.2f}' for x in avg_depth_per_chain]}")
    print("=" * 110)
    print(f"Gradient accounting (across all 4 chains):")
    print(f"  Pilot evals               : {training_evals}")
    print(f"  Sum of per-chain real grads (MH + cooldown): "
          f"{sum(real_grads_per_chain)}")
    print(f"  Total real grad evals     : {total_real_grad_evals}")
    print(f"  Vanilla HMC L=5 equivalent: {vanilla_evals}")
    print(f"  Saved                     : {saved} ({pct_saved:.1f}%)")
    print("=" * 110)

    # --- Stage 4: KS vs vanilla baseline ---
    names = ["mu", "phi", "sigma_eta_sq"]
    truth_arr = np.asarray([args.mu, args.phi, args.sigma_eta ** 2])
    comparison_rows = []
    base_con = None

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
                json.dumps({"baseline_npz": str(bp),
                              "comparison": comparison_rows}, indent=2)
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
        "SVSSM L-HNN NUTS Recovery Report",
        "=" * 110,
        f"TF {tf.__version__}, TFP {tfp.__version__}",
        f"truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}",
        f"T={args.T} N={args.N} step_size={args.step_size} "
        f"max_treedepth={args.max_treedepth}",
        f"num_chains={args.num_chains} burnin={args.num_burnin} "
        f"results={args.num_results}",
        f"Pilot: {args.num_pilot_trajectories} traj × "
        f"{args.pilot_steps_per_trajectory} steps (single anchor)",
        f"Total wall: {total_wall:.1f}s "
        f"(pilot {pilot_wall:.1f}, train {train_wall:.1f}, "
        f"sample {sample_wall:.1f})",
        "",
        f"NUTS diagnostics:",
        f"  Error triggers per chain : {error_triggers_per_chain}",
        f"  Real grad evals per chain: {real_grads_per_chain}",
        f"  LHNN leapfrog steps/chain: {lhnn_steps_per_chain}",
        f"  Avg tree depth per chain : {[f'{x:.2f}' for x in avg_depth_per_chain]}",
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
        f"  Pilot evals               : {training_evals}",
        f"  Sum chain real grads      : {sum(real_grads_per_chain)}",
        f"  Total real grad evals     : {total_real_grad_evals}",
        f"  Vanilla HMC L=5 equivalent: {vanilla_evals}",
        f"  Saved                     : {saved} ({pct_saved:.1f}%)",
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
    (out_dir / "svssm_lhnn_results.txt").write_text("\n".join(text_lines))
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
        "training_gradient_evals": int(training_evals),
        "sampling_gradient_evals": int(sum(real_grads_per_chain)),
        "total_lhnn_gradient_evals": int(total_real_grad_evals),
        "vanilla_hmc_equivalent_gradient_evals": int(vanilla_evals),
        "fraction_saved": float(saved) / max(vanilla_evals, 1),
        "nuts_diagnostics": {
            "error_triggers_per_chain": error_triggers_per_chain,
            "real_grads_per_chain": real_grads_per_chain,
            "lhnn_steps_per_chain": lhnn_steps_per_chain,
            "avg_depth_per_chain": avg_depth_per_chain,
        },
        "comparison": comparison_rows,
    }, indent=2))
    print(f"  [save] {out_dir / 'svssm_lhnn_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
