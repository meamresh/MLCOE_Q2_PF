"""
exp_hmc_svssm_multivariate_full_phi_lhnn_nuts.py
================================================
L-HNN NUTS parameter recovery for the V1 multivariate SVSSM with an
UPPER-TRIANGULAR Phi (cross-asset persistence) + diagonal Sigma_eta.

This is the multivariate sister of `exp_hmc_svssm_lhnn_nuts.py` (which is
univariate, d=3 parameters) and a drop-in NUTS replacement for the
fixed-L windowed-HMC driver `exp_hmc_svssm_multivariate_full_phi.py`.

Why this script exists
----------------------
At d>=3 (e.g. d=4, an 18-parameter target) fixed-L windowed HMC hits its
preconditioning ceiling: the d=4 T=200 Sinkhorn AND NN-OT runs both
failed to mix (rank-Rhat 1.5-3.4, ESS ~25-33, chains parked on different
basins of the mu/sigma^2 ridge). The failure was sampler-geometry, not
operator quality -- Sinkhorn (exact OT) failed just as badly as NN-OT.

The L-HNN writeup's interview rule: *fixed-L HMC when you can precondition
(real grad + dense mass); NUTS when you can't.* The 18-D mu/sigma^2 ridge
is exactly the "can't precondition with a single dense mass" regime, so
NUTS's adaptive tree-doubling is the right tool. We reuse the L-HNN NUTS
engine (surrogate gradient + online H-error monitor + final-state MH
correction against the REAL particle-filter target) verbatim -- it is
dimension-generic (d = initial_state.shape[0]).

Pipeline
--------
  1. Same data / target / priors as the full-Phi windowed driver.
  2. Single-anchor pilot at truth_raw with the REAL target gradient.
  3. Train LatentHNN on the (3d + d(d-1)/2)-D phase space.
  4. NUTS-with-LHNN multi-chain sampling. Final MH correction uses the
     real ll.call_mat_phi target -> unbiased even if the surrogate is off.
  5. Save in the EXACT full-Phi npz format so the existing d-generic
     tooling works unchanged:
       - scripts/save_diagnostics_multi_full_phi.py
       - scripts/plot_trace_multi_full_phi.py
       - scripts/plot_trace_stationary_cov.py
       - scripts/convert_full_phi_npz_for_aggregate.py

Outputs (under --out_dir), full-Phi-compatible names:
  svssm_hmc_multi_full_phi_samples.npz     mu/phi_diag/phi_off/sigma_eta_sq
                                           + *_truth + Phi_truth + accept
                                           + log_prob + NUTS diagnostics
  svssm_hmc_multi_full_phi_summary.json    config + rows + nuts diagnostics
  (+ diagnostics .txt/.json via save_diagnostics_multi_full_phi.py)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
)
from src.filters.bonus.lhnn_hmc_pf import (
    LatentHNN,
    generate_training_data,
    train_lhnn,
)
from src.filters.bonus.lhnn_nuts import run_lhnn_nuts, run_lhnn_nuts_multi_chain
from scripts.exp.exp_hmc_svssm_multivariate_full_phi import (
    build_phi_matrix_np,
    constrain,
    gen_svssm_multi_phi_mat,
    make_target_log_prob,
    num_off_diag,
    unconstrain,
    upper_tri_indices,
)
from scripts.exp.compare_svssm_hmc_methods import (
    rank_rhat as _rank_rhat,
    bulk_ess as _bulk_ess,
    tail_ess as _tail_ess,
)


def main() -> int:
    p = argparse.ArgumentParser()
    # ---- model / data (mirror full-Phi driver) ----
    p.add_argument("-d", "--d", type=int, default=4, dest="d", metavar="D")
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--mu", type=str, default="0.0,-0.3,0.2,-0.1")
    p.add_argument("--phi_diag", type=str, default="0.95,0.85,0.90,0.80")
    p.add_argument("--phi_off", type=str, default="0.05,0.0,0.0,0.05,0.0,0.05",
                   help="comma-separated upper off-diagonals, row-major "
                        "(0,1),(0,2),...; length d(d-1)/2. Empty '' for d=1.")
    p.add_argument("--sigma_eta", type=str, default="0.3,0.4,0.35,0.3")
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    # ---- priors (mirror full-Phi driver) ----
    p.add_argument("--prior_mu_loc", type=float, default=-0.2)
    p.add_argument("--prior_mu_scale", type=float, default=1.0)
    p.add_argument("--prior_phi_raw_loc", type=float, default=1.5)
    p.add_argument("--prior_phi_raw_scale", type=float, default=1.0)
    p.add_argument("--prior_log_sigma_eta_sq_loc", type=float, default=-2.0)
    p.add_argument("--prior_log_sigma_eta_sq_scale", type=float, default=1.0)
    p.add_argument("--prior_phi_off_scale", type=float, default=0.5)
    # ---- filter numerical guards (full-Phi LEDH flow) ----
    p.add_argument("--mat_phi_ridge", type=float, default=1e-3)
    p.add_argument("--mat_phi_clip_particle", type=float, default=50.0)
    p.add_argument("--mat_phi_clip_P", type=float, default=1e3)
    # ---- optional NN-OT operator (any d) ----
    p.add_argument("--nnot_weights", type=str, default=None,
                   help="Path to trained DeepONet weights (.h5). If set, "
                        "the Sinkhorn resampler is replaced by the operator.")
    p.add_argument("--nnot_n_basis", type=int, default=64)
    # ---- NUTS sampling ----
    p.add_argument("--step_size", type=float, default=0.01)
    p.add_argument("--max_treedepth", type=int, default=8,
                   help="2^max_treedepth caps trajectory length (default 8 -> 256).")
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--num_burnin", type=int, default=200)
    p.add_argument("--num_results", type=int, default=2000)
    p.add_argument("--dispersion", type=float, default=0.05)
    p.add_argument("--target_accept_prob", type=float, default=0.65)
    p.add_argument("--adapt_step_size", action="store_true", default=True)
    p.add_argument("--error_threshold", type=float, default=10.0)
    p.add_argument("--cooldown_steps", type=int, default=10)
    p.add_argument("--kinetic_mh", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use the proper Hamiltonian MH acceptance "
                        "(-H_prop_real + H_init_real, including the kinetic "
                        "energy at the selected state) instead of the legacy "
                        "potential-only ratio. DEFAULT ON: the d=2 A/B showed "
                        "it recovers the posterior dispersion (mean CI-width "
                        "vs windowed 0.70x -> 0.86x; mu fully calibrated) at "
                        "no R-hat or wall cost. Pass --no-kinetic_mh for the "
                        "legacy potential-only behaviour.")
    # ---- L-HNN architecture + training ----
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--num_hidden", type=int, default=3)
    p.add_argument("--lhnn_epochs", type=int, default=2500)
    p.add_argument("--lhnn_lr", type=float, default=1e-3)
    p.add_argument("--lhnn_batch_size", type=int, default=64)
    p.add_argument("--num_pilot_trajectories", type=int, default=60)
    p.add_argument("--pilot_steps_per_trajectory", type=int, default=60)
    p.add_argument("--weights_cache", type=str, default="",
                   help="Path to .weights.h5. If it exists, load and skip "
                        "pilot+training; else train and save here.")
    # ---- parallelism (chains are independent -> run as parallel processes) ----
    p.add_argument("--chain_id", type=int, default=-1,
                   help="If >=0, run ONLY this chain (zero-indexed) with the "
                        "same init_raw / chain_seed / shared CRN it would have "
                        "had in a sequential --num_chains run, and save a "
                        "single-chain npz (chains-axis=1) to --out_dir. Launch "
                        "one process per chain in parallel, then combine with "
                        "scripts/stitch_multi_full_phi_chains.py. REQUIRES a "
                        "pre-warmed --weights_cache (run --train_only first) so "
                        "the pilot+train happens ONCE, not per chain.")
    p.add_argument("--train_only", action="store_true",
                   help="Run pilot + train the L-HNN, save to --weights_cache, "
                        "and exit WITHOUT sampling. Use this once to pre-warm "
                        "the cache before launching parallel --chain_id chains.")
    # ---- I/O ----
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "svssm_hmc_multi_full_phi_lhnn_nuts")
    p.add_argument("--progress_every", type=int, default=10)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.train_only and not args.weights_cache:
        raise ValueError("--train_only requires --weights_cache (nowhere to "
                         "save the trained L-HNN otherwise).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- parse truth ----
    mu_truth        = np.array([float(x) for x in args.mu.split(",")], dtype=np.float32)
    phi_diag_truth  = np.array([float(x) for x in args.phi_diag.split(",")], dtype=np.float32)
    _phi_off_parts  = [x for x in args.phi_off.split(",") if x.strip() != ""]
    phi_off_truth   = np.array([float(x) for x in _phi_off_parts], dtype=np.float32)
    sigma_eta_truth = np.array([float(x) for x in args.sigma_eta.split(",")], dtype=np.float32)
    if not (len(mu_truth) == len(phi_diag_truth) == len(sigma_eta_truth) == args.d):
        raise ValueError(f"mu/phi_diag/sigma_eta must be length d={args.d}")
    if len(phi_off_truth) != num_off_diag(args.d):
        raise ValueError(f"phi_off must be length d(d-1)/2 = {num_off_diag(args.d)}; "
                         f"got {len(phi_off_truth)}")
    sigma_eta_sq_truth = sigma_eta_truth ** 2
    Phi_truth = build_phi_matrix_np(phi_diag_truth, phi_off_truth, args.d)

    print(f"[hmc-multi-full-phi-LHNN-NUTS] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  d={args.d}  T={args.T}  N={args.N}  step_size={args.step_size}  "
          f"max_treedepth={args.max_treedepth}")
    print(f"  chains={args.num_chains}  burnin={args.num_burnin}  "
          f"results={args.num_results}  dispersion={args.dispersion}")
    print(f"  truth mu             : {mu_truth.tolist()}")
    print(f"  truth phi_diag       : {phi_diag_truth.tolist()}")
    print(f"  truth phi_off (upper): {phi_off_truth.tolist()}")
    print(f"  truth sigma_eta      : {sigma_eta_truth.tolist()}")
    eigs = np.linalg.eigvals(Phi_truth)
    print(f"  eigvals(Phi)         : {[round(float(np.abs(e)), 4) for e in eigs]}")
    print(f"  LHNN: hidden_units={args.hidden_units} num_hidden={args.num_hidden} "
          f"epochs={args.lhnn_epochs}")
    n_pilot_evals = args.num_pilot_trajectories * (args.pilot_steps_per_trajectory + 1)
    print(f"  pilot: single-anchor x {args.num_pilot_trajectories} traj x "
          f"{args.pilot_steps_per_trajectory} steps = ~{n_pilot_evals} real grad evals")
    print(f"  out_dir={out_dir}")

    # ---- data ----
    y_obs = gen_svssm_multi_phi_mat(args.T, mu_truth, Phi_truth, sigma_eta_truth,
                                     seed=args.data_seed)
    print(f"  y_obs shape: {tuple(y_obs.shape)}  range: "
          f"[{float(tf.reduce_min(y_obs)):.3f}, {float(tf.reduce_max(y_obs)):.3f}]")

    # ---- filter (Sinkhorn or NN-OT) ----
    if args.nnot_weights:
        from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
        from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_multivariate import (
            DifferentiableLEDHNeuralOTSVSSMmulti, svssm_multi_ctx_dim,
        )
        ctx_dim = svssm_multi_ctx_dim(args.d)
        print(f"\n  [nnot] loading DeepONet from {args.nnot_weights} "
              f"(d={args.d}, ctx_dim={ctx_dim})")
        model = DeepONetMonotoneOT(
            state_dim=args.d, n_basis=args.nnot_n_basis,
            d_branch=64, d_trunk=64, n_scalar_ctx=ctx_dim,
        )
        _ = model(tf.zeros([1, args.N, args.d]),
                   tf.ones([1, args.N]) / args.N, tf.zeros([1, ctx_dim]))
        model.load_weights(args.nnot_weights)
        ll = DifferentiableLEDHNeuralOTSVSSMmulti(
            neural_ot_model=model,
            state_dim=args.d, num_particles=args.N, n_lambda=args.n_lambda,
            sinkhorn_epsilon=1.0, grad_window=4, jit_compile=True,
        )
    else:
        ll = DifferentiableLEDHLogLikelihoodSVSSMmulti(
            state_dim=args.d, num_particles=args.N, n_lambda=args.n_lambda,
            sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
            grad_window=4, jit_compile=True,
            mat_phi_ridge=args.mat_phi_ridge,
            mat_phi_clip_particle=args.mat_phi_clip_particle,
            mat_phi_clip_P=args.mat_phi_clip_P,
        )

    # ---- target ----
    crn_seed = args.base_seed
    target_log_prob_fn = make_target_log_prob(
        ll, y_obs, crn_seed=crn_seed, d=args.d,
        prior_mu_loc=args.prior_mu_loc, prior_mu_scale=args.prior_mu_scale,
        prior_phi_raw_loc=args.prior_phi_raw_loc,
        prior_phi_raw_scale=args.prior_phi_raw_scale,
        prior_log_sigma_eta_sq_loc=args.prior_log_sigma_eta_sq_loc,
        prior_log_sigma_eta_sq_scale=args.prior_log_sigma_eta_sq_scale,
        prior_phi_off_scale=args.prior_phi_off_scale,
        no_likelihood=False,
    )

    truth_raw = unconstrain(mu_truth, phi_diag_truth, phi_off_truth, sigma_eta_sq_truth)
    d_param = int(truth_raw.shape[0])
    lp_at_truth = float(target_log_prob_fn(tf.constant(truth_raw)).numpy())
    print(f"  theta_raw dim = {d_param}  (3d + d(d-1)/2)")
    print(f"  target_log_prob(truth_raw) = {lp_at_truth:.4f}")

    # ---- L-HNN: cache load OR pilot+train ----
    cache_path = Path(args.weights_cache) if args.weights_cache else None
    use_cache = (cache_path is not None and cache_path.exists())

    if use_cache:
        print(f"\n[cache] loading L-HNN weights from {cache_path} "
              f"(skipping pilot + training)")
        lhnn = LatentHNN(d_param, args.hidden_units, args.num_hidden)
        _ = lhnn(tf.zeros([1, 2 * d_param]))
        lhnn.load_weights(str(cache_path))
        pilot_wall = train_wall = 0.0
        training_evals = 0
    else:
        print(f"\n[stage 1/3] Single-anchor pilot @ truth_raw "
              f"({n_pilot_evals} real grad evals; this is the expensive part "
              f"at high T) ...")
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

        print(f"\n[stage 2/3] Training LatentHNN on {int(q_data.shape[0])} "
              f"points for {args.lhnn_epochs} epochs ...")
        t_train = time.perf_counter()
        lhnn = LatentHNN(d_param, args.hidden_units, args.num_hidden)
        _ = lhnn(tf.zeros([1, 2 * d_param]))
        train_lhnn(
            lhnn=lhnn, q_data=q_data, p_data=p_data,
            dq_dt=dq_data, dp_dt=dp_data,
            epochs=args.lhnn_epochs, lr=args.lhnn_lr,
            batch_size=args.lhnn_batch_size, verbose=(not args.quiet),
        )
        train_wall = time.perf_counter() - t_train
        print(f"  [stage 2 done] training wall: {train_wall:.1f}s")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            lhnn.save_weights(str(cache_path))
            print(f"  [cache] saved L-HNN weights to {cache_path}")

    # ---- train_only: pilot+train+cache done, exit before sampling ----
    if args.train_only:
        print(f"\n[train_only] L-HNN trained and cached to {cache_path}. "
              f"Launch parallel chains with --chain_id 0..{args.num_chains-1} "
              f"--weights_cache {cache_path}. Exiting without sampling.")
        return 0

    # ---- dispersed inits (same rng pattern as run_lhnn_nuts_multi_chain) ----
    # In --chain_id mode we still generate ALL target_num_chains inits with the
    # SAME rng(base_seed) and index by chain_id, so a parallel per-chain launch
    # is bit-identical to the sequential multi-chain run (init_raw, chain_seed,
    # and shared CRN all match).
    if args.chain_id >= 0:
        target_num_chains = max(args.num_chains, args.chain_id + 1)
        chain_indices = [args.chain_id]
        if not use_cache:
            print(f"  [warn] --chain_id set but --weights_cache was not "
                  f"pre-warmed; this process trained its own L-HNN. For a "
                  f"true parallel launch, run --train_only first.")
    else:
        target_num_chains = args.num_chains
        chain_indices = list(range(args.num_chains))

    rng = np.random.default_rng(args.base_seed)
    init_raws = np.stack([
        truth_raw + args.dispersion * rng.standard_normal(d_param).astype(np.float32)
        for _ in range(target_num_chains)
    ], axis=0)

    mode = (f"chain {args.chain_id} only (parallel-process mode)"
            if args.chain_id >= 0 else f"{args.num_chains} chains (sequential)")
    print(f"\n[stage 3/3] NUTS-with-LHNN, {mode}, "
          f"max_treedepth={args.max_treedepth} ...")
    for c in chain_indices:
        ic = constrain(init_raws[c], args.d)
        print(f"    chain {c+1} init: mu={ic['mu'].tolist()} "
              f"phi_diag={ic['phi_diag'].tolist()}")

    # Per-chain seeds replicate run_lhnn_nuts_multi_chain exactly:
    #   chain_seed = base_seed + 1009*(c+1);  crn_offset = base_seed (shared).
    t_sample = time.perf_counter()
    per_chain_results = []
    for c in chain_indices:
        res = run_lhnn_nuts(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=tf.constant(init_raws[c], dtype=tf.float32),
            lhnn=lhnn,
            num_results=args.num_results,
            num_burnin=args.num_burnin,
            step_size=args.step_size,
            max_treedepth=args.max_treedepth,
            error_threshold=args.error_threshold,
            cooldown_steps=args.cooldown_steps,
            adapt_step_size=args.adapt_step_size,
            target_accept_prob=args.target_accept_prob,
            seed=args.base_seed + 1009 * (c + 1),
            crn_offset=args.base_seed,
            verbose=(not args.quiet),
            progress_every=args.progress_every,
            kinetic_mh=args.kinetic_mh,
        )
        per_chain_results.append(res)
    sample_wall = time.perf_counter() - t_sample
    total_wall = pilot_wall + train_wall + sample_wall

    # ---- extract (stack the chain(s) we ran -> leading axis = len(chain_indices)) ----
    samples_raw = np.stack([r.samples.numpy() for r in per_chain_results], axis=0)
    accs = np.stack([r.is_accepted.numpy().astype(np.float32)
                     for r in per_chain_results], axis=0)
    lps = np.stack([r.target_log_probs.numpy() for r in per_chain_results], axis=0)
    step_arr = np.stack([r.step_sizes.numpy() for r in per_chain_results], axis=0)
    depth_arr = np.stack([r.tree_depths.numpy() for r in per_chain_results], axis=0)

    class _R:  # lightweight shim so downstream code reads per_chain_diagnostics
        per_chain_diagnostics = per_chain_results
    result = _R()
    cons = constrain(samples_raw, args.d)

    error_triggers_per_chain = [dg.total_error_triggers
                                 for dg in result.per_chain_diagnostics]
    real_grads_per_chain = [dg.total_real_grad_evals
                             for dg in result.per_chain_diagnostics]
    lhnn_steps_per_chain = [dg.total_lhnn_leapfrog_evals
                             for dg in result.per_chain_diagnostics]
    avg_depth_per_chain = [float(dg.avg_tree_depth.numpy())
                            for dg in result.per_chain_diagnostics]
    per_chain_acc = [float(a.mean()) for a in accs]
    total_real_grad_evals = int(training_evals) + int(sum(real_grads_per_chain))

    # ---- per-parameter Vehtari diagnostics (constrained) ----
    def _vehtari(x: np.ndarray) -> tuple:
        x_clean = np.where(np.isfinite(x), x, float(np.nanmedian(x)))
        return (float(_rank_rhat(x_clean)),
                float(_bulk_ess(x_clean)),
                float(_tail_ess(x_clean)))

    print(f"\n  total wall: {total_wall:.1f}s "
          f"(pilot {pilot_wall:.1f}, train {train_wall:.1f}, sample {sample_wall:.1f})")
    print(f"  overall accept rate: {float(accs.mean()):.3f}")
    print(f"  avg tree depth/chain: {[round(x,2) for x in avg_depth_per_chain]}  "
          f"err_trigs/chain: {error_triggers_per_chain}")
    print(f"\n{'param':<24} {'comp':>6} {'truth':>10} {'mean':>10} {'std':>10} "
          f"{'median':>10} {'2.5%':>10} {'97.5%':>10} {'rankR':>7} "
          f"{'bulkESS':>8} {'tailESS':>8} cov")

    rows = []

    def _emit(label, comp_index, tval, *, row_param_name, chain_x):
        samp = chain_x.ravel()
        med = float(np.median(samp))
        q025 = float(np.quantile(samp, 0.025))
        q975 = float(np.quantile(samp, 0.975))
        rhat, be, te = _vehtari(chain_x)
        cov = "OK" if q025 <= tval <= q975 else "OUT"
        print(f"{label:<24} {comp_index:>6} {tval:>10.4f} "
              f"{float(samp.mean()):>10.4f} {float(samp.std()):>10.4f} "
              f"{med:>10.4f} {q025:>10.4f} {q975:>10.4f} "
              f"{rhat:>7.3f} {be:>8.0f} {te:>8.0f} {cov:>3}")
        rows.append({
            "param": row_param_name, "component": int(comp_index),
            "truth": tval, "median": med, "q025": q025, "q975": q975,
            "rhat": rhat, "bulk_ess": be, "tail_ess": te,
            "covered_95ci": (cov == "OK"),
        })

    for i in range(args.d):
        _emit("mu", i, float(mu_truth[i]), row_param_name="mu",
              chain_x=cons["mu"][:, :, i])
    for i in range(args.d):
        _emit("phi_diag", i, float(phi_diag_truth[i]), row_param_name="phi_diag",
              chain_x=cons["phi_diag"][:, :, i])
    for k, (i, j) in enumerate(upper_tri_indices(args.d)):
        _emit(f"phi_off ({i},{j})", k, float(phi_off_truth[k]),
              row_param_name=f"phi_off_{i}{j}", chain_x=cons["phi_off"][:, :, k])
    for i in range(args.d):
        _emit("sigma_eta_sq", i, float(sigma_eta_sq_truth[i]),
              row_param_name="sigma_eta_sq", chain_x=cons["sigma_eta_sq"][:, :, i])

    n_covered = sum(1 for r in rows if r["covered_95ci"])
    valid_rhats = [r["rhat"] for r in rows if not np.isnan(r["rhat"])]
    max_rhat = max(valid_rhats) if valid_rhats else float("nan")
    print(f"\n  coverage: {n_covered}/{len(rows)} params covered at 95% CI")
    print(f"  max rank-Rhat: {max_rhat:.3f}")

    # ---- save (full-Phi-compatible npz) ----
    np.savez_compressed(
        out_dir / "svssm_hmc_multi_full_phi_samples.npz",
        samples_raw=samples_raw,
        mu=cons["mu"], phi_diag=cons["phi_diag"], phi_off=cons["phi_off"],
        sigma_eta_sq=cons["sigma_eta_sq"],
        accept=accs, log_prob=lps,
        step_size=step_arr, tree_depth=depth_arr,
        mu_truth=mu_truth, phi_diag_truth=phi_diag_truth,
        phi_off_truth=phi_off_truth, sigma_eta_sq_truth=sigma_eta_sq_truth,
        Phi_truth=Phi_truth,
    )

    # config keyed so save_diagnostics_multi_full_phi.py stays self-documenting.
    # diagonal_mass=True so it prints dense_mass=False (NUTS uses unit mass,
    # no dense preconditioner -- that's the whole point vs the windowed driver).
    cfg = dict(vars(args))
    cfg["sampler"] = "lhnn_nuts"
    cfg["L"] = None
    cfg["diagonal_mass"] = True
    (out_dir / "svssm_hmc_multi_full_phi_summary.json").write_text(json.dumps({
        "tf": tf.__version__, "tfp": tfp.__version__,
        "config": cfg, "rows": rows,
        "elapsed_s": total_wall,
        "pilot_wall_s": pilot_wall,
        "training_wall_s": train_wall,
        "sampling_wall_s": sample_wall,
        "accept_rate_overall": float(accs.mean()),
        "accept_rate_per_chain": per_chain_acc,
        "max_rhat": max_rhat,
        "n_covered_95ci": n_covered,
        "n_total_params": len(rows),
        "nuts_diagnostics": {
            "error_triggers_per_chain": error_triggers_per_chain,
            "real_grads_per_chain": real_grads_per_chain,
            "lhnn_steps_per_chain": lhnn_steps_per_chain,
            "avg_depth_per_chain": avg_depth_per_chain,
            "pilot_grad_evals": int(training_evals),
            "total_real_grad_evals": total_real_grad_evals,
        },
    }, indent=2))
    print(f"\n  [save] {out_dir / 'svssm_hmc_multi_full_phi_samples.npz'}")
    print(f"  [save] {out_dir / 'svssm_hmc_multi_full_phi_summary.json'}")

    # ---- d-generic diagnostics .txt/.json (reuse existing tooling) ----
    # Skip in single-chain (--chain_id) mode: rank-Rhat needs >=2 chains, so
    # run save_diagnostics on the STITCHED dir after combining the per-chain
    # npzs with scripts/stitch_multi_full_phi_chains.py.
    if args.chain_id < 0:
        try:
            import subprocess
            subprocess.run([
                "python", "scripts/save_diagnostics_multi_full_phi.py",
                "--out_dir", str(out_dir),
            ], check=True)
        except Exception as e:
            print(f"  [save_diagnostics warning] {e}")
    else:
        print(f"\n  [chain_id={args.chain_id}] single-chain npz saved to "
              f"{out_dir}. Stitch all chains, then run "
              f"save_diagnostics_multi_full_phi.py on the combined dir.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
