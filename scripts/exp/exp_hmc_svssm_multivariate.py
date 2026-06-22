"""
HMC parameter recovery for the multivariate V1 SVSSM.

Vector parameterisation:
    theta_raw ∈ R^{3d} = [ mu (d) ; phi_raw (d) ; log_sigma_eta_sq (d) ]
    phi_diag        = tanh(phi_raw)            (per-component stationarity)
    sigma_eta_diag² = exp(log_sigma_eta_sq)    (per-component positivity)

Per-component priors (independent across d):
    mu_i              ~ N(prior_mu_loc, prior_mu_scale²)
    phi_raw_i         ~ N(prior_phi_raw_loc, prior_phi_raw_scale²)
    log_sigma_eta_sq_i ~ N(prior_log_sigma_eta_sq_loc, prior_log_sigma_eta_sq_scale²)

Output:
    reports/.../svssm_hmc_multivariate_smoke/
      svssm_hmc_multi_samples.npz
      svssm_hmc_multi_results.txt
      svssm_hmc_multi_traces.png  (per-component traces)

Usage:
    PYTHONPATH=. python scripts/exp/exp_hmc_svssm_multivariate.py \
        --d 2 --T 20 --N 64 \
        --num_chains 1 --num_burnin 20 --num_results 20
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

# v7 windowed-adaptive kernel (PreconditionedHMC + dense mass matrix in
# expanding windows with step-size adapter reset). Dimension-agnostic, so
# applies to the 3d-dimensional multivariate state space the same way it
# applies to univariate. See section3_kernel_upgrade.tex §4 for design
# and §6 for the univariate HPC validation.
from scripts.exp.exp_hmc_svssm import run_chain_windowed_proper

tfd = tfp.distributions
tfm = tfp.mcmc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    plt = None
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# Data generation (multivariate SVSSM, diagonal Phi + diagonal Sigma_eta)
# ---------------------------------------------------------------------------

def gen_svssm_multi(T, mu_vec, phi_vec, sigma_eta_vec, seed=42):
    tf.random.set_seed(seed)
    d = len(mu_vec)
    mu = tf.constant(mu_vec, tf.float32)
    phi = tf.constant(phi_vec, tf.float32)
    sigma_eta = tf.constant(sigma_eta_vec, tf.float32)
    h = tf.identity(mu)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta * tf.random.normal([d])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([d]))
    return tf.stack(ys, axis=0)  # (T, d)


# ---------------------------------------------------------------------------
# Target log-prob
# ---------------------------------------------------------------------------

def make_target_log_prob(ll, y_obs, crn_seed,
                          prior_mu_loc=0.0, prior_mu_scale=1.0,
                          prior_phi_raw_loc=2.0, prior_phi_raw_scale=0.5,
                          prior_log_sigma_eta_sq_loc=-2.0,
                          prior_log_sigma_eta_sq_scale=1.0):
    """Build target_log_prob_fn for theta_raw ∈ R^{3d}.

    Uses the same per-component priors as the univariate driver (Phases 5-12)
    applied independently across d.
    """
    d = ll.state_dim

    prior_mu = tfd.Normal(loc=tf.constant(prior_mu_loc, tf.float32),
                          scale=tf.constant(prior_mu_scale, tf.float32))
    prior_phi_raw = tfd.Normal(loc=tf.constant(prior_phi_raw_loc, tf.float32),
                                scale=tf.constant(prior_phi_raw_scale, tf.float32))
    prior_log_sig = tfd.Normal(loc=tf.constant(prior_log_sigma_eta_sq_loc, tf.float32),
                                scale=tf.constant(prior_log_sigma_eta_sq_scale, tf.float32))

    def target_log_prob(theta_raw):
        # theta_raw: (3d,)
        mu               = theta_raw[:d]                # (d,)
        phi_raw          = theta_raw[d:2*d]             # (d,)
        log_sigma_eta_sq = theta_raw[2*d:3*d]           # (d,)

        phi_diag             = tf.tanh(phi_raw)
        sigma_eta_diag_sq    = tf.exp(log_sigma_eta_sq)

        log_prior = (
            tf.reduce_sum(prior_mu.log_prob(mu))
            + tf.reduce_sum(prior_phi_raw.log_prob(phi_raw))
            + tf.reduce_sum(prior_log_sig.log_prob(log_sigma_eta_sq))
        )

        # Use shared CRN inside the filter — eliminates target drift inside leapfrog.
        tf.random.set_seed(crn_seed)
        log_lik = ll(mu, phi_diag, sigma_eta_diag_sq, y_obs)
        log_lik = tf.cast(tf.math.real(log_lik), tf.float32)
        # NaN guard: use -inf (not -1e6) so MH cleanly rejects bad proposals.
        # The -1e6 sentinel saturates the MH ratio between two NaN states and
        # fools dual averaging into thinking giant step sizes are safe. This
        # is the same safeguard documented in section3_kernel_upgrade.tex §4.
        log_lik = tf.where(tf.math.is_finite(log_lik), log_lik,
                           tf.constant(-np.inf, tf.float32))

        return log_prior + log_lik

    return target_log_prob


def unconstrain(mu_vec, phi_vec, sigma_eta_sq_vec):
    """(d,)+(d,)+(d,) ↦ (3d,) unconstrained theta_raw."""
    mu = np.asarray(mu_vec, dtype=np.float32)
    phi_clipped = np.clip(np.asarray(phi_vec), -0.9999, 0.9999)
    phi_raw = np.arctanh(phi_clipped).astype(np.float32)
    log_sig = np.log(np.maximum(np.asarray(sigma_eta_sq_vec), 1e-8)).astype(np.float32)
    return np.concatenate([mu, phi_raw, log_sig])


def constrain(theta_raw, d):
    """theta_raw (..., 3d) ↦ (..., 3, d) constrained (mu, phi, sigma_eta_sq)."""
    th = np.asarray(theta_raw)
    mu = th[..., :d]
    phi = np.tanh(th[..., d:2*d])
    sigma_eta_sq = np.exp(th[..., 2*d:3*d])
    return np.stack([mu, phi, sigma_eta_sq], axis=-2)   # (..., 3, d)


# ---------------------------------------------------------------------------
# HMC chain driver (single chain)
# ---------------------------------------------------------------------------

def run_chain(target_log_prob_fn, init_raw, num_results, num_burnin,
              step_size, num_leapfrog, seed, progress_every=10,
              use_windowed_adaptive=False, dense_mass=True,
              target_accept_prob=None):
    """Multivariate SVSSM HMC. Two kernels supported.

    Vanilla (use_windowed_adaptive=False):
        tfm.HamiltonianMonteCarlo + DualAveragingStepSizeAdaptation,
        identity mass matrix. The Phase 14 production setup.

    v7 windowed (use_windowed_adaptive=True):
        PreconditionedHMC + adapted mass matrix in expanding windows,
        with step-size adapter reset at each window boundary. Five
        safeguards (NaN guard, variance clip [0.1, 10], step clip
        [1e-4, 0.3], last window as term, log_averaging_step smoothing).
        dense_mass=True  (default): full inverse-covariance mass matrix
                         at d=3*d state dim. Handles per-component
                         (phi_i, sigma²_i) ridges + cross-component
                         correlations.
        dense_mass=False: diagonal mass.
        Returns sampling-phase data only (no burnin).
    """
    if use_windowed_adaptive:
        # Dispatch to the shared v7 implementation. Dimension-agnostic.
        tap = 0.65 if target_accept_prob is None else float(target_accept_prob)
        samples, acc, lp, step_arr = run_chain_windowed_proper(
            target_log_prob_fn=target_log_prob_fn,
            init_raw=np.asarray(init_raw, np.float32),
            num_results=int(num_results),
            num_burnin=int(num_burnin),
            step_size=float(step_size),
            num_leapfrog=int(num_leapfrog),
            seed=int(seed),
            target_accept_prob=tap,
            progress_every=int(progress_every),
            dense_mass=bool(dense_mass),
        )
        # Match the legacy return shape: (samples (R, 3d), accepts (R,),
        # log_probs (R,), step_sizes (R,)). The vanilla path historically
        # returned (R+B,) for accepts/lps/steps, but the only downstream
        # use is `accs.mean()` for the overall acceptance rate, which
        # behaves equivalently on the sampling-only arrays.
        return (samples.astype(np.float32),
                acc.astype(np.float32),
                lp.astype(np.float32),
                step_arr.astype(np.float32))

    # --- Vanilla HMC path (Phase 14 baseline) ---
    inner = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog,
    )
    kernel = tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=inner,
        num_adaptation_steps=int(num_burnin),
    )

    @tf.function
    def _one_step(state, results):
        new_state, new_results = kernel.one_step(state, results)
        return new_state, new_results

    state = tf.constant(np.asarray(init_raw, np.float32))
    results = kernel.bootstrap_results(state)

    samples = []
    accepts = []
    step_sizes = []
    log_probs = []
    total = int(num_burnin) + int(num_results)
    t0 = time.perf_counter()
    for s in range(total):
        state, results = _one_step(state, results)
        # extract diagnostics
        inner_r = results.inner_results
        acc_p = float(np.exp(min(0.0, float(inner_r.log_accept_ratio.numpy()))))
        lp = float(inner_r.accepted_results.target_log_prob.numpy())
        ss = float(results.new_step_size.numpy()) if hasattr(results, "new_step_size") else float(inner_r.accepted_results.step_size.numpy())
        accepts.append(acc_p)
        step_sizes.append(ss)
        log_probs.append(lp)
        if s >= int(num_burnin):
            samples.append(state.numpy().copy())
        if progress_every and (s + 1) % progress_every == 0:
            phase = "burn" if s < int(num_burnin) else "samp"
            print(f"      [{phase}] step {s+1}/{total}  eps={ss:.5f}  acc={acc_p:.1f}", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"      chain done in {elapsed:.1f}s")

    return (
        np.asarray(samples, dtype=np.float32),       # (num_results, 3d)
        np.asarray(accepts, dtype=np.float32),
        np.asarray(log_probs, dtype=np.float32),
        np.asarray(step_sizes, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--d", type=int, default=2,
                   dest="d", metavar="D",
                   help="State dimension. Both -d and --d accepted.")
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--step_size", type=float, default=0.05)
    p.add_argument("--num_chains", type=int, default=1)
    p.add_argument("--num_burnin", type=int, default=20)
    p.add_argument("--num_results", type=int, default=20)
    p.add_argument("--dispersion", type=float, default=0.15)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--progress_every", type=int, default=5)
    # Per-component truth (passed as comma-separated lists or single-vals broadcast).
    p.add_argument("--mu", type=str, default="0.0,-0.3",
                   help="comma-separated truth for mu (length d)")
    p.add_argument("--phi", type=str, default="0.95,0.85",
                   help="comma-separated truth for phi_diag (length d)")
    p.add_argument("--sigma_eta", type=str, default="0.3,0.4",
                   help="comma-separated truth for sigma_eta_diag (length d)")
    # Priors (same form as univariate driver, applied per component).
    p.add_argument("--prior_mu_loc", type=float, default=-0.2)
    p.add_argument("--prior_mu_scale", type=float, default=1.0)
    p.add_argument("--prior_phi_raw_loc", type=float, default=1.5)
    p.add_argument("--prior_phi_raw_scale", type=float, default=1.0)
    p.add_argument("--prior_log_sigma_eta_sq_loc", type=float, default=-2.0)
    p.add_argument("--prior_log_sigma_eta_sq_scale", type=float, default=1.0)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "svssm_hmc_multivariate_smoke")
    p.add_argument("--use_windowed_adaptive", action="store_true",
                   help="Use the v7 windowed-adaptive PreconditionedHMC kernel "
                        "(dense mass matrix by default). Matches the kernel "
                        "validated in the univariate Phase 16/17 HPC sweeps. "
                        "Strongly recommended for wide-shifted priors where "
                        "vanilla HMC's identity mass cannot navigate the "
                        "multivariate (phi_i, sigma_i^2) ridges.")
    p.add_argument("--diagonal_mass", action="store_true",
                   help="With --use_windowed_adaptive: use a DIAGONAL mass "
                        "matrix (per-dim variance only) instead of the default "
                        "DENSE mass matrix (full inverse covariance).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_truth        = np.array([float(x) for x in args.mu.split(",")], dtype=np.float32)
    phi_truth       = np.array([float(x) for x in args.phi.split(",")], dtype=np.float32)
    sigma_eta_truth = np.array([float(x) for x in args.sigma_eta.split(",")], dtype=np.float32)
    if not (len(mu_truth) == len(phi_truth) == len(sigma_eta_truth) == args.d):
        raise ValueError(f"truth vectors must have length d={args.d}; got "
                         f"mu={mu_truth.shape}, phi={phi_truth.shape}, sigma={sigma_eta_truth.shape}")
    sigma_eta_sq_truth = sigma_eta_truth ** 2

    print(f"[hmc-multi] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  d={args.d}  T={args.T}  N={args.N}  L={args.L}  step_size={args.step_size}")
    print(f"  chains={args.num_chains}  burnin={args.num_burnin}  results={args.num_results}")
    print(f"  truth: mu={mu_truth.tolist()}  phi={phi_truth.tolist()}  "
          f"sigma_eta={sigma_eta_truth.tolist()}")

    # Generate data
    y_obs = gen_svssm_multi(args.T, mu_truth, phi_truth, sigma_eta_truth, seed=args.data_seed)
    print(f"  y_obs shape: {tuple(y_obs.shape)}  range: "
          f"[{float(tf.reduce_min(y_obs)):.3f}, {float(tf.reduce_max(y_obs)):.3f}]")

    # Build filter
    ll = DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=args.d, num_particles=args.N, n_lambda=args.n_lambda,
        sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
        grad_window=4, jit_compile=True,
    )

    # Target log-prob
    crn_seed = args.base_seed
    target_log_prob_fn = make_target_log_prob(
        ll, y_obs, crn_seed=crn_seed,
        prior_mu_loc=args.prior_mu_loc, prior_mu_scale=args.prior_mu_scale,
        prior_phi_raw_loc=args.prior_phi_raw_loc,
        prior_phi_raw_scale=args.prior_phi_raw_scale,
        prior_log_sigma_eta_sq_loc=args.prior_log_sigma_eta_sq_loc,
        prior_log_sigma_eta_sq_scale=args.prior_log_sigma_eta_sq_scale,
    )

    # Sanity: one forward+grad
    mu_t  = tf.constant(mu_truth, tf.float32)
    phi_t = tf.constant(phi_truth, tf.float32)
    sig_t = tf.constant(sigma_eta_sq_truth, tf.float32)
    tf.random.set_seed(args.base_seed)
    v0 = float(ll(mu_t, phi_t, sig_t, y_obs).numpy())
    print(f"  sanity: log p at truth = {v0:.4f}")

    truth_raw = unconstrain(mu_truth, phi_truth, sigma_eta_sq_truth)
    print(f"  truth_raw (3d-vector): {truth_raw}")
    print(f"  target_log_prob(truth_raw) = {float(target_log_prob_fn(tf.constant(truth_raw)).numpy()):.4f}")

    # Initial states (dispersed around truth in unconstrained space)
    rng = np.random.default_rng(args.base_seed)
    init_raws = [
        truth_raw + args.dispersion * rng.standard_normal(3 * args.d).astype(np.float32)
        for _ in range(args.num_chains)
    ]
    chain_seeds = [args.base_seed + 1009 * (c + 1) for c in range(args.num_chains)]

    # Run chains sequentially
    all_samples = []
    all_accs = []
    all_lps = []
    all_steps = []
    t_total = time.perf_counter()
    for c in range(args.num_chains):
        print(f"\n  [chain {c+1}/{args.num_chains}] init_raw = {init_raws[c].tolist()}")
        s, acc, lp, st = run_chain(
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
        all_samples.append(s)
        all_accs.append(acc)
        all_lps.append(lp)
        all_steps.append(st)
    elapsed = time.perf_counter() - t_total

    samples_raw = np.stack(all_samples, axis=0)                      # (chains, results, 3d)
    samples_constrained = constrain(samples_raw, args.d)             # (chains, results, 3, d)
    accs = np.stack(all_accs, axis=0)
    lps = np.stack(all_lps, axis=0)

    # ----- Posterior summary per-component -----
    truth_constrained = np.stack([mu_truth, phi_truth, sigma_eta_sq_truth], axis=0)  # (3, d)
    param_names = ["mu", "phi", "sigma_eta_sq"]
    print(f"\n  total wall: {elapsed:.1f}s")
    print(f"  overall accept rate: {float(accs.mean()):.3f}")
    print()
    print(f"{'param':<20s} {'comp':>4} {'truth':>10} {'mean':>10} {'std':>10} "
          f"{'median':>10} {'2.5%':>10} {'97.5%':>10} {'cov?':>6}")
    rows = []
    for pi, name in enumerate(param_names):
        for ci in range(args.d):
            samp = samples_constrained[:, :, pi, ci].ravel()
            tval = float(truth_constrained[pi, ci])
            mean = float(samp.mean())
            std = float(samp.std())
            med = float(np.median(samp))
            q025 = float(np.quantile(samp, 0.025))
            q975 = float(np.quantile(samp, 0.975))
            covered = q025 <= tval <= q975
            ok = "OK" if covered else "OUT"
            print(f"{name:<20s} {ci:>4d} {tval:>10.4f} {mean:>10.4f} {std:>10.4f} "
                  f"{med:>10.4f} {q025:>10.4f} {q975:>10.4f} {ok:>6}")
            rows.append({
                "param": name, "component": ci, "truth": tval,
                "mean": mean, "std": std, "median": med,
                "q025": q025, "q975": q975, "covered_95ci": bool(covered),
            })

    # Persist
    np.savez_compressed(
        out_dir / "svssm_hmc_multi_samples.npz",
        samples_raw=samples_raw.astype(np.float32),
        samples_constrained=samples_constrained.astype(np.float32),
        accept=accs.astype(np.float32),
        log_prob=lps.astype(np.float32),
        step_size=np.stack(all_steps, axis=0).astype(np.float32),
        truth=truth_constrained.astype(np.float32),
    )

    summary = {
        "tf": tf.__version__, "tfp": tfp.__version__,
        "config": vars(args), "d": args.d,
        "truth": truth_constrained.tolist(),
        "rows": rows,
        "elapsed_s": elapsed,
        "accept_rate_overall": float(accs.mean()),
    }
    (out_dir / "svssm_hmc_multi_summary.json").write_text(json.dumps(summary, indent=2))

    text_lines = [
        "=" * 100,
        f"Multivariate SVSSM HMC Smoke Test  (d={args.d})",
        "=" * 100,
        f"TF {tf.__version__}, TFP {tfp.__version__}",
        f"truth (per-component): mu={mu_truth.tolist()}, "
        f"phi={phi_truth.tolist()}, sigma_eta={sigma_eta_truth.tolist()}",
        f"T={args.T} N={args.N} L={args.L} step_size={args.step_size}",
        f"chains={args.num_chains} burnin={args.num_burnin} results={args.num_results}",
        f"wall: {elapsed:.1f}s   overall accept: {float(accs.mean()):.3f}",
        "",
        f"{'param':<20s} {'comp':>4} {'truth':>10} {'mean':>10} {'std':>10} "
        f"{'median':>10} {'2.5%':>10} {'97.5%':>10} {'cov?':>6}",
        "-" * 100,
    ]
    for r in rows:
        text_lines.append(
            f"{r['param']:<20s} {r['component']:>4d} {r['truth']:>10.4f} "
            f"{r['mean']:>10.4f} {r['std']:>10.4f} {r['median']:>10.4f} "
            f"{r['q025']:>10.4f} {r['q975']:>10.4f} "
            f"{'OK' if r['covered_95ci'] else 'OUT':>6}"
        )
    text_lines.append("=" * 100)
    (out_dir / "svssm_hmc_multi_results.txt").write_text("\n".join(text_lines))

    # Plots
    if _HAVE_MPL:
        fig, axes = plt.subplots(3, args.d, figsize=(4 * args.d, 9), sharex=True)
        # Handle the d=1 case where axes is 1D
        if args.d == 1:
            axes = axes[:, np.newaxis]
        for pi, name in enumerate(param_names):
            for ci in range(args.d):
                ax = axes[pi, ci]
                for cc in range(samples_constrained.shape[0]):
                    ax.plot(samples_constrained[cc, :, pi, ci], alpha=0.7,
                            label=f"chain {cc+1}")
                ax.axhline(truth_constrained[pi, ci], color="red", linestyle="--",
                            lw=1.2, label="truth")
                ax.set_ylabel(f"{name}[{ci}]")
                ax.grid(alpha=0.3)
                if pi == 0 and ci == 0:
                    ax.legend(loc="upper right", fontsize=8)
        for ax in axes[-1, :]:
            ax.set_xlabel("iteration")
        fig.suptitle(f"Multivariate SVSSM HMC trace (d={args.d})")
        fig.tight_layout()
        fig.savefig(out_dir / "svssm_hmc_multi_traces.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[plot] wrote {out_dir / 'svssm_hmc_multi_traces.png'}")

    print(f"\n[done] wrote {out_dir / 'svssm_hmc_multi_results.txt'}")
    print(f"       wrote {out_dir / 'svssm_hmc_multi_samples.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
