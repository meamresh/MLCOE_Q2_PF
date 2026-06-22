"""
HMC parameter recovery for the canonical SVSSM, using
DifferentiableLEDHLogLikelihoodSVSSM as the differentiable target.

Pipeline:
  1. Generate T observations from y_t = exp(h_t/2) eps_t with known truth
     (mu*, phi*, sigma_eta*).
  2. Define a target log-posterior in unconstrained parameter space
        theta_raw = [mu, phi_raw, log_sigma_eta_sq]
        phi = tanh(phi_raw),      sigma_eta_sq = exp(log_sigma_eta_sq)
     with weakly-informative priors.
  3. Run TFP HMC with dual-averaging step-size adaptation, multiple chains
     started from dispersed initial states around truth.
  4. Diagnose: per-parameter R-hat, ESS, posterior mean/std/CIs, accept
     rate, and a coverage check against truth. Save samples + plots.

This is the headline Section 3 deliverable for SVSSM V1. Caveat: the
Harvey-Ruiz-Shephard Gaussian quasi-likelihood biases sigma_eta upward by
roughly 30-50% (documented in reports/.../svssm_validation/), so the
posterior mean of sigma_eta is expected to be HIGHER than the true value;
mu and phi should be recovered close to truth.

Outputs:
  reports/.../svssm_hmc/
    - svssm_hmc_results.txt    : human-readable summary
    - svssm_hmc_samples.npz    : (n_chains, n_samples, 3) tensor of samples
    - svssm_hmc_corner.png     : pair-plot of posterior samples (if matplotlib)
    - svssm_hmc_traces.png     : per-chain traceplots (if matplotlib)
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
# Data generation
# ---------------------------------------------------------------------------

def gen_svssm(T: int, mu: float, phi: float, sigma_eta: float, seed: int) -> tf.Tensor:
    tf.random.set_seed(seed)
    sigma_eta_t = tf.constant(sigma_eta, tf.float32)
    h = tf.constant(float(mu), tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta_t * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


# ---------------------------------------------------------------------------
# Unconstrained parameterisation and target log-posterior
# ---------------------------------------------------------------------------

def make_target_log_prob(ll: DifferentiableLEDHLogLikelihoodSVSSM,
                         y_obs: tf.Tensor,
                         crn_seed: int,
                         prior_mu_loc: float = 0.0,
                         prior_mu_scale: float = 5.0,
                         prior_phi_raw_loc: float = 0.0,
                         prior_phi_raw_scale: float = 2.0,
                         prior_log_sigma_eta_sq_loc: float = -2.0,
                         prior_log_sigma_eta_sq_scale: float = 2.0,
                         no_likelihood: bool = False) -> callable:
    """Return target_log_prob_fn(theta_raw) in unconstrained space.

    theta_raw layout: [mu, phi_raw, log_sigma_eta_sq]
        mu             unbounded                 (prior N(prior_mu_loc, prior_mu_scale^2))
        phi_raw        unbounded, phi=tanh()     (prior on phi_raw)
        log_sigma_eta_sq  unbounded              (prior on log_sigma_eta_sq)

    Default priors are weakly informative. For SVSSM at small T, tightening
    the phi_raw and log_sigma_eta_sq priors (e.g., phi_raw_loc=2,
    phi_raw_scale=0.5 to concentrate phi near tanh(2)~0.964) prevents the
    chain from wandering into the near-non-stationary mode.
    """
    prior_mu = tfd.Normal(loc=tf.constant(prior_mu_loc, tf.float32),
                          scale=tf.constant(prior_mu_scale, tf.float32))
    prior_phi_raw = tfd.Normal(loc=tf.constant(prior_phi_raw_loc, tf.float32),
                               scale=tf.constant(prior_phi_raw_scale, tf.float32))
    prior_log_sigma_eta_sq = tfd.Normal(
        loc=tf.constant(prior_log_sigma_eta_sq_loc, tf.float32),
        scale=tf.constant(prior_log_sigma_eta_sq_scale, tf.float32),
    )

    def target_log_prob(theta_raw):
        # Fix common-random-number seed across HMC evaluations so the PF
        # estimator is theta-only (deterministic in theta within a chain),
        # restoring leapfrog energy conservation. See §5 of the HMC
        # derivations report.
        tf.random.set_seed(int(crn_seed))

        mu = theta_raw[0]
        phi_raw = theta_raw[1]
        log_sigma_eta_sq = theta_raw[2]

        phi = tf.tanh(phi_raw)
        sigma_eta_sq = tf.exp(log_sigma_eta_sq)

        log_prior = (prior_mu.log_prob(mu)
                     + prior_phi_raw.log_prob(phi_raw)
                     + prior_log_sigma_eta_sq.log_prob(log_sigma_eta_sq))

        if no_likelihood:
            # Prior-only baseline: skip the filter entirely. Used to measure
            # the empirical "attractor" location in prior space.
            return log_prior

        log_lik = ll(mu, phi, sigma_eta_sq, y_obs)
        log_lik = tf.cast(tf.math.real(log_lik), tf.float32)
        # Use -inf (not -1e6) for non-finite likelihood. The -1e6 sentinel
        # saturates the MH ratio between two bad states to 0 (accept=1),
        # which fools dual averaging into thinking giant step sizes are
        # safe. -inf makes MH reject the proposal cleanly.
        log_lik = tf.where(tf.math.is_finite(log_lik), log_lik,
                           tf.constant(-np.inf, tf.float32))

        return log_prior + log_lik

    return target_log_prob


def unconstrain(mu: float, phi: float, sigma_eta_sq: float) -> np.ndarray:
    """Map (mu, phi, sigma_eta_sq) -> unconstrained theta_raw."""
    return np.asarray([mu, np.arctanh(np.clip(phi, -0.9999, 0.9999)),
                       np.log(max(sigma_eta_sq, 1e-8))], dtype=np.float32)


def constrain(theta_raw: np.ndarray) -> np.ndarray:
    """Map theta_raw -> (mu, phi, sigma_eta_sq) (vectorised)."""
    th = np.asarray(theta_raw)
    mu = th[..., 0]
    phi = np.tanh(th[..., 1])
    sigma_eta_sq = np.exp(th[..., 2])
    return np.stack([mu, phi, sigma_eta_sq], axis=-1)


# ---------------------------------------------------------------------------
# Single-chain HMC
# ---------------------------------------------------------------------------

def _build_preconditioned_kernel(target_log_prob_fn, step_size, num_leapfrog,
                                  momentum_dist, target_accept_prob,
                                  num_adaptation_steps):
    """PreconditionedHMC, optionally wrapped in DualAveragingStepSizeAdaptation.

    If num_adaptation_steps is None, returns the bare PreconditionedHMC (no
    step-size adapt — used in the sampling phase after warmup).

    Layering when adapted:
        DualAverage.results
          .inner_results = MetropolisHastingsKernelResults
            .accepted_results = UncalibratedPreconditionedHMC.results
              .step_size
            .log_accept_ratio
    """
    inner = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=tf.constant(step_size, tf.float32),
        num_leapfrog_steps=int(num_leapfrog),
        momentum_distribution=momentum_dist,
    )
    if num_adaptation_steps is None:
        return inner
    return tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=inner,
        num_adaptation_steps=int(num_adaptation_steps),
        target_accept_prob=tf.constant(target_accept_prob, tf.float32),
        step_size_setter_fn=(
            lambda kr, ss: kr._replace(
                accepted_results=kr.accepted_results._replace(step_size=ss))
        ),
        step_size_getter_fn=lambda kr: kr.accepted_results.step_size,
        log_accept_prob_getter_fn=lambda kr: kr.log_accept_ratio,
    )


def _trace_preconditioned(_, kr):
    """Unified trace_fn that handles both adapted and non-adapted kernels."""
    if hasattr(kr, "new_step_size"):
        step = kr.new_step_size
        mh = kr.inner_results
    else:
        mh = kr
        step = mh.accepted_results.step_size
    accepted = mh.is_accepted
    lp = mh.accepted_results.target_log_prob
    return accepted, tf.cast(lp, tf.float32), tf.cast(step, tf.float32)


def _run_window(target_log_prob_fn, init_state, num_steps, step_size,
                num_leapfrog, momentum_dist, target_accept_prob,
                adapt_step_size, seed):
    """Run num_steps HMC steps under PreconditionedHMC (with optional
    step-size adapt). Returns (final_state, samples, acc, lp, step_arr,
    final_step_size).

    When adapt_step_size=True, the returned final_step_size is the SMOOTHED
    estimate (exp(log_averaging_step) from dual averaging) — Hoffman &
    Gelman's recommendation, what Stan uses. This is much more robust than
    the noisy last `new_step_size`, especially for short windows where DA
    hasn't fully converged.
    """
    num_adapt = int(num_steps) if adapt_step_size else None
    kernel = _build_preconditioned_kernel(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog=num_leapfrog,
        momentum_dist=momentum_dist,
        target_accept_prob=target_accept_prob,
        num_adaptation_steps=num_adapt,
    )
    if adapt_step_size:
        samples, (acc, lp, step_arr), final_kr = tfm.sample_chain(
            num_results=int(num_steps),
            num_burnin_steps=0,
            current_state=init_state,
            kernel=kernel,
            trace_fn=_trace_preconditioned,
            seed=int(seed),
            return_final_kernel_results=True,
        )
        # log_averaging_step: smoothed log step size from Nesterov dual
        # averaging. Falls back to last `new_step_size` if zero/missing.
        try:
            arr = np.asarray(tf.exp(final_kr.log_averaging_step).numpy())
            smoothed = float(arr.flatten()[0]) if arr.size > 0 else 0.0
        except (AttributeError, ValueError):
            smoothed = 0.0
        if smoothed > 0.0:
            final_step = smoothed
        else:
            final_step = float(np.asarray(step_arr[-1].numpy()).flatten()[-1])
    else:
        samples, (acc, lp, step_arr) = tfm.sample_chain(
            num_results=int(num_steps),
            num_burnin_steps=0,
            current_state=init_state,
            kernel=kernel,
            trace_fn=_trace_preconditioned,
            seed=int(seed),
        )
        # Sampling phase: step size is frozen. No need to read from step_arr.
        final_step = float(step_size)
    final_state = samples[-1]
    return final_state, samples, acc, lp, step_arr, final_step


def run_chain_windowed_proper(target_log_prob_fn, init_raw, num_results,
                                num_burnin, step_size, num_leapfrog, seed,
                                target_accept_prob=0.65, progress_every=0,
                                dense_mass=True):
    """Stan-style windowed adaptive HMC (single chain).

    Three warmup phases:
      1. Init window (~10% of burnin): identity mass, step-size adapt only.
      2. Expanding windows (~80% of burnin), three of them in ~1:2:4 ratio.
         In each window: PreconditionedHMC with current mass, step-size
         adapt. At the end of each window: estimate posterior covariance
         from window samples, build new momentum_distribution, RESET
         step-size adapter for the next window.
      3. Term window (~10% of burnin): step-size adapt with frozen mass.
    Sampling: no adaptation; final step size + final mass matrix.

    Mass matrix convention (Stan/TFP standard):
        mass matrix M := posterior_covariance^{-1}
        momentum_distribution covariance := M = posterior_covariance^{-1}
    So scale of the momentum distribution is the INVERSE-cov (precision),
    not the covariance itself. The leapfrog position update q += eps*M^{-1}*p
    then scales by posterior covariance, so high-variance dims take
    proportionally larger steps. DiagonalMassMatrixAdaptation in TFP does
    this inversion automatically; our hand-rolled version must do it
    explicitly. (An earlier v4 implementation passed sample variance
    directly as momentum scale^2 — wrong direction — but the bug was
    partially masked because all three SVSSM parameters happen to have
    similar posterior scale.)

    Two mass-matrix modes:
      dense_mass=True  (default): MultivariateNormalTriL momentum with
        full inverse-covariance matrix. Handles off-diagonal correlations
        like the (phi, sigma_eta^2) ridge (~-0.79 at T=100).
      dense_mass=False: diagonal-only. Cheaper per window but cannot
        correct off-diagonal correlations.

    Resetting the step-size adapter at each window boundary prevents
    the runaway pathology observed with the monolithic
    DualAverage[DiagonalMass[PreconditionedHMC]] stack: there, the two
    adapters update independently every step and can co-amplify, sending
    step size to inf when one chain hits a bad geometry.
    """
    init = tf.constant(init_raw, dtype=tf.float32)
    state_dim = int(init.shape[0])
    B = int(num_burnin)

    # Budget: 10% init, then 3 expanding windows in roughly 1:2:5 ratio.
    # KEY DESIGN: there is NO separate "term" phase, and the LAST expanding
    # window does NOT update the mass matrix at its end. Instead, the last
    # window is enlarged so it serves the dual role of (a) last mass-matrix
    # estimation window and (b) Stan's "term" window where the step size
    # converges against the final mass matrix.
    #
    # Earlier versions had a separate short term phase. That failed because:
    # mass-from-W3 differs from mass-from-W2 (which adapted W3's step), so
    # at the start of term the step had to re-equilibrate. With only ~8-16
    # term steps, dual averaging couldn't fully settle and one chain would
    # land at a step size that worked locally but tanked sampling acceptance.
    #
    # In the new scheme, the last expanding window IS the term window: step
    # size is adapted against the SAME mass matrix that sampling will use,
    # and the smoothed log_averaging_step from that window is the final step.
    phase1_n = max(int(B * 0.10), 5)
    phase2_total = max(B - phase1_n, 18)
    w1 = max(phase2_total // 8, 4)
    w2 = max(phase2_total // 4, 6)
    w3 = max(phase2_total - w1 - w2, 8)  # the super-window (W3 + term role)

    # ---- Mass-matrix helpers ----
    # All factories return a momentum distribution whose covariance is the
    # MASS matrix M = posterior_cov^{-1}. The position update inside
    # PreconditionedHMC then uses M^{-1} = posterior_cov, taking larger
    # steps in high-variance dims (Stan's convention).
    def make_momentum_diag(precision_diag):
        """precision_diag: (d,) elements = 1/posterior_variance_per_dim."""
        return tfd.Independent(
            tfd.Normal(loc=tf.zeros([state_dim], tf.float32),
                       scale=tf.sqrt(precision_diag)),
            reinterpreted_batch_ndims=1,
        )

    def make_momentum_dense(precision):
        """precision: (d, d) = inverse posterior covariance.

        scale_tril for MVNTriL must be the Cholesky of the DISTRIBUTION
        covariance, which here equals the mass matrix precision.
        """
        return tfd.MultivariateNormalTriL(
            loc=tf.zeros([state_dim], tf.float32),
            scale_tril=tf.linalg.cholesky(precision),
        )

    def diag_precision_from_window(win_samples):
        var = tf.math.reduce_variance(win_samples, axis=0)
        # Clip variance to [0.1, 10] -> precision (1/var) in [0.1, 10].
        # Identity mass corresponds to var=1, so [0.1, 10] gives a 10x
        # rescale range either side. Prevents the briefly-stuck-chain
        # pathology where var -> 0 makes precision (and trajectory
        # length) explode.
        var = tf.clip_by_value(var, 0.1, 10.0)
        return 1.0 / var

    def dense_precision_from_window(win_samples):
        cov = tfp.stats.covariance(win_samples)
        # Add a tiny ridge for numerical stability before eigh.
        cov = cov + tf.eye(state_dim, dtype=tf.float32) * 1e-4
        cov = 0.5 * (cov + tf.linalg.matrix_transpose(cov))
        # Eigen-clip: bound eigenvalues of the posterior-cov estimate to
        # [0.1, 10]. This is the dense analogue of the diagonal clip,
        # preserving the EIGENVECTORS (correlation structure) but
        # bounding the EIGENVALUES (per-direction scale).
        e, v = tf.linalg.eigh(cov)
        e_clip = tf.clip_by_value(e, 0.1, 10.0)
        cov_clip = tf.matmul(
            tf.matmul(v, tf.linalg.diag(e_clip)),
            tf.linalg.matrix_transpose(v),
        )
        cov_clip = 0.5 * (cov_clip + tf.linalg.matrix_transpose(cov_clip))
        precision = tf.linalg.inv(cov_clip)
        precision = 0.5 * (precision + tf.linalg.matrix_transpose(precision))
        return precision

    if progress_every > 0:
        mass_type = "DENSE (full Sigma^-1)" if dense_mass else "diagonal (1/var)"
        print(f"      [windowed] phases: init={phase1_n}, "
              f"win=[{w1},{w2},{w3}*], samp={num_results}  "
              f"mass={mass_type}  "
              f"(* = last window does not update mass; serves as term)",
              flush=True)

    state = init
    current_step = float(step_size)
    # Initial momentum: identity mass (no adaptation yet).
    if dense_mass:
        momentum = make_momentum_dense(
            tf.eye(state_dim, dtype=tf.float32))
    else:
        momentum = make_momentum_diag(
            tf.ones([state_dim], tf.float32))

    t0 = time.perf_counter()

    state, _, _, _, _, current_step = _run_window(
        target_log_prob_fn, state, phase1_n, current_step, num_leapfrog,
        momentum, target_accept_prob, adapt_step_size=True, seed=seed,
    )
    current_step = float(np.clip(current_step, 1e-4, 0.3))
    if progress_every > 0:
        print(f"      [windowed] init done   eps={current_step:.5f}", flush=True)

    windows = [w1, w2, w3]
    last_idx = len(windows) - 1
    for wi, wsize in enumerate(windows):
        state, samples_w, acc_w, _, _, current_step = _run_window(
            target_log_prob_fn, state, wsize, current_step, num_leapfrog,
            momentum, target_accept_prob, adapt_step_size=True,
            seed=seed + 10_000 * (wi + 1),
        )
        # Cap step size to a sensible range. Healthy adapted step sizes for
        # this target are in [0.03, 0.30]; capping at [1e-4, 0.3] keeps
        # DA within the empirically-healthy regime. The 0.5 cap used
        # earlier let DA grow into a non-mixing regime (eps=0.5 with
        # mass-frozen sampling gave accept <5% on 2/4 chains in a 72-min
        # laptop test); 0.3 cap + target_accept 0.8 keeps all chains
        # in the productive range.
        current_step = float(np.clip(current_step, 1e-4, 0.3))
        # CRITICAL: the LAST window does NOT update mass. Its end-step is
        # already tuned against the mass matrix that sampling will use.
        win_diag_info = ""
        if wi < last_idx:
            skip = max(int(int(wsize) * 0.25), 1)
            win_post = samples_w[skip:]
            if dense_mass:
                # Full covariance estimate; precision = inv(cov_clipped)
                cov_est = tfp.stats.covariance(win_post)
                # Diag + max off-diagonal correlation for logging
                d_vec = tf.linalg.diag_part(cov_est).numpy()
                # Cor matrix (off-diag) for logging only
                std_outer = np.sqrt(np.outer(d_vec, d_vec))
                cor = cov_est.numpy() / np.maximum(std_outer, 1e-9)
                np.fill_diagonal(cor, 0.0)
                max_abs_cor = float(np.max(np.abs(cor)))
                precision = dense_precision_from_window(win_post)
                momentum = make_momentum_dense(precision)
                win_diag_info = (
                    f"  diag_var=[{d_vec[0]:.3f},{d_vec[1]:.3f},"
                    f"{d_vec[2]:.3f}]  max|corr|={max_abs_cor:.2f}"
                )
            else:
                precision_diag = diag_precision_from_window(win_post)
                # diag_var for logging (precision_diag = 1/var_clipped)
                d_vec = (1.0 / precision_diag).numpy()
                momentum = make_momentum_diag(precision_diag)
                win_diag_info = (
                    f"  diag_var=[{d_vec[0]:.3f},{d_vec[1]:.3f},"
                    f"{d_vec[2]:.3f}]"
                )
        if progress_every > 0:
            acc_rate = float(tf.reduce_mean(tf.cast(acc_w, tf.float32)).numpy())
            mass_tag = "" if wi < last_idx else "  (mass frozen)"
            print(f"      [windowed] win {wi+1}/{len(windows)} (n={wsize}) "
                  f" eps={current_step:.5f}  acc={acc_rate:.2f}{mass_tag}"
                  f"{win_diag_info}",
                  flush=True)

    # Sampling: no adaptation, frozen step size and frozen mass matrix.
    # Chunk into progress_every-sized blocks so we get heartbeat events
    # during sampling (otherwise the chain runs silent between burnin
    # end and final "sampling done", which can be hours at high T).
    # When progress_every == 0 we still call once (no logging path).
    num_results_int = int(num_results)
    chunk_size = progress_every if progress_every > 0 else num_results_int
    chunk_size = max(min(chunk_size, num_results_int), 1)

    samples_chunks, acc_chunks, lp_chunks = [], [], []
    state_sample = state
    samples_done = 0
    sample_t0 = time.perf_counter()
    chunk_idx = 0
    while samples_done < num_results_int:
        this_chunk = min(chunk_size, num_results_int - samples_done)
        state_sample, samp_c, acc_c, lp_c, _, _ = _run_window(
            target_log_prob_fn, state_sample, this_chunk, current_step,
            num_leapfrog, momentum, target_accept_prob,
            adapt_step_size=False,
            seed=seed + 1_234_567 + chunk_idx * 7919,
        )
        samples_chunks.append(samp_c)
        acc_chunks.append(acc_c)
        lp_chunks.append(lp_c)
        samples_done += this_chunk
        chunk_idx += 1

        if progress_every > 0 and samples_done < num_results_int:
            wall_so_far = time.perf_counter() - sample_t0
            avg_per_step = wall_so_far / samples_done
            eta_min = (num_results_int - samples_done) * avg_per_step / 60.0
            acc_so_far = float(tf.reduce_mean(tf.cast(
                tf.concat(acc_chunks, axis=0), tf.float32)).numpy())
            print(f"      [windowed] sampled {samples_done}/{num_results_int}"
                  f"  eps={current_step:.5f}  acc={acc_so_far:.2f}  "
                  f"avg/step={avg_per_step:.1f}s  eta={eta_min:.1f}min",
                  flush=True)

    samples = tf.concat(samples_chunks, axis=0)
    acc = tf.concat(acc_chunks, axis=0)
    lp = tf.concat(lp_chunks, axis=0)
    elapsed = time.perf_counter() - t0
    if progress_every > 0:
        acc_rate = float(tf.reduce_mean(tf.cast(acc, tf.float32)).numpy())
        print(f"      [windowed] sampling done eps={current_step:.5f} "
              f"acc={acc_rate:.2f}  chain done in {elapsed:.1f}s",
              flush=True)

    # Step size is frozen during sampling — synthesize the (num_results,)
    # array directly (the kernel-traced step_arr can have inconsistent shape
    # when the bare PreconditionedHMC is used without DualAverage wrap).
    step_arr_np = np.full(int(num_results), float(current_step), dtype=np.float32)
    return samples.numpy(), acc.numpy(), lp.numpy(), step_arr_np


def run_chain(target_log_prob_fn, init_raw, num_results, num_burnin,
              step_size, num_leapfrog, seed, target_accept_prob=None,
              progress_every=0, use_windowed_adaptive=False,
              dense_mass=True):
    """Run a single HMC chain.

    Two kernels supported:
      - Default (use_windowed_adaptive=False): vanilla HMC + dual-averaging
        step-size adaptation. Identity mass matrix. The Phases-5-12
        production setup. Default target_accept_prob = 0.65.
      - use_windowed_adaptive=True: Stan-style windowed adaptive HMC.
        See run_chain_windowed_proper for the algorithm.
        - dense_mass=True (default): full inverse-covariance mass matrix.
        - dense_mass=False: diagonal-only mass matrix.
    """
    if use_windowed_adaptive:
        # 0.65: Hoffman & Gelman's original recommendation for fixed-L HMC.
        # v5 laptop A/B (target=0.80, cap=0.3) was over-cautious: chain 1
        # adapted eps=0.014 with acc=0.87, barely moving. 0.65 + cap=0.3
        # gives 3/4 healthy chains at moderate steps (v4 22-min profile).
        tap = 0.65 if target_accept_prob is None else float(target_accept_prob)
        return run_chain_windowed_proper(
            target_log_prob_fn=target_log_prob_fn,
            init_raw=init_raw,
            num_results=num_results,
            num_burnin=num_burnin,
            step_size=step_size,
            num_leapfrog=num_leapfrog,
            seed=seed,
            target_accept_prob=tap,
            progress_every=progress_every,
            dense_mass=dense_mass,
        )
    init = tf.constant(init_raw, dtype=tf.float32)
    # Vanilla HMC default: 0.65 (Hoffman & Gelman's original recommendation
    # for fixed-L HMC; less conservative than the windowed branch's 0.80
    # because vanilla has no mass-matrix-induced geometry shifts).
    tap = 0.65 if target_accept_prob is None else float(target_accept_prob)

    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=tf.constant(step_size, tf.float32),
        num_leapfrog_steps=int(num_leapfrog),
    )
    kernel = tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(num_burnin),
        target_accept_prob=tf.constant(tap, tf.float32),
    )

    def trace_fn(_, kr):
        # Drill through all wrapper layers to reach the base sampler's results.
        inner = kr
        while hasattr(inner, "inner_results"):
            inner = inner.inner_results
        accepted = tf.cast(inner.is_accepted, tf.bool) \
            if hasattr(inner, "is_accepted") else tf.constant(False)
        if hasattr(inner, "target_log_prob"):
            lp = inner.target_log_prob
        elif hasattr(inner, "accepted_results") and hasattr(inner.accepted_results, "target_log_prob"):
            lp = inner.accepted_results.target_log_prob
        else:
            lp = tf.constant(float("nan"), tf.float32)
        step = kr.new_step_size if hasattr(kr, "new_step_size") \
            else tf.constant(float("nan"), tf.float32)
        return accepted, tf.cast(lp, tf.float32), tf.cast(step, tf.float32)

    if progress_every <= 0:
        samples, (acc, lp, step) = tfm.sample_chain(
            num_results=int(num_results),
            num_burnin_steps=int(num_burnin),
            current_state=init,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=int(seed),
        )
        return (samples.numpy(), acc.numpy(), lp.numpy(), step.numpy())

    # Step-by-step path with progress
    state = init
    kr = kernel.bootstrap_results(state)
    samples_list, acc_list, lp_list, step_list = [], [], [], []
    total = int(num_burnin) + int(num_results)
    t0 = time.perf_counter()
    for s in range(total):
        state, kr = kernel.one_step(state, kr, seed=[int(seed), s + 1])
        a, lp_s, st_s = trace_fn(state, kr)
        acc_list.append(tf.identity(a))
        lp_list.append(tf.identity(lp_s))
        step_list.append(tf.identity(st_s))
        if s >= int(num_burnin):
            samples_list.append(tf.identity(state))
        if progress_every > 0 and (s + 1) % progress_every == 0:
            phase = "burn" if s < int(num_burnin) else "samp"
            print(f"      [{phase}] step {s+1}/{total}  "
                  f"eps={float(st_s):.5f}  acc={float(a)}",
                  flush=True)
    elapsed = time.perf_counter() - t0
    print(f"      chain done in {elapsed:.1f}s")
    samples = tf.stack(samples_list, axis=0).numpy()
    acc = tf.stack(acc_list, axis=0).numpy()
    lp = tf.stack(lp_list, axis=0).numpy()
    step_arr = tf.stack(step_list, axis=0).numpy()
    return samples, acc, lp, step_arr


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def split_rhat(samples: np.ndarray) -> np.ndarray:
    """Split-Rhat across chains, per parameter. samples: (chains, draws, dim).

    Returns NaN per parameter if fewer than 2 chains (Rhat is undefined).
    """
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


def ess_bulk(samples: np.ndarray) -> np.ndarray:
    """Effective sample size, bulk (per-parameter).

    Loops parameter-by-parameter so TFP's effective_sample_size sees a
    clean (chains, draws) tensor and cross_chain_dims=[0] broadcasts. With
    1 chain, computes per-chain ESS.
    """
    chains, draws, dim = samples.shape
    out = np.zeros(dim, dtype=np.float64)
    for p in range(dim):
        slice_p = tf.constant(samples[:, :, p], dtype=tf.float32)  # (chains, draws)
        if chains >= 2:
            ess_p = tfp.mcmc.effective_sample_size(slice_p, cross_chain_dims=[0])
        else:
            ess_p = tfp.mcmc.effective_sample_size(slice_p[0])
        out[p] = float(ess_p.numpy())
    return out


def summarise_posterior(samples_constrained: np.ndarray, truth: dict) -> list[dict]:
    """Posterior summary per parameter. samples: (chains, draws, 3)."""
    names = ["mu", "phi", "sigma_eta_sq"]
    truth_arr = np.asarray([truth["mu"], truth["phi"], truth["sigma_eta"] ** 2])
    rhat = split_rhat(samples_constrained)
    ess = ess_bulk(samples_constrained)

    flat = samples_constrained.reshape(-1, 3)
    rows = []
    for i, name in enumerate(names):
        x = flat[:, i]
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1))
        q025, q50, q975 = np.percentile(x, [2.5, 50.0, 97.5])
        truth_val = float(truth_arr[i])
        covered = bool(q025 <= truth_val <= q975)
        rows.append({
            "param": name, "truth": truth_val,
            "mean": mean, "std": std, "median": float(q50),
            "q025": float(q025), "q975": float(q975),
            "rhat": float(rhat[i]), "ess_bulk": float(ess[i]),
            "covered_95ci": covered,
            "bias_pct": (float(100.0 * (mean - truth_val) / abs(truth_val))
                         if abs(truth_val) > 1e-3 else float("nan")),
            "abs_bias": float(mean - truth_val),
        })
    # sigma_eta (not _sq) summary derived from sqrt
    se = np.sqrt(np.maximum(flat[:, 2], 1e-12))
    rows.append({
        "param": "sigma_eta (derived)",
        "truth": truth["sigma_eta"],
        "mean": float(np.mean(se)), "std": float(np.std(se, ddof=1)),
        "median": float(np.percentile(se, 50.0)),
        "q025": float(np.percentile(se, 2.5)),
        "q975": float(np.percentile(se, 97.5)),
        "rhat": float("nan"), "ess_bulk": float("nan"),
        "covered_95ci": bool(np.percentile(se, 2.5) <= truth["sigma_eta"] <= np.percentile(se, 97.5)),
        "bias_pct": (float(100.0 * (np.mean(se) - truth["sigma_eta"]) / abs(truth["sigma_eta"]))
                     if abs(truth["sigma_eta"]) > 1e-3 else float("nan")),
        "abs_bias": float(np.mean(se) - truth["sigma_eta"]),
    })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_reports_and_plots(args, out_dir: Path, truth: dict, rows: list,
                              samples_raw: np.ndarray, samples_constrained: np.ndarray,
                              accs: np.ndarray, lps: np.ndarray,
                              all_step_sizes: list, elapsed: float) -> int:
    """Print summary, save .npz (defensive), text + JSON reports, and plots."""
    # Defensive .npz of the stacked arrays (alongside the per-chain files if
    # they exist; the per-chain ones survive aggregate-mode crashes too).
    np.savez_compressed(
        out_dir / "svssm_hmc_samples.npz",
        samples_raw=samples_raw,
        samples_constrained=samples_constrained,
        accept=accs,
        log_prob=lps,
        step_size=np.stack(all_step_sizes, axis=0)
            if isinstance(all_step_sizes, list) and all_step_sizes and isinstance(all_step_sizes[0], np.ndarray)
            else np.asarray(all_step_sizes),
        truth=np.asarray([args.mu, args.phi, args.sigma_eta]),
    )
    print(f"\n  [defensive save] wrote {out_dir / 'svssm_hmc_samples.npz'}")

    print("\n" + "=" * 100)
    print(f"SVSSM HMC parameter recovery (total wall: {elapsed:.1f}s)")
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
    final_steps = [float(s[-1]) for s in all_step_sizes]
    print(f"Final step sizes: {final_steps}")
    print("=" * 100)

    text_rows = [
        "=" * 100,
        "SVSSM HMC Parameter Recovery Report",
        "=" * 100,
        f"TF {tf.__version__}, TFP {tfp.__version__}",
        f"truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}",
        f"T={args.T} N={args.N} n_lambda={args.n_lambda} K={args.K} L={args.L} "
        f"step_size={args.step_size} dispersion={args.dispersion}",
        f"num_chains={args.num_chains} burnin={args.num_burnin} results={args.num_results}",
        f"Wall (max-chain if parallel; sum if sequential): {elapsed:.1f}s",
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
        f"Final step sizes:      {final_steps}",
        "",
        "Notes:",
        " - sigma_eta bias under the HRS Gaussian quasi-likelihood is",
        "   ~+30-50% on the likelihood maximum; the posterior bias is",
        "   T-dependent (about -15% at T=50, +3% at T=100 in our runs).",
        " - For exact inference, swap in the Kim-Shephard-Chib (1998)",
        "   7-component Gaussian-mixture refinement of the log(eps^2) noise.",
        "=" * 100,
    ]
    (out_dir / "svssm_hmc_results.txt").write_text("\n".join(text_rows))

    json_rows = [{k: v for k, v in r.items()} for r in rows]
    (out_dir / "svssm_hmc_summary.json").write_text(json.dumps({
        "tf": tf.__version__, "tfp": tfp.__version__,
        "config": vars(args), "truth": truth, "rows": json_rows,
        "accept_rate_per_chain": per_chain_acc,
        "accept_rate_overall": float(accs.mean()),
        "final_step_sizes": final_steps,
        "elapsed_s": elapsed,
    }, indent=2))

    if _HAVE_MPL:
        names = ["mu", "phi", "sigma_eta_sq"]
        truth_arr = np.asarray([args.mu, args.phi, args.sigma_eta ** 2])

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
        fig.suptitle(f"SVSSM HMC traces (T={args.T}, N={args.N})")
        fig.tight_layout()
        fig.savefig(out_dir / "svssm_hmc_traces.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        flat = samples_constrained.reshape(-1, 3)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, (ax, name) in enumerate(zip(axes, names)):
            ax.hist(flat[:, i], bins=40, alpha=0.7, color="steelblue", edgecolor="white")
            ax.axvline(truth_arr[i], color="red", linestyle="--", lw=1.2,
                       label=f"truth = {truth_arr[i]:.3f}")
            ax.axvline(flat[:, i].mean(), color="black", linestyle=":", lw=1.2,
                       label=f"post mean = {flat[:, i].mean():.3f}")
            ax.set_xlabel(name)
            ax.set_ylabel("count")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
        fig.suptitle(f"SVSSM HMC posterior marginals (T={args.T}, N={args.N})")
        fig.tight_layout()
        fig.savefig(out_dir / "svssm_hmc_marginals.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"\nPlots: {out_dir / 'svssm_hmc_traces.png'} "
              f"{out_dir / 'svssm_hmc_marginals.png'}")

    print(f"\nWrote: {out_dir / 'svssm_hmc_results.txt'}")
    print(f"       {out_dir / 'svssm_hmc_summary.json'}")
    print(f"       {out_dir / 'svssm_hmc_samples.npz'}")
    return 0


def _aggregate_from_chains(_args, out_dir: Path) -> tuple:
    """Load all chains/chain_*.npz, stack into the same arrays as the
    sequential path, return (samples_raw, samples_constrained, accs, lps,
    all_step_sizes, elapsed). _args is reserved for future use."""
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--phi", type=float, default=0.95)
    p.add_argument("--sigma_eta", type=float, default=0.3)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--step_size", type=float, default=0.05)
    p.add_argument("--num_chains", type=int, default=2)
    p.add_argument("--num_burnin", type=int, default=200)
    p.add_argument("--num_results", type=int, default=400)
    p.add_argument("--dispersion", type=float, default=0.15)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--progress_every", type=int, default=50)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc")
    p.add_argument("--prior_mu_loc", type=float, default=0.0)
    p.add_argument("--prior_mu_scale", type=float, default=1.0)
    p.add_argument("--prior_phi_raw_loc", type=float, default=2.0)
    p.add_argument("--prior_phi_raw_scale", type=float, default=0.5)
    p.add_argument("--prior_log_sigma_eta_sq_loc", type=float, default=-2.0)
    p.add_argument("--prior_log_sigma_eta_sq_scale", type=float, default=1.0)
    # Per-chain process isolation (for parallel launchers)
    p.add_argument("--chain_id", type=int, default=None,
                   help="If set, run ONLY this chain index (0-based) and save to "
                        "<out_dir>/chains/chain_{id}.npz. Use with the parallel launcher.")
    p.add_argument("--aggregate", action="store_true",
                   help="Skip running; load all <out_dir>/chains/chain_*.npz, "
                        "compute diagnostics, write combined report.")
    # Modern HMC kernel toggle: NUTS + diagonal mass-matrix adaptation +
    # dual-averaging step-size adaptation. Default is the Phase 5-12
    # production setup (vanilla HMC + step-size only). Use this for posteriors
    # with non-spherical geometry (e.g. the SVSSM (phi, sigma_eta) ridge).
    p.add_argument("--use_windowed_adaptive", action="store_true",
                   help="Stan-style windowed adaptive HMC: PreconditionedHMC "
                        "with adapted mass matrix, in expanding windows "
                        "with the step-size adapter reset at each window "
                        "boundary. Fixes the runaway-step-size pathology of "
                        "the monolithic DA[DiagMass[PHMC]] stack.")
    p.add_argument("--diagonal_mass", action="store_true",
                   help="With --use_windowed_adaptive: use a DIAGONAL mass "
                        "matrix (per-dim variance only) instead of the "
                        "default DENSE mass matrix (full inverse "
                        "covariance, handles the (phi, sigma_eta^2) ridge).")
    p.add_argument("--no_likelihood", action="store_true",
                   help="Prior-only baseline: zero out the PF log-likelihood "
                        "so the chain samples from the prior alone. Used to "
                        "measure the empirical attractor location and to "
                        "verify that with-data posteriors actually differ "
                        "from the prior (i.e., the likelihood is informing).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chains_dir = out_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    truth = {"mu": args.mu, "phi": args.phi, "sigma_eta": args.sigma_eta}
    print(f"[svssm-hmc] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  truth: mu={args.mu}, phi={args.phi}, sigma_eta={args.sigma_eta}")
    print(f"  T={args.T} N={args.N} n_lambda={args.n_lambda} K={args.K} "
          f"L={args.L} step_size={args.step_size}")
    if args.use_windowed_adaptive:
        mass_label = "diagonal" if args.diagonal_mass else "DENSE"
        kernel_label = (f"windowed adaptive PreconditionedHMC "
                        f"(Stan-style, {mass_label} mass, expanding "
                        f"windows with step-size adapter reset)")
    else:
        kernel_label = "vanilla HMC + step-size adapt"
    print(f"  kernel: {kernel_label}")
    print(f"  num_chains={args.num_chains} burnin={args.num_burnin} "
          f"results={args.num_results} dispersion={args.dispersion}")
    print(f"  out_dir={out_dir}")
    if args.aggregate:
        print(f"  MODE: aggregate (load chains, compute diagnostics)")
    elif args.chain_id is not None:
        print(f"  MODE: single-chain  chain_id={args.chain_id} of {args.num_chains}")
    else:
        print(f"  MODE: sequential (all {args.num_chains} chains in this process)")

    # === Aggregate mode: skip the run, load chains and go straight to diagnostics ===
    if args.aggregate:
        samples_raw, samples_constrained, accs, lps, all_step_sizes, elapsed = \
            _aggregate_from_chains(args, out_dir)
        rows = summarise_posterior(samples_constrained, truth)
        # jump to report-writing below
        return _write_reports_and_plots(
            args, out_dir, truth, rows,
            samples_raw, samples_constrained, accs, lps, all_step_sizes, elapsed,
        )

    # Data
    y_obs = gen_svssm(args.T, args.mu, args.phi, args.sigma_eta, seed=args.data_seed)
    print(f"  y_obs range: [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")

    # Filter
    ll = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=args.N, n_lambda=args.n_lambda,
        sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
        grad_window=4, jit_compile=True, integrator="exp",
    )

    # Initial states: dispersed around truth in unconstrained space
    truth_raw = unconstrain(args.mu, args.phi, args.sigma_eta ** 2)
    rng = np.random.default_rng(args.base_seed)
    init_raws = [truth_raw + args.dispersion * rng.standard_normal(3).astype(np.float32)
                 for _ in range(args.num_chains)]

    # Run chains sequentially
    all_samples = []
    all_accs = []
    all_lps = []
    all_step_sizes = []
    chain_seeds = [args.base_seed + 1009 * (c + 1) for c in range(args.num_chains)]
    crn_seed = args.base_seed  # shared CRN across chains; eliminates target drift in leapfrog

    t_total = time.perf_counter()
    target_log_prob_fn = make_target_log_prob(
        ll, y_obs, crn_seed=crn_seed,
        prior_mu_loc=args.prior_mu_loc,
        prior_mu_scale=args.prior_mu_scale,
        prior_phi_raw_loc=args.prior_phi_raw_loc,
        prior_phi_raw_scale=args.prior_phi_raw_scale,
        prior_log_sigma_eta_sq_loc=args.prior_log_sigma_eta_sq_loc,
        prior_log_sigma_eta_sq_scale=args.prior_log_sigma_eta_sq_scale,
        no_likelihood=args.no_likelihood,
    )
    if args.no_likelihood:
        print(f"  [PRIOR-ONLY MODE] likelihood term zeroed; sampling prior")
    print(f"  priors: "
          f"mu~N({args.prior_mu_loc},{args.prior_mu_scale}^2), "
          f"phi_raw~N({args.prior_phi_raw_loc},{args.prior_phi_raw_scale}^2), "
          f"log_sigma_eta_sq~N({args.prior_log_sigma_eta_sq_loc},{args.prior_log_sigma_eta_sq_scale}^2)")

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
        elapsed = time.perf_counter() - t0
        chain_path = chains_dir / f"chain_{c}.npz"
        np.savez_compressed(
            chain_path,
            samples_raw=samples,                              # (draws, 3)
            samples_constrained=constrain(samples),           # (draws, 3)
            accept=acc,                                       # (draws,)
            log_prob=lp,                                      # (draws,)
            step_size=step,                                   # (draws,)
            truth=np.asarray([args.mu, args.phi, args.sigma_eta]),
            elapsed_s=np.asarray(elapsed),
            chain_id=np.asarray(c),
        )
        print(f"    chain {c+1} done: {elapsed:.1f}s, accept rate={acc.mean():.3f}, "
              f"final eps={float(step[-1]):.5f}")
        print(f"    saved {chain_path}")
        return 0

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
        print(f"    chain {c+1} done: {time.perf_counter() - t0:.1f}s, "
              f"accept rate={acc.mean():.3f}, final eps={float(step[-1]):.5f}")
        all_samples.append(samples)
        all_accs.append(acc)
        all_lps.append(lp)
        all_step_sizes.append(step)
    elapsed = time.perf_counter() - t_total

    # Stack: (chains, draws, dim)
    samples_raw = np.stack(all_samples, axis=0)
    samples_constrained = constrain(samples_raw)
    accs = np.stack(all_accs, axis=0)
    lps = np.stack(all_lps, axis=0)

    rows = summarise_posterior(samples_constrained, truth)
    return _write_reports_and_plots(
        args, out_dir, truth, rows,
        samples_raw, samples_constrained, accs, lps, all_step_sizes, elapsed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
