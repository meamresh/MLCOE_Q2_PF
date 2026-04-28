"""
Particle Marginal Metropolis-Hastings (PMMH) sampler.

Implements the PMMH algorithm from Andrieu, Doucet & Holenstein (2010)
for Bayesian parameter inference in state-space models. The marginal
likelihood ``p(y_{1:T} | theta)`` is estimated unbiasedly by a bootstrap
particle filter, and a random-walk Metropolis kernel proposes new
parameter values.

This module provides:
    - ``bootstrap_pf_log_likelihood``: bootstrap PF log-likelihood estimator
    - ``make_kitagawa_bootstrap_target_log_prob``: log-posterior for the Bonus-1b SSM
    - ``run_pmmh``: PMMH sampler using ``tfp.mcmc.RandomWalkMetropolis``
    - ``run_pmmh_multi_chain``: optional multi-process CPU parallel chains
      (``parallel=True`` with ``y_obs`` + ``bootstrap_num_particles``)

All computations use TensorFlow / TensorFlow Probability.

References
----------
- Andrieu, Doucet & Holenstein (2010), "Particle Markov chain Monte
  Carlo methods", JRSSB.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, NamedTuple, Optional

import tensorflow as tf

from src.models.ssm_katigawa import PMCMCNonlinearSSM
import tensorflow_probability as tfp

tfd = tfp.distributions


# =====================================================================
# Bootstrap Particle Filter log-likelihood
# =====================================================================


def bootstrap_pf_log_likelihood(
    ssm,
    y_obs: tf.Tensor,
    num_particles: int = 200,
) -> tf.Tensor:
    """
    Unbiased log marginal likelihood estimate via bootstrap particle filter.

    Parameters
    ----------
    ssm
        State-space model exposing:
        ``motion_model``, ``measurement_model``,
        ``full_measurement_cov``, ``Q``, and ``state_dim``.
    y_obs : tf.Tensor
        Observations of shape ``(T,)`` or ``(T, obs_dim)``.
    num_particles : int
        Number of particles *N*.

    Returns
    -------
    tf.Tensor
        Scalar log marginal likelihood estimate.
    """

    N = num_particles
    T = int(y_obs.shape[0])
    state_dim = ssm.state_dim
    N_f = tf.cast(N, tf.float32)

    if len(y_obs.shape) == 1:
        y_obs = y_obs[:, tf.newaxis]
    obs_dim = int(y_obs.shape[1]) if y_obs.shape[1] is not None else 1

    I_sd = tf.eye(state_dim, dtype=tf.float32)
    init_var = tf.cast(
        getattr(ssm, "initial_var", float(ssm.Q[0, 0])), tf.float32
    )
    particles = tf.random.normal([N, state_dim]) * tf.sqrt(init_var)
    log_ml = tf.constant(0.0)

    R = ssm.full_measurement_cov(1)
    R_reg = R + tf.eye(obs_dim, dtype=tf.float32) * 1e-8
    R_inv = tf.linalg.inv(R_reg)
    log_det_R = tf.linalg.slogdet(R)[1]
    obs_dim_f = tf.cast(obs_dim, tf.float32)
    log_norm = -0.5 * (
        log_det_R + obs_dim_f * tf.math.log(2.0 * 3.141592653589793)
    )

    for t_int in range(1, T + 1):
        # Transition for X_t when t >= 2
        if t_int >= 2:
            t_f = tf.cast(t_int, tf.float32)
            control = tf.fill([N, 1], t_f)
            preds = ssm.motion_model(particles, control)
            L_Q = tf.linalg.cholesky(ssm.Q + I_sd * 1e-8)
            noise = tf.random.normal([N, state_dim])
            particles = preds + noise @ tf.transpose(L_Q)

        # Full Gaussian log-likelihood (normalisation constant matters for PMMH)
        y_pred = tf.reshape(ssm.measurement_model(particles, None), [N, obs_dim])
        innov = y_obs[t_int - 1][tf.newaxis, :] - y_pred
        log_weights = -0.5 * tf.reduce_sum((innov @ R_inv) * innov, axis=1) + log_norm

        # Log marginal likelihood increment: log( (1/N) sum_k w_k )
        max_lw = tf.reduce_max(log_weights)
        log_sum_w = max_lw + tf.math.log(
            tf.reduce_sum(tf.exp(log_weights - max_lw)) + 1e-30
        )
        log_ml = log_ml + log_sum_w - tf.math.log(N_f)

        # Multinomial resampling (non-differentiable — fine for PMMH)
        norm_log_w = log_weights - log_sum_w
        indices = tf.random.categorical(norm_log_w[tf.newaxis, :], N)[0]
        particles = tf.gather(particles, indices)

    return log_ml


def make_kitagawa_bootstrap_target_log_prob(
    y_obs: tf.Tensor,
    num_particles: int,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Unnormalised log-posterior for ``(log sigma_v^2, log sigma_w^2)`` for the
    nonlinear SSM in Andrieu et al. (2010, Eqs. 14–15) with a bootstrap PF
    marginal likelihood.

    Priors: ``sigma^2 ~ InvGamma(0.01, 0.01)`` with Jacobian for the log
    parameterisation — matches ``exp_part3_bonus1b_*`` experiments.
    """
    prior_v = tfd.InverseGamma(concentration=0.01, scale=0.01)
    prior_w = tfd.InverseGamma(concentration=0.01, scale=0.01)
    y_obs = tf.convert_to_tensor(y_obs, dtype=tf.float32)

    def target(theta: tf.Tensor) -> tf.Tensor:
        log_sv2, log_sw2 = theta[0], theta[1]
        sv2, sw2 = tf.exp(log_sv2), tf.exp(log_sw2)
        lp_prior = prior_v.log_prob(sv2) + prior_w.log_prob(sw2)
        jacobian = log_sv2 + log_sw2
        ssm = PMCMCNonlinearSSM(sigma_v_sq=sv2, sigma_w_sq=sw2)
        ll = bootstrap_pf_log_likelihood(ssm, y_obs, num_particles=num_particles)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-1e6, tf.float32))
        result = lp_prior + jacobian + ll
        result = tf.math.real(result)
        result = tf.cast(result, tf.float32)
        return tf.where(tf.math.is_finite(result), result, tf.constant(-1e6, tf.float32))

    return target


# =====================================================================
# PMMH result container
# =====================================================================


class PMMHResult(NamedTuple):
    """Container for PMMH sampler output."""

    samples: tf.Tensor  # (num_results, d)
    is_accepted: tf.Tensor  # (num_results,)
    accept_rate: tf.Tensor  # scalar
    target_log_probs: tf.Tensor  # (num_results,)


class MultiChainPMMHResult(NamedTuple):
    """Container for multi-chain PMMH output (chain axis stacked first)."""

    samples: tf.Tensor          # (num_chains, num_results, d)
    is_accepted: tf.Tensor      # (num_chains, num_results)
    accept_rate: tf.Tensor      # (num_chains,)
    target_log_probs: tf.Tensor # (num_chains, num_results)


# =====================================================================
# PMMH sampler
# =====================================================================


def run_pmmh(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_state: tf.Tensor,
    num_results: int = 1000,
    num_burnin: int = 500,
    step_size: float = 0.25,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> PMMHResult:
    """
    Run PMMH using ``tfp.mcmc.RandomWalkMetropolis``.

    Parameters
    ----------
    target_log_prob_fn : callable
        Maps parameter vector ``theta`` to scalar log-posterior (including
        log-likelihood estimated by bootstrap PF + log-prior).
    initial_state : tf.Tensor
        Starting point in parameter space, shape ``(d,)``.
    num_results : int
        Number of post-burn-in samples.
    num_burnin : int
        Number of burn-in iterations (discarded).
    step_size : float
        Standard deviation of the Gaussian random-walk proposal.
    seed : int, optional
        RNG seed for reproducibility.
    verbose : bool
        Print progress every 100 iterations.

    Returns
    -------
    PMMHResult
        Named tuple with ``samples``, ``is_accepted``, ``accept_rate``,
        and ``target_log_probs``.
    """

    if seed is not None:
        tf.random.set_seed(seed)

    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size),
    )

    state = tf.identity(initial_state)
    kr = kernel.bootstrap_results(state)

    samples_buf = tf.TensorArray(tf.float32, size=num_results, dynamic_size=False)
    accepted_buf = tf.TensorArray(tf.bool, size=num_results, dynamic_size=False)
    log_prob_buf = tf.TensorArray(tf.float32, size=num_results, dynamic_size=False)
    idx = 0

    total = num_burnin + num_results
    for i in range(total):
        state, kr = kernel.one_step(state, kr)
        if i >= num_burnin:
            samples_buf = samples_buf.write(idx, state)
            accepted_buf = accepted_buf.write(idx, kr.is_accepted)
            # Be robust across TFP versions: prefer a direct field if present.
            lp = getattr(kr, "target_log_prob", None)
            if lp is None:
                accepted = getattr(kr, "accepted_results", None)
                lp = (
                    getattr(accepted, "target_log_prob", None)
                    if accepted is not None
                    else None
                )
                if lp is None:
                    lp = getattr(kr, "log_prob", None)
                if lp is None:
                    lp = tf.constant(0.0, dtype=tf.float32)
            log_prob_buf = log_prob_buf.write(idx, lp)
            idx += 1

        if verbose and (i + 1) % 100 == 0:
            phase = "burn" if i < num_burnin else "sample"
            lp = getattr(kr, "target_log_prob", None)
            if lp is None:
                # TFP versions often store the value under `accepted_results`.
                accepted = getattr(kr, "accepted_results", None)
                lp = getattr(accepted, "target_log_prob", None) if accepted is not None else None
                if lp is None:
                    lp = getattr(kr, "log_prob", None)
                if lp is None:
                    lp = tf.constant(0.0, dtype=tf.float32)
            tf.print(
                f"  PMMH {i + 1}/{total} ({phase})",
                "  lp =",
                lp,
            )

    samples = samples_buf.stack()
    is_accepted = accepted_buf.stack()
    accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
    target_log_probs = log_prob_buf.stack()

    return PMMHResult(
        samples=samples,
        is_accepted=is_accepted,
        accept_rate=accept_rate,
        target_log_probs=target_log_probs,
    )


def _pmmh_kitagawa_chain_worker(payload: tuple) -> dict:
    """Picklable entry point for ``ProcessPoolExecutor`` (one PMMH chain)."""
    (
        y_obs_np,
        init_state_np,
        num_results,
        num_burnin,
        step_size,
        chain_seed,
        num_particles,
        verbose,
    ) = payload
    import os as _os

    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as _tf

    try:
        _tf.config.threading.set_intra_op_parallelism_threads(1)
        _tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    y_obs = _tf.constant(y_obs_np, dtype=_tf.float32)
    initial_state = _tf.constant(init_state_np, dtype=_tf.float32)
    target = make_kitagawa_bootstrap_target_log_prob(y_obs, num_particles)
    result = run_pmmh(
        target_log_prob_fn=target,
        initial_state=initial_state,
        num_results=num_results,
        num_burnin=num_burnin,
        step_size=step_size,
        seed=chain_seed,
        verbose=verbose,
    )
    return {
        "samples": result.samples.numpy(),
        "is_accepted": result.is_accepted.numpy(),
        "accept_rate": float(result.accept_rate.numpy()),
        "target_log_probs": result.target_log_probs.numpy(),
    }


def run_pmmh_multi_chain(
    target_log_prob_fn: Optional[Callable[[tf.Tensor], tf.Tensor]],
    initial_states: tf.Tensor,
    num_results: int = 1000,
    num_burnin: int = 500,
    step_size: float = 0.25,
    seed: Optional[int] = None,
    verbose: bool = True,
    *,
    parallel: bool = False,
    num_workers: Optional[int] = None,
    y_obs: Optional[tf.Tensor] = None,
    bootstrap_num_particles: Optional[int] = None,
) -> MultiChainPMMHResult:
    """Run independent PMMH chains with dispersed initial points.

    Per-chain seeds are derived from ``seed + 1009 * chain_idx`` so the
    bootstrap PF realisations and proposal noise are different across
    chains — required for valid R-hat / cross-chain ESS computation.

    Parameters
    ----------
    initial_states : tf.Tensor, shape ``(num_chains, d)``
        Dispersed starting points (e.g. via ``disperse_initial_states``).
    target_log_prob_fn :
        Required when ``parallel=False``. Ignored when ``parallel=True``
        (the target is rebuilt in each process from ``y_obs`` and
        ``bootstrap_num_particles``).
    parallel : bool
        If True and ``num_chains > 1``, run chains in separate processes
        (good for multi-core CPU). Requires ``y_obs`` and
        ``bootstrap_num_particles`` (Kitagawa / Andrieu SSM + bootstrap PF).
    num_workers : optional int
        Process pool size; default ``min(num_chains, os.cpu_count())``.
    y_obs, bootstrap_num_particles
        Passed to :func:`make_kitagawa_bootstrap_target_log_prob` in worker
        processes when ``parallel=True``.

    Other args forwarded to :func:`run_pmmh`.
    """
    base_seed = seed if seed is not None else 42
    num_chains = int(initial_states.shape[0])
    use_parallel = bool(parallel) and num_chains > 1

    if use_parallel:
        if y_obs is None or bootstrap_num_particles is None:
            raise ValueError(
                "parallel=True requires y_obs and bootstrap_num_particles "
                "(Kitagawa bootstrap-PF target is rebuilt per worker)."
            )
        y_obs_np = tf.convert_to_tensor(y_obs, dtype=tf.float32).numpy()
        workers = num_workers if num_workers is not None else min(
            num_chains, os.cpu_count() or 1
        )
        workers = max(1, min(workers, num_chains))
        if verbose:
            print(
                f"\n[PMMH parallel]  {num_chains} chains  workers={workers}  "
                f"(TF intra/inter threads capped at 1 per worker)"
            )

        payloads: list[tuple] = []
        for c in range(num_chains):
            chain_seed = base_seed + 1009 * (c + 1)
            if verbose:
                print(
                    f"  chain {c + 1}/{num_chains}  seed={chain_seed}  "
                    f"init={initial_states[c].numpy().tolist()}"
                )
            payloads.append(
                (
                    y_obs_np,
                    initial_states[c].numpy(),
                    num_results,
                    num_burnin,
                    step_size,
                    chain_seed,
                    int(bootstrap_num_particles),
                    False,
                )
            )

        with ProcessPoolExecutor(max_workers=workers) as executor:
            raw_results = list(executor.map(_pmmh_kitagawa_chain_worker, payloads))

        samples_list = [tf.constant(r["samples"], dtype=tf.float32) for r in raw_results]
        accepted_list = [
            tf.constant(r["is_accepted"], dtype=tf.bool) for r in raw_results
        ]
        accept_rate_list = [
            tf.constant(r["accept_rate"], dtype=tf.float32) for r in raw_results
        ]
        lp_list = [
            tf.constant(r["target_log_probs"], dtype=tf.float32) for r in raw_results
        ]

        return MultiChainPMMHResult(
            samples=tf.stack(samples_list, axis=0),
            is_accepted=tf.stack(accepted_list, axis=0),
            accept_rate=tf.stack(accept_rate_list, axis=0),
            target_log_probs=tf.stack(lp_list, axis=0),
        )

    if target_log_prob_fn is None:
        raise ValueError("target_log_prob_fn is required when parallel=False")

    samples_list = []
    accepted_list = []
    accept_rate_list = []
    lp_list = []

    for c in range(num_chains):
        chain_seed = base_seed + 1009 * (c + 1)
        if verbose:
            print(f"\n[PMMH chain {c + 1}/{num_chains}]  seed={chain_seed}"
                  f"  init={initial_states[c].numpy().tolist()}")
        result = run_pmmh(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=initial_states[c],
            num_results=num_results,
            num_burnin=num_burnin,
            step_size=step_size,
            seed=chain_seed,
            verbose=verbose,
        )
        samples_list.append(result.samples)
        accepted_list.append(result.is_accepted)
        accept_rate_list.append(result.accept_rate)
        lp_list.append(result.target_log_probs)

    return MultiChainPMMHResult(
        samples=tf.stack(samples_list, axis=0),
        is_accepted=tf.stack(accepted_list, axis=0),
        accept_rate=tf.stack(accept_rate_list, axis=0),
        target_log_probs=tf.stack(lp_list, axis=0),
    )

