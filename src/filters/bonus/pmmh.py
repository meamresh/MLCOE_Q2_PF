"""
Particle Marginal Metropolis-Hastings (PMMH) sampler.

Implements the PMMH algorithm from Andrieu, Doucet & Holenstein (2010)
for Bayesian parameter inference in state-space models. The marginal
likelihood ``p(y_{1:T} | theta)`` is estimated unbiasedly by a bootstrap
particle filter, and a random-walk Metropolis kernel proposes new
parameter values.

This module provides:
    - ``bootstrap_pf_log_likelihood``: bootstrap PF log-likelihood estimator
    - ``run_pmmh``: PMMH sampler using ``tfp.mcmc.RandomWalkMetropolis``

All computations use TensorFlow / TensorFlow Probability.

References
----------
- Andrieu, Doucet & Holenstein (2010), "Particle Markov chain Monte
  Carlo methods", JRSSB.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import tensorflow as tf
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


# =====================================================================
# PMMH result container
# =====================================================================


class PMMHResult(NamedTuple):
    """Container for PMMH sampler output."""

    samples: tf.Tensor  # (num_results, d)
    is_accepted: tf.Tensor  # (num_results,)
    accept_rate: tf.Tensor  # scalar
    target_log_probs: tf.Tensor  # (num_results,)


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

