"""
lhnn_nuts.py
============
No-U-Turn Sampler (NUTS) using a trained Latent-HNN gradient surrogate
with online H_theta error monitoring + real-gradient fallback +
final-state MH correction against the REAL target.

Why this exists
---------------
Our existing `run_lhnn_hmc` uses *fixed-L* leapfrog HMC with L=5. At
step_size=0.01, that gives trajectory length 0.05 in q-space — far too
short for |H_theta - H_0| to drift past the paper's error threshold
(~10). Result: the online error monitor never fires (0 fallbacks across
54,000 surrogate gradient evaluations in the T=200 wide-shifted run),
and the chain gets stuck in surrogate-induced fake potential basins
without any warning signal.

Dhulipala et al. 2022 explicitly pair L-HNN with NUTS. NUTS's tree
doubling lets trajectories grow up to 2^max_treedepth leapfrog steps
(default 2^10 = 1024). When the chain enters a region where the
surrogate is biased, the trajectory keeps growing until the
|H_theta drift| exceeds the threshold and the monitor fires. With
fixed L=5, that mechanism cannot trigger.

This file implements the missing pairing: recursive build_tree NUTS
that uses the L-HNN's `potential_grad_and_hamiltonian` inside leapfrog,
checks |H_theta - H_0| at each leaf, and terminates the tree (s=0)
when the threshold is breached. Multinomial sampling inside the tree
uses L-HNN's H_theta for weights (efficient). The final-state MH
correction uses the REAL target evaluation — so the posterior is
unbiased even when the surrogate is wrong. Mathematically: a surrogate-
proposal-density / true-target-density MH ratio (cf. surrogate MCMC,
Conrad et al. 2016).

Sampler outline
---------------
At each iteration i:
  1. Sample momentum p ~ N(0, I).
  2. Evaluate H_init_real = -log pi(q) + 0.5 ||p||^2 with REAL target.
  3. Evaluate H_0_lhnn = lhnn.hamiltonian(q, p)   (for drift monitor).
  4. NUTS tree build, doubling forward or backward in time per coin
     flip, using L-HNN gradients in leapfrog. If at any leaf
     |H_theta - H_0_lhnn| > error_threshold:
        - terminate that sub-tree (s = 0)
        - record that this iteration needs a cooldown for the NEXT
          iteration's leapfrog calls.
     Multinomial-sample within the tree using L-HNN H_theta weights to
     pick a candidate sample (q_cand, p_cand).
  5. Final MH correction:
        log_alpha = -H_prop_real + H_init_real
     where H_prop_real = -log pi(q_cand) + 0.5 ||p_cand||^2 with REAL
     target. Accept q_cand with min(1, exp(log_alpha)).
  6. Use alpha for dual-averaging step-size adaptation during burn-in.

The cooldown across iterations is handled by a per-chain
`fallback_remaining` counter. While > 0, leapfrog steps fall back to
real-gradient evaluations one at a time, decrementing the counter.

References
----------
- Hoffman & Gelman (2014), "The No-U-Turn Sampler", JMLR.
- Betancourt (2017), "A Conceptual Introduction to HMC", arXiv 1701.02434
  (multinomial sampling within trajectory).
- Dhulipala, Che & Shields (2022), arXiv 2208.06120.
- Conrad et al. (2016), "Accelerating asymptotically exact MCMC for
  computationally intensive models via local approximations", JASA.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.hmc_pf import _eval_target_and_grad
from src.filters.bonus.lhnn_hmc_pf import LatentHNN, LHNNConfig


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NUTSResult:
    samples: tf.Tensor                # (num_results, d)
    is_accepted: tf.Tensor            # (num_results,)
    accept_rate: tf.Tensor            # scalar
    target_log_probs: tf.Tensor       # (num_results,)
    step_sizes: tf.Tensor             # (total,)
    avg_tree_depth: tf.Tensor         # scalar, mean tree depth over iters
    tree_depths: tf.Tensor            # (total,) per-iteration depth
    real_grad_evals_per_iter: tf.Tensor   # (total,), cumulative
    total_real_grad_evals: int
    total_lhnn_leapfrog_evals: int
    total_error_triggers: int


@dataclass
class MultiChainNUTSResult:
    samples: tf.Tensor                # (num_chains, num_results, d)
    is_accepted: tf.Tensor            # (num_chains, num_results)
    accept_rate: tf.Tensor            # (num_chains,)
    target_log_probs: tf.Tensor       # (num_chains, num_results)
    step_sizes: tf.Tensor             # (num_chains, total)
    tree_depths: tf.Tensor            # (num_chains, total)
    per_chain_diagnostics: List[NUTSResult]


# ---------------------------------------------------------------------------
# Leapfrog primitives
# ---------------------------------------------------------------------------

def _real_target_value(target_log_prob_fn, q, crn_seed) -> float:
    """Real target log-density at q, with CRN."""
    tf.random.set_seed(int(crn_seed))
    lp_t = target_log_prob_fn(q)
    lp = tf.cast(tf.math.real(tf.cast(lp_t, tf.complex64)), tf.float32)
    val = float(lp.numpy())
    if not math.isfinite(val):
        return -1e9
    return val


def _real_hamiltonian(target_log_prob_fn, q, p, crn_seed) -> float:
    """Real Hamiltonian H_real = -log pi(q) + 0.5 ||p||^2."""
    lp = _real_target_value(target_log_prob_fn, q, crn_seed)
    K = 0.5 * float(tf.reduce_sum(p ** 2).numpy())
    return -lp + K


def _leapfrog_step_lhnn(lhnn: LatentHNN, q: tf.Tensor, p: tf.Tensor,
                          eps: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """Synchronized leapfrog with L-HNN gradient (no real target call).

    Standard Euclidean kinetic K = 0.5 ||p||^2 → dq/dt = p exact, so we
    only need dp/dt = -∂H_theta/∂q from the L-HNN.

      q_new = q + eps * p + (eps^2 / 2) * grad_lhnn(q)
      p_new = p + (eps / 2) * (grad_lhnn(q) + grad_lhnn(q_new))

    where grad_lhnn = -∂H_theta/∂q (L-HNN's view of ∇ log pi).
    """
    dH_dq, _ = lhnn.potential_grad_and_hamiltonian(q, p)
    grad = -dH_dq
    q_new = q + eps * p + (eps ** 2 / 2.0) * grad
    dH_dq_new, _ = lhnn.potential_grad_and_hamiltonian(q_new, p)
    grad_new = -dH_dq_new
    p_new = p + (eps / 2.0) * (grad + grad_new)
    return q_new, p_new


def _leapfrog_step_real(target_log_prob_fn, q: tf.Tensor, p: tf.Tensor,
                          eps: float, crn_seed: int
                          ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Synchronized leapfrog with REAL target gradient (no L-HNN call)."""
    _, grad = _eval_target_and_grad(target_log_prob_fn, q, crn_seed)
    q_new = q + eps * p + (eps ** 2 / 2.0) * grad
    _, grad_new = _eval_target_and_grad(target_log_prob_fn, q_new, crn_seed)
    p_new = p + (eps / 2.0) * (grad + grad_new)
    return q_new, p_new


# ---------------------------------------------------------------------------
# Sampler-level mutable state (passed through recursion)
# ---------------------------------------------------------------------------

class _SamplerState:
    """Carries the cooldown counter and grad-eval counters across the
    recursive build_tree calls within a single NUTS iteration.

    Necessary because Python recursion doesn't naturally support mutable
    out-of-band state without globals.
    """
    __slots__ = ("fallback_remaining", "real_grad_count",
                 "lhnn_grad_count", "error_triggered_this_iter",
                 "cooldown_steps", "rng")

    def __init__(self, fallback_remaining: int, cooldown_steps: int,
                 rng: np.random.Generator):
        self.fallback_remaining = int(fallback_remaining)
        self.cooldown_steps = int(cooldown_steps)
        self.real_grad_count = 0     # per-iter
        self.lhnn_grad_count = 0     # per-iter
        self.error_triggered_this_iter = False
        self.rng = rng


# ---------------------------------------------------------------------------
# NUTS tree building (recursive)
# ---------------------------------------------------------------------------

def _no_u_turn(q_minus: tf.Tensor, q_plus: tf.Tensor,
                p_minus: tf.Tensor, p_plus: tf.Tensor) -> bool:
    """Standard NUTS U-turn criterion.

      (q_plus - q_minus) . p_minus >= 0  AND
      (q_plus - q_minus) . p_plus  >= 0

    If either dot product is < 0, the trajectory has turned back on
    itself and we should stop growing.
    """
    delta = q_plus - q_minus
    d_pm = float(tf.reduce_sum(delta * p_minus).numpy())
    d_pp = float(tf.reduce_sum(delta * p_plus).numpy())
    return (d_pm >= 0.0) and (d_pp >= 0.0)


def _leapfrog_with_monitor(lhnn, target_log_prob_fn, q, p, eps,
                             direction, crn_seed, H0_lhnn,
                             error_threshold, state: _SamplerState):
    """One leapfrog step with fallback logic. Returns (q_new, p_new,
    H_new_lhnn, s) where s = 1 if drift OK else 0.

    Cooldown semantics: while state.fallback_remaining > 0, use real
    gradient and decrement. Otherwise use L-HNN; if |H_theta - H_0| >
    threshold post-step, set s = 0, mark error_triggered, set
    fallback_remaining = cooldown_steps so the NEXT iteration starts
    with real gradients.
    """
    if state.fallback_remaining > 0:
        q_new, p_new = _leapfrog_step_real(
            target_log_prob_fn, q, p, direction * eps, crn_seed,
        )
        state.fallback_remaining -= 1
        state.real_grad_count += 2
        # Compute L-HNN view of new Hamiltonian for drift bookkeeping
        H_new_lhnn = float(lhnn.hamiltonian(q_new, p_new).numpy())
        return q_new, p_new, H_new_lhnn, 1
    # L-HNN leapfrog
    q_new, p_new = _leapfrog_step_lhnn(lhnn, q, p, direction * eps)
    state.lhnn_grad_count += 2
    H_new_lhnn = float(lhnn.hamiltonian(q_new, p_new).numpy())
    drift = abs(H_new_lhnn - H0_lhnn)
    if drift > error_threshold:
        # Trigger fallback for NEXT iteration; terminate this sub-tree
        state.error_triggered_this_iter = True
        # We don't recompute this step — terminate cleanly
        # Mark fallback_remaining so the next ITERATION starts with real
        # gradients. (Within the current trajectory we let multinomial
        # sampling pick a state from before the divergence.)
        state.fallback_remaining = state.cooldown_steps
        return q_new, p_new, H_new_lhnn, 0
    return q_new, p_new, H_new_lhnn, 1


def _build_tree(lhnn, target_log_prob_fn, q, p, direction, depth, eps,
                  H0_lhnn, crn_seed, error_threshold,
                  state: _SamplerState):
    """Recursive NUTS build_tree.

    Returns:
      q_minus, p_minus, q_plus, p_plus,
      q_sample, log_w_sample,         (multinomial-selected sample so far)
      n_states,                       (number of valid leaves)
      s                               (continue flag: 1 = continue, 0 = stop)
    """
    if depth == 0:
        q_new, p_new, H_new_lhnn, s = _leapfrog_with_monitor(
            lhnn, target_log_prob_fn, q, p, eps, direction, crn_seed,
            H0_lhnn, error_threshold, state,
        )
        log_w = -H_new_lhnn  # log of multinomial weight (proportional to exp(-H_theta))
        n_states = 1
        # Carry p_sample alongside q_sample so the caller can compute the
        # kinetic energy at the selected state for a proper H-based MH.
        return (q_new, p_new, q_new, p_new,
                q_new, p_new, log_w, n_states, s)
    # Recurse: build first sub-tree
    (q_m, p_m, q_p, p_p,
     q_sample, p_sample, log_w, n, s) = _build_tree(
        lhnn, target_log_prob_fn, q, p, direction, depth - 1, eps,
        H0_lhnn, crn_seed, error_threshold, state,
    )
    if s == 0:
        return q_m, p_m, q_p, p_p, q_sample, p_sample, log_w, n, 0
    # Build second sub-tree from the appropriate boundary
    if direction == -1:
        (q_m2, p_m2, _, _,
         q_sample_2, p_sample_2, log_w_2, n_2, s_2) = _build_tree(
            lhnn, target_log_prob_fn, q_m, p_m, direction, depth - 1, eps,
            H0_lhnn, crn_seed, error_threshold, state,
        )
        q_m, p_m = q_m2, p_m2
    else:
        (_, _, q_p2, p_p2,
         q_sample_2, p_sample_2, log_w_2, n_2, s_2) = _build_tree(
            lhnn, target_log_prob_fn, q_p, p_p, direction, depth - 1, eps,
            H0_lhnn, crn_seed, error_threshold, state,
        )
        q_p, p_p = q_p2, p_p2
    # Multinomial sample combining the two halves
    log_w_total = float(np.logaddexp(log_w, log_w_2))
    if log_w_total > -1e30:
        prob_swap = math.exp(log_w_2 - log_w_total)
        if state.rng.random() < prob_swap:
            q_sample = q_sample_2
            p_sample = p_sample_2
    n_total = n + n_2
    # Continue flag: both sub-trees must be alive AND no overall U-turn
    s_combined = (s_2 * (1 if _no_u_turn(q_m, q_p, p_m, p_p) else 0))
    return (q_m, p_m, q_p, p_p,
            q_sample, p_sample, log_w_total, n_total, s_combined)


# ---------------------------------------------------------------------------
# Single-chain NUTS with L-HNN
# ---------------------------------------------------------------------------

def run_lhnn_nuts(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_state: tf.Tensor,
    lhnn: LatentHNN,
    num_results: int = 1_000,
    num_burnin: int = 200,
    step_size: float = 0.01,
    max_treedepth: int = 8,
    error_threshold: float = 10.0,
    cooldown_steps: int = 10,
    adapt_step_size: bool = True,
    target_accept_prob: float = 0.65,
    seed: int = 42,
    verbose: bool = True,
    crn_offset: Optional[int] = None,
    progress_every: int = 10,
    kinetic_mh: bool = False,
) -> NUTSResult:
    """Run NUTS with L-HNN gradient and final-state MH correction
    against the REAL target.

    kinetic_mh : bool
        If True, the final-state MH acceptance uses the proper Hamiltonian
        ratio log_alpha = -H_prop_real + H_init_real (i.e. it includes the
        kinetic energy 0.5||p_sample||^2 at the multinomial-selected state),
        as the module docstring prescribes. If False (default, legacy
        behaviour), it uses the potential-only ratio lp_sample - lp_init,
        which rejects energy-conserving tail-ward moves and biases the
        posterior toward under-dispersion. Default False preserves the
        univariate L-HNN NUTS results; the multivariate driver passes True.

    Parameters
    ----------
    max_treedepth : int
        2^max_treedepth caps the trajectory length. Default 8 = 256.
        Plenty long enough for |H_theta - H_0| to drift past
        error_threshold whenever the surrogate is unreliable.
    error_threshold : float
        |H_theta(t) - H_theta(0)| past this triggers fallback cooldown.
    cooldown_steps : int
        Number of real-gradient leapfrog steps before resuming L-HNN.
    target_accept_prob : float
        Dual-averaging target. 0.65 is the H&G original recommendation
        for fixed-L HMC; NUTS papers often use 0.80. Default 0.65 here
        because acceptance is via REAL-target MH and stochastic-target
        noise rewards a lower target.
    """
    d = int(initial_state.shape[0])
    q = tf.cast(initial_state, tf.float32)
    base_seed = int(seed)
    crn_base = int(crn_offset) if crn_offset is not None else base_seed
    total = int(num_burnin) + int(num_results)
    rng = np.random.default_rng(base_seed)

    # Dual averaging state (Hoffman & Gelman 2014, Alg. 5)
    mu = math.log(10.0 * step_size)
    log_eps = math.log(step_size)
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma, t0_da, kappa = 0.05, 10.0, 0.75
    delta = float(target_accept_prob)

    samples_list: List[tf.Tensor] = []
    accepted_list: List[bool] = []
    lp_list: List[float] = []
    eps_list: List[float] = []
    depth_list: List[int] = []
    real_grad_cum_list: List[int] = []

    fallback_remaining = 0
    total_real = 0
    total_lhnn = 0
    total_errors = 0
    t0 = time.perf_counter()

    for i in range(total):
        iter_seed = base_seed + (i + 1) * 7_919
        crn_seed = crn_base + (i + 1) * 7_919
        eps = math.exp(log_eps)

        # Sample momentum
        tf.random.set_seed(iter_seed + 3_000_000)
        p = tf.random.normal([d], dtype=tf.float32)
        K_init = 0.5 * float(tf.reduce_sum(p ** 2).numpy())

        # Real Hamiltonian at trajectory start (for MH correction)
        lp_init = _real_target_value(target_log_prob_fn, q, crn_seed)
        H_init_real = -lp_init + K_init
        total_real += 1  # one real-target value eval per iter

        # L-HNN view of starting Hamiltonian (for drift monitor)
        H0_lhnn = float(lhnn.hamiltonian(q, p).numpy())

        # NUTS tree-building loop
        q_minus, p_minus = tf.identity(q), tf.identity(p)
        q_plus, p_plus = tf.identity(q), tf.identity(p)
        q_sample = tf.identity(q)
        p_sample = tf.identity(p)   # momentum at the selected state (for kinetic_mh)
        log_w_total = float(-H0_lhnn)   # initial single-leaf "tree"
        n_total = 1
        depth = 0
        s = 1
        state = _SamplerState(
            fallback_remaining=fallback_remaining,
            cooldown_steps=int(cooldown_steps),
            rng=rng,
        )

        while s == 1 and depth < max_treedepth:
            direction = -1 if rng.random() < 0.5 else 1
            if direction == -1:
                (q_minus, p_minus, _, _,
                 q_sample_new, p_sample_new, log_w_new, n_new, s_new) = _build_tree(
                    lhnn, target_log_prob_fn, q_minus, p_minus,
                    direction, depth, eps, H0_lhnn, crn_seed,
                    error_threshold, state,
                )
            else:
                (_, _, q_plus, p_plus,
                 q_sample_new, p_sample_new, log_w_new, n_new, s_new) = _build_tree(
                    lhnn, target_log_prob_fn, q_plus, p_plus,
                    direction, depth, eps, H0_lhnn, crn_seed,
                    error_threshold, state,
                )
            if s_new == 1:
                log_w_total_new = float(np.logaddexp(log_w_total, log_w_new))
                prob_swap = math.exp(log_w_new - log_w_total_new)
                if rng.random() < prob_swap:
                    q_sample = q_sample_new
                    p_sample = p_sample_new
                log_w_total = log_w_total_new
                n_total += n_new
            # Combined U-turn check
            s = s_new * (1 if _no_u_turn(q_minus, q_plus, p_minus, p_plus) else 0)
            depth += 1

        # Update fallback bookkeeping for next iter
        fallback_remaining = state.fallback_remaining
        total_real += state.real_grad_count
        total_lhnn += state.lhnn_grad_count
        if state.error_triggered_this_iter:
            total_errors += 1

        # Final-state MH correction against REAL target (Conrad-style:
        # surrogate-driven proposal, accept against the real posterior).
        #
        # Two acceptance forms (selected by kinetic_mh):
        #   legacy (kinetic_mh=False): potential-only ratio
        #       log_alpha = log_pi_real(q_sample) - log_pi_real(q_init)
        #     This drops the kinetic term, so it rejects energy-conserving
        #     tail-ward moves (ones where the trajectory traded kinetic for
        #     potential) and biases the posterior toward UNDER-dispersion.
        #   proper (kinetic_mh=True): full Hamiltonian ratio
        #       log_alpha = -H_prop_real + H_init_real
        #                 = (lp_sample - lp_init) + (K_init - K_sample)
        #     where K_sample = 0.5||p_sample||^2 is the kinetic energy at the
        #     multinomial-selected phase-space point (now threaded through
        #     _build_tree). This is the form the module docstring prescribes.
        lp_sample = _real_target_value(target_log_prob_fn, q_sample, crn_seed)
        total_real += 1
        if kinetic_mh:
            K_sample = 0.5 * float(tf.reduce_sum(p_sample ** 2).numpy())
            log_alpha = (lp_sample - lp_init) + (K_init - K_sample)
        else:
            log_alpha = lp_sample - lp_init
        if not math.isfinite(log_alpha):
            log_alpha = -1e9
        alpha = min(math.exp(min(log_alpha, 0.0)), 1.0)

        tf.random.set_seed(iter_seed + 4_000_000)
        accept = float(tf.random.uniform([]).numpy()) < alpha

        if accept:
            q = tf.identity(q_sample)
            lp_q = lp_sample
        else:
            lp_q = lp_init

        # Dual averaging for step size
        if adapt_step_size and i < num_burnin:
            m = float(i + 1)
            w = 1.0 / (m + t0_da)
            H_bar = (1.0 - w) * H_bar + w * (delta - alpha)
            log_eps = mu - (m ** 0.5 / gamma) * H_bar
            # Clip to a reasonable band so noisy alpha doesn't push eps
            # into unproductive extremes
            log_eps = max(log_eps, math.log(step_size * 0.25))
            log_eps = min(log_eps, math.log(step_size * 4.0))
            log_eps_bar = (m ** (-kappa) * log_eps
                            + (1.0 - m ** (-kappa)) * log_eps_bar)
        elif adapt_step_size and i >= num_burnin:
            log_eps = log_eps_bar

        eps_list.append(eps)
        depth_list.append(depth)
        real_grad_cum_list.append(total_real)

        if i >= num_burnin:
            samples_list.append(tf.identity(q))
            accepted_list.append(bool(accept))
            lp_list.append(float(lp_q))

        if verbose and progress_every > 0 and (i + 1) % progress_every == 0:
            phase = "burn" if i < num_burnin else "samp"
            elapsed = time.perf_counter() - t0
            print(f"  NUTS-LHNN {i+1}/{total} ({phase})  eps={eps:.5f}  "
                  f"alpha={alpha:.3f}  depth={depth}  err_trigs={total_errors}  "
                  f"real_grads={total_real}  lhnn_steps={total_lhnn}  "
                  f"({elapsed:.1f}s)",
                  flush=True)

    samples = tf.stack(samples_list, axis=0)
    is_accepted = tf.constant(accepted_list)
    accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
    target_log_probs = tf.constant(lp_list, dtype=tf.float32)
    step_sizes = tf.constant(eps_list, dtype=tf.float32)
    tree_depths = tf.constant(depth_list, dtype=tf.int32)
    real_grad_evals_per_iter = tf.constant(real_grad_cum_list, dtype=tf.int32)
    avg_depth = tf.reduce_mean(tf.cast(tree_depths, tf.float32))

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  [chain done]  wall={elapsed:.1f}s  accept_rate="
              f"{float(accept_rate.numpy()):.3f}  avg_depth="
              f"{float(avg_depth.numpy()):.2f}  err_trigs={total_errors}  "
              f"total_real_grads={total_real}  total_lhnn_steps={total_lhnn}",
              flush=True)

    return NUTSResult(
        samples=samples,
        is_accepted=is_accepted,
        accept_rate=accept_rate,
        target_log_probs=target_log_probs,
        step_sizes=step_sizes,
        avg_tree_depth=avg_depth,
        tree_depths=tree_depths,
        real_grad_evals_per_iter=real_grad_evals_per_iter,
        total_real_grad_evals=int(total_real),
        total_lhnn_leapfrog_evals=int(total_lhnn),
        total_error_triggers=int(total_errors),
    )


def run_lhnn_nuts_multi_chain(
    target_log_prob_fn,
    initial_states: tf.Tensor,
    lhnn: LatentHNN,
    num_results: int = 1_000,
    num_burnin: int = 200,
    step_size: float = 0.01,
    max_treedepth: int = 8,
    error_threshold: float = 10.0,
    cooldown_steps: int = 10,
    adapt_step_size: bool = True,
    target_accept_prob: float = 0.65,
    seed: int = 42,
    verbose: bool = True,
    share_crn_across_chains: bool = True,
    progress_every: int = 10,
    kinetic_mh: bool = False,
) -> MultiChainNUTSResult:
    """Independent NUTS-with-LHNN chains. CRN shared across chains by
    default (same stochastic-target realisation, varying inits).

    kinetic_mh forwards to run_lhnn_nuts (default False = legacy
    potential-only acceptance)."""
    base_seed = int(seed)
    num_chains = int(initial_states.shape[0])
    shared_crn = base_seed if share_crn_across_chains else None

    s_list, a_list, ar_list, lp_list, eps_list, depth_list, diag_list = (
        [], [], [], [], [], [], []
    )

    for c in range(num_chains):
        chain_seed = base_seed + 1009 * (c + 1)
        if verbose:
            print(f"\n[NUTS-LHNN chain {c+1}/{num_chains}]  seed={chain_seed}  "
                  f"init={initial_states[c].numpy().tolist()}", flush=True)
        result = run_lhnn_nuts(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=initial_states[c],
            lhnn=lhnn,
            num_results=num_results,
            num_burnin=num_burnin,
            step_size=step_size,
            max_treedepth=max_treedepth,
            error_threshold=error_threshold,
            cooldown_steps=cooldown_steps,
            adapt_step_size=adapt_step_size,
            target_accept_prob=target_accept_prob,
            seed=chain_seed,
            verbose=verbose,
            crn_offset=shared_crn,
            progress_every=progress_every,
            kinetic_mh=kinetic_mh,
        )
        s_list.append(result.samples)
        a_list.append(result.is_accepted)
        ar_list.append(result.accept_rate)
        lp_list.append(result.target_log_probs)
        eps_list.append(result.step_sizes)
        depth_list.append(result.tree_depths)
        diag_list.append(result)

    return MultiChainNUTSResult(
        samples=tf.stack(s_list, axis=0),
        is_accepted=tf.stack(a_list, axis=0),
        accept_rate=tf.stack(ar_list, axis=0),
        target_log_probs=tf.stack(lp_list, axis=0),
        step_sizes=tf.stack(eps_list, axis=0),
        tree_depths=tf.stack(depth_list, axis=0),
        per_chain_diagnostics=diag_list,
    )
