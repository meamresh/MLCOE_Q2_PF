"""
TensorFlow Probability HMC wrappers for differentiable particle-filter targets.

This module is intentionally separate from ``hmc_pf.py``.  It uses TFP's
standard HMC implementation and dual-averaging step-size adaptation, while
returning result containers compatible with the existing experiment/report code.

CRN policy
----------
TFP kernels do not expose the outer MCMC iteration index to ``target_log_prob``.
For stochastic particle-filter likelihoods we therefore make the target
deterministic by resetting TensorFlow's RNG to a fixed CRN seed on every target
evaluation.  This tests sampler correctness against one fixed particle/noise
realisation.  It is not the same as the custom sampler's per-iteration CRN
schedule, but it avoids target drift inside leapfrog trajectories.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.hmc_pf import HMCResult, MultiChainHMCResult

tfm = tfp.mcmc


def _finite_log_prob(value: tf.Tensor) -> tf.Tensor:
    """Return a finite scalar/vector log-probability tensor."""
    value = tf.cast(tf.math.real(value), tf.float32)
    return tf.where(tf.math.is_finite(value), value, tf.constant(-1e6, tf.float32))


def _with_fixed_crn(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    crn_seed: Optional[int],
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Wrap a target so stochastic PF noise is fixed across TFP evaluations."""

    def target(q: tf.Tensor) -> tf.Tensor:
        if crn_seed is not None:
            tf.random.set_seed(int(crn_seed))
        return _finite_log_prob(target_log_prob_fn(q))

    return target


def _kernel_step_size(kernel_results) -> tf.Tensor:
    """Best-effort extraction of TFP's current step size."""
    if hasattr(kernel_results, "new_step_size"):
        return tf.cast(kernel_results.new_step_size, tf.float32)
    inner = kernel_results.inner_results if hasattr(kernel_results, "inner_results") else kernel_results
    if hasattr(inner, "accepted_results") and hasattr(inner.accepted_results, "step_size"):
        return tf.cast(inner.accepted_results.step_size, tf.float32)
    if hasattr(inner, "step_size"):
        return tf.cast(inner.step_size, tf.float32)
    return tf.constant(float("nan"), tf.float32)


def _kernel_is_accepted(kernel_results) -> tf.Tensor:
    """Best-effort extraction of HMC accept/reject indicators."""
    inner = kernel_results.inner_results if hasattr(kernel_results, "inner_results") else kernel_results
    if hasattr(inner, "is_accepted"):
        return tf.cast(inner.is_accepted, tf.bool)
    return tf.constant(False)


def _kernel_target_log_prob(kernel_results) -> tf.Tensor:
    """Best-effort extraction of accepted-state target log probability."""
    inner = kernel_results.inner_results if hasattr(kernel_results, "inner_results") else kernel_results
    if hasattr(inner, "accepted_results") and hasattr(inner.accepted_results, "target_log_prob"):
        return _finite_log_prob(inner.accepted_results.target_log_prob)
    if hasattr(inner, "target_log_prob"):
        return _finite_log_prob(inner.target_log_prob)
    return tf.constant(float("nan"), tf.float32)


def _trace_fn(_, kernel_results):
    """Trace fields needed by existing experiment reports."""
    return (
        _kernel_is_accepted(kernel_results),
        _kernel_target_log_prob(kernel_results),
        _kernel_step_size(kernel_results),
    )


def run_hmc_tfp(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_state: tf.Tensor,
    num_results: int = 1000,
    num_burnin: int = 500,
    step_size: float = 0.001,
    num_leapfrog_steps: int = 10,
    target_accept_prob: float = 0.65,
    seed: Optional[int] = None,
    adapt_step_size: bool = True,
    crn_seed: Optional[int] = None,
    verbose: bool = True,
    progress_every: int = 10,
) -> HMCResult:
    """
    Run one TFP HMC chain.

    Parameters mirror ``hmc_pf.run_hmc`` where possible.  ``adapt_mass_matrix``
    is deliberately omitted: TFP's standard HMC kernel here uses identity mass.
    That avoids the custom diagonal-mass ambiguity while testing whether the
    sampler implementation was the bottleneck.

    Progress lines
    ----------------
    ``tfp.mcmc.sample_chain`` executes the whole trajectory as one fused graph,
    so nothing appears on stdout until the chain finishes.  When ``verbose``
    is True and ``progress_every > 0``, we instead call ``kernel.one_step`` in
    a Python loop and print every ``progress_every`` iterations (similar cadence
    to ``run_hmc``).  Set ``progress_every=0`` to keep the fused ``sample_chain``
    path while still printing the header and final summary.
    """
    initial_state = tf.cast(initial_state, tf.float32)
    base_seed = int(seed if seed is not None else 42)
    crn = int(crn_seed if crn_seed is not None else base_seed + 99_991)
    target = _with_fixed_crn(target_log_prob_fn, crn)

    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=tf.cast(step_size, tf.float32),
        num_leapfrog_steps=int(num_leapfrog_steps),
    )

    if adapt_step_size and num_burnin > 0:
        kernel = tfm.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=int(num_burnin),
            target_accept_prob=tf.cast(target_accept_prob, tf.float32),
        )

    if verbose:
        print(
            "\n[TFP HMC]"
            f" results={num_results} burnin={num_burnin}"
            f" step_size={step_size:g} L={num_leapfrog_steps}"
            f" adapt_step_size={adapt_step_size} crn_seed={crn}",
            flush=True,
        )

    num_results_i = int(num_results)
    num_burnin_i = int(num_burnin)
    total_steps = num_burnin_i + num_results_i

    # ``sample_chain`` fuses the whole trajectory into one graph; nothing reaches
    # stdout until completion.  When ``verbose`` is True, step explicitly so we
    # can emit progress like ``run_hmc`` (every ``progress_every`` iterations).
    use_step_loop = (
        verbose and progress_every > 0 and total_steps > 0 and num_results_i > 0
    )

    if not use_step_loop:
        samples, trace = tfm.sample_chain(
            num_results=num_results_i,
            num_burnin_steps=num_burnin_i,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=_trace_fn,
            seed=base_seed,
        )
        is_accepted, target_log_probs, step_sizes = trace
        is_accepted = tf.cast(is_accepted, tf.bool)
        target_log_probs = tf.cast(target_log_probs, tf.float32)
        step_sizes = tf.cast(step_sizes, tf.float32)
        accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
        if verbose:
            tf.print(
                "  TFP HMC done",
                "accept=",
                accept_rate,
                "final_step=",
                step_sizes[-1],
            )
        return HMCResult(
            samples=tf.cast(samples, tf.float32),
            is_accepted=is_accepted,
            accept_rate=accept_rate,
            target_log_probs=target_log_probs,
            step_sizes=step_sizes,
        )

    kernel_results = kernel.bootstrap_results(initial_state)
    state = tf.identity(initial_state)
    samples_list: list[tf.Tensor] = []
    accepted_trace: list[tf.Tensor] = []
    log_prob_trace: list[tf.Tensor] = []
    step_size_list: list[tf.Tensor] = []

    pe = max(1, int(progress_every))
    for step in range(total_steps):
        # Chain RNG stream: distinct from ``crn`` used inside the wrapped target.
        step_seed = base_seed + (step + 1) * 79_127
        state, kernel_results = kernel.one_step(state, kernel_results, seed=step_seed)

        acc = _kernel_is_accepted(kernel_results)
        lp_t = _kernel_target_log_prob(kernel_results)
        eps_t = _kernel_step_size(kernel_results)

        step_size_list.append(tf.identity(eps_t))
        accepted_trace.append(tf.squeeze(acc))
        log_prob_trace.append(tf.squeeze(lp_t))

        if step >= num_burnin_i:
            samples_list.append(tf.identity(state))

        if (step + 1) % pe == 0:
            phase = "burn" if step < num_burnin_i else "sample"
            tf.print(
                "  TFP HMC",
                step + 1,
                "/",
                total_steps,
                "(" + phase + ")",
                " eps =",
                eps_t,
                " accept =",
                tf.cast(acc, tf.float32),
            )

    samples = tf.stack(samples_list, axis=0)
    is_accepted = tf.cast(tf.stack(accepted_trace), tf.bool)
    accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
    target_log_probs = tf.cast(tf.stack(log_prob_trace), tf.float32)
    step_sizes = tf.cast(tf.stack(step_size_list), tf.float32)

    if verbose:
        tf.print(
            "  TFP HMC done accept=",
            accept_rate,
            "final_step=",
            step_sizes[-1],
        )

    return HMCResult(
        samples=tf.cast(samples, tf.float32),
        is_accepted=is_accepted,
        accept_rate=accept_rate,
        target_log_probs=target_log_probs,
        step_sizes=step_sizes,
    )


def run_hmc_tfp_multi_chain(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_states: tf.Tensor,
    num_results: int = 1000,
    num_burnin: int = 500,
    step_size: float = 0.001,
    num_leapfrog_steps: int = 10,
    target_accept_prob: float = 0.65,
    seed: Optional[int] = None,
    adapt_step_size: bool = True,
    share_crn_across_chains: bool = True,
    verbose: bool = True,
    progress_every: int = 10,
) -> MultiChainHMCResult:
    """Run TFP HMC chains sequentially and stack results."""
    initial_states = tf.cast(initial_states, tf.float32)
    num_chains = int(initial_states.shape[0])
    base_seed = int(seed if seed is not None else 42)
    shared_crn = base_seed + 99_991

    samples_list: list[tf.Tensor] = []
    accepted_list: list[tf.Tensor] = []
    accept_rate_list: list[tf.Tensor] = []
    lp_list: list[tf.Tensor] = []
    step_list: list[tf.Tensor] = []

    for c in range(num_chains):
        chain_seed = base_seed + 1009 * (c + 1)
        crn_seed = shared_crn if share_crn_across_chains else chain_seed + 99_991
        if verbose:
            print(
                f"\n[TFP HMC chain {c + 1}/{num_chains}]"
                f" seed={chain_seed} crn_seed={crn_seed}"
                f" init={initial_states[c].numpy().tolist()}",
                flush=True,
            )
        result = run_hmc_tfp(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=initial_states[c],
            num_results=num_results,
            num_burnin=num_burnin,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            target_accept_prob=target_accept_prob,
            seed=chain_seed,
            adapt_step_size=adapt_step_size,
            crn_seed=crn_seed,
            verbose=verbose,
            progress_every=progress_every,
        )
        samples_list.append(result.samples)
        accepted_list.append(result.is_accepted)
        accept_rate_list.append(result.accept_rate)
        lp_list.append(result.target_log_probs)
        step_list.append(result.step_sizes)

    return MultiChainHMCResult(
        samples=tf.stack(samples_list, axis=0),
        is_accepted=tf.stack(accepted_list, axis=0),
        accept_rate=tf.stack(accept_rate_list, axis=0),
        target_log_probs=tf.stack(lp_list, axis=0),
        step_sizes=tf.stack(step_list, axis=0),
    )
