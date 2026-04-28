"""
Hamiltonian Monte Carlo with a differentiable particle filter target.

Implements genuine HMC — momentum sampling, leapfrog integration, and
Hamiltonian accept/reject — on a single differentiable log-posterior
built from the LEDH particle flow filter with OT-Sinkhorn resampling
(Li & Coates 2017 + Corenflos et al. 2021).

Because the particle filter introduces stochasticity (random particle
initialisation, process noise), Common Random Numbers (CRN) are used:
the TF random seed is fixed within each HMC trajectory so every
leapfrog sub-step evaluates the same deterministic realisation of the
target. Different MCMC iterations use different seeds.

Adaptation
----------
- *Step size*: Nesterov dual averaging (Hoffman & Gelman 2014, Alg. 5)
  via ``adapt_step_size=True``.
- *Mass matrix*: Stan-style 3-window warmup with diagonal ``M``
  (Stan Reference Manual §15.2), via ``adapt_mass_matrix=True``.
  Window I (~15% of burn-in) tunes the step size with ``M = I``;
  Window II (~75%) collects post-buffer samples to estimate
  ``diag(M) = Var(samples)`` then resets dual averaging; Window III
  (~10%) re-tunes the step size with ``M`` frozen.
- A multi-chain wrapper ``run_hmc_multi_chain`` runs ``num_chains``
  independent chains with dispersed initial points (one per chain seed),
  returning samples shaped ``(num_chains, num_results, d)`` for use with
  ``src/utils/mcmc_diagnostics`` (R-hat, bulk/tail ESS, etc.).

References
----------
- Neal (2011), "MCMC using Hamiltonian dynamics", Handbook of MCMC.
- Hoffman & Gelman (2014), "The No-U-Turn Sampler", JMLR.
- Stan Development Team, "Stan Reference Manual" §15 (HMC sampler config).
- Li & Coates (2017), "Particle Filtering with Invertible Particle Flow"
- Corenflos et al. (2021), "Differentiable Particle Filtering via
  Entropy-Regularized Optimal Transport"
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import tensorflow as tf


class HMCResult(NamedTuple):
    """Container returned by :func:`run_hmc`."""

    samples: tf.Tensor  # (num_results, d)
    is_accepted: tf.Tensor  # (num_results,)
    accept_rate: tf.Tensor  # scalar
    target_log_probs: tf.Tensor  # (num_results,)
    step_sizes: tf.Tensor  # (total,)


class MultiChainHMCResult(NamedTuple):
    """Container returned by :func:`run_hmc_multi_chain`.

    All tensors stack the per-chain results along a new leading axis.
    """

    samples: tf.Tensor          # (num_chains, num_results, d)
    is_accepted: tf.Tensor      # (num_chains, num_results)
    accept_rate: tf.Tensor      # (num_chains,)
    target_log_probs: tf.Tensor # (num_chains, num_results)
    step_sizes: tf.Tensor       # (num_chains, num_burnin + num_results)


_MAX_GRAD_NORM = 1e3


def _safe_grad(grad: Optional[tf.Tensor]) -> Optional[tf.Tensor]:
    """Replace non-finite entries; clip extreme L2 norm for safety.

    The gradient *scale* is preserved up to ``_MAX_GRAD_NORM`` — no
    unit-norm normalisation is applied, so the leapfrog integrator
    sees the true curvature of the target.
    """

    if grad is None:
        return None
    grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
    return tf.clip_by_norm(grad, _MAX_GRAD_NORM)


def _eval_target_and_grad(
    target_fn: Callable[[tf.Tensor], tf.Tensor],
    q: tf.Tensor,
    crn_seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate log pi(q) and nabla_q log pi(q) under Common Random Numbers."""

    tf.random.set_seed(crn_seed)
    q_t = tf.identity(q)
    with tf.GradientTape() as tape:
        tape.watch(q_t)
        lp = target_fn(q_t)
    raw = tape.gradient(lp, q_t)
    grad = _safe_grad(raw) if raw is not None else tf.zeros_like(q)
    return lp, grad


def _stan_warmup_windows(num_burnin: int) -> tuple[int, int, int]:
    """Stan-style 3-window split of warmup: ``(initial, mass, final)``.

    Approximate ratios 15% / 75% / 10%, with floors so a short burn-in
    still gets a non-empty mass-matrix window.  Stan's defaults are
    75 / 925 / 50 out of 1000 — we keep the same proportions but scale
    to whatever ``num_burnin`` is requested.
    """
    if num_burnin < 30:
        # Too short for meaningful mass-matrix estimation — give it all to
        # the step-size adaptation buffer to preserve current behaviour.
        return num_burnin, 0, 0
    init_buf = max(15, int(round(0.15 * num_burnin)))
    final_buf = max(10, int(round(0.10 * num_burnin)))
    mass_window = max(num_burnin - init_buf - final_buf, 0)
    if mass_window == 0:
        # Pathological — collapse final buffer.
        mass_window = num_burnin - init_buf
        final_buf = 0
    return init_buf, mass_window, final_buf


def run_hmc(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_state: tf.Tensor,
    num_results: int = 1000,
    num_burnin: int = 500,
    step_size: float = 0.001,
    num_leapfrog_steps: int = 10,
    target_accept_prob: float = 0.65,
    seed: Optional[int] = None,
    verbose: bool = True,
    adapt_step_size: bool = False,
    adapt_mass_matrix: bool = False,
    mass_matrix: Optional[tf.Tensor] = None,
    crn_offset: Optional[int] = None,
    **kwargs,
) -> HMCResult:
    """
    Hamiltonian Monte Carlo with CRN for stochastic targets.

    A **single** ``target_log_prob_fn`` is used for both the leapfrog
    gradient computation and the Hamiltonian accept/reject step.

    Parameters
    ----------
    target_log_prob_fn
        Differentiable log-posterior log pi(theta).
    initial_state
        Starting point in parameter space, shape ``(d,)``.
    num_results
        Post-burn-in samples to collect.
    num_burnin
        Burn-in iterations (step-size adaptation happens here).
    step_size
        Leapfrog step size epsilon. Fixed unless ``adapt_step_size=True``.
    num_leapfrog_steps
        Number of leapfrog steps *L* per proposal.
    target_accept_prob
        Target acceptance probability for dual averaging (if adapting).
    seed
        Base random seed.  Controls momentum draws and MH-accept randomness.
    adapt_step_size
        If True, use Nesterov dual averaging during burn-in.
        For stochastic targets (particle filters) a fixed step size
        is usually more stable.
    adapt_mass_matrix
        If True, estimate a *diagonal* mass matrix from samples collected
        during the middle of warmup (Stan §15.2 style 3-window scheme).
        Step-size dual averaging is reset after the mass-matrix update so
        epsilon adapts to the new preconditioned geometry.
    mass_matrix
        Optional pre-specified diagonal mass matrix as a length-``d``
        tensor.  Defaults to all-ones (identity).  Ignored if
        ``adapt_mass_matrix=True`` after the adaptation window.
    crn_offset
        Optional override for the CRN seed used inside
        ``_eval_target_and_grad``.  When ``None`` (default) the CRN seed
        equals the chain-private ``iter_seed`` derived from ``seed`` —
        bit-exact identical to the previous behaviour.  When set
        explicitly, the per-iteration CRN seed becomes
        ``crn_offset + (i+1) * 7919``, decoupled from ``seed``.  This is
        used by :func:`run_hmc_multi_chain` to share the stochastic-target
        realisation across chains so R-hat measures MCMC convergence
        rather than CRN drift between chains (see that function's
        docstring for the rationale).
    """

    _ = kwargs  # Reserved for future options (kept to avoid breaking callers.)

    base_seed = seed if seed is not None else 42
    d = int(initial_state.shape[0])
    total = num_burnin + num_results
    L = num_leapfrog_steps

    # Mass matrix (diagonal).  Float32, shape (d,).
    if mass_matrix is None:
        M = tf.ones([d], dtype=tf.float32)
    else:
        M = tf.cast(tf.reshape(mass_matrix, [d]), tf.float32)
    M_inv = 1.0 / M
    sqrt_M = tf.sqrt(M)

    # Stan-style warmup windows for mass-matrix adaptation.
    init_buf, mass_window, final_buf = _stan_warmup_windows(num_burnin)
    mass_start = init_buf
    mass_end = init_buf + mass_window  # exclusive
    mass_collect: list[tf.Tensor] = []

    # Dual-averaging state (Hoffman & Gelman 2014, Alg. 5)
    def _reset_dual_averaging(eps0: float) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            tf.math.log(10.0 * eps0),
            tf.math.log(tf.cast(eps0, tf.float32)),
            tf.constant(0.0),
            tf.constant(0.0),
        )

    mu, log_eps, log_eps_bar, H_bar = _reset_dual_averaging(step_size)
    gamma, t0, kappa = 0.05, 10.0, 0.75
    delta = target_accept_prob

    q = tf.identity(initial_state)

    samples_list: list[tf.Tensor] = []
    accepted_list: list[bool] = []
    log_prob_list: list[tf.Tensor] = []
    step_size_list: list[float] = []
    da_step = 0  # local dual-averaging step counter (reset after mass-matrix update)

    crn_base = crn_offset if crn_offset is not None else base_seed

    for i in range(total):
        iter_seed = base_seed + (i + 1) * 7919   # chain-private (momentum, MH)
        crn_seed = crn_base + (i + 1) * 7919     # shared across chains when crn_offset is set
        eps = tf.exp(log_eps)

        # 1. Sample momentum p ~ N(0, M)  (diag M -> p_i = sqrt(M_i) * eta_i)
        tf.random.set_seed(iter_seed + 3_000_000)
        eta = tf.random.normal([d])
        p = sqrt_M * eta
        current_ke = 0.5 * tf.reduce_sum(p * p * M_inv)

        # 2. Evaluate U(q) = -log pi(q) and grad log pi(q) with CRN
        lp_cur, grad_cur = _eval_target_and_grad(target_log_prob_fn, q, crn_seed)

        # 3. Leapfrog integration (L steps) preconditioned by M
        q_prop = tf.identity(q)
        p_prop = p + (eps / 2.0) * grad_cur  # half-step momentum

        for l_step in range(L):
            q_prop = q_prop + eps * (M_inv * p_prop)  # full-step position with M^{-1}
            if l_step < L - 1:
                _, grad_l = _eval_target_and_grad(target_log_prob_fn, q_prop, crn_seed)
                p_prop = p_prop + eps * grad_l  # full-step momentum

        lp_prop, grad_final = _eval_target_and_grad(target_log_prob_fn, q_prop, crn_seed)
        p_prop = p_prop + (eps / 2.0) * grad_final  # final half-step

        p_prop = -p_prop  # negate for reversibility
        proposed_ke = 0.5 * tf.reduce_sum(p_prop * p_prop * M_inv)

        # 4. Hamiltonian accept/reject
        #    H = U + K = -log pi + 0.5 p^T M^{-1} p.
        log_alpha = (lp_prop - lp_cur) + (current_ke - proposed_ke)
        log_alpha = tf.where(
            tf.math.is_finite(log_alpha), log_alpha, tf.constant(-1e6)
        )

        alpha = tf.minimum(tf.exp(tf.minimum(log_alpha, 0.0)), 1.0)

        tf.random.set_seed(iter_seed + 4_000_000)
        accept = tf.random.uniform([]) < alpha

        if bool(accept.numpy()):
            q = tf.identity(q_prop)
            current_lp = lp_prop
        else:
            current_lp = lp_cur

        # 5a. Mass-matrix collection during the middle warmup window.
        if adapt_mass_matrix and mass_start <= i < mass_end:
            mass_collect.append(tf.identity(q))

        # 5b. Mass-matrix update at the end of the collection window.
        if (
            adapt_mass_matrix
            and mass_window > 0
            and i + 1 == mass_end
            and len(mass_collect) >= 5
        ):
            collected = tf.stack(mass_collect, axis=0)  # (n, d)
            # Stan-style shrinkage toward identity (variance bound away from 0):
            #   sigma_hat^2 = ((n / (n+5)) * Var + 1e-3 * (5 / (n+5)))
            n_c = float(collected.shape[0])
            var_emp = tf.math.reduce_variance(collected, axis=0)
            sigma2 = (n_c / (n_c + 5.0)) * var_emp + (5.0 / (n_c + 5.0)) * 1e-3
            sigma2 = tf.maximum(sigma2, 1e-6)
            # Mass matrix = inverse posterior covariance (diag).
            M = 1.0 / sigma2
            M_inv = 1.0 / M
            sqrt_M = tf.sqrt(M)
            # Reset dual averaging so eps re-tunes for the new geometry.
            mu, log_eps, log_eps_bar, H_bar = _reset_dual_averaging(
                float(tf.exp(log_eps).numpy())
            )
            da_step = 0
            if verbose:
                tf.print(
                    f"  HMC mass-matrix update at iter {i + 1}/{total}",
                    "  diag(M)=",
                    M,
                )

        # 6. Step-size adaptation (during steps where dual averaging is active).
        in_init_buf = i < init_buf
        in_final_buf = i >= mass_end and i < num_burnin
        in_mass_window = (
            adapt_mass_matrix and mass_start <= i < mass_end
        )
        # Without mass-matrix adaptation, dual averaging runs through the
        # entire burn-in (preserves backward compat with adapt_step_size=True).
        do_da = adapt_step_size and i < num_burnin and (
            (not adapt_mass_matrix)
            or in_init_buf
            or in_final_buf
            or in_mass_window
        )

        if do_da:
            da_step += 1
            m = tf.cast(da_step, tf.float32)
            w = 1.0 / (m + t0)
            H_bar = (1.0 - w) * H_bar + w * (delta - float(alpha.numpy()))
            log_eps = mu - tf.sqrt(m) / gamma * H_bar
            log_eps = tf.maximum(log_eps, tf.math.log(step_size * 0.1))
            log_eps = tf.minimum(log_eps, tf.math.log(step_size * 10.0))
            m_kappa = m**(-kappa)
            log_eps_bar = m_kappa * log_eps + (1.0 - m_kappa) * log_eps_bar
        elif adapt_step_size and i >= num_burnin:
            log_eps = log_eps_bar

        # 7. Bookkeeping
        step_size_list.append(float(tf.exp(log_eps).numpy()))

        if i >= num_burnin:
            samples_list.append(tf.identity(q))
            accepted_list.append(bool(accept.numpy()))
            log_prob_list.append(current_lp)

        if verbose and (i + 1) % 10 == 0:
            phase = "burn" if i < num_burnin else "sample"
            tf.print(
                f"  HMC {i + 1}/{total} ({phase})",
                "  eps =",
                tf.exp(log_eps),
                "  alpha =",
                alpha,
            )

    samples = tf.stack(samples_list)
    is_accepted = tf.constant(accepted_list)
    accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
    target_log_probs = tf.stack(log_prob_list)
    step_sizes = tf.constant(step_size_list)

    return HMCResult(
        samples=samples,
        is_accepted=is_accepted,
        accept_rate=accept_rate,
        target_log_probs=target_log_probs,
        step_sizes=step_sizes,
    )


def run_hmc_multi_chain(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_states: tf.Tensor,
    num_results: int = 1000,
    num_burnin: int = 500,
    step_size: float = 0.001,
    num_leapfrog_steps: int = 10,
    target_accept_prob: float = 0.65,
    seed: Optional[int] = None,
    verbose: bool = True,
    adapt_step_size: bool = False,
    adapt_mass_matrix: bool = False,
    mass_matrix: Optional[tf.Tensor] = None,
    share_crn_across_chains: bool = True,
    **kwargs,
) -> MultiChainHMCResult:
    """Run independent HMC chains with dispersed initial states.

    Each chain gets a distinct chain-private seed
    (``base_seed + 1009 * chain_idx``) controlling its momentum draws and
    MH-accept randomness, so the chains explore the target independently.
    Per-chain mass-matrix adaptation is also independent: every chain
    estimates its own ``diag(M)``, matching how multi-chain warmup works
    in Stan and PyMC.

    CRN policy across chains
    ------------------------
    When the target is a *stochastic* function (LEDH / particle-filter
    likelihood with finite particles N), each evaluation depends on a
    Common-Random-Numbers seed.  By default
    (``share_crn_across_chains=True``) every chain shares the *same* CRN
    seed sequence ``crn_offset = base_seed + (i+1)*7919`` derived from
    the global seed, so all chains see the *same* stochastic-target
    realisation at iteration ``i``.  This is the right policy for
    multi-chain R-hat: without it, each chain locks onto a local maximum
    of *its own* CRN realisation and rank-R-hat measures CRN drift
    rather than MCMC convergence.  See ``trace_posterior.png`` from the
    pre-CRN-fix run for the failure mode and Report_II_Addendum_Diagnostics
    §4.5 for the discussion.

    Setting ``share_crn_across_chains=False`` reverts to the legacy
    behaviour where each chain has its own independent CRN sequence —
    appropriate only when the target is *deterministic*.

    Parameters
    ----------
    initial_states : tf.Tensor, shape ``(num_chains, d)``
        Dispersed starting points.  Must be over-dispersed relative to the
        posterior to give R-hat its diagnostic power.
    seed : optional base seed (per-chain seeds derived from this).
    share_crn_across_chains : bool
        See discussion above.  Default ``True``.
    All remaining args are forwarded to :func:`run_hmc`.
    """
    base_seed = seed if seed is not None else 42
    num_chains = int(initial_states.shape[0])
    shared_crn = base_seed if share_crn_across_chains else None

    samples_list: list[tf.Tensor] = []
    accepted_list: list[tf.Tensor] = []
    accept_rate_list: list[tf.Tensor] = []
    lp_list: list[tf.Tensor] = []
    eps_list: list[tf.Tensor] = []

    for c in range(num_chains):
        chain_seed = base_seed + 1009 * (c + 1)
        if verbose:
            crn_tag = f"shared({base_seed})" if share_crn_across_chains else "private"
            print(f"\n[HMC chain {c + 1}/{num_chains}]  seed={chain_seed}"
                  f"  crn={crn_tag}"
                  f"  init={initial_states[c].numpy().tolist()}")
        result = run_hmc(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=initial_states[c],
            num_results=num_results,
            num_burnin=num_burnin,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            target_accept_prob=target_accept_prob,
            seed=chain_seed,
            verbose=verbose,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            mass_matrix=mass_matrix,
            crn_offset=shared_crn,
            **kwargs,
        )
        samples_list.append(result.samples)
        accepted_list.append(result.is_accepted)
        accept_rate_list.append(result.accept_rate)
        lp_list.append(result.target_log_probs)
        eps_list.append(result.step_sizes)

    return MultiChainHMCResult(
        samples=tf.stack(samples_list, axis=0),
        is_accepted=tf.stack(accepted_list, axis=0),
        accept_rate=tf.stack(accept_rate_list, axis=0),
        target_log_probs=tf.stack(lp_list, axis=0),
        step_sizes=tf.stack(eps_list, axis=0),
    )


def disperse_initial_states(
    init: tf.Tensor,
    num_chains: int,
    scale: float = 0.5,
    seed: int = 0,
) -> tf.Tensor:
    """Build dispersed initial points around a central guess.

    Returns a ``(num_chains, d)`` tensor where chain 0 is the original
    ``init`` and chains 1..num_chains-1 are perturbed by Gaussian noise
    with standard deviation ``scale * |init|`` (a multiplicative scale on
    the parameter magnitudes — over-dispersion in *log*-parameter space
    when ``init`` already lives there).

    The dispersion is what gives R-hat its diagnostic power: if the chains
    haven't all gravitated to the same posterior region by the end of
    warmup, the between-chain variance dominates and ``R-hat >> 1``.
    """
    init = tf.cast(init, tf.float32)
    d = int(init.shape[0])
    tf.random.set_seed(seed)
    inits = [init]
    for c in range(1, num_chains):
        noise = tf.random.normal([d]) * scale * (tf.abs(init) + 0.1)
        inits.append(init + noise)
    return tf.stack(inits, axis=0)
