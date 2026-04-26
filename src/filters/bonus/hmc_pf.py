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

Step-size adaptation follows Nesterov dual averaging (Hoffman &
Gelman, 2014, Algorithm 5).

References
----------
- Neal (2011), "MCMC using Hamiltonian dynamics", Handbook of MCMC.
- Hoffman & Gelman (2014), "The No-U-Turn Sampler", JMLR.
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
        Base random seed.
    adapt_step_size
        If True, use Nesterov dual averaging during burn-in.
        For stochastic targets (particle filters) a fixed step size
        is usually more stable.
    """

    _ = kwargs  # Reserved for future options (kept to avoid breaking callers.)

    base_seed = seed if seed is not None else 42
    d = int(initial_state.shape[0])
    total = num_burnin + num_results
    L = num_leapfrog_steps

    # Dual-averaging state (Hoffman & Gelman 2014, Alg. 5)
    mu = tf.math.log(10.0 * step_size)
    log_eps = tf.math.log(tf.cast(step_size, tf.float32))
    log_eps_bar = tf.constant(0.0)
    H_bar = tf.constant(0.0)
    gamma, t0, kappa = 0.05, 10.0, 0.75
    delta = target_accept_prob

    q = tf.identity(initial_state)

    samples_list: list[tf.Tensor] = []
    accepted_list: list[bool] = []
    log_prob_list: list[tf.Tensor] = []
    step_size_list: list[float] = []

    for i in range(total):
        iter_seed = base_seed + (i + 1) * 7919
        eps = tf.exp(log_eps)

        # 1. Sample momentum p ~ N(0, I)
        tf.random.set_seed(iter_seed + 3_000_000)
        p = tf.random.normal([d])
        current_ke = 0.5 * tf.reduce_sum(p**2)

        # 2. Evaluate U(q) = -log pi(q) and grad log pi(q) with CRN
        lp_cur, grad_cur = _eval_target_and_grad(target_log_prob_fn, q, iter_seed)

        # 3. Leapfrog integration (L steps)
        q_prop = tf.identity(q)
        p_prop = p + (eps / 2.0) * grad_cur  # half-step momentum

        for l_step in range(L):
            q_prop = q_prop + eps * p_prop  # full-step position
            if l_step < L - 1:
                _, grad_l = _eval_target_and_grad(target_log_prob_fn, q_prop, iter_seed)
                p_prop = p_prop + eps * grad_l  # full-step momentum

        lp_prop, grad_final = _eval_target_and_grad(target_log_prob_fn, q_prop, iter_seed)
        p_prop = p_prop + (eps / 2.0) * grad_final  # final half-step

        p_prop = -p_prop  # negate for reversibility
        proposed_ke = 0.5 * tf.reduce_sum(p_prop**2)

        # 4. Hamiltonian accept/reject
        #    H = U + K = -log pi + 0.5 ||p||^2.
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

        # 5. Step-size adaptation (burn-in only, if enabled)
        if adapt_step_size and i < num_burnin:
            m = tf.cast(i + 1, tf.float32)
            w = 1.0 / (m + t0)
            # Note: alpha is a tensor; convert to python float for stability.
            H_bar = (1.0 - w) * H_bar + w * (delta - float(alpha.numpy()))
            log_eps = mu - tf.sqrt(m) / gamma * H_bar
            log_eps = tf.maximum(log_eps, tf.math.log(step_size * 0.5))
            log_eps = tf.minimum(log_eps, tf.math.log(step_size * 5.0))
            m_kappa = m**(-kappa)
            log_eps_bar = m_kappa * log_eps + (1.0 - m_kappa) * log_eps_bar
        elif adapt_step_size and i >= num_burnin:
            log_eps = log_eps_bar

        # 6. Bookkeeping
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
