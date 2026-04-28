"""
MCMC convergence diagnostics following Vehtari, Gelman, Simpson, Carpenter,
& Bürkner (2021), "Rank-normalization, folding, and localization: An improved
R-hat for assessing convergence of MCMC", *Bayesian Analysis* 16(2):667-718.

Provides:

- ``bulk_ess``       : effective sample size on rank-normalized split chains
                       (Vehtari 2021 §4.2).
- ``tail_ess``       : ESS at the 5%/95% quantile indicators (Vehtari §4.3),
                       sensitive to tail mixing.
- ``split_rhat``     : classical split-R-hat (Gelman & Rubin 1992 + half-chain
                       split), via :func:`tfp.mcmc.potential_scale_reduction`.
- ``rank_rhat``      : rank-normalized + folded split-R-hat (Vehtari §3) — the
                       robust statistic recommended in modern practice.
- ``credible_interval`` : flat-quantile 95% (configurable) interval.
- ``coverage``       : whether the interval contains the truth, per parameter.
- ``convergence_verdict`` : Stan/PyMC default thresholds (R-hat<1.01,
                       bulk-ESS>400, tail-ESS>400) per parameter.
- ``diagnostics_summary`` : convenience helper that bundles every metric into
                       a single dataclass.
- ``format_diagnostics_table`` : human-readable string for ``results.txt``.

Canonical input shape used throughout this module is ``(num_chains,
num_samples, num_params)`` (single-chain inputs of shape ``(num_samples,
num_params)`` are auto-promoted with a leading 1).

References
----------
- Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021).
- Gelman & Rubin (1992), "Inference from iterative simulation using multiple
  sequences", *Statistical Science* 7(4):457-472.
- Stan Development Team, *Stan Reference Manual* §15.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_3d(samples_chains: tf.Tensor) -> tf.Tensor:
    """Promote ``(S, D)`` and ``(S,)`` inputs to canonical ``(C, S, D)`` shape."""
    x = tf.cast(samples_chains, tf.float32)
    rank = len(x.shape)
    if rank == 1:
        # (S,) -> (1, S, 1)
        x = x[tf.newaxis, :, tf.newaxis]
    elif rank == 2:
        # (S, D) -> (1, S, D)  (treated as a single chain with D parameters)
        x = x[tf.newaxis, :, :]
    elif rank != 3:
        raise ValueError(
            f"Expected samples of rank 1/2/3 (got rank {rank}, shape {x.shape})."
        )
    return x


def _split_chains(samples_chains: tf.Tensor) -> tf.Tensor:
    """Split each chain in half: ``(C, S, D) -> (2C, S//2, D)``.

    Half-chain splitting (Stan's "split-R-hat") protects against chains that
    look stationary because their first and second halves cancel out.
    """
    C = int(samples_chains.shape[0])
    S = int(samples_chains.shape[1])
    half = S // 2
    if half == 0:
        # Too short to split — return as-is (caller should request more samples).
        return samples_chains
    first = samples_chains[:, :half, :]
    second = samples_chains[:, half:2 * half, :]
    return tf.concat([first, second], axis=0)


def _rank_normalize(samples_chains: tf.Tensor) -> tf.Tensor:
    """Vehtari 2021 rank-normalization (Eq. 14).

    Pools all chain×sample values per parameter, ranks them with average ties,
    then maps ranks to standard-normal scores via
    ``Phi^{-1}((r - 3/8) / (N - 1/4))``.

    Parameters
    ----------
    samples_chains : tf.Tensor, shape ``(C, S, D)``

    Returns
    -------
    tf.Tensor, shape ``(C, S, D)``, rank-normalized scores.
    """
    x = tf.cast(samples_chains, tf.float32)
    C, S, D = x.shape[0], x.shape[1], x.shape[2]
    flat = tf.reshape(tf.transpose(x, [2, 0, 1]), [D, C * S])  # (D, N)
    N = tf.cast(tf.shape(flat)[1], tf.float32)
    # argsort(argsort(.)) gives 0-based ranks; +1 to make 1-based.
    ranks = tf.argsort(tf.argsort(flat, axis=1), axis=1)
    ranks_f = tf.cast(ranks, tf.float32) + 1.0
    # Vehtari Eq. 14: u = (rank - 3/8) / (N - 1/4)
    u = (ranks_f - 3.0 / 8.0) / (N - 0.25)
    # Clip away from {0, 1} to keep the inverse CDF finite.
    u = tf.clip_by_value(u, 1e-7, 1.0 - 1e-7)
    z = tfd.Normal(0.0, 1.0).quantile(u)  # (D, N)
    z = tf.reshape(z, [D, C, S])
    return tf.transpose(z, [1, 2, 0])  # (C, S, D)


def _per_chain_ess(samples_chains: tf.Tensor) -> tf.Tensor:
    """Per-chain ESS via :func:`tfp.mcmc.effective_sample_size`.

    Returns a tensor of shape ``(C, D)``.  TFP's API expects samples-first,
    so we transpose ``(C, S, D)`` → ``(S, C, D)`` and let TFP treat ``C`` and
    ``D`` as independent batch dimensions.
    """
    samples_first = tf.transpose(samples_chains, [1, 0, 2])  # (S, C, D)
    ess = tfp.mcmc.effective_sample_size(samples_first)       # (C, D)
    return tf.maximum(ess, 0.0)


# ---------------------------------------------------------------------------
# Public diagnostics
# ---------------------------------------------------------------------------

def bulk_ess(samples_chains: tf.Tensor) -> tf.Tensor:
    """Vehtari 2021 bulk-ESS: per-chain ESS of rank-normalized split chains, summed.

    "Bulk-ESS" measures how informative the chains are about the posterior
    *mean* — i.e. the variance of the central tendency estimator.  Following
    Stan's convention we (1) split each chain in half, (2) rank-normalize
    across the resulting ``2C`` half-chains, (3) compute per-half-chain ESS,
    and (4) sum across chains.

    Parameters
    ----------
    samples_chains : tf.Tensor, shape ``(C, S, D)`` (or ``(S, D)`` / ``(S,)``).

    Returns
    -------
    tf.Tensor, shape ``(D,)``.
    """
    x = _ensure_3d(samples_chains)
    z = _rank_normalize(_split_chains(x))
    per = _per_chain_ess(z)            # (2C, D)
    return tf.reduce_sum(per, axis=0)  # (D,)


def tail_ess(
    samples_chains: tf.Tensor,
    quantiles: Sequence[float] = (0.05, 0.95),
) -> tf.Tensor:
    """Vehtari 2021 tail-ESS at extreme quantiles (§4.3).

    For each requested quantile ``q``, replaces samples by the indicator
    ``1{x <= quantile_q}`` (which is the score function of the empirical
    quantile estimator), then computes ESS as in :func:`bulk_ess`.  The
    *tail-ESS* returned per parameter is the minimum across the requested
    quantiles — this is the worst-case mixing for tail summaries.

    Parameters
    ----------
    samples_chains : tf.Tensor, shape ``(C, S, D)``.
    quantiles : pair of floats in (0, 1).

    Returns
    -------
    tf.Tensor, shape ``(D,)``.
    """
    x = _ensure_3d(samples_chains)
    C, S, D = x.shape[0], x.shape[1], x.shape[2]
    flat = tf.reshape(tf.transpose(x, [2, 0, 1]), [D, C * S])  # (D, N)

    ess_list: List[tf.Tensor] = []
    for q in quantiles:
        q_f = float(q)
        thresh = tfp.stats.percentile(flat, 100.0 * q_f, axis=1)  # (D,)
        # Indicator I{x <= thresh} per parameter, broadcast back to (C, S, D).
        thresh_b = tf.reshape(thresh, [1, 1, D])
        ind = tf.cast(x <= thresh_b, tf.float32)
        # Rank-normalize the indicator and compute split-ESS.
        z = _rank_normalize(_split_chains(ind))
        per = _per_chain_ess(z)
        ess_list.append(tf.reduce_sum(per, axis=0))
    stacked = tf.stack(ess_list, axis=0)  # (n_quantiles, D)
    return tf.reduce_min(stacked, axis=0)


def _rhat_via_tfp(samples_chains: tf.Tensor) -> tf.Tensor:
    """TFP wrapper that handles the ``(C, S, D) -> (S, C, D)`` transpose.

    TFP's :func:`tfp.mcmc.potential_scale_reduction` treats axis 0 as the
    sample axis and the next ``independent_chain_ndims`` axes as chain axes.
    Our canonical layout is chain-first, so we transpose before calling.
    """
    samples_first = tf.transpose(samples_chains, [1, 0, 2])  # (S, C, D)
    return tfp.mcmc.potential_scale_reduction(
        samples_first, independent_chain_ndims=1, split_chains=True
    )


def split_rhat(samples_chains: tf.Tensor) -> tf.Tensor:
    """Classical split-R-hat (Gelman-Rubin with half-chain split).

    Wraps :func:`tfp.mcmc.potential_scale_reduction` with ``split_chains=True``.
    Convergence threshold ``< 1.01`` (Vehtari 2021).
    """
    x = _ensure_3d(samples_chains)
    return _rhat_via_tfp(x)


def rank_rhat(samples_chains: tf.Tensor) -> tf.Tensor:
    """Rank-normalized + folded split-R-hat (Vehtari 2021 §3).

    Returns ``max(rank_R-hat, folded_rank_R-hat)`` per parameter.  The folded
    variant is sensitive to multi-modality / scale mismatch that the
    rank-normalized version on the raw samples might miss.

    Convergence threshold ``< 1.01``.
    """
    x = _ensure_3d(samples_chains)

    # Rank-normalized split-R-hat on the raw samples.
    z = _rank_normalize(x)
    rhat_rank = _rhat_via_tfp(z)

    # Folded version: |x - median(x)| (median pooled across all chains).
    C, S, D = x.shape[0], x.shape[1], x.shape[2]
    flat = tf.reshape(tf.transpose(x, [2, 0, 1]), [D, C * S])  # (D, N)
    median = tfp.stats.percentile(flat, 50.0, axis=1)          # (D,)
    folded = tf.abs(x - tf.reshape(median, [1, 1, D]))
    z_folded = _rank_normalize(folded)
    rhat_folded = _rhat_via_tfp(z_folded)

    return tf.maximum(rhat_rank, rhat_folded)


def credible_interval(
    samples_chains: tf.Tensor,
    level: float = 0.95,
) -> tf.Tensor:
    """Flat-quantile credible interval pooled across chains.

    Parameters
    ----------
    samples_chains : tf.Tensor, shape ``(C, S, D)``.
    level : float in (0, 1).  Default 0.95.

    Returns
    -------
    tf.Tensor, shape ``(D, 2)`` — columns are ``[lower, upper]``.
    """
    x = _ensure_3d(samples_chains)
    C, S, D = x.shape[0], x.shape[1], x.shape[2]
    flat = tf.reshape(tf.transpose(x, [2, 0, 1]), [D, C * S])  # (D, N)
    lo_pct = 100.0 * (1.0 - level) / 2.0
    hi_pct = 100.0 * (1.0 + level) / 2.0
    lo = tfp.stats.percentile(flat, lo_pct, axis=1)
    hi = tfp.stats.percentile(flat, hi_pct, axis=1)
    return tf.stack([lo, hi], axis=1)  # (D, 2)


def coverage(
    ci: tf.Tensor,
    truth: tf.Tensor,
) -> tf.Tensor:
    """Boolean per-parameter test ``ci_low <= truth <= ci_high``.

    Parameters
    ----------
    ci : tf.Tensor, shape ``(D, 2)`` — output of :func:`credible_interval`.
    truth : tf.Tensor, shape ``(D,)``.
    """
    truth = tf.cast(truth, ci.dtype)
    return tf.logical_and(ci[:, 0] <= truth, truth <= ci[:, 1])


def convergence_verdict(
    rhat: tf.Tensor,
    bulk_ess_val: tf.Tensor,
    tail_ess_val: tf.Tensor,
    rhat_thresh: float = 1.01,
    ess_thresh: float = 400.0,
) -> tf.Tensor:
    """Per-parameter pass/fail.

    Stan / PyMC defaults: a parameter is "well-mixed" if ``R-hat < 1.01`` AND
    ``bulk-ESS > 400`` AND ``tail-ESS > 400``.  See Vehtari 2021 §6 for the
    rationale (400 ESS gives ~5% MCSE on a unit-variance estimator).
    """
    return tf.logical_and(
        rhat < rhat_thresh,
        tf.logical_and(bulk_ess_val > ess_thresh, tail_ess_val > ess_thresh),
    )


# ---------------------------------------------------------------------------
# Bundle + formatter
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsSummary:
    """Container returned by :func:`diagnostics_summary`."""

    param_names: List[str]
    posterior_mean: List[float]
    posterior_std: List[float]
    bulk_ess: List[float]
    tail_ess: List[float]
    ess_pct: List[float]                 # bulk_ess / (C * S)
    split_rhat: List[float]
    rank_rhat: List[float]
    ci_low: List[float]
    ci_high: List[float]
    truth: Optional[List[float]] = None
    covered: Optional[List[bool]] = None
    converged: List[bool] = field(default_factory=list)
    n_chains: int = 0
    n_samples_per_chain: int = 0
    ci_level: float = 0.95


def diagnostics_summary(
    samples_chains: tf.Tensor,
    param_names: Sequence[str],
    truth: Optional[Sequence[float]] = None,
    ci_level: float = 0.95,
    rhat_thresh: float = 1.01,
    ess_thresh: float = 400.0,
) -> DiagnosticsSummary:
    """One-shot computation of every diagnostic in this module.

    Parameters
    ----------
    samples_chains : tf.Tensor, shape ``(C, S, D)``.
    param_names : list of D names matching the last axis.
    truth : optional length-D ground-truth values (enables coverage column).
    ci_level : credible-interval level (default 0.95).
    rhat_thresh, ess_thresh : convergence verdict thresholds.

    Returns
    -------
    DiagnosticsSummary
    """
    x = _ensure_3d(samples_chains)
    C, S, D = x.shape[0], x.shape[1], x.shape[2]
    if len(param_names) != D:
        raise ValueError(
            f"Got {len(param_names)} param_names but samples have D={D}."
        )

    flat = tf.reshape(tf.transpose(x, [2, 0, 1]), [D, C * S])
    mean = tf.reduce_mean(flat, axis=1)
    std = tf.math.reduce_std(flat, axis=1)

    b_ess = bulk_ess(x)
    t_ess = tail_ess(x)
    ess_pct = 100.0 * b_ess / float(C * S)
    s_rhat = split_rhat(x)
    r_rhat = rank_rhat(x)
    ci = credible_interval(x, level=ci_level)

    truth_t = None
    cov_t = None
    if truth is not None:
        truth_t = tf.cast(truth, tf.float32)
        cov_t = coverage(ci, truth_t)

    converged = convergence_verdict(r_rhat, b_ess, t_ess, rhat_thresh, ess_thresh)

    return DiagnosticsSummary(
        param_names=list(param_names),
        posterior_mean=mean.numpy().tolist(),
        posterior_std=std.numpy().tolist(),
        bulk_ess=b_ess.numpy().tolist(),
        tail_ess=t_ess.numpy().tolist(),
        ess_pct=ess_pct.numpy().tolist(),
        split_rhat=s_rhat.numpy().tolist(),
        rank_rhat=r_rhat.numpy().tolist(),
        ci_low=ci[:, 0].numpy().tolist(),
        ci_high=ci[:, 1].numpy().tolist(),
        truth=None if truth_t is None else truth_t.numpy().tolist(),
        covered=None if cov_t is None else cov_t.numpy().tolist(),
        converged=converged.numpy().tolist(),
        n_chains=int(C),
        n_samples_per_chain=int(S),
        ci_level=float(ci_level),
    )


def format_diagnostics_table(
    method_name: str,
    summary: DiagnosticsSummary,
    width: int = 92,
) -> str:
    """Format a :class:`DiagnosticsSummary` as a printable text block."""
    lines: List[str] = []
    lines.append("=" * width)
    lines.append(
        f"  Diagnostics — {method_name}   "
        f"(chains={summary.n_chains}, samples/chain={summary.n_samples_per_chain},"
        f" CI level={int(summary.ci_level * 100)}%)"
    )
    lines.append("-" * width)

    has_truth = summary.truth is not None
    header = (
        f"{'param':<12} {'mean':>10} {'std':>10} "
        f"{'bulk-ESS':>10} {'tail-ESS':>10} {'ESS%':>7} "
        f"{'splitR^':>9} {'rankR^':>9} {'95% CI':>22}"
    )
    if has_truth:
        header += f" {'truth':>10} {'covers?':>9}"
    header += f" {'ok':>5}"
    lines.append(header)
    lines.append("-" * width)

    for i, name in enumerate(summary.param_names):
        ci_str = f"[{summary.ci_low[i]:.3f}, {summary.ci_high[i]:.3f}]"
        row = (
            f"{name:<12} "
            f"{summary.posterior_mean[i]:>10.3f} {summary.posterior_std[i]:>10.3f} "
            f"{summary.bulk_ess[i]:>10.1f} {summary.tail_ess[i]:>10.1f} "
            f"{summary.ess_pct[i]:>6.1f}% "
            f"{summary.split_rhat[i]:>9.3f} {summary.rank_rhat[i]:>9.3f} "
            f"{ci_str:>22}"
        )
        if has_truth:
            cov = summary.covered[i] if summary.covered is not None else False
            row += f" {summary.truth[i]:>10.3f} {('YES' if cov else 'NO'):>9}"
        row += f" {('PASS' if summary.converged[i] else 'FAIL'):>5}"
        lines.append(row)

    lines.append("-" * width)
    overall = all(summary.converged)
    cov_overall = (
        all(summary.covered) if summary.covered is not None else None
    )
    verdict = "CONVERGED" if overall else "NOT CONVERGED (rank-R^>=1.01 or ESS<400)"
    lines.append(f"  Overall verdict : {verdict}")
    if cov_overall is not None:
        cov_str = "all true values inside 95% CI" if cov_overall else "some truth not covered"
        lines.append(f"  Coverage        : {cov_str}")
    lines.append("=" * width)
    return "\n".join(lines)
