"""
Particle filter metrics for assessing filter performance and degeneracy.

This module provides metrics specific to particle filters, including effective
sample size (ESS) and weight distribution entropy, which are crucial for
diagnosing filter degeneracy and resampling decisions.
"""

from __future__ import annotations

import tensorflow as tf


def compute_effective_sample_size(weights: tf.Tensor) -> tf.Tensor:
    """
    Compute Effective Sample Size (ESS) for particle weights.

    The ESS measures the number of independent samples that the weighted
    particles represent. It is defined as:
        ESS = 1 / sum(w_i^2)

    where w_i are the normalized particle weights.

    A low ESS indicates that only a few particles have significant weight,
    which is a sign of filter degeneracy. Typically, resampling is triggered
    when ESS falls below a threshold (e.g., 0.5 * num_particles).

    Parameters
    ----------
    weights : tf.Tensor
        Normalized particle weights of shape (num_particles,).

    Returns
    -------
    ess : tf.Tensor
        Effective sample size (scalar).

    Examples
    --------
    >>> weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
    >>> ess = compute_effective_sample_size(weights)
    >>> print(f"ESS: {ess:.2f}")
    ESS: 3.33

    Notes
    -----
    - ESS ranges from 1 (worst case: one particle has all weight) to
      num_particles (best case: uniform weights).
    - Lower ESS indicates higher degeneracy.
    """
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    return 1.0 / tf.reduce_sum(weights ** 2)


def compute_weight_entropy(weights: tf.Tensor, normalize: bool = True) -> tf.Tensor:
    """
    Compute Shannon entropy of particle weight distribution.

    The entropy measures the diversity of the weight distribution. Higher
    entropy indicates more uniform weights (less degeneracy), while lower
    entropy indicates concentrated weights (more degeneracy).

    The entropy is defined as:
        H = -sum(w_i * log(w_i))

    where w_i are the normalized particle weights.

    Parameters
    ----------
    weights : tf.Tensor
        Normalized particle weights of shape (num_particles,).
    normalize : bool, optional
        If True, returns normalized entropy H / log(N) in [0, 1], where N is
        the number of particles. This makes the metric independent of the
        number of particles. Defaults to True.

    Returns
    -------
    entropy : tf.Tensor
        Entropy value (normalized if normalize=True, otherwise in nats).

    Examples
    --------
    >>> weights = tf.ones(10, dtype=tf.float32) / 10.0
    >>> entropy = compute_weight_entropy(weights, normalize=True)
    >>> print(f"Normalized entropy: {entropy:.3f}")
    Normalized entropy: 1.000

    Notes
    -----
    - Normalized entropy of 1.0 indicates uniform weights (no degeneracy).
    - Normalized entropy of 0.0 indicates complete degeneracy (one particle
      has all weight).
    - Weights near zero are filtered out to avoid numerical issues.
    """
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    
    # Filter out near-zero weights to avoid numerical issues
    weights_clean = tf.boolean_mask(weights, weights > 1e-12)
    
    if tf.size(weights_clean) == 0:
        return tf.constant(0.0, dtype=tf.float32)

    # Compute entropy: -sum(w_i * log(w_i))
    entropy = -tf.reduce_sum(weights_clean * tf.math.log(weights_clean + 1e-12))
    
    if normalize:
        num_particles = tf.cast(tf.size(weights), tf.float32)
        max_entropy = tf.math.log(num_particles)
        entropy = tf.cond(
            max_entropy > 0.0,
            lambda: entropy / max_entropy,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
    
    return entropy


def compute_weight_variance(weights: tf.Tensor) -> tf.Tensor:
    """
    Compute variance of particle weights.

    This metric measures the spread of weights. Higher variance indicates
    more uneven weight distribution (more degeneracy).

    Parameters
    ----------
    weights : tf.Tensor
        Normalized particle weights of shape (num_particles,).

    Returns
    -------
    variance : tf.Tensor
        Weight variance (scalar).

    Examples
    --------
    >>> weights = tf.ones(10, dtype=tf.float32) / 10.0
    >>> var = compute_weight_variance(weights)
    >>> print(f"Variance: {var:.6f}")
    Variance: 0.000000
    """
    import tensorflow_probability as tfp
    return tfp.stats.variance(weights)

