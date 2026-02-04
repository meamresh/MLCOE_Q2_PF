"""
Resampling algorithms for particle filters.

This module provides common resampling strategies used across
particle filter implementations for addressing weight degeneracy.

Supported Methods
-----------------
- Systematic: Low variance, O(N), most commonly used
- Multinomial: Simple but higher variance
- Stratified: Balance between systematic and multinomial
- Residual: Deterministic + stochastic hybrid

All functions use TensorFlow only (no NumPy) and are JIT-compatible.

Key Concept: Effective Sample Size (ESS)
----------------------------------------
ESS = 1 / Σᵢ(wᵢ)² measures how many particles have significant weight.
- ESS = N: All weights equal (ideal)
- ESS = 1: One particle has all weight (severe degeneracy)
- Typical threshold: Resample when ESS < 0.5N or 0.7N

References
----------
- Douc, R., Cappe, O., & Moulines, E. (2005). "Comparison of resampling schemes"
- Liu, J. S., & Chen, R. (1998). "Sequential Monte Carlo methods"

"""

from __future__ import annotations

import tensorflow as tf


# =============================================================================
# Core Resampling Algorithms
# =============================================================================


@tf.function
def systematic_resample(weights: tf.Tensor) -> tf.Tensor:
    """
    Systematic resampling with low variance.
    
    This is the most commonly used resampling method in particle filters
    due to its low variance properties and O(N) complexity.
    
    Parameters
    ----------
    weights : tf.Tensor
        Normalized particle weights of shape (N,).
        
    Returns
    -------
    tf.Tensor
        Resampled indices of shape (N,), dtype int32.
        
    Examples
    --------
    >>> weights = tf.constant([0.1, 0.2, 0.3, 0.4])
    >>> indices = systematic_resample(weights)
    >>> resampled_particles = tf.gather(particles, indices)
    
    Notes
    -----
    The algorithm generates N positions using:
        positions[i] = (u + i) / N
    where u ~ Uniform(0, 1/N), then selects indices based on cumulative weights.
    """
    N = tf.shape(weights)[0]
    N_float = tf.cast(N, tf.float32)
    
    # Normalize weights if not already normalized
    weights = weights / (tf.reduce_sum(weights) + 1e-15)
    
    # Cumulative sum
    cumsum = tf.cumsum(weights)
    
    # Generate systematic positions
    u = tf.random.uniform([], dtype=tf.float32) / N_float
    positions = u + tf.cast(tf.range(N), tf.float32) / N_float
    
    # Find indices using searchsorted
    indices = tf.searchsorted(cumsum, positions, side='right')
    indices = tf.minimum(indices, N - 1)
    
    return indices


@tf.function
def multinomial_resample(weights: tf.Tensor) -> tf.Tensor:
    """
    Multinomial resampling.
    
    Samples indices with replacement according to the weight distribution.
    Higher variance than systematic resampling but simpler to implement.
    
    Parameters
    ----------
    weights : tf.Tensor
        Normalized particle weights of shape (N,).
        
    Returns
    -------
    tf.Tensor
        Resampled indices of shape (N,), dtype int32.
    """
    N = tf.shape(weights)[0]
    
    # Normalize weights
    weights = weights / (tf.reduce_sum(weights) + 1e-15)
    
    # Use categorical distribution
    log_weights = tf.math.log(weights + 1e-15)
    indices = tf.random.categorical(
        log_weights[tf.newaxis, :], 
        N
    )[0]
    
    return tf.cast(indices, tf.int32)


@tf.function
def stratified_resample(weights: tf.Tensor) -> tf.Tensor:
    """
    Stratified resampling.
    
    Divides the CDF into N equal strata and samples one particle per stratum.
    Lower variance than multinomial but slightly higher than systematic.
    
    Parameters
    ----------
    weights : tf.Tensor
        Normalized particle weights of shape (N,).
        
    Returns
    -------
    tf.Tensor
        Resampled indices of shape (N,), dtype int32.
    """
    N = tf.shape(weights)[0]
    N_float = tf.cast(N, tf.float32)
    
    # Normalize weights
    weights = weights / (tf.reduce_sum(weights) + 1e-15)
    
    # Cumulative sum
    cumsum = tf.cumsum(weights)
    
    # Generate stratified positions: u_i ~ Uniform((i-1)/N, i/N)
    base = tf.cast(tf.range(N), tf.float32) / N_float
    u = tf.random.uniform([N], dtype=tf.float32) / N_float
    positions = base + u
    
    # Find indices
    indices = tf.searchsorted(cumsum, positions, side='right')
    indices = tf.minimum(indices, N - 1)
    
    return indices


def residual_resample(weights: tf.Tensor) -> tf.Tensor:
    """
    Residual resampling (Liu & Chen, 1998).
    
    First deterministically copies floor(N * w_i) of each particle,
    then samples the remaining particles multinomially.
    
    Parameters
    ----------
    weights : tf.Tensor
        Normalized particle weights of shape (N,).
        
    Returns
    -------
    tf.Tensor
        Resampled indices of shape (N,), dtype int32.
    """
    N = tf.shape(weights)[0]
    N_float = tf.cast(N, tf.float32)
    
    # Normalize weights
    weights = weights / (tf.reduce_sum(weights) + 1e-15)
    
    # Compute expected copies
    expected_copies = N_float * weights
    
    # Deterministic part: floor of expected copies
    deterministic_copies = tf.cast(tf.floor(expected_copies), tf.int32)
    
    # Create indices for deterministic copies
    indices_list = []
    for i in tf.range(N):
        copies = deterministic_copies[i]
        indices_list.append(tf.fill([copies], i))
    
    deterministic_indices = tf.concat(indices_list, axis=0)
    n_deterministic = tf.shape(deterministic_indices)[0]
    
    # Residual part
    n_residual = N - n_deterministic
    residual_weights = expected_copies - tf.cast(deterministic_copies, tf.float32)
    residual_weights = residual_weights / (tf.reduce_sum(residual_weights) + 1e-15)
    
    if n_residual > 0:
        log_residual = tf.math.log(residual_weights + 1e-15)
        residual_indices = tf.random.categorical(
            log_residual[tf.newaxis, :], 
            n_residual
        )[0]
        residual_indices = tf.cast(residual_indices, tf.int32)
        all_indices = tf.concat([deterministic_indices, residual_indices], axis=0)
    else:
        all_indices = deterministic_indices
    
    # Shuffle to avoid ordering artifacts
    all_indices = tf.random.shuffle(all_indices)
    
    return all_indices[:N]


def resample_particles(
    particles: tf.Tensor,
    weights: tf.Tensor,
    method: str = 'systematic'
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Resample particles using the specified method.
    
    This is a convenience function that handles the full resampling process:
    1. Compute resampling indices
    2. Gather resampled particles
    3. Reset weights to uniform
    
    Parameters
    ----------
    particles : tf.Tensor
        Particle states of shape (N, d).
    weights : tf.Tensor
        Particle weights of shape (N,).
    method : str, optional
        Resampling method. One of 'systematic', 'multinomial', 'stratified',
        'residual'. Default 'systematic'.
        
    Returns
    -------
    resampled_particles : tf.Tensor
        Resampled particles of shape (N, d).
    uniform_weights : tf.Tensor
        Uniform weights of shape (N,).
        
    Examples
    --------
    >>> particles_new, weights_new = resample_particles(particles, weights)
    """
    N = tf.shape(particles)[0]
    
    if method == 'systematic':
        indices = systematic_resample(weights)
    elif method == 'multinomial':
        indices = multinomial_resample(weights)
    elif method == 'stratified':
        indices = stratified_resample(weights)
    elif method == 'residual':
        indices = residual_resample(weights)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    resampled = tf.gather(particles, indices)
    uniform_weights = tf.ones(N, dtype=tf.float32) / tf.cast(N, tf.float32)
    
    return resampled, uniform_weights


@tf.function(jit_compile=True)
def compute_ess(weights: tf.Tensor) -> tf.Tensor:
    """
    Compute Effective Sample Size (ESS).
    
    ESS = 1 / sum(w_i^2)
    
    A low ESS indicates that only a few particles have significant weight
    (degeneracy). Typically, resampling is triggered when ESS < threshold * N.
    
    Parameters
    ----------
    weights : tf.Tensor
        Normalized particle weights of shape (N,).
        
    Returns
    -------
    tf.Tensor
        Effective sample size (scalar).
        
    Notes
    -----
    ESS ranges from 1 (complete degeneracy) to N (uniform weights).
    Common threshold: resample when ESS < 0.5 * N or ESS < 0.7 * N.
    """
    weights = weights / (tf.reduce_sum(weights) + 1e-15)
    return 1.0 / (tf.reduce_sum(weights ** 2) + 1e-15)


def should_resample(weights: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """
    Determine if resampling should be performed based on ESS.
    
    Parameters
    ----------
    weights : tf.Tensor
        Particle weights of shape (N,).
    threshold : float, optional
        ESS threshold as fraction of N. Default 0.5.
        
    Returns
    -------
    tf.Tensor
        Boolean scalar indicating whether to resample.
    """
    N = tf.shape(weights)[0]
    ess = compute_ess(weights)
    ess_threshold = threshold * tf.cast(N, tf.float32)
    return ess < ess_threshold
