"""Deterministic and soft resampling strategies for differentiable PFs."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from .sinkhorn import entropy_regularized_transport


def det_resample(
    x: tf.Tensor,
    log_w: tf.Tensor,
    epsilon: float = 0.5,
    n_iters: int = 50,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Differentiable Ensemble Transform (DET) resampling.

    Implements Algorithm 3 in the paper using entropy-regularized OT.

    Args:
        x: particles at time t, shape (N, d).
        log_w: log-weights at time t, shape (N,).
        epsilon: entropic regularization strength.
        n_iters: Sinkhorn iterations.

    Returns:
        x_tilde: resampled particles, shape (N, d), approximately corresponding
                 to drawing from the weighted empirical measure.
        w_tilde: uniform weights, shape (N,), i.e. 1/N.
    """
    N = x.shape[0] if x.shape[0] is not None else tf.shape(x)[0]
    N_float = tf.cast(N, dtype=x.dtype)

    # Normalize weights using softmax
    w = tf.nn.softmax(log_w, axis=0)  # (N,)

    # Transport from weighted (source) to uniform (target) so that resampled particles approximate the weighted measure.
    # Source α = sum_i w_i δ_{x_i}, target β = (1/N) on N slots.
    # P_{i,j} = mass from weighted particle i to uniform slot j; sum_j P_ij = w_i, sum_i P_ij = 1/N.
    a = w  # source: weighted
    b = tf.fill(tf.shape(w), 1.0 / N_float)  # target: uniform


    P = entropy_regularized_transport(a, b, x, x, epsilon=epsilon, n_iters=n_iters)

    # New particle at uniform slot j = barycenter of source positions that sent to j: x_tilde_j = N sum_i P_{i,j} x_i = N (P.T @ x)_j
    x_tilde = N_float * tf.matmul(P, x, transpose_a=True)  # (N, d)
    w_tilde = tf.fill(tf.shape(w), 1.0 / N_float)  # uniform weights after resampling

    return x_tilde, w_tilde


def soft_resample(
    x: tf.Tensor,
    log_w: tf.Tensor,
    alpha: float = 0.5,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Soft-resampling (mixture with uniform) as described in arXiv:2302.09639 and Karkus et al. (2018), 1805.08975.
    
    Instead of sampling from the weights w, we sample from a mixture:
        q = (1 - alpha) * w + alpha * (1/N)
    
    The new weights are updated to keep the estimator unbiased:
        w_new = w / q
    
    This allows gradients to flow back to the original weights even through the 
    stochastic sampling step (which is treated as a constant during backprop).

    Args:
        x: particles, shape (N, d).
        log_w: log-weights, shape (N,).
        alpha: mixture parameter (0 = hard resampling, 1 = no resampling/uniform sampling).

    Returns:
        x_resampled: resampled particles, shape (N, d).
        log_w_resampled: adjusted log-weights, shape (N,).
    """
    N = x.shape[0] if x.shape[0] is not None else tf.shape(x)[0]
    N_float = tf.cast(N, dtype=x.dtype)
    
    w = tf.nn.softmax(log_w, axis=0)
    
    # 1. Compute mixture distribution q
    q = (1.0 - alpha) * w + alpha * (1.0 / N_float) 
    # alpha=0 → hard resampling, alpha=1 → uniform, or no resampling
    
    # 2. Sample indices from q
    idx = tf.random.categorical(tf.math.log(tf.expand_dims(q, 0) + 1e-20), num_samples=N)[0]
    
    # 3. Gather particles
    x_resampled = tf.gather(x, idx, axis=0)
    
    # 4. Compute new weights: w_new = w / q (at the sampled indices)
    # We take log of w/q = log(w) - log(q)
    log_q = tf.math.log(q + 1e-20)
    log_w_new = tf.gather(log_w - log_q, idx, axis=0)
    
    # Normalize weights so they sum to 1 (in log space)
    log_w_new = log_w_new - tf.reduce_logsumexp(log_w_new)
    
    return x_resampled, log_w_new
