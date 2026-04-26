"""Entropy-regularised optimal transport via the Sinkhorn algorithm."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _cost_matrix(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Half squared Euclidean cost matrix C_ij = (1/2) ||x_i - y_j||^2.
    Matches filterflow's cost (squared_distances / 2) for comparable epsilon.

    Args:
        x: (N, d) tensor
        y: (N, d) tensor
    """
    x_norm = tf.reduce_sum(x**2, axis=1, keepdims=True)  # (N, 1)
    y_norm = tf.reduce_sum(y**2, axis=1, keepdims=True)  # (N, 1)
    sq_dist = x_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(x, y, transpose_b=True)
    # Clamp: tiny negative values from float32 cancellation break sqrt/exp in OT grads.
    sq_dist = tf.maximum(sq_dist, 0.0)
    return 0.5 * sq_dist  # (N, N)

@tf.function(jit_compile=True)
def sinkhorn_potentials(
    a: tf.Tensor,
    b: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
    epsilon: float = 0.5,
    n_iters: int = 50,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute dual Sinkhorn potentials (f, g) for entropy-regularized OT.

    This implements Algorithm 2 (Potentials) from the paper in a stable
    log-sum-exp form, following Feydy et al. (2019).

    Args:
        a: source weights, shape (N,), non-negative, sum to 1.
        b: target weights, shape (N,), non-negative, sum to 1.
        x: source locations, shape (N, d).
        y: target locations, shape (N, d).
        epsilon: entropic regularization strength.
        n_iters: number of Sinkhorn fixed-point iterations.

    Returns:
        f, g: dual potentials, each of shape (N,).
    """
    # Get static or dynamic shape
    N = x.shape[0] if x.shape[0] is not None else tf.shape(x)[0]

    # (N, N) cost matrix: half squared Euclidean distance
    C = _cost_matrix(x, y)

    # Handle both static and dynamic shapes
    if isinstance(N, int):
        f = tf.zeros(N, dtype=x.dtype)
        g = tf.zeros(N, dtype=x.dtype)
    else:
        f = tf.zeros_like(a)
        g = tf.zeros_like(b)

    # Helper: T_eps(a, f, C_:)
    def T_eps(weights: tf.Tensor, potentials: tf.Tensor, costs_row_or_col: tf.Tensor) -> tf.Tensor:
        # weights: (N,), potentials: (N,), costs_row_or_col: (N,)
        # T_eps(a, f, C_:,i) = -eps * log sum_k exp(log a_k + (f_k - C_k,i)/eps)
        log_w = tf.math.log(weights + 1e-20)  # (N,)
        scaled = (potentials - costs_row_or_col) / epsilon + log_w
        m = tf.reduce_max(scaled)  # scalar, for numerical stability
        return -epsilon * (tf.math.log(tf.reduce_sum(tf.exp(scaled - m))) + m)

    # Use tf.while_loop for better performance, but for clarity use Python loop
    # In practice, n_iters is small (20-50), so Python loop is fine
    for _ in range(n_iters):
        # Update f: f_i <- 0.5 * (f_i + T_eps(b, g, C_{i,:}))
        # Vectorized version: log_b_vec + (g_j - C_ij) / epsilon
        log_b_vec = tf.math.log(b + 1e-20)
        scaled_f = (tf.expand_dims(g, axis=0) - C) / epsilon + tf.expand_dims(log_b_vec, axis=0)
        new_f = -epsilon * tf.reduce_logsumexp(scaled_f, axis=1)
        f = 0.5 * (f + new_f)

        # Update g: g_j <- 0.5 * (g_j + T_eps(a, f, C_{:,j}))
        # Vectorized version: log_a_vec + (f_i - C_ij) / epsilon
        log_a_vec = tf.math.log(a + 1e-20)
        scaled_g = (tf.expand_dims(f, axis=1) - C) / epsilon + tf.expand_dims(log_a_vec, axis=1)
        new_g = -epsilon * tf.reduce_logsumexp(scaled_g, axis=0)
        g = 0.5 * (g + new_g)

    return f, g



def entropy_regularized_transport(
    a: tf.Tensor,
    b: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
    epsilon: float = 0.5,
    n_iters: int = 50,
) -> tf.Tensor:
    """
    Compute entropy-regularized OT transport matrix P^OT_ε.

    Args:
        a: source weights, shape (N,)
        b: target weights, shape (N,)
        x: source locations, shape (N, d)
        y: target locations, shape (N, d)
        epsilon: entropic regularization strength
        n_iters: Sinkhorn iterations

    Returns:
        P: (N, N) transport matrix, rows sum to a, columns sum to b (up to
           numerical tolerance).
    """
    f, g = sinkhorn_potentials(a, b, x, y, epsilon=epsilon, n_iters=n_iters)
    C = _cost_matrix(x, y)

    # P_ij = a_i b_j * exp((f_i + g_j - C_ij)/eps)
    log_a = tf.expand_dims(tf.math.log(a + 1e-20), axis=1)  # (N,1)
    log_b = tf.expand_dims(tf.math.log(b + 1e-20), axis=0)  # (1,N)
    f_expanded = tf.expand_dims(f, axis=1)  # (N, 1)
    g_expanded = tf.expand_dims(g, axis=0)  # (1, N)
    exponent = (f_expanded + g_expanded - C) / epsilon
    P = tf.exp(log_a + log_b + exponent)

    return P