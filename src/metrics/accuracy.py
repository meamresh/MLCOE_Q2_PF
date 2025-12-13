"""
Accuracy metrics for state-space model filtering.

This module provides metrics to assess the accuracy and consistency of
filtered state estimates against ground truth, including RMSE and NEES.

"""

from __future__ import annotations

import tensorflow as tf


def compute_rmse(estimates: tf.Tensor, ground_truth: tf.Tensor) -> float:
    """
    Compute root mean squared error between estimates and ground truth.

    Parameters
    ----------
    estimates : tf.Tensor
        Estimated states of shape (N, n) or (N, n, 1).
    ground_truth : tf.Tensor
        True states of shape (N, n).

    Returns
    -------
    float
        RMSE value.

    Examples
    --------
    >>> x_true = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> x_est = tf.constant([[1.1, 2.1], [2.9, 4.1]])
    >>> rmse = compute_rmse(x_est, x_true)
    >>> print(f"RMSE: {rmse:.4f}")
    RMSE: 0.1414
    """
    estimates = tf.convert_to_tensor(estimates)
    ground_truth = tf.convert_to_tensor(ground_truth)
    try:
        if estimates.shape.ndims == 3:
            estimates = tf.squeeze(estimates, axis=-1)
    except (AttributeError, ValueError):
        if tf.rank(estimates) == 3:
            estimates = tf.squeeze(estimates, axis=-1)
    return float(tf.sqrt(tf.reduce_mean((estimates - ground_truth) ** 2)))


def compute_mae(estimates: tf.Tensor, ground_truth: tf.Tensor) -> float:
    """
    Compute mean absolute error between estimates and ground truth.

    Parameters
    ----------
    estimates : tf.Tensor
        Estimated states of shape (N, n) or (N, n, 1).
    ground_truth : tf.Tensor
        True states of shape (N, n).

    Returns
    -------
    float
        MAE value.
    """
    estimates = tf.convert_to_tensor(estimates)
    ground_truth = tf.convert_to_tensor(ground_truth)
    try:
        if estimates.shape.ndims == 3:
            estimates = tf.squeeze(estimates, axis=-1)
    except (AttributeError, ValueError):
        if tf.rank(estimates) == 3:
            estimates = tf.squeeze(estimates, axis=-1)
    return float(tf.reduce_mean(tf.abs(estimates - ground_truth)))


def compute_nees(estimates: tf.Tensor, 
                 covariances: tf.Tensor,
                 ground_truth: tf.Tensor) -> tf.Tensor:
    """
    Compute Normalized Estimation Error Squared (NEES) for consistency check.

    NEES at time t is defined as:
        NEES_t = (x_t - x̂_t)^T P_t^{-1} (x_t - x̂_t)

    Under optimal filtering with a correct model, NEES follows a chi-squared
    distribution with n_x degrees of freedom: NEES_t ~ χ²(n_x).

    Parameters
    ----------
    estimates : tf.Tensor
        Filtered state estimates, shape (N, n, 1) or (N, n).
    covariances : tf.Tensor
        Filtered covariances, shape (N, n, n).
    ground_truth : tf.Tensor
        True states, shape (N, n).

    Returns
    -------
    nees : tf.Tensor
        NEES values at each time step, shape (N,).

    Notes
    -----
    For a filter to be consistent:
        - Mean NEES ≈ n_x (expected value of χ²(n_x))
        - Mean NEES ≪ n_x suggests overconfidence (underestimated uncertainty)
        - Mean NEES ≫ n_x suggests model mismatch or overestimated uncertainty

    References
    ----------
    Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). 
    Estimation with Applications to Tracking and Navigation. Wiley. Chapter 5.
    """
    estimates = tf.convert_to_tensor(estimates)
    covariances = tf.convert_to_tensor(covariances)
    ground_truth = tf.convert_to_tensor(ground_truth)
    if len(tf.shape(estimates)) == 3:
        estimates = tf.squeeze(estimates, axis=-1)

    N = tf.shape(estimates)[0]
    nees_list = []

    for t in range(N):
        error = tf.expand_dims(ground_truth[t] - estimates[t], axis=-1)
        P_inv = tf.linalg.inv(covariances[t])
        nees_t = tf.squeeze(tf.matmul(tf.transpose(error), tf.matmul(P_inv, error)))
        nees_list.append(nees_t)

    return tf.stack(nees_list, axis=0)


def compute_nll(observations: tf.Tensor,
                predicted_means: tf.Tensor,
                innovation_covariances: tf.Tensor) -> float:
    """
    Compute negative log-likelihood of observations given predictions.

    Evaluates the predictive distribution:
        p(y_t | y_{1:t-1}) = N(C x̂_{t|t-1}, S_t)
    where S_t = C P_{t|t-1} C^T + R is the innovation covariance.

    Parameters
    ----------
    observations : tf.Tensor
        Observed measurements, shape (N, m).
    predicted_means : tf.Tensor
        Predicted observation means (C x̂_{t|t-1}), shape (N, m) or (N, m, 1).
    innovation_covariances : tf.Tensor
        Innovation covariances S_t, shape (N, m, m).

    Returns
    -------
    float
        Average negative log-likelihood per time step.

    Notes
    -----
    Lower NLL indicates better predictive performance. This metric is
    particularly useful when ground truth states are unavailable.
    """
    observations = tf.convert_to_tensor(observations)
    predicted_means = tf.convert_to_tensor(predicted_means)
    innovation_covariances = tf.convert_to_tensor(innovation_covariances)
    try:
        if predicted_means.shape.ndims == 3:
            predicted_means = tf.squeeze(predicted_means, axis=-1)
    except (AttributeError, ValueError):
        if tf.rank(predicted_means) == 3:
            predicted_means = tf.squeeze(predicted_means, axis=-1)

    N = tf.shape(observations)[0]
    log_likelihood = 0.0

    for t in range(N):
        innovation = observations[t] - predicted_means[t]
        S = innovation_covariances[t]

        # Log-likelihood of Gaussian: -0.5 * [log|S| + innov^T S^{-1} innov + m*log(2π)]
        sign, logdet = tf.linalg.slogdet(S)
        mahal = tf.squeeze(tf.matmul(tf.expand_dims(innovation, 0), tf.linalg.solve(S, tf.expand_dims(innovation, -1))))
        m = tf.cast(tf.shape(innovation)[0], tf.float32)

        log_likelihood += -0.5 * (logdet + mahal + m * tf.math.log(2.0 * 3.141592653589793))

    return float(-log_likelihood / tf.cast(N, tf.float32))


def compute_per_dimension_rmse(estimates: tf.Tensor, 
                                ground_truth: tf.Tensor) -> tf.Tensor:
    """
    Compute RMSE for each state dimension separately.

    Useful for identifying which state components are harder to estimate.

    Parameters
    ----------
    estimates : tf.Tensor
        Estimated states, shape (N, n) or (N, n, 1).
    ground_truth : tf.Tensor
        True states, shape (N, n).

    Returns
    -------
    rmse_per_dim : tf.Tensor
        RMSE for each dimension, shape (n,).
    """
    estimates = tf.convert_to_tensor(estimates)
    ground_truth = tf.convert_to_tensor(ground_truth)
    try:
        if estimates.shape.ndims == 3:
            estimates = tf.squeeze(estimates, axis=-1)
    except (AttributeError, ValueError):
        if tf.rank(estimates) == 3:
            estimates = tf.squeeze(estimates, axis=-1)

    return tf.sqrt(tf.reduce_mean((estimates - ground_truth) ** 2, axis=0))