"""
Accuracy metrics for state-space model filtering.

This module provides metrics to assess the accuracy and consistency of
filtered state estimates against ground truth, including RMSE and NEES.

"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp


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



"""
Mostly for UKF and EKF Tests
"""

"""
Filter consistency tests: NEES, NIS, and innovation whiteness.

These metrics quantify whether filter covariance estimates match actual errors,
detecting linearization breakdown or sigma-point approximation failures.
"""


def compute_nis(innovations: list[tf.Tensor],
                innovation_covariances: list[tf.Tensor],
                n_meas_per_landmark: int = 2) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute Normalized Innovation Squared (NIS).

    NIS tests filter consistency using measurement residuals (innovations).
    Unlike NEES, NIS doesn't require ground truth.

    Parameters
    ----------
    innovations : list[tf.Tensor]
        Innovation sequences, each shape (n_landmarks, n_meas).
    innovation_covariances : list[tf.Tensor]
        Innovation covariances, each shape (n_landmarks, n_meas, n_meas).
    n_meas_per_landmark : int
        Number of measurements per landmark (2 for range-bearing).

    Returns
    -------
    nis : tf.Tensor
        NIS values, shape (N * n_landmarks,).
    lower_bound : tf.Tensor
        Lower confidence bound (scalar).
    upper_bound : tf.Tensor
        Upper confidence bound (scalar).
    """
    nis_list = []

    for innov, S_list in zip(innovations, innovation_covariances):
        n_landmarks = int(tf.shape(innov)[0].numpy())
        for i in range(n_landmarks):
            S = S_list[i]
            # Add regularization
            S_reg = S + 1e-6 * tf.eye(n_meas_per_landmark, dtype=tf.float32)
            
            S_inv = tf.linalg.inv(S_reg)
            innov_i = innov[i]
            # Ensure innov_i is 1D (not scalar)
            if tf.rank(innov_i) == 0:
                # If scalar, expand to 1D
                innov_i = tf.expand_dims(innov_i, 0)
            # Compute y^T S^-1 y
            nis_val = tf.einsum('i,ij,j->', innov_i, S_inv, innov_i)
            nis_list.append(nis_val)

    nis = tf.stack(nis_list)

    # Chi-squared bounds (95% confidence interval)
    chi2_dist = tfp.distributions.Chi2(df=n_meas_per_landmark)
    lower_bound = chi2_dist.quantile(0.025)
    upper_bound = chi2_dist.quantile(0.975)

    return nis, lower_bound, upper_bound


def compute_autocorrelation(x: tf.Tensor, nlags: int = 20) -> tf.Tensor:
    """
    Compute autocorrelation function using TensorFlow.

    Parameters
    ----------
    x : tf.Tensor
        Time series, shape (N,).
    nlags : int
        Number of lags.

    Returns
    -------
    acf : tf.Tensor
        Autocorrelation function, shape (nlags+1,).
    """
    x = x - tf.reduce_mean(x)
    n = tf.shape(x)[0]
    
    acf_list = []
    variance = tf.reduce_sum(x ** 2) / tf.cast(n, tf.float32)
    
    for lag in range(nlags + 1):
        if lag == 0:
            acf_list.append(tf.constant(1.0, dtype=tf.float32))
        else:
            x1 = x[:-lag]
            x2 = x[lag:]
            covariance = tf.reduce_sum(x1 * x2) / tf.cast(n, tf.float32)
            acf_list.append(covariance / variance)
    
    return tf.stack(acf_list)


def test_innovation_whiteness(innovations: tf.Tensor,
                              nlags: int = 20) -> dict:
    """
    Test if innovation sequence is white (temporally uncorrelated).

    White innovations indicate optimal filtering. Non-white innovations
    suggest model mismatch or filter suboptimality.

    Parameters
    ----------
    innovations : tf.Tensor
        Flattened innovation sequence, shape (N,).
    nlags : int
        Number of lags for autocorrelation test.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'autocorrelation': Autocorrelation function
        - 'is_white': Boolean indicating whiteness
        - 'mean': Sample mean
        - 'std': Sample standard deviation
        - 'is_zero_mean': Boolean indicating zero-mean property
        - 'confidence_bound': Confidence bound for whiteness test
    """
    # Autocorrelation
    autocorr = compute_autocorrelation(innovations, nlags)

    # Zero-mean test
    mean = tf.reduce_mean(innovations)
    std = tf.math.reduce_std(innovations)
    is_zero_mean = tf.abs(mean) < 0.1 * std

    # Whiteness test: autocorrelation should be near zero for lags > 0
    # Use 95% confidence bound: ±1.96/sqrt(N)
    n_samples = tf.cast(tf.shape(innovations)[0], tf.float32)
    conf_bound = 1.96 / tf.sqrt(n_samples)
    is_white = tf.reduce_all(tf.abs(autocorr[1:]) < conf_bound)

    return {
        'autocorrelation': autocorr,
        'is_white': is_white,
        'mean': mean,
        'std': std,
        'is_zero_mean': is_zero_mean,
        'confidence_bound': conf_bound
    }


def analyze_filter_consistency(true_states: tf.Tensor,
                               ekf_results: dict,
                               ukf_results: dict,
                               compute_nees_fn) -> dict:
    """
    Comprehensive consistency analysis for EKF and UKF.

    Parameters
    ----------
    true_states : tf.Tensor
        Ground truth states, shape (N, n_states).
    ekf_results : dict
        EKF results containing 'states', 'covariances', 'innovations', 'S'.
    ukf_results : dict
        UKF results containing 'states', 'covariances', 'innovations', 'S'.
    compute_nees_fn : callable
        Function to compute NEES: compute_nees(estimates, covariances, ground_truth).

    Returns
    -------
    analysis : dict
        Comprehensive consistency metrics for both filters.
    """
    # NEES
    ekf_nees = compute_nees_fn(
        ekf_results['states'], ekf_results['covariances'], true_states
    )
    ukf_nees = compute_nees_fn(
        ukf_results['states'], ukf_results['covariances'], true_states
    )
    
    n_states = tf.shape(true_states)[1]
    chi2_dist = tfp.distributions.Chi2(df=tf.cast(n_states, tf.float32))
    nees_lower = chi2_dist.quantile(0.025)
    nees_upper = chi2_dist.quantile(0.975)

    # NIS
    ekf_nis, nis_lower, nis_upper = compute_nis(
        ekf_results['innovations'], ekf_results['S']
    )
    ukf_nis, _, _ = compute_nis(
        ukf_results['innovations'], ukf_results['S']
    )

    # Innovation whiteness
    ekf_innov_flat = tf.concat([tf.reshape(inn, [-1]) 
                                for inn in ekf_results['innovations']], axis=0)
    ukf_innov_flat = tf.concat([tf.reshape(inn, [-1]) 
                                for inn in ukf_results['innovations']], axis=0)

    ekf_whiteness = test_innovation_whiteness(ekf_innov_flat)
    ukf_whiteness = test_innovation_whiteness(ukf_innov_flat)

    # Consistency percentage (within bounds)
    ekf_nees_in_bounds = tf.logical_and(
        ekf_nees >= nees_lower, ekf_nees <= nees_upper
    )
    ukf_nees_in_bounds = tf.logical_and(
        ukf_nees >= nees_lower, ukf_nees <= nees_upper
    )
    
    ekf_nees_consistent = tf.reduce_mean(
        tf.cast(ekf_nees_in_bounds, tf.float32)
    ) * 100.0
    ukf_nees_consistent = tf.reduce_mean(
        tf.cast(ukf_nees_in_bounds, tf.float32)
    ) * 100.0

    ekf_nis_in_bounds = tf.logical_and(
        ekf_nis >= nis_lower, ekf_nis <= nis_upper
    )
    ukf_nis_in_bounds = tf.logical_and(
        ukf_nis >= nis_lower, ukf_nis <= nis_upper
    )
    
    ekf_nis_consistent = tf.reduce_mean(
        tf.cast(ekf_nis_in_bounds, tf.float32)
    ) * 100.0
    ukf_nis_consistent = tf.reduce_mean(
        tf.cast(ukf_nis_in_bounds, tf.float32)
    ) * 100.0

    return {
        'ekf': {
            'nees': ekf_nees,
            'nis': ekf_nis,
            'nees_consistent_pct': ekf_nees_consistent,
            'nis_consistent_pct': ekf_nis_consistent,
            'whiteness': ekf_whiteness
        },
        'ukf': {
            'nees': ukf_nees,
            'nis': ukf_nis,
            'nees_consistent_pct': ukf_nees_consistent,
            'nis_consistent_pct': ukf_nis_consistent,
            'whiteness': ukf_whiteness
        },
        'bounds': {
            'nees_lower': nees_lower,
            'nees_upper': nees_upper,
            'nis_lower': nis_lower,
            'nis_upper': nis_upper
        }
    }
