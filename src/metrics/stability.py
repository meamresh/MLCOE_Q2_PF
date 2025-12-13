"""
Numerical stability metrics for covariance matrices and filtering algorithms.

This module provides diagnostics for assessing numerical stability.

"""

from __future__ import annotations

import tensorflow as tf


def compute_condition_numbers(covariances: tf.Tensor) -> tf.Tensor:
    """
    Compute condition number for each covariance matrix in a sequence.

    The condition number κ(P) = λ_max(P) / λ_min(P) measures the ratio of
    largest to smallest eigenvalues, indicating how ill-conditioned the matrix
    is for numerical operations (especially inversion).

    Parameters
    ----------
    covariances : tf.Tensor
        Covariance matrices, shape (N, n, n).

    Returns
    -------
    cond_numbers : tf.Tensor
        Condition numbers, shape (N,).

    Notes
    -----
    Interpretation:
        - κ < 10³: Excellent conditioning
        - κ ∈ [10³, 10⁶]: Acceptable
        - κ ∈ [10⁶, 10¹⁰]: Caution—monitor for numerical issues
        - κ > 10¹⁰: Dangerous—near-singular, expect numerical errors

    Large condition numbers indicate that small perturbations (rounding errors)
    can lead to large errors in computed quantities.

    References
    ----------
    Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
    Johns Hopkins University Press.
    """
    covariances = tf.convert_to_tensor(covariances)
    N = tf.shape(covariances)[0]
    cond_numbers_list = []

    for t in range(N):
        eigvals = tf.linalg.eigvalsh(covariances[t])
        cond_numbers_list.append(tf.reduce_max(eigvals) / tf.reduce_min(eigvals))

    return tf.stack(cond_numbers_list, axis=0)


def check_symmetry(matrices: tf.Tensor) -> tf.Tensor:
    """
    Check symmetry of matrices via max|P - P^T|.

    Covariance matrices should be symmetric by definition. This metric
    quantifies symmetry violations due to finite-precision arithmetic.

    Parameters
    ----------
    matrices : tf.Tensor
        Matrices to check, shape (N, n, n).

    Returns
    -------
    symmetry_error : tf.Tensor
        Maximum absolute element-wise error |P - P^T|, shape (N,).

    Notes
    -----
    For IEEE 754 double precision (float64), expect:
        - Standard Riccati update: ~10⁻¹³ to 10⁻¹²
        - Joseph-stabilized update: ~10⁻¹⁵ (near machine epsilon)

    If symmetry error exceeds 10⁻¹⁰, consider enforcing symmetry:
        P = 0.5 * (P + P.T)
    """
    matrices = tf.convert_to_tensor(matrices)
    N = tf.shape(matrices)[0]
    symmetry_errors = []
    for t in range(N):
        P = matrices[t]
        symmetry_errors.append(tf.reduce_max(tf.abs(P - tf.transpose(P))))
    return tf.stack(symmetry_errors, axis=0)


def check_positive_definite(matrices: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Check positive definiteness via eigenvalue analysis.

    A covariance matrix must be positive semi-definite (all eigenvalues ≥ 0).
    Negative eigenvalues indicate numerical errors or algorithmic issues.

    Parameters
    ----------
    matrices : tf.Tensor
        Covariance matrices, shape (N, n, n).

    Returns
    -------
    min_eigvals : tf.Tensor
        Minimum eigenvalue at each step, shape (N,).
    is_pd : tf.Tensor
        Boolean array indicating positive definiteness (λ_min > 0), shape (N,).

    Notes
    -----
    Joseph-stabilized covariance updates guarantee P ≽ 0 even under finite
    precision, while standard Riccati updates may occasionally produce
    slightly negative eigenvalues (~-10⁻¹⁴) due to rounding.

    If λ_min < -10⁻¹⁰, this signals a serious numerical issue requiring
    algorithmic correction (e.g., switching to Joseph form or adding
    regularization).
    """
    matrices = tf.convert_to_tensor(matrices)
    N = tf.shape(matrices)[0]
    min_eigvals_list = []
    is_pd_list = []

    for t in range(N):
        eigvals = tf.linalg.eigvalsh(matrices[t])
        min_eigval = tf.reduce_min(eigvals)
        min_eigvals_list.append(min_eigval)
        is_pd_list.append(min_eigval > 0)

    return tf.stack(min_eigvals_list, axis=0), tf.stack(is_pd_list, axis=0)


def compute_frobenius_norm_difference(matrices_a: tf.Tensor,
                                       matrices_b: tf.Tensor) -> tf.Tensor:
    """
    Compute Frobenius norm of matrix differences.

    Useful for comparing two covariance sequences (e.g., Riccati vs Joseph).

    Parameters
    ----------
    matrices_a : tf.Tensor
        First sequence of matrices, shape (N, n, n).
    matrices_b : tf.Tensor
        Second sequence of matrices, shape (N, n, n).

    Returns
    -------
    norm_diff : tf.Tensor
        ||A - B||_F at each time step, shape (N,).

    Notes
    -----
    The Frobenius norm is defined as:
        ||A||_F = sqrt(sum(A_ij^2))

    For numerically equivalent algorithms, expect ||P_a - P_b||_F < 10⁻¹⁰.
    """
    matrices_a = tf.convert_to_tensor(matrices_a)
    matrices_b = tf.convert_to_tensor(matrices_b)
    N = tf.shape(matrices_a)[0]
    norm_diff_list = []

    for t in range(N):
        # Frobenius norm: sqrt(sum of squares of all elements)
        # For matrices, tf.linalg.norm without ord parameter computes Frobenius norm
        diff = matrices_a[t] - matrices_b[t]
        norm_diff_list.append(tf.linalg.norm(diff))

    return tf.stack(norm_diff_list, axis=0)


def compute_trace(matrices: tf.Tensor) -> tf.Tensor:
    """
    Compute trace of each matrix (sum of diagonal elements).

    For covariance matrices, tr(P) represents total variance across all states.
    Useful for monitoring uncertainty evolution over time.

    Parameters
    ----------
    matrices : tf.Tensor
        Matrices, shape (N, n, n).

    Returns
    -------
    traces : tf.Tensor
        tr(P_t) at each time step, shape (N,).
    """
    matrices = tf.convert_to_tensor(matrices)
    N = tf.shape(matrices)[0]
    traces = []
    for t in range(N):
        traces.append(tf.linalg.trace(matrices[t]))
    return tf.stack(traces, axis=0)


def compute_log_determinant(matrices: tf.Tensor) -> tf.Tensor:
    """
    Compute log-determinant of each matrix.

    log|P| measures the volume of the uncertainty ellipsoid. Useful for
    detecting covariance collapse (when log|P| → -∞) or divergence.

    Parameters
    ----------
    matrices : tf.Tensor
        Positive definite matrices, shape (N, n, n).

    Returns
    -------
    log_dets : tf.Tensor
        log|P_t| at each time step, shape (N,).

    Notes
    -----
    Uses numerically stable slogdet instead of log(det(P)).
    Returns -inf if matrix is singular or not positive definite.
    """
    matrices = tf.convert_to_tensor(matrices)
    N = tf.shape(matrices)[0]
    log_dets_list = []

    for t in range(N):
        sign, logdet = tf.linalg.slogdet(matrices[t])
        log_dets_list.append(tf.where(sign > 0, logdet, tf.constant(-float('inf'), dtype=logdet.dtype)))

    return tf.stack(log_dets_list, axis=0)


def enforce_symmetry(matrix: tf.Tensor) -> tf.Tensor:
    """
    Enforce symmetry by averaging P and P^T.

    Parameters
    ----------
    matrix : tf.Tensor
        Matrix to symmetrize, shape (n, n).

    Returns
    -------
    symmetric_matrix : tf.Tensor
        Symmetrized matrix, shape (n, n).

    Notes
    -----
    Use this if symmetry violations exceed acceptable tolerance (~10⁻¹⁰).
    """
    matrix = tf.convert_to_tensor(matrix)
    return 0.5 * (matrix + tf.transpose(matrix))


def nearest_psd(matrix: tf.Tensor, epsilon: float = 1e-10) -> tf.Tensor:
    """
    Project a symmetric matrix to the nearest positive semi-definite matrix.

    Uses eigenvalue thresholding: negative eigenvalues are set to epsilon.

    Parameters
    ----------
    matrix : tf.Tensor
        Symmetric matrix, shape (n, n).
    epsilon : float, optional
        Minimum eigenvalue floor. Default 1e-10.

    Returns
    -------
    psd_matrix : tf.Tensor
        Nearest PSD matrix, shape (n, n).

    References
    ----------
    Higham, N. J. (1988). Computing a nearest symmetric positive semidefinite
    matrix. Linear Algebra and its Applications, 103, 103-118.
    """
    matrix = tf.convert_to_tensor(matrix)
    # Ensure symmetry first
    A = 0.5 * (matrix + tf.transpose(matrix))

    # Eigenvalue decomposition
    eigvals, eigvecs = tf.linalg.eigh(A)

    # Threshold negative eigenvalues
    eigvals = tf.maximum(eigvals, epsilon)

    # Reconstruct
    return tf.matmul(eigvecs, tf.matmul(tf.linalg.diag(eigvals), eigvecs, transpose_b=True))