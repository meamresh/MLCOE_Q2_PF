"""
Linear algebra utilities for state-space models and filtering.

This module provides shared linear algebra operations used across
filters and experiments, eliminating code duplication.

"""

from __future__ import annotations

import tensorflow as tf


def regularize_covariance(P: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """
    Regularize a covariance matrix for numerical stability.
    
    Ensures the matrix is symmetric and positive semi-definite by:
    1. Symmetrizing: P <- 0.5 * (P + P^T)
    2. Adding small regularization: P <- P + eps * I
    
    Parameters
    ----------
    P : tf.Tensor
        Covariance matrix of shape (n, n).
    eps : float, optional
        Small positive value for regularization. Default 1e-6.
        
    Returns
    -------
    tf.Tensor
        Regularized covariance matrix of shape (n, n).
        
    Examples
    --------
    >>> P = tf.constant([[1.0, 0.1], [0.1001, 1.0]])  # Slightly asymmetric
    >>> P_reg = regularize_covariance(P)
    >>> # P_reg is now symmetric with small diagonal boost
    """
    P = tf.convert_to_tensor(P, dtype=tf.float32)
    n = tf.shape(P)[0]
    P = 0.5 * (P + tf.transpose(P))
    P = P + eps * tf.eye(n, dtype=P.dtype)
    return P


def sample_from_gaussian(
    mean: tf.Tensor, 
    cov: tf.Tensor, 
    n_samples: int
) -> tf.Tensor:
    """
    Sample from a multivariate Gaussian distribution.
    
    Uses Cholesky decomposition for efficient sampling. Falls back to
    eigenvalue decomposition if Cholesky fails (non-positive-definite matrix).
    
    Parameters
    ----------
    mean : tf.Tensor
        Mean vector of shape (d,).
    cov : tf.Tensor
        Covariance matrix of shape (d, d).
    n_samples : int
        Number of samples to draw.
        
    Returns
    -------
    tf.Tensor
        Samples of shape (n_samples, d).
        
    Examples
    --------
    >>> mean = tf.constant([0.0, 0.0])
    >>> cov = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    >>> samples = sample_from_gaussian(mean, cov, 100)
    >>> samples.shape
    TensorShape([100, 2])
    """
    mean = tf.convert_to_tensor(mean, dtype=tf.float32)
    cov = tf.convert_to_tensor(cov, dtype=tf.float32)
    d = tf.shape(mean)[0]
    
    # Ensure mean is 1D
    mean = tf.reshape(mean, [-1])
    
    try:
        L = tf.linalg.cholesky(cov)
    except Exception:
        # Fallback: use eigenvalue decomposition for non-PD matrices
        eigenvalues, eigenvectors = tf.linalg.eigh(cov)
        eigenvalues = tf.maximum(eigenvalues, 1e-6)
        L = eigenvectors @ tf.linalg.diag(tf.sqrt(eigenvalues))
    
    noise = tf.random.normal([n_samples, d], dtype=tf.float32)
    samples = mean + noise @ tf.transpose(L)
    return samples


def compute_condition_number(matrix: tf.Tensor) -> float:
    """
    Compute the condition number of a matrix using SVD.
    
    The condition number is defined as the ratio of the largest to
    smallest singular value: κ(A) = σ_max / σ_min.
    
    Parameters
    ----------
    matrix : tf.Tensor
        Input matrix of shape (m, n) or (n, n).
        If 3D tensor is provided, uses the first element.
        
    Returns
    -------
    float
        Condition number. Returns NaN if computation fails.
        
    Notes
    -----
    Interpretation:
        - κ < 10³: Excellent conditioning
        - κ ∈ [10³, 10⁶]: Acceptable
        - κ ∈ [10⁶, 10¹⁰]: Caution—monitor for numerical issues
        - κ > 10¹⁰: Dangerous—near-singular
    """
    import math
    
    try:
        matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
        
        # Handle 3D tensors (e.g., batched covariances)
        if len(matrix.shape) > 2:
            matrix = matrix[0]
            
        s = tf.linalg.svd(matrix, compute_uv=False)
        s_max = tf.reduce_max(s)
        s_min = tf.reduce_min(s)
        cond = s_max / (s_min + 1e-10)
        
        result = float(cond.numpy()) if hasattr(cond, 'numpy') else float(cond)
        return result if math.isfinite(result) else float('nan')
    except Exception:
        return float('nan')


def localization_matrix(
    d: int, 
    r_in: float, 
    periodic: bool = False
) -> tf.Tensor:
    """
    Compute Gaussian localization matrix for covariance localization.
    
    The localization matrix has entries:
        C_ij = exp(-d_ij² / r_in²)
    
    where d_ij is the distance between indices i and j.
    
    Parameters
    ----------
    d : int
        Dimension of the state space.
    r_in : float
        Localization radius (influence length scale).
    periodic : bool, optional
        If True, use periodic (circular) distance for Lorenz-96 type systems.
        Default False.
        
    Returns
    -------
    tf.Tensor
        Localization matrix of shape (d, d).
        
    Examples
    --------
    >>> C = localization_matrix(10, 3.0, periodic=True)
    >>> C.shape
    TensorShape([10, 10])
    
    References
    ----------
    Hu, C. C., & van Leeuwen, P. J. (2021). A particle flow filter for
    high-dimensional system applications. Q. J. R. Meteorol. Soc.
    """
    indices = tf.cast(tf.range(d), tf.float32)
    i_grid, j_grid = tf.meshgrid(indices, indices, indexing='ij')
    
    if periodic:
        # Periodic (circular) distance for systems like Lorenz-96
        diff = tf.abs(i_grid - j_grid)
        d_float = tf.cast(d, tf.float32)
        dist = tf.minimum(diff, d_float - diff)
    else:
        # Standard Euclidean distance
        dist = tf.abs(i_grid - j_grid)
    
    C = tf.exp(-(dist ** 2) / (r_in ** 2))
    return C


def nearest_psd(matrix: tf.Tensor, epsilon: float = 1e-10) -> tf.Tensor:
    """
    Project a symmetric matrix to the nearest positive semi-definite matrix.
    
    Uses eigenvalue thresholding: negative eigenvalues are set to epsilon.
    
    Parameters
    ----------
    matrix : tf.Tensor
        Symmetric matrix of shape (n, n).
    epsilon : float, optional
        Minimum eigenvalue floor. Default 1e-10.
        
    Returns
    -------
    tf.Tensor
        Nearest PSD matrix of shape (n, n).
        
    References
    ----------
    Higham, N. J. (1988). Computing a nearest symmetric positive semidefinite
    matrix. Linear Algebra and its Applications, 103, 103-118.
    """
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
    
    # Ensure symmetry first
    A = 0.5 * (matrix + tf.transpose(matrix))
    
    # Eigenvalue decomposition
    eigvals, eigvecs = tf.linalg.eigh(A)
    
    # Threshold negative eigenvalues
    eigvals = tf.maximum(eigvals, epsilon)
    
    # Reconstruct
    return tf.matmul(eigvecs, tf.matmul(tf.linalg.diag(eigvals), eigvecs, transpose_b=True))


def ensure_symmetric(matrix: tf.Tensor) -> tf.Tensor:
    """
    Enforce symmetry by averaging matrix with its transpose.
    
    Parameters
    ----------
    matrix : tf.Tensor
        Matrix of shape (n, n).
        
    Returns
    -------
    tf.Tensor
        Symmetric matrix of shape (n, n).
    """
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
    return 0.5 * (matrix + tf.transpose(matrix))


def safe_cholesky(matrix: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """
    Compute Cholesky decomposition with fallback for ill-conditioned matrices.
    
    Parameters
    ----------
    matrix : tf.Tensor
        Positive semi-definite matrix of shape (n, n).
    eps : float, optional
        Regularization to add if initial Cholesky fails.
        
    Returns
    -------
    tf.Tensor
        Lower triangular Cholesky factor L such that matrix = L @ L^T.
    """
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
    n = tf.shape(matrix)[0]
    
    try:
        return tf.linalg.cholesky(matrix)
    except Exception:
        # Add regularization and retry
        matrix_reg = matrix + eps * tf.eye(n, dtype=matrix.dtype)
        try:
            return tf.linalg.cholesky(matrix_reg)
        except Exception:
            # Last resort: use eigenvalue decomposition
            eigvals, eigvecs = tf.linalg.eigh(matrix)
            eigvals = tf.maximum(eigvals, eps)
            return eigvecs @ tf.linalg.diag(tf.sqrt(eigvals))


def compute_particle_covariance(particles: tf.Tensor, weights: tf.Tensor = None) -> tf.Tensor:
    """
    Compute covariance matrix from weighted particles.
    
    Parameters
    ----------
    particles : tf.Tensor
        Particle states of shape (N, d).
    weights : tf.Tensor, optional
        Particle weights of shape (N,). If None, uses uniform weights.
        
    Returns
    -------
    tf.Tensor
        Covariance matrix of shape (d, d).
    """
    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    N = tf.shape(particles)[0]
    
    if weights is None:
        # Uniform weights
        mean = tf.reduce_mean(particles, axis=0)
        centered = particles - mean[tf.newaxis, :]
        cov = tf.reduce_mean(
            centered[:, :, tf.newaxis] * centered[:, tf.newaxis, :],
            axis=0
        )
    else:
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        weights = weights / tf.reduce_sum(weights)
        mean = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
        centered = particles - mean[tf.newaxis, :]
        cov = tf.reduce_sum(
            weights[:, tf.newaxis, tf.newaxis] * 
            centered[:, :, tf.newaxis] * centered[:, tf.newaxis, :],
            axis=0
        )
    
    return regularize_covariance(cov)
