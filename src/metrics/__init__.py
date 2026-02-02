"""
Metrics for evaluating state-space model filtering algorithms.

This package provides metrics for:
- Accuracy: RMSE, MAE, NEES, NIS, NLL
- Particle filter diagnostics: ESS, weight entropy
- Stability: condition numbers, symmetry checks, positive definiteness
"""

from __future__ import annotations

from src.metrics.accuracy import (
    compute_rmse,
    compute_mae,
    compute_nees,
    compute_nll,
    compute_per_dimension_rmse,
    compute_nis,
    compute_autocorrelation,
    test_innovation_whiteness,
    analyze_filter_consistency,
)

from src.metrics.particle_filter_metrics import (
    compute_effective_sample_size,
    compute_weight_entropy,
    compute_weight_variance,
)

from src.metrics.stability import (
    compute_condition_numbers,
    check_symmetry,
    check_positive_definite,
    compute_frobenius_norm_difference,
    compute_trace,
    compute_log_determinant,
    enforce_symmetry,
    nearest_psd,
)

__all__ = [
    # Accuracy metrics
    'compute_rmse',
    'compute_mae',
    'compute_nees',
    'compute_nll',
    'compute_per_dimension_rmse',
    'compute_nis',
    'compute_autocorrelation',
    'test_innovation_whiteness',
    'analyze_filter_consistency',
    # Particle filter metrics
    'compute_effective_sample_size',
    'compute_weight_entropy',
    'compute_weight_variance',
    # Stability metrics
    'compute_condition_numbers',
    'check_symmetry',
    'check_positive_definite',
    'compute_frobenius_norm_difference',
    'compute_trace',
    'compute_log_determinant',
    'enforce_symmetry',
    'nearest_psd',
]
