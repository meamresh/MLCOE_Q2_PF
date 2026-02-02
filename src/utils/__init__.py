"""
Utility modules for shared functionality across the codebase.

This package provides common utilities for:
- Linear algebra operations (linalg.py)
- Experiment runners (experiment.py)
"""

from __future__ import annotations

from src.utils.linalg import (
    regularize_covariance,
    sample_from_gaussian,
    compute_condition_number,
    localization_matrix,
    nearest_psd,
)

__all__ = [
    "regularize_covariance",
    "sample_from_gaussian",
    "compute_condition_number",
    "localization_matrix",
    "nearest_psd",
]
