"""
Utility modules for shared functionality across the codebase.

This package provides common utilities for:
- Linear algebra operations (linalg.py)
- Experiment runners (experiment.py)
- Logging configuration (logging_config.py)
"""

from __future__ import annotations

from src.utils.linalg import (
    regularize_covariance,
    sample_from_gaussian,
    compute_condition_number,
    localization_matrix,
    nearest_psd,
)

from src.utils.logging_config import (
    get_logger,
    setup_logging,
    set_level,
    LoggerAdapter,
)

__all__ = [
    # Linear algebra
    "regularize_covariance",
    "sample_from_gaussian",
    "compute_condition_number",
    "localization_matrix",
    "nearest_psd",
    # Logging
    "get_logger",
    "setup_logging",
    "set_level",
    "LoggerAdapter",
]
