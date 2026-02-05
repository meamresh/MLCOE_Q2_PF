"""
Filtering algorithms for state-space models.

This package provides implementations of various filtering algorithms:
- Kalman Filter (KF) - for linear Gaussian systems
- Extended Kalman Filter (EKF) - linearization-based
- Unscented Kalman Filter (UKF) - sigma-point based
- Particle Filter (PF) - sequential Monte Carlo
- EDH/LEDH - Exact/Local Daum-Huang flow filters
- PFPF - Particle Flow Particle Filters
- PFF - Particle Flow Filters with kernel methods

Also provides:
- Base classes: BaseFilter, BaseParticleFilter
- Resampling utilities: systematic_resample(), etc.
"""

from src.filters.base import BaseFilter, BaseParticleFilter, StateSpaceModel
from src.filters.kalman import KalmanFilter
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.filters.particle_filter import ParticleFilter
from src.filters.pfpf_filter import PFPFLEDHFilter, PFPFEDHFilter
from src.filters.ledh import LEDH
from src.filters.edh import EDH
from src.filters.pff_kernel import ScalarPFF, MatrixPFF
from src.filters.resampling import (
    systematic_resample,
    multinomial_resample,
    stratified_resample,
    residual_resample,
    resample_particles,
    compute_ess,
    should_resample,
)

__all__ = [
    # Base classes
    'BaseFilter',
    'BaseParticleFilter',
    'StateSpaceModel',
    # Filters
    'KalmanFilter',
    'ExtendedKalmanFilter',
    'UnscentedKalmanFilter',
    'ParticleFilter',
    'PFPFLEDHFilter',
    'PFPFEDHFilter',
    'LEDH',
    'EDH',
    'ScalarPFF',
    'MatrixPFF',
    # Resampling
    'systematic_resample',
    'multinomial_resample',
    'stratified_resample',
    'residual_resample',
    'resample_particles',
    'compute_ess',
    'should_resample',
]

