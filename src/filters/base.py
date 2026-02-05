"""
Base classes for state-space filters.

This module defines abstract base classes that establish a common interface
for all filter implementations, ensuring consistency and reducing code duplication.

Classes:
    StateSpaceModel: Protocol for state-space models (motion + measurement)
    BaseFilter: Abstract base class for all filters
    BaseParticleFilter: Abstract base class for particle-based filters
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import tensorflow as tf


@runtime_checkable
class StateSpaceModel(Protocol):
    """
    Protocol defining the interface for state-space models.
    
    All state-space models must implement:
    - motion_model: State transition function
    - measurement_model: Observation function
    - Q: Process noise covariance
    - R: Measurement noise covariance
    
    Optional:
    - motion_jacobian: Jacobian of motion model (for EKF)
    - measurement_jacobian: Jacobian of measurement model (for EKF)
    """
    
    Q: tf.Tensor  # Process noise covariance
    R: tf.Tensor  # Measurement noise covariance
    
    def motion_model(
        self,
        state: tf.Tensor,
        control: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Propagate state through motion model.
        
        Parameters
        ----------
        state : tf.Tensor
            Current state (state_dim,) or (batch, state_dim).
        control : tf.Tensor, optional
            Control input.
            
        Returns
        -------
        next_state : tf.Tensor
            Predicted next state.
        jacobian : tf.Tensor
            Jacobian of motion model (for linearization).
        """
        ...
    
    def measurement_model(
        self,
        state: tf.Tensor,
        *args: Any
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute expected measurement from state.
        
        Parameters
        ----------
        state : tf.Tensor
            Current state.
        *args : Any
            Additional arguments (e.g., landmarks).
            
        Returns
        -------
        measurement : tf.Tensor
            Expected measurement.
        jacobian : tf.Tensor
            Jacobian of measurement model.
        """
        ...


class BaseFilter(ABC):
    """
    Abstract base class for all state estimation filters.
    
    Provides common interface and shared functionality for:
    - Kalman-type filters (KF, EKF, UKF)
    - Particle filters (PF, PFPF, EDH, LEDH)
    
    Attributes
    ----------
    ssm : StateSpaceModel
        State-space model defining dynamics and observations.
    state : tf.Tensor
        Current state estimate (state_dim,).
    covariance : tf.Tensor
        Current state uncertainty (state_dim, state_dim).
    state_dim : int
        Dimension of state vector.
        
    Properties
    ----------
    x_hat : tf.Tensor
        Alias for state estimate.
    P : tf.Tensor
        Alias for covariance.
    """
    
    def __init__(
        self,
        ssm: StateSpaceModel,
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor
    ):
        """
        Initialize filter with state-space model and initial conditions.
        
        Parameters
        ----------
        ssm : StateSpaceModel
            State-space model instance.
        initial_state : tf.Tensor
            Initial state estimate (state_dim,).
        initial_covariance : tf.Tensor
            Initial covariance matrix (state_dim, state_dim).
        """
        self.ssm = ssm
        self._state = tf.Variable(
            tf.cast(initial_state, tf.float32),
            trainable=False,
            name="state"
        )
        self._covariance = tf.Variable(
            tf.cast(initial_covariance, tf.float32),
            trainable=False,
            name="covariance"
        )
        self.state_dim = int(initial_state.shape[0])
    
    @property
    def state(self) -> tf.Tensor:
        """Current state estimate."""
        return self._state.value()
    
    @state.setter
    def state(self, value: tf.Tensor) -> None:
        self._state.assign(tf.cast(value, tf.float32))
    
    @property
    def x_hat(self) -> tf.Tensor:
        """Alias for state estimate."""
        return self.state
    
    @property
    def covariance(self) -> tf.Tensor:
        """Current state covariance."""
        return self._covariance.value()
    
    @covariance.setter
    def covariance(self, value: tf.Tensor) -> None:
        self._covariance.assign(tf.cast(value, tf.float32))
    
    @property
    def P(self) -> tf.Tensor:
        """Alias for covariance."""
        return self.covariance
    
    @abstractmethod
    def predict(
        self,
        control: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform prediction step.
        
        Parameters
        ----------
        control : tf.Tensor, optional
            Control input.
            
        Returns
        -------
        state_pred : tf.Tensor
            Predicted state.
        cov_pred : tf.Tensor
            Predicted covariance.
        """
        pass
    
    @abstractmethod
    def update(
        self,
        measurement: tf.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Perform measurement update step.
        
        Parameters
        ----------
        measurement : tf.Tensor
            Measurement vector.
        *args, **kwargs : Any
            Additional arguments (e.g., landmarks).
            
        Returns
        -------
        state_upd : tf.Tensor
            Updated state estimate.
        cov_upd : tf.Tensor
            Updated covariance.
        residual : tf.Tensor
            Innovation/residual.
        """
        pass
    
    def reset(
        self,
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor
    ) -> None:
        """
        Reset filter to new initial conditions.
        
        Parameters
        ----------
        initial_state : tf.Tensor
            New initial state.
        initial_covariance : tf.Tensor
            New initial covariance.
        """
        self.state = initial_state
        self.covariance = initial_covariance


class BaseParticleFilter(BaseFilter):
    """
    Abstract base class for particle-based filters.
    
    Extends BaseFilter with particle-specific attributes and methods.
    
    Additional Attributes
    ---------------------
    num_particles : int
        Number of particles.
    particles : tf.Tensor
        Particle states (num_particles, state_dim).
    weights : tf.Tensor
        Particle weights (num_particles,).
    resample_threshold : float
        ESS threshold for resampling (0-1).
    """
    
    def __init__(
        self,
        ssm: StateSpaceModel,
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor,
        num_particles: int = 1000,
        resample_threshold: float = 0.5
    ):
        """
        Initialize particle filter.
        
        Parameters
        ----------
        ssm : StateSpaceModel
            State-space model instance.
        initial_state : tf.Tensor
            Initial state estimate.
        initial_covariance : tf.Tensor
            Initial covariance.
        num_particles : int
            Number of particles.
        resample_threshold : float
            ESS threshold (relative to num_particles).
        """
        super().__init__(ssm, initial_state, initial_covariance)
        
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        
        # Initialize particles from prior
        self._particles = tf.Variable(
            self._sample_initial_particles(initial_state, initial_covariance),
            trainable=False,
            name="particles"
        )
        
        # Initialize uniform weights
        self._weights = tf.Variable(
            tf.ones(num_particles, dtype=tf.float32) / num_particles,
            trainable=False,
            name="weights"
        )
    
    def _sample_initial_particles(
        self,
        mean: tf.Tensor,
        covariance: tf.Tensor
    ) -> tf.Tensor:
        """Sample initial particles from Gaussian prior."""
        from src.utils.linalg import sample_from_gaussian
        return sample_from_gaussian(mean, covariance, self.num_particles)
    
    @property
    def particles(self) -> tf.Tensor:
        """Current particle states."""
        return self._particles.value()
    
    @particles.setter
    def particles(self, value: tf.Tensor) -> None:
        self._particles.assign(tf.cast(value, tf.float32))
    
    @property
    def weights(self) -> tf.Tensor:
        """Current particle weights."""
        return self._weights.value()
    
    @weights.setter
    def weights(self, value: tf.Tensor) -> None:
        self._weights.assign(tf.cast(value, tf.float32))
    
    def compute_ess(self) -> tf.Tensor:
        """
        Compute Effective Sample Size.
        
        ESS = 1 / Σ(w_i²)
        
        Returns
        -------
        tf.Tensor
            Effective sample size (scalar).
        """
        return 1.0 / tf.reduce_sum(tf.square(self.weights))
    
    def should_resample(self) -> bool:
        """
        Check if resampling is needed based on ESS.
        
        Returns
        -------
        bool
            True if ESS < threshold * num_particles.
        """
        ess = self.compute_ess()
        threshold = self.resample_threshold * self.num_particles
        return float(ess) < threshold
    
    @abstractmethod
    def resample(self) -> None:
        """
        Resample particles based on weights.
        
        After resampling, weights should be uniform.
        """
        pass
    
    def reset(
        self,
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor
    ) -> None:
        """Reset filter including particles."""
        super().reset(initial_state, initial_covariance)
        self.particles = self._sample_initial_particles(
            initial_state, initial_covariance
        )
        self.weights = tf.ones(self.num_particles, dtype=tf.float32) / self.num_particles
