# Range-Bearing Localization Filters

This document describes the nonlinear filtering implementations for range-bearing sensor localization.

## Overview

The range-bearing localization system implements robot localization using range and bearing measurements to known landmarks. Three filtering approaches are provided:

1. **Extended Kalman Filter (EKF)** - Linearizes nonlinear models using Jacobians
2. **Unscented Kalman Filter (UKF)** - Uses sigma points to propagate statistics through nonlinearities

## Models

### RangeBearingSSM

The `RangeBearingSSM` class implements a nonlinear state-space model for robot localization.

**State**: `[x, y, theta]` - position (x, y) and orientation (theta)

**Control**: `[v, omega]` - velocity and angular velocity

**Measurement**: `[range, bearing]` - distance and angle to each landmark

#### Key Methods

- `motion_model(state, control)` - Unicycle motion model
- `measurement_model(state, landmarks)` - Range-bearing measurements
- `motion_jacobian(state, control)` - Jacobian of motion model
- `measurement_jacobian(state, landmarks)` - Jacobian of measurement model
- `full_measurement_cov(num_landmarks)` - Block-diagonal measurement covariance

## Filters

### ExtendedKalmanFilter

The EKF linearizes the nonlinear motion and measurement models using their Jacobians, then applies standard Kalman filter equations.

**Usage**:
```python
from src.models.ssm_range_bearing import RangeBearingSSM
from src.filters.ekf import ExtendedKalmanFilter

ssm = RangeBearingSSM(dt=0.1)
initial_state = tf.constant([0.0, 0.0, 0.0])
initial_cov = tf.eye(3) * 0.1

ekf = ExtendedKalmanFilter(ssm, initial_state, initial_cov)

# Predict step
ekf.predict(control)

# Update step
ekf.update(measurement, landmarks)
```

### UnscentedKalmanFilter

The UKF uses sigma points and the unscented transform to propagate mean and covariance through nonlinear transformations without linearization.

**Usage**:
```python
from src.filters.ukf import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(ssm, initial_state, initial_cov,
                            alpha=0.1, beta=1.0, kappa=0)

# Predict step
ukf.predict(control)

# Update step
ukf.update(measurement, landmarks)
```

**Parameters**:
- `alpha`: Spread parameter for sigma points (typically 0.001 to 1)
- `beta`: Parameter for incorporating prior knowledge (typically 2 for Gaussian)
- `kappa`: Secondary scaling parameter (typically 0 for state estimation)

## Example

See `examples/range_bearing_example.py` for a complete example demonstrating:
- Robot trajectory simulation
- Measurement generation
- EKF and UKF filtering
- Performance comparison

Run the example:
```bash
python examples/range_bearing_example.py
```

## Implementation Details

### Bearing Wrapping

Both filters handle bearing measurements correctly by wrapping angles to `[-π, π]` using `atan2(sin(θ), cos(θ))`.

### Numerical Stability

- **EKF**: Uses Joseph-stabilized covariance update and regularization
- **UKF**: Uses Cholesky decomposition for matrix square root (with SVD fallback), eigenvalue checking for positive definiteness, and regularization

### Matrix Square Root

The UKF uses Cholesky decomposition for computing matrix square roots, which is more stable than SVD for positive definite matrices. SVD is used as a fallback if Cholesky fails.

## References

- Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation. *Proceedings of the IEEE*, 92(3), 401-422.
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic robotics*. MIT press.

