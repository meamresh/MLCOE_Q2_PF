"""
Example: Range-Bearing Localization with EKF and UKF.

This example demonstrates basic usage of the range-bearing localization
filters. For full experiments with parameter tuning, see
src/experiments/exp_range_bearing_ekf_ukf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ssm_range_bearing import RangeBearingSSM
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter


def simulate_robot_trajectory(ssm: RangeBearingSSM, num_steps: int,
                              seed: int | None = None) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Simulate robot trajectory with control inputs.

    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    num_steps : int
        Number of time steps.
    seed : int, optional
        Random seed.

    Returns
    -------
    true_states : tf.Tensor
        True states of shape (num_steps, 3).
    controls : tf.Tensor
        Control inputs of shape (num_steps, 2).
    """
    if seed is not None:
        tf.random.set_seed(seed)

    true_states = []
    controls = []

    # Initial state
    state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    true_states.append(state)

    # Generate control sequence (circular motion)
    for t in range(num_steps - 1):
        v = 1.0  # Constant velocity
        omega = 0.1  # Constant angular velocity
        control = tf.constant([v, omega], dtype=tf.float32)
        controls.append(control)

        # Predict next state
        state = ssm.motion_model(state[tf.newaxis, :], control[tf.newaxis, :])[0]

        # Add process noise
        process_noise = tf.random.normal([3], dtype=tf.float32)
        process_noise = tf.linalg.cholesky(ssm.Q) @ process_noise
        state = state + process_noise

        true_states.append(state)

    return tf.stack(true_states, axis=0), tf.stack(controls, axis=0)


def generate_measurements(ssm: RangeBearingSSM, true_states: tf.Tensor,
                         landmarks: tf.Tensor) -> tf.Tensor:
    """
    Generate noisy range-bearing measurements.

    Parameters
    ----------
    ssm : RangeBearingSSM
        State-space model.
    true_states : tf.Tensor
        True states of shape (num_steps, 3).
    landmarks : tf.Tensor
        Landmark positions of shape (num_landmarks, 2).

    Returns
    -------
    measurements : tf.Tensor
        Measurements of shape (num_steps, num_landmarks, 2).
    """
    measurements = []

    for state in true_states:
        meas = ssm.measurement_model(state[tf.newaxis, :], landmarks)[0]

        # Add measurement noise
        meas_noise = tf.random.normal([tf.shape(landmarks)[0], 2],
                                     dtype=tf.float32)
        meas_noise = meas_noise @ tf.linalg.cholesky(ssm.R)
        meas = meas + meas_noise

        measurements.append(meas)

    return tf.stack(measurements, axis=0)


def main():
    """Run simple range-bearing localization example."""
    # Setup
    dt = 0.1
    num_steps = 100
    num_landmarks = 3

    # Create landmarks
    landmarks = tf.constant([
        [5.0, 5.0],
        [-5.0, 5.0],
        [0.0, -5.0]
    ], dtype=tf.float32)

    # Create state-space model
    ssm = RangeBearingSSM(dt=dt)

    # Simulate trajectory
    print("Simulating robot trajectory...")
    true_states, controls = simulate_robot_trajectory(ssm, num_steps, seed=42)
    measurements = generate_measurements(ssm, true_states, landmarks)

    # Initialize filters
    initial_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    initial_cov = tf.eye(3, dtype=tf.float32) * 0.1

    ekf = ExtendedKalmanFilter(ssm, initial_state, initial_cov)
    ukf = UnscentedKalmanFilter(ssm, initial_state, initial_cov)

    # Run filters
    print("Running EKF and UKF...")
    ekf_states = []
    ukf_states = []

    for t in range(num_steps):
        # Predict
        ekf.predict(controls[t])
        ukf.predict(controls[t])

        # Update
        if t < num_steps:
            ekf.update(measurements[t], landmarks)
            ukf.update(measurements[t], landmarks)

        ekf_states.append(ekf.state.numpy())
        ukf_states.append(ukf.state.numpy())

    ekf_states = tf.stack(ekf_states, axis=0)
    ukf_states = tf.stack(ukf_states, axis=0)

    # Compute RMSE
    ekf_rmse = tf.sqrt(tf.reduce_mean((ekf_states[:, :2] -
                                      true_states[:, :2]) ** 2))
    ukf_rmse = tf.sqrt(tf.reduce_mean((ukf_states[:, :2] -
                                       true_states[:, :2]) ** 2))

    print(f"\nEKF Position RMSE: {float(ekf_rmse):.4f}")
    print(f"UKF Position RMSE: {float(ukf_rmse):.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))

    # Trajectory plot
    plt.subplot(1, 2, 1)
    plt.plot(true_states[:, 0].numpy(), true_states[:, 1].numpy(),
            'k-', label='True', linewidth=2)
    plt.plot(ekf_states[:, 0].numpy(), ekf_states[:, 1].numpy(),
            'b--', label='EKF', alpha=0.7)
    plt.plot(ukf_states[:, 0].numpy(), ukf_states[:, 1].numpy(),
            'r--', label='UKF', alpha=0.7)
    plt.scatter(landmarks[:, 0].numpy(), landmarks[:, 1].numpy(),
               c='red', s=100, marker='*', label='Landmarks', zorder=5)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Robot Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Error plot
    plt.subplot(1, 2, 2)
    time = tf.range(num_steps, dtype=tf.float32) * dt
    ekf_error = tf.norm(ekf_states[:, :2] - true_states[:, :2], axis=1)
    ukf_error = tf.norm(ukf_states[:, :2] - true_states[:, :2], axis=1)

    plt.plot(time.numpy(), ekf_error.numpy(), 'b-', label='EKF', alpha=0.7)
    plt.plot(time.numpy(), ukf_error.numpy(), 'r-', label='UKF', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error')
    plt.title('Filtering Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('range_bearing_localization.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to 'range_bearing_localization.png'")
    plt.close()


if __name__ == '__main__':
    main()
