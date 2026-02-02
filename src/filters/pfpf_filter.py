"""
Particle Flow Particle Filter (PF-PF) for nonlinear state estimation.

Based on "Particle Filtering with Invertible Particle Flow"
by Yunpeng Li and Mark Coates (2017): https://arxiv.org/abs/1607.08799

This module provides:
- PFPFLEDHFilter: Local Exact Daum-Huang (per-particle linearization)
- PFPFEDHFilter: Exact Daum-Huang (global linearization)

"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from src.utils.linalg import regularize_covariance, sample_from_gaussian
from src.filters.resampling import systematic_resample

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

tfd = tfp.distributions


def _to_tensor(x, dtype=tf.float32):
    """Convert numpy array or tensor to tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype) if x.dtype != dtype else x
    return tf.convert_to_tensor(x, dtype=dtype)


def _get_shape_dim(x, dim=0):
    """Get shape dimension as integer (works with tensors, numpy, lists)."""
    if isinstance(x, tf.Tensor):
        shape = x.shape
        if shape[dim] is not None:
            return int(shape[dim])
        return int(tf.shape(x)[dim])
    if hasattr(x, 'shape') and x.shape[dim] is not None:
        return int(x.shape[dim])
    if hasattr(x, '__len__'):
        if dim == 0:
            return len(x)
        x_tensor = tf.convert_to_tensor(x)
        return int(tf.shape(x_tensor)[dim])
    x_tensor = tf.convert_to_tensor(x)
    return int(tf.shape(x_tensor)[dim])


class PFPFLEDHFilter:
    """
    Particle Flow Particle Filter with LEDH (Local Exact Daum-Huang).

    Uses per-particle local linearization for flow computation.
    Reference: Li & Coates (2017), PF with invertible particle flow.
    """

    def __init__(self, ssm, initial_state, initial_cov, num_particles=500,
                 n_lambda=29, filter_type='ekf', ukf_alpha=0.001, ukf_beta=2.0,
                 show_progress=False):
        self.ssm = ssm
        self.num_particles = num_particles
        self.n_lambda = n_lambda
        self.q = 1.2
        self.epsilon_1 = (1.0 - self.q) / (1.0 - self.q ** self.n_lambda)
        self.filter_type = filter_type
        self.ukf_alpha = ukf_alpha
        self.ukf_beta = ukf_beta
        self.show_progress = show_progress
        self.state_dim = _get_shape_dim(initial_state, dim=0)
        initial_state = _to_tensor(initial_state)
        initial_cov = _to_tensor(initial_cov)

        mvn = tfd.MultivariateNormalTriL(
            loc=initial_state,
            scale_tril=tf.linalg.cholesky(
                initial_cov + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6),
        )
        self.particles = tf.Variable(
            mvn.sample(num_particles), trainable=False, dtype=tf.float32)
        self.particle_covs = tf.Variable(
            tf.tile(initial_cov[tf.newaxis, :, :], [num_particles, 1, 1]),
            trainable=False)
        self.particles_prev = None
        self.control_prev = None
        uniform_weight = 1.0 / tf.cast(num_particles, tf.float32)
        self.weights = tf.Variable(
            tf.ones(num_particles, dtype=tf.float32) * uniform_weight,
            trainable=False)
        self.log_weights = tf.Variable(
            tf.math.log(self.weights), trainable=False)
        self.state = tf.Variable(
            tf.reduce_mean(self.particles, axis=0), trainable=False)
        self.ess_before_resample = tf.constant(
            float(num_particles), dtype=tf.float32)

        if filter_type == 'ukf':
            kappa = 3.0 - tf.cast(self.state_dim, tf.float32)
            self.ukf_lambda = tf.constant(
                ukf_alpha**2 * (tf.cast(self.state_dim, tf.float32) + kappa)
                - tf.cast(self.state_dim, tf.float32),
                dtype=tf.float32)
            self.ukf_gamma = tf.sqrt(
                tf.cast(self.state_dim, tf.float32) + self.ukf_lambda)
        else:
            self.ukf_lambda = None
            self.ukf_gamma = None

        Q = self.ssm.Q
        try:
            self.Q_inv = tf.linalg.inv(Q)
            self.log_det_Q = tf.linalg.slogdet(Q)[1]
        except Exception:
            self.Q_inv = tf.linalg.pinv(Q)
            self.log_det_Q = tf.constant(0.0, dtype=tf.float32)
        self.Q_mvn = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.state_dim, dtype=tf.float32),
            scale_tril=tf.linalg.cholesky(
                Q + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6),
        )

    def predict(self, control):
        """Prediction step."""
        control = _to_tensor(control)
        self.particles_prev = tf.Variable(self.particles, trainable=False)
        self.control_prev = tf.Variable(control, trainable=False)
        control_batch = tf.tile(control[tf.newaxis, :], [self.num_particles, 1])
        preds = self.ssm.motion_model(self.particles, control_batch)
        if len(preds.shape) > 2:
            preds = tf.reshape(preds, [self.num_particles, -1])
        noise = self.Q_mvn.sample(self.num_particles)
        self.particles.assign(preds + noise)
        F_batch = self.ssm.motion_jacobian(self.particles, control_batch)
        if len(F_batch.shape) > 3:
            F_batch = tf.reshape(
                F_batch, [self.num_particles, self.state_dim, self.state_dim])
        elif len(F_batch.shape) == 2:
            F_batch = tf.tile(
                F_batch[tf.newaxis, :, :], [self.num_particles, 1, 1])
        F_T = tf.transpose(F_batch, [0, 2, 1])
        Q = self.ssm.Q
        P_new_batch = tf.einsum(
            'bij,bjk,bkl->bil', F_batch, self.particle_covs, F_T) + Q
        self.particle_covs.assign(P_new_batch)
        self.state.assign(
            tf.reduce_sum(self.particles * self.weights[:, tf.newaxis], axis=0))

    def update(self, measurement, landmarks):
        """PF-PF-LEDH update with importance weights."""
        measurement = _to_tensor(measurement)
        landmarks = _to_tensor(landmarks)
        particles_before_flow = tf.Variable(self.particles, trainable=False)
        log_det_jacobians = self._ledh_flow_with_jacobian(
            measurement, landmarks)
        self._check_jacobian_consistency(
            particles_before_flow, log_det_jacobians,
            measurement, landmarks, do_finite_diff=False)
        log_weight_increments = self._compute_weight_increments(
            measurement, landmarks, particles_before_flow, log_det_jacobians)
        self.log_weights.assign_add(log_weight_increments)
        self._normalize_weights()
        self._resample()
        self.state.assign(
            tf.reduce_sum(self.particles * self.weights[:, tf.newaxis], axis=0))

    def _compute_weight_increments(self, measurement, landmarks,
                                   particles_before_flow, log_det_jacobians):
        """Compute weight increments for all particles."""
        failed_mask = ~tf.math.is_finite(log_det_jacobians)
        log_weight_increments = tf.where(
            failed_mask,
            tf.constant(-1e10, dtype=tf.float32),
            tf.zeros(self.num_particles, dtype=tf.float32))
        valid_mask = ~failed_mask & tf.reduce_all(
            tf.abs(self.particles[:, :2]) <= 200.0, axis=1)
        if tf.reduce_any(valid_mask):
            valid_indices = tf.where(valid_mask)[:, 0]
            particles_valid = tf.gather(self.particles, valid_indices)
            particles_prev_valid = tf.gather(self.particles_prev, valid_indices)
            particles_before_valid = tf.gather(
                particles_before_flow, valid_indices)
            log_liks = self._compute_log_likelihood_batch(
                particles_valid, measurement, landmarks)
            log_liks = tf.clip_by_value(log_liks, -100.0, 100.0)
            log_p_plus = self._compute_log_transition_batch(
                particles_valid, particles_prev_valid)
            log_p_minus = self._compute_log_transition_batch(
                particles_before_valid, particles_prev_valid)
            transition_ratios = tf.clip_by_value(
                log_p_plus - log_p_minus, -20.0, 20.0)
            log_det_Js = tf.clip_by_value(
                tf.gather(log_det_jacobians, valid_indices), -20.0, 20.0)
            increments_valid = log_liks + transition_ratios + log_det_Js
            log_weight_increments = tf.tensor_scatter_nd_update(
                log_weight_increments,
                valid_indices[:, tf.newaxis], increments_valid)
        return log_weight_increments

    def _normalize_weights(self):
        """Normalize weights with regularization."""
        finite_mask = tf.math.is_finite(self.log_weights)
        if tf.reduce_any(finite_mask):
            max_log_weight = tf.reduce_max(tf.where(
                finite_mask, self.log_weights,
                tf.constant(-1e10, dtype=tf.float32)))
            self.log_weights.assign(self.log_weights - max_log_weight)
            weights_unnorm = tf.where(
                finite_mask, tf.exp(self.log_weights),
                tf.zeros_like(self.log_weights))
            weight_sum = tf.reduce_sum(weights_unnorm)
            num_p = tf.cast(self.num_particles, tf.float32)
            if weight_sum > 1e-10:
                self.weights.assign(weights_unnorm / weight_sum)
                weight_sq_sum = tf.reduce_sum(self.weights ** 2)
                ess = (tf.reduce_sum(self.weights) ** 2) / (weight_sq_sum + 1e-15)
                alpha = tf.cond(
                    ess < num_p / 10.0, lambda: 0.4,
                    lambda: tf.cond(
                        ess < num_p / 5.0, lambda: 0.3,
                        lambda: tf.cond(
                            ess < num_p / 3.0, lambda: 0.15,
                            lambda: tf.cond(
                                ess < num_p / 2.0, lambda: 0.05, lambda: 0.02))))
                uniform_weight = 1.0 / num_p
                self.weights.assign(
                    (1.0 - alpha) * self.weights + alpha * uniform_weight)
                self.weights.assign(
                    self.weights / tf.reduce_sum(self.weights))
                self.log_weights.assign(tf.math.log(self.weights + 1e-10))
            else:
                uniform_weight = 1.0 / num_p
                self.weights.assign(tf.ones(self.num_particles) * uniform_weight)
                self.log_weights.assign(tf.math.log(self.weights))
        else:
            uniform_weight = 1.0 / tf.cast(self.num_particles, tf.float32)
            self.weights.assign(tf.ones(self.num_particles) * uniform_weight)
            self.log_weights.assign(tf.math.log(self.weights))

    def _ledh_flow_with_jacobian(self, measurement, landmarks):
        """Flow particles using LEDH; return accumulated log|det(J)|."""
        log_det_jacobians = tf.Variable(
            tf.zeros(self.num_particles, dtype=tf.float32), trainable=False)
        eta_bars = tf.Variable(self.particles, trainable=False)
        P_preds = self.particle_covs
        R_full = self.ssm.full_measurement_cov(tf.shape(landmarks)[0])
        R_inv = tf.linalg.inv(
            R_full + tf.eye(tf.shape(R_full)[0], dtype=tf.float32) * 1e-6)
        z_meas = tf.reshape(measurement, [-1])
        epsilon_steps = [
            self.epsilon_1 * (self.q ** (j - 1))
            for j in range(1, self.n_lambda + 1)]
        iterator = range(self.n_lambda)
        if self.show_progress:
            iterator = tqdm(
                iterator, desc=f'PF-PF-LEDH λ (0/{self.n_lambda})',
                leave=False, total=self.n_lambda)
        lambda_cumulative = 0.0
        for step in iterator:
            if self.show_progress and hasattr(iterator, 'set_description'):
                iterator.set_description(
                    f'PF-PF-LEDH λ ({step+1}/{self.n_lambda})')
            epsilon_j = epsilon_steps[step]
            lambda_k = lambda_cumulative + epsilon_j / 2.0
            lambda_cumulative += epsilon_j
            is_already_failed = log_det_jacobians < -1e9
            eigvals_batch = tf.linalg.eigvalsh(P_preds)
            min_eigvals = tf.reduce_min(eigvals_batch, axis=1)
            max_eigvals = tf.reduce_max(eigvals_batch, axis=1)
            cond_nums = max_eigvals / (min_eigvals + 1e-10)
            is_finite_P = tf.reduce_all(
                tf.math.is_finite(P_preds), axis=[1, 2])
            is_finite_eta = tf.reduce_all(tf.math.is_finite(eta_bars), axis=1)
            in_bounds = tf.reduce_all(
                tf.abs(eta_bars[:, :2]) <= 200.0, axis=1)
            should_skip = (
                (cond_nums > 1e8) | (~is_finite_P) | (~in_bounds)
                | (~is_finite_eta))
            log_det_jacobians.assign(tf.where(
                should_skip & (~is_already_failed),
                tf.constant(-1e10, dtype=tf.float32), log_det_jacobians))
            is_already_failed = log_det_jacobians < -1e9
            valid_mask = ~is_already_failed & ~should_skip
            if tf.reduce_any(valid_mask):
                H_batch = self.ssm.measurement_jacobian(eta_bars, landmarks)
                if len(H_batch.shape) == 2:
                    H_batch = tf.tile(
                        H_batch[tf.newaxis, :, :],
                        [self.num_particles, 1, 1])
                H_batch = tf.reshape(
                    H_batch, [self.num_particles, -1, self.state_dim])
                meas_dim = tf.shape(H_batch)[1]
                h_eta_batch = self.ssm.measurement_model(
                    eta_bars, landmarks)
                h_eta_batch = tf.reshape(
                    h_eta_batch, [self.num_particles, meas_dim])
                eta_bars_expanded = eta_bars[:, tf.newaxis, :]
                H_eta = tf.squeeze(tf.matmul(
                    H_batch,
                    tf.transpose(eta_bars_expanded, [0, 2, 1])), axis=2)
                e_lambda_batch = h_eta_batch - H_eta
                H_T = tf.transpose(H_batch, [0, 2, 1])
                HPH = tf.einsum('bij,bjk,bkl->bil', H_batch, P_preds, H_T)
                S_lambda_batch = lambda_k * HPH + R_full[tf.newaxis, :, :]
                eigvals_S = tf.linalg.eigvalsh(S_lambda_batch)
                min_eigvals_S = tf.reduce_min(eigvals_S, axis=1)
                reg = tf.where(
                    min_eigvals_S < 1e-6,
                    tf.maximum(1e-5, tf.abs(min_eigvals_S) * 0.1),
                    tf.where(lambda_k > 0.1, 1e-6, 1e-5))
                reg_expanded = reg[:, tf.newaxis, tf.newaxis]
                S_lambda_batch = S_lambda_batch + reg_expanded * tf.eye(
                    meas_dim, dtype=tf.float32)[tf.newaxis, :, :]
                is_finite_S = tf.reduce_all(
                    tf.math.is_finite(S_lambda_batch), axis=[1, 2])
                valid_S_mask = valid_mask & is_finite_S
                S_inv_batch = tf.zeros_like(S_lambda_batch)
                if tf.reduce_any(valid_S_mask):
                    try:
                        S_inv_valid = tf.linalg.inv(S_lambda_batch)
                        is_finite_S_inv = tf.reduce_all(
                            tf.math.is_finite(S_inv_valid), axis=[1, 2])
                        valid_S_inv_mask = valid_S_mask & is_finite_S_inv
                        S_inv_batch = tf.where(
                            valid_S_inv_mask[:, tf.newaxis, tf.newaxis],
                            S_inv_valid, tf.linalg.pinv(S_lambda_batch))
                    except Exception:
                        S_inv_batch = tf.linalg.pinv(S_lambda_batch)
                A_batch = -0.5 * tf.einsum(
                    'bij,bjk,bkl,blm->bim',
                    P_preds, H_T, S_inv_batch, H_batch)
                is_finite_A = tf.reduce_all(
                    tf.math.is_finite(A_batch), axis=[1, 2])
                valid_A_mask = valid_S_mask & is_finite_A
                if tf.reduce_any(valid_A_mask):
                    I = tf.eye(self.state_dim, dtype=tf.float32)
                    I_batch = tf.tile(
                        I[tf.newaxis, :, :],
                        [self.num_particles, 1, 1])
                    I_plus_lambda_A = I_batch + lambda_k * A_batch
                    I_plus_2lambda_A = I_batch + 2.0 * lambda_k * A_batch
                    is_finite_I = (
                        tf.reduce_all(tf.math.is_finite(I_plus_lambda_A),
                                      axis=[1, 2])
                        & tf.reduce_all(tf.math.is_finite(I_plus_2lambda_A),
                                        axis=[1, 2]))
                    valid_I_mask = valid_A_mask & is_finite_I
                    if tf.reduce_any(valid_I_mask):
                        z_minus_e_batch = z_meas[tf.newaxis, :] - e_lambda_batch
                        if hasattr(self.ssm, 'meas_per_landmark') and self.ssm.meas_per_landmark == 2:
                            z_minus_e_reshaped = tf.reshape(
                                z_minus_e_batch,
                                [self.num_particles, -1, 2])
                            bearings = z_minus_e_reshaped[:, :, 1]
                            bearings_wrapped = tf.math.atan2(
                                tf.sin(bearings), tf.cos(bearings))
                            z_minus_e_reshaped = tf.concat([
                                z_minus_e_reshaped[:, :, 0:1],
                                bearings_wrapped[:, :, tf.newaxis]], axis=2)
                            z_minus_e_batch = tf.reshape(
                                z_minus_e_reshaped,
                                [self.num_particles, meas_dim])
                        z_minus_e_batch = tf.clip_by_value(
                            z_minus_e_batch, -100.0, 100.0)
                        z_minus_e_col = z_minus_e_batch[:, :, tf.newaxis]
                        temp1 = tf.matmul(
                            R_inv[tf.newaxis, :, :], z_minus_e_col)
                        temp2 = tf.matmul(H_T, temp1)
                        temp3 = tf.matmul(P_preds, temp2)
                        term1 = tf.squeeze(
                            tf.matmul(I_plus_lambda_A, temp3), axis=2)
                        term2 = tf.einsum('bij,bj->bi', A_batch, eta_bars)
                        is_finite_terms = (
                            tf.reduce_all(tf.math.is_finite(term1), axis=1)
                            & tf.reduce_all(tf.math.is_finite(term2), axis=1))
                        valid_terms_mask = valid_I_mask & is_finite_terms
                        if tf.reduce_any(valid_terms_mask):
                            term_sum = term1 + term2
                            b_batch = tf.squeeze(tf.matmul(
                                I_plus_2lambda_A,
                                term_sum[:, :, tf.newaxis]), axis=2)
                            is_finite_b = tf.reduce_all(
                                tf.math.is_finite(b_batch), axis=1)
                            valid_b_mask = valid_terms_mask & is_finite_b
                            if tf.reduce_any(valid_b_mask):
                                particle_std = tf.cond(
                                    self.num_particles > 1,
                                    lambda: tf.math.reduce_std(
                                        self.particles[:, :2], axis=0),
                                    lambda: tf.constant(
                                        [10.0, 10.0], dtype=tf.float32))
                                max_velocity_pos = tf.maximum(
                                    particle_std * 2.0,
                                    tf.constant([10.0, 10.0], dtype=tf.float32))
                                if self.state_dim == 2:
                                    max_velocity = max_velocity_pos
                                elif self.state_dim == 3:
                                    max_velocity = tf.concat([
                                        max_velocity_pos,
                                        tf.constant([3.14159], dtype=tf.float32)
                                    ], axis=0)
                                else:
                                    n_targets = self.state_dim // 4
                                    max_velocity_parts = []
                                    for _ in range(n_targets):
                                        max_velocity_parts.append(max_velocity_pos)
                                        max_velocity_parts.append(
                                            tf.constant([10.0, 10.0],
                                                       dtype=tf.float32))
                                    max_velocity = tf.concat(
                                        max_velocity_parts, axis=0)
                                    remainder = self.state_dim - n_targets * 4
                                    if remainder > 0:
                                        max_velocity = tf.concat([
                                            max_velocity,
                                            tf.ones([remainder],
                                                   dtype=tf.float32) * 10.0
                                        ], axis=0)
                                velocities_batch = (
                                    tf.einsum('bij,bj->bi', A_batch,
                                              self.particles) + b_batch)
                                velocities_batch = tf.clip_by_value(
                                    velocities_batch,
                                    -max_velocity, max_velocity)
                                is_finite_velocity = tf.reduce_all(
                                    tf.math.is_finite(velocities_batch), axis=1)
                                valid_velocity_mask = (
                                    valid_b_mask & is_finite_velocity)
                                if tf.reduce_any(valid_velocity_mask):
                                    new_particles = (
                                        self.particles
                                        + epsilon_j * velocities_batch)
                                    in_bounds_new = tf.reduce_all(
                                        tf.abs(new_particles[:, :2]) <= 200.0,
                                        axis=1)
                                    valid_update_mask = (
                                        valid_velocity_mask & in_bounds_new)
                                    if self.state_dim > 2:
                                        angles = new_particles[:, 2]
                                        angles_wrapped = tf.math.atan2(
                                            tf.sin(angles), tf.cos(angles))
                                        new_particles = tf.concat([
                                            new_particles[:, :2],
                                            angles_wrapped[:, tf.newaxis],
                                            new_particles[:, 3:]
                                            if self.state_dim > 3
                                            else tf.zeros([self.num_particles, 0],
                                                          dtype=tf.float32)
                                        ], axis=1)
                                    self.particles.assign(tf.where(
                                        valid_update_mask[:, tf.newaxis],
                                        new_particles, self.particles))
                                    velocity_bar_batch = (
                                        tf.einsum('bij,bj->bi', A_batch,
                                                  eta_bars) + b_batch)
                                    velocity_bar_batch = tf.clip_by_value(
                                        velocity_bar_batch,
                                        -max_velocity, max_velocity)
                                    is_finite_velocity_bar = tf.reduce_all(
                                        tf.math.is_finite(velocity_bar_batch),
                                        axis=1)
                                    valid_eta_mask = (
                                        valid_update_mask
                                        & is_finite_velocity_bar)
                                    if tf.reduce_any(valid_eta_mask):
                                        new_eta_bars = (
                                            eta_bars
                                            + epsilon_j * velocity_bar_batch)
                                        in_bounds_eta = tf.reduce_all(
                                            tf.abs(new_eta_bars[:, :2])
                                            <= 200.0, axis=1)
                                        valid_eta_update_mask = (
                                            valid_eta_mask & in_bounds_eta)
                                        if self.state_dim > 2:
                                            angles_eta = new_eta_bars[:, 2]
                                            angles_eta_wrapped = tf.math.atan2(
                                                tf.sin(angles_eta),
                                                tf.cos(angles_eta))
                                            new_eta_bars = tf.concat([
                                                new_eta_bars[:, :2],
                                                angles_eta_wrapped[:, tf.newaxis],
                                                new_eta_bars[:, 3:]
                                                if self.state_dim > 3
                                                else tf.zeros([self.num_particles, 0],
                                                              dtype=tf.float32)
                                            ], axis=1)
                                        eta_bars.assign(tf.where(
                                            valid_eta_update_mask[:, tf.newaxis],
                                            new_eta_bars, eta_bars))
                                    J_step_batch = (
                                        I_batch + epsilon_j * A_batch)
                                    signs, log_dets = tf.linalg.slogdet(
                                        J_step_batch)
                                    is_negative = signs < 0.0
                                    is_finite_step = tf.math.is_finite(log_dets)
                                    is_valid_step = ~is_negative & is_finite_step
                                    can_update = (
                                        valid_eta_mask & is_valid_step
                                        & (~is_already_failed))
                                    log_det_abs_step = tf.abs(log_dets)
                                    new_log_dets = tf.where(
                                        can_update,
                                        log_det_jacobians + log_det_abs_step,
                                        tf.where(
                                            is_negative | (~is_finite_step),
                                            tf.constant(-1e10,
                                                        dtype=tf.float32),
                                            log_det_jacobians))
                                    log_det_jacobians.assign(new_log_dets)
                                    new_failures = (
                                        (is_negative | (~is_finite_step))
                                        & valid_eta_mask)
                                    if tf.reduce_any(new_failures):
                                        log_det_jacobians.assign(tf.where(
                                            new_failures,
                                            tf.constant(-1e10,
                                                        dtype=tf.float32),
                                            log_det_jacobians))
                                velocity_failures = (
                                    valid_b_mask & (~is_finite_velocity))
                                if tf.reduce_any(velocity_failures):
                                    log_det_jacobians.assign(tf.where(
                                        velocity_failures & (~is_already_failed),
                                        tf.constant(-1e10, dtype=tf.float32),
                                        log_det_jacobians))
                            terms_failures = (
                                valid_I_mask & (~is_finite_terms))
                            if tf.reduce_any(terms_failures):
                                log_det_jacobians.assign(tf.where(
                                    terms_failures & (~is_already_failed),
                                    tf.constant(-1e10, dtype=tf.float32),
                                    log_det_jacobians))
                        I_failures = valid_A_mask & (~is_finite_I)
                        if tf.reduce_any(I_failures):
                            log_det_jacobians.assign(tf.where(
                                I_failures & (~is_already_failed),
                                tf.constant(-1e10, dtype=tf.float32),
                                log_det_jacobians))
                A_failures = valid_S_mask & (~is_finite_A)
                if tf.reduce_any(A_failures):
                    log_det_jacobians.assign(tf.where(
                        A_failures & (~is_already_failed),
                        tf.constant(-1e10, dtype=tf.float32),
                        log_det_jacobians))
        return log_det_jacobians

    def _check_jacobian_consistency(self, particles_before_flow,
                                    log_det_jacobians, measurement=None,
                                    landmarks=None, do_finite_diff=False):
        """Check accumulated log|det(J)| (no-op unless do_finite_diff)."""
        pass

    def _compute_log_likelihood_batch(self, particles, measurement, landmarks):
        """Vectorized log-likelihood."""
        R_full = self.ssm.full_measurement_cov(tf.shape(landmarks)[0])
        R_inv = tf.linalg.inv(
            R_full + tf.eye(tf.shape(R_full)[0], dtype=tf.float32) * 1e-6)
        log_det_R = tf.linalg.slogdet(R_full)[1]
        z_preds = self.ssm.measurement_model(particles, landmarks)
        if len(z_preds.shape) > 2:
            z_preds = tf.reshape(z_preds, [tf.shape(particles)[0], -1])
        z_meas = tf.reshape(measurement, [1, -1])
        innovations = z_meas - z_preds
        if tf.shape(innovations)[1] > 1:
            num_meas = tf.shape(innovations)[1]
            bearing_cols = tf.range(1, num_meas, 2, dtype=tf.int32)
            if tf.size(bearing_cols) > 0:
                bearings = tf.gather(innovations, bearing_cols, axis=1)
                bearings_wrapped = tf.math.atan2(
                    tf.sin(bearings), tf.cos(bearings))
                batch_size = tf.shape(innovations)[0]
                num_bearings = tf.size(bearing_cols)
                row_indices = tf.repeat(
                    tf.range(batch_size), num_bearings)
                col_indices = tf.tile(bearing_cols, [batch_size])
                indices = tf.stack([row_indices, col_indices], axis=1)
                updates = tf.reshape(
                    tf.transpose(bearings_wrapped), [-1])
                innovations = tf.tensor_scatter_nd_update(
                    innovations, indices, updates)
        weighted_innovations = innovations @ R_inv
        mahalanobis_dists = tf.reduce_sum(
            weighted_innovations * innovations, axis=1)
        meas_dim = tf.cast(tf.shape(z_meas)[1], tf.float32)
        log_liks = -0.5 * (
            mahalanobis_dists + log_det_R
            + meas_dim * tf.math.log(2.0 * 3.14159))
        return log_liks

    def _compute_log_transition_batch(self, x_currents, x_previouss):
        """Vectorized transition density."""
        if self.control_prev is None:
            control = tf.zeros([1, 2], dtype=tf.float32)
        else:
            control = tf.reshape(self.control_prev, [1, -1])
        control_batch = tf.tile(
            control, [tf.shape(x_previouss)[0], 1])
        x_preds = self.ssm.motion_model(x_previouss, control_batch)
        if len(x_preds.shape) > 2:
            x_preds = tf.reshape(x_preds, tf.shape(x_previouss))
        diffs = x_currents - x_preds
        if tf.shape(diffs)[1] > 2:
            angles = diffs[:, 2]
            diffs = tf.concat([
                diffs[:, :2],
                tf.math.atan2(tf.sin(angles), tf.cos(angles))[:, tf.newaxis],
                diffs[:, 3:]
                if tf.shape(diffs)[1] > 3
                else tf.zeros([tf.shape(diffs)[0], 0], dtype=tf.float32)
            ], axis=1)
        weighted_diffs = diffs @ self.Q_inv
        mahalanobis_dists = tf.reduce_sum(weighted_diffs * diffs, axis=1)
        state_dim_f = tf.cast(self.state_dim, tf.float32)
        log_ps = -0.5 * (
            mahalanobis_dists + self.log_det_Q
            + state_dim_f * tf.math.log(2.0 * 3.14159))
        return log_ps

    def _systematic_resample(self):
        """Systematic resampling."""
        u = tf.random.uniform(
            [], 0.0, 1.0 / tf.cast(self.num_particles, tf.float32))
        cumulative_weights = tf.cumsum(self.weights)
        indices = []
        j = 0
        for i in range(self.num_particles):
            threshold = (
                u + tf.cast(i, tf.float32)
                / tf.cast(self.num_particles, tf.float32))
            while j < self.num_particles and cumulative_weights[j] < threshold:
                j += 1
            indices.append(tf.minimum(j, self.num_particles - 1))
        return tf.stack(indices)

    def _resample(self):
        """Resample when ESS is low; add jitter."""
        weight_sum = tf.reduce_sum(self.weights)
        weight_sq_sum = tf.reduce_sum(self.weights ** 2)
        ess = (weight_sum ** 2) / (weight_sq_sum + 1e-15)
        self.ess_before_resample = ess
        num_p = tf.cast(self.num_particles, tf.float32)
        if ess < num_p * 0.7:
            indices = self._systematic_resample()
            self.particles.assign(tf.gather(self.particles, indices))
            self.particle_covs.assign(tf.gather(self.particle_covs, indices))
            uniform_weight = 1.0 / num_p
            self.weights.assign(
                tf.ones(self.num_particles, dtype=tf.float32) * uniform_weight)
            self.log_weights.assign(tf.math.log(self.weights))
            particle_std = tf.math.reduce_std(self.particles, axis=0)
            jitter_scale = tf.maximum(particle_std * 0.02, 0.02)
            jitter_dist = tfd.Normal(
                loc=tf.zeros(self.state_dim, dtype=tf.float32),
                scale=jitter_scale)
            jitter = jitter_dist.sample(self.num_particles)
            self.particles.assign_add(jitter)
            if self.state_dim > 2:
                angles = self.particles[:, 2]
                angles_wrapped = tf.math.atan2(
                    tf.sin(angles), tf.cos(angles))
                particles_new = tf.concat([
                    self.particles[:, :2],
                    angles_wrapped[:, tf.newaxis],
                    self.particles[:, 3:]
                    if self.state_dim > 3
                    else tf.zeros([self.num_particles, 0], dtype=tf.float32)
                ], axis=1)
                self.particles.assign(particles_new)


class PFPFEDHFilter:
    """
    Particle Flow Particle Filter with EDH (Exact Daum-Huang).

    Uses global Gaussian approximation (EKF/UKF) for flow; all particles
    share the same linearization point.
    """

    def __init__(self, ssm, initial_state, initial_cov, num_particles=500,
                 n_lambda=20, filter_type='ekf', ukf_alpha=0.001, ukf_beta=2.0,
                 show_progress=False, redraw_particles=False):
        self.ssm = ssm
        self.num_particles = num_particles
        self.n_lambda = n_lambda
        self.q = 1.2
        self.epsilon_1 = (1.0 - self.q) / (1.0 - self.q ** self.n_lambda)
        self.filter_type = filter_type
        self.ukf_alpha = ukf_alpha
        self.ukf_beta = ukf_beta
        self.show_progress = show_progress
        self.redraw_particles = redraw_particles
        self.state_dim = _get_shape_dim(initial_state, dim=0)
        initial_state = _to_tensor(initial_state)
        initial_cov = _to_tensor(initial_cov)
        self.x_hat = tf.Variable(initial_state, trainable=False)
        self.P = tf.Variable(initial_cov, trainable=False)
        self.P_pred = tf.Variable(initial_cov, trainable=False)
        mvn = tfd.MultivariateNormalTriL(
            loc=initial_state,
            scale_tril=tf.linalg.cholesky(
                initial_cov + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6))
        self.particles = tf.Variable(
            mvn.sample(num_particles), trainable=False, dtype=tf.float32)
        self.particles_prev = None
        self.control_prev = None
        uniform_weight = 1.0 / tf.cast(num_particles, tf.float32)
        self.weights = tf.Variable(
            tf.ones(num_particles, dtype=tf.float32) * uniform_weight,
            trainable=False)
        self.log_weights = tf.Variable(
            tf.math.log(self.weights), trainable=False)
        self.state = tf.Variable(self.x_hat, trainable=False)
        self.ess_before_resample = tf.Variable(
            tf.constant(float(num_particles), dtype=tf.float32))
        if filter_type == 'ukf':
            kappa = 3.0 - tf.cast(self.state_dim, tf.float32)
            self.ukf_lambda = tf.constant(
                float(ukf_alpha**2 * (self.state_dim + kappa) - self.state_dim),
                dtype=tf.float32)
            self.ukf_gamma = tf.sqrt(
                tf.cast(self.state_dim, tf.float32) + self.ukf_lambda)
        else:
            self.ukf_lambda = None
            self.ukf_gamma = None
        Q = self.ssm.Q
        try:
            self.Q_inv = tf.linalg.inv(Q)
            self.log_det_Q = tf.linalg.slogdet(Q)[1]
        except Exception:
            self.Q_inv = tf.linalg.pinv(Q)
            self.log_det_Q = tf.constant(0.0, dtype=tf.float32)
        self.Q_mvn = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.state_dim, dtype=tf.float32),
            scale_tril=tf.linalg.cholesky(
                Q + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6))

    def predict(self, control):
        """Prediction step."""
        control = _to_tensor(control)
        self.particles_prev = tf.Variable(self.particles, trainable=False)
        self.control_prev = tf.Variable(control, trainable=False)
        if self.filter_type == 'ekf':
            self._ekf_predict(control)
        elif self.filter_type == 'ukf':
            self._ukf_predict(control)
        Q = self.ssm.Q
        control_batch = tf.tile(
            control[tf.newaxis, :], [self.num_particles, 1])
        preds = self.ssm.motion_model(self.particles, control_batch)
        if len(preds.shape) > 2:
            preds = tf.reshape(preds, [self.num_particles, -1])
        noise = tf.random.normal(
            [self.num_particles, self.state_dim], dtype=tf.float32)
        chol_Q = tf.linalg.cholesky(
            Q + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6)
        noise_multivariate = noise @ tf.transpose(chol_Q)
        self.particles.assign(preds + noise_multivariate)
        self.state.assign(self.x_hat)

    def update(self, measurement, landmarks):
        """PF-PF-EDH update."""
        measurement = _to_tensor(measurement)
        landmarks = _to_tensor(landmarks)
        if self.filter_type == 'ekf':
            self._ekf_update(measurement, landmarks)
        elif self.filter_type == 'ukf':
            self._ukf_update(measurement, landmarks)
        if self.redraw_particles:
            try:
                mvn_P = tfd.MultivariateNormalTriL(
                    loc=self.x_hat,
                    scale_tril=tf.linalg.cholesky(
                        self.P + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6))
                self.particles.assign(mvn_P.sample(self.num_particles))
            except Exception:
                eigvals, eigvecs = tf.linalg.eigh(self.P)
                eigvals = tf.maximum(eigvals, 1e-6)
                P_fixed = eigvecs @ tf.linalg.diag(eigvals) @ tf.transpose(eigvecs)
                mvn_P = tfd.MultivariateNormalTriL(
                    loc=self.x_hat,
                    scale_tril=tf.linalg.cholesky(
                        P_fixed + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6))
                self.particles.assign(mvn_P.sample(self.num_particles))
        particles_before_flow = tf.Variable(self.particles, trainable=False)
        log_det_jacobians = self._edh_flow_with_jacobian(
            measurement, landmarks)
        self._check_jacobian_consistency_edh(
            particles_before_flow, log_det_jacobians)
        log_weight_increments = self._compute_weight_increments_robust(
            measurement, landmarks, particles_before_flow, log_det_jacobians)
        self.log_weights.assign_add(log_weight_increments)
        self._normalize_weights_edh()
        self._resample_edh()
        self.state.assign(
            tf.reduce_sum(self.particles * self.weights[:, tf.newaxis], axis=0))

    def _compute_weight_increments_robust(self, measurement, landmarks,
                                          particles_before_flow,
                                          log_det_jacobians):
        """Compute weight increments with robust handling."""
        failed_mask = ~tf.math.is_finite(log_det_jacobians)
        log_weight_increments = tf.where(
            failed_mask, tf.constant(-1e10, dtype=tf.float32),
            tf.zeros(self.num_particles, dtype=tf.float32))
        valid_mask = ~failed_mask & tf.reduce_all(
            tf.abs(self.particles[:, :2]) <= 200.0, axis=1)
        if tf.reduce_any(valid_mask):
            valid_indices = tf.where(valid_mask)[:, 0]
            particles_valid = tf.gather(self.particles, valid_indices)
            particles_prev_valid = tf.gather(self.particles_prev, valid_indices)
            particles_before_valid = tf.gather(
                particles_before_flow, valid_indices)
            log_liks = self._compute_log_likelihood_batch_edh(
                particles_valid, measurement, landmarks)
            log_liks = tf.clip_by_value(log_liks, -100.0, 100.0)
            log_p_plus = self._compute_log_transition_batch_edh(
                particles_valid, particles_prev_valid)
            log_p_minus = self._compute_log_transition_batch_edh(
                particles_before_valid, particles_prev_valid)
            transition_ratios = tf.clip_by_value(
                log_p_plus - log_p_minus, -20.0, 20.0)
            log_det_Js = tf.clip_by_value(
                tf.gather(log_det_jacobians, valid_indices), -20.0, 20.0)
            increments_valid = log_liks + transition_ratios + log_det_Js
            log_weight_increments = tf.tensor_scatter_nd_update(
                log_weight_increments,
                valid_indices[:, tf.newaxis], increments_valid)
        return log_weight_increments

    def _ekf_predict(self, control):
        """EKF prediction."""
        pred = self.ssm.motion_model(
            self.x_hat[tf.newaxis, :], control[tf.newaxis, :])
        if len(pred.shape) > 1:
            pred = pred[0]
        self.x_hat.assign(pred)
        F = self.ssm.motion_jacobian(
            self.x_hat[tf.newaxis, :], control[tf.newaxis, :])
        if len(F.shape) > 2:
            F = F[0]
        Q = self.ssm.Q
        self.P_pred.assign(F @ self.P @ tf.transpose(F) + Q)
        self.P.assign(self.P_pred)

    def _ekf_update(self, measurement, landmarks):
        """EKF update."""
        z_pred = self.ssm.measurement_model(
            self.x_hat[tf.newaxis, :], landmarks)
        z_pred = tf.reshape(z_pred, [-1])
        z_meas = tf.reshape(measurement, [-1])
        innovation = z_meas - z_pred
        if hasattr(self.ssm, 'meas_per_landmark') and self.ssm.meas_per_landmark == 2:
            if tf.shape(innovation)[0] > 1:
                innovation_reshaped = tf.reshape(innovation, [-1, 2])
                bearings = innovation_reshaped[:, 1]
                bearings_wrapped = tf.math.atan2(
                    tf.sin(bearings), tf.cos(bearings))
                innovation_reshaped = tf.concat([
                    innovation_reshaped[:, 0:1],
                    bearings_wrapped[:, tf.newaxis]], axis=1)
                innovation = tf.reshape(innovation_reshaped, [-1])
        H = self.ssm.measurement_jacobian(
            self.x_hat[tf.newaxis, :], landmarks)
        if len(H.shape) > 2:
            H = tf.reshape(H, [-1, self.state_dim])
        R_full = self.ssm.full_measurement_cov(tf.shape(landmarks)[0])
        S = H @ self.P_pred @ tf.transpose(H) + R_full
        S_inv = tf.linalg.inv(
            S + tf.eye(tf.shape(S)[0], dtype=tf.float32) * 1e-6)
        K = self.P_pred @ tf.transpose(H) @ S_inv
        self.x_hat.assign_add(tf.linalg.matvec(K, innovation))
        if self.state_dim > 2:
            angle = self.x_hat[2]
            x_hat_new = tf.concat([
                self.x_hat[:2],
                [tf.math.atan2(tf.sin(angle), tf.cos(angle))],
                self.x_hat[3:]
                if self.state_dim > 3
                else tf.zeros([0], dtype=tf.float32)], axis=0)
            self.x_hat.assign(x_hat_new)
        I_KH = tf.eye(self.state_dim, dtype=tf.float32) - K @ H
        self.P.assign(
            I_KH @ self.P_pred @ tf.transpose(I_KH)
            + K @ R_full @ tf.transpose(K))

    def _ukf_predict(self, control):
        """UKF prediction."""
        try:
            L = tf.linalg.cholesky(self.P)
        except Exception:
            L = tf.linalg.cholesky(
                self.P + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6)
        sigma_points = tf.zeros(
            [2 * self.state_dim + 1, self.state_dim], dtype=tf.float32)
        sigma_points = tf.tensor_scatter_nd_update(
            sigma_points, [[0]], self.x_hat[tf.newaxis, :])
        for i in range(self.state_dim):
            sigma_points = tf.tensor_scatter_nd_update(
                sigma_points, [[i + 1]],
                (self.x_hat + self.ukf_gamma * L[:, i])[tf.newaxis, :])
            sigma_points = tf.tensor_scatter_nd_update(
                sigma_points, [[self.state_dim + i + 1]],
                (self.x_hat - self.ukf_gamma * L[:, i])[tf.newaxis, :])
        sigma_points_pred = []
        for i in range(2 * self.state_dim + 1):
            pred = self.ssm.motion_model(
                sigma_points[i:i+1], control[tf.newaxis, :])
            if len(pred.shape) > 1:
                pred = pred[0]
            sigma_points_pred.append(pred)
        sigma_points_pred = tf.stack(sigma_points_pred)
        w_m = tf.zeros(2 * self.state_dim + 1, dtype=tf.float32)
        w_c = tf.zeros(2 * self.state_dim + 1, dtype=tf.float32)
        w_m = tf.tensor_scatter_nd_update(
            w_m, [[0]],
            [self.ukf_lambda / (tf.cast(self.state_dim, tf.float32)
             + self.ukf_lambda)])
        w_c = tf.tensor_scatter_nd_update(
            w_c, [[0]], [w_m[0] + (1.0 - self.ukf_alpha**2 + self.ukf_beta)])
        for i in range(1, 2 * self.state_dim + 1):
            weight_val = 1.0 / (2.0 * (tf.cast(self.state_dim, tf.float32)
                                        + self.ukf_lambda))
            w_m = tf.tensor_scatter_nd_update(w_m, [[i]], [weight_val])
            w_c = tf.tensor_scatter_nd_update(w_c, [[i]], [weight_val])
        self.x_hat.assign(
            tf.reduce_sum(w_m[:, tf.newaxis] * sigma_points_pred, axis=0))
        Q = self.ssm.Q
        self.P_pred.assign(Q)
        for i in range(2 * self.state_dim + 1):
            diff = sigma_points_pred[i] - self.x_hat
            outer_prod = diff[:, tf.newaxis] @ diff[tf.newaxis, :]
            if tf.reduce_all(tf.math.is_finite(outer_prod)):
                self.P_pred.assign_add(w_c[i] * outer_prod)
        self.P_pred.assign(
            (self.P_pred + tf.transpose(self.P_pred)) / 2.0
            + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6)
        self.P.assign(self.P_pred)

    def _ukf_update(self, measurement, landmarks):
        """UKF update."""
        try:
            L = tf.linalg.cholesky(self.P_pred)
        except Exception:
            L = tf.linalg.cholesky(
                self.P_pred + tf.eye(self.state_dim, dtype=tf.float32) * 1e-6)
        sigma_points = tf.zeros(
            [2 * self.state_dim + 1, self.state_dim], dtype=tf.float32)
        sigma_points = tf.tensor_scatter_nd_update(
            sigma_points, [[0]], self.x_hat[tf.newaxis, :])
        for i in range(self.state_dim):
            sigma_points = tf.tensor_scatter_nd_update(
                sigma_points, [[i + 1]],
                (self.x_hat + self.ukf_gamma * L[:, i])[tf.newaxis, :])
            sigma_points = tf.tensor_scatter_nd_update(
                sigma_points, [[self.state_dim + i + 1]],
                (self.x_hat - self.ukf_gamma * L[:, i])[tf.newaxis, :])
        z_sigma = []
        for i in range(2 * self.state_dim + 1):
            z_pred = self.ssm.measurement_model(
                sigma_points[i:i+1], landmarks)
            z_sigma.append(tf.reshape(z_pred, [-1]))
        z_sigma = tf.stack(z_sigma)
        w_m = tf.zeros(2 * self.state_dim + 1, dtype=tf.float32)
        w_c = tf.zeros(2 * self.state_dim + 1, dtype=tf.float32)
        w_m = tf.tensor_scatter_nd_update(
            w_m, [[0]],
            [self.ukf_lambda / (tf.cast(self.state_dim, tf.float32)
             + self.ukf_lambda)])
        w_c = tf.tensor_scatter_nd_update(
            w_c, [[0]], [w_m[0] + (1.0 - self.ukf_alpha**2 + self.ukf_beta)])
        for i in range(1, 2 * self.state_dim + 1):
            weight_val = 1.0 / (2.0 * (tf.cast(self.state_dim, tf.float32)
                                        + self.ukf_lambda))
            w_m = tf.tensor_scatter_nd_update(w_m, [[i]], [weight_val])
            w_c = tf.tensor_scatter_nd_update(w_c, [[i]], [weight_val])
        z_mean = tf.reduce_sum(w_m[:, tf.newaxis] * z_sigma, axis=0)
        R_full = self.ssm.full_measurement_cov(tf.shape(landmarks)[0])
        S = R_full
        for i in range(2 * self.state_dim + 1):
            diff = z_sigma[i] - z_mean
            S = S + w_c[i] * (diff[:, tf.newaxis] @ diff[tf.newaxis, :])
        P_xz = tf.zeros(
            [self.state_dim, tf.shape(z_sigma)[1]], dtype=tf.float32)
        for i in range(2 * self.state_dim + 1):
            diff_x = sigma_points[i] - self.x_hat
            diff_z = z_sigma[i] - z_mean
            P_xz = P_xz + w_c[i] * (diff_x[:, tf.newaxis] @ diff_z[tf.newaxis, :])
        S_reg = (S + tf.transpose(S)) / 2.0 + tf.eye(
            tf.shape(S)[0], dtype=tf.float32) * 1e-6
        try:
            S_inv = tf.linalg.inv(S_reg)
        except Exception:
            S_inv = tf.linalg.pinv(S_reg)
        K = P_xz @ S_inv
        z_meas = tf.reshape(measurement, [-1])
        innovation = z_meas - z_mean
        if hasattr(self.ssm, 'meas_per_landmark') and self.ssm.meas_per_landmark == 2:
            if tf.shape(innovation)[0] > 1:
                innovation_reshaped = tf.reshape(innovation, [-1, 2])
                bearings = innovation_reshaped[:, 1]
                bearings_wrapped = tf.math.atan2(
                    tf.sin(bearings), tf.cos(bearings))
                innovation_reshaped = tf.concat([
                    innovation_reshaped[:, 0:1],
                    bearings_wrapped[:, tf.newaxis]], axis=1)
                innovation = tf.reshape(innovation_reshaped, [-1])
        self.x_hat.assign_add(tf.linalg.matvec(K, innovation))
        if self.state_dim > 2:
            angle = self.x_hat[2]
            x_hat_new = tf.concat([
                self.x_hat[:2],
                [tf.math.atan2(tf.sin(angle), tf.cos(angle))],
                self.x_hat[3:]
                if self.state_dim > 3
                else tf.zeros([0], dtype=tf.float32)], axis=0)
            self.x_hat.assign(x_hat_new)
        self.P.assign(self.P_pred - K @ S @ tf.transpose(K))

    def _edh_flow_with_jacobian(self, measurement, landmarks):
        """Flow particles using global covariance (EDH)."""
        log_det_jacobians = tf.Variable(
            tf.zeros(self.num_particles, dtype=tf.float32), trainable=False)
        eta_bar = tf.Variable(self.x_hat, trainable=False)
        R_full = self.ssm.full_measurement_cov(tf.shape(landmarks)[0])
        R_inv = tf.linalg.inv(
            R_full + tf.eye(tf.shape(R_full)[0], dtype=tf.float32) * 1e-6)
        z_meas = tf.reshape(measurement, [-1])
        epsilon_steps = [
            self.epsilon_1 * (self.q ** (j - 1))
            for j in range(1, self.n_lambda + 1)]
        iterator = range(self.n_lambda)
        if self.show_progress:
            iterator = tqdm(
                iterator, desc=f'PF-PF-EDH λ (0/{self.n_lambda})',
                leave=False, total=self.n_lambda)
        lambda_cumulative = 0.0
        for step in iterator:
            if self.show_progress and hasattr(iterator, 'set_description'):
                iterator.set_description(
                    f'PF-PF-EDH λ ({step+1}/{self.n_lambda})')
            epsilon_j = epsilon_steps[step]
            lambda_k = lambda_cumulative + epsilon_j / 2.0
            lambda_cumulative += epsilon_j
            P_pred = self.P_pred
            H = self.ssm.measurement_jacobian(
                eta_bar[tf.newaxis, :], landmarks)
            if len(H.shape) > 2:
                H = tf.reshape(H, [-1, self.state_dim])
            h_eta = self.ssm.measurement_model(
                eta_bar[tf.newaxis, :], landmarks)
            h_eta = tf.reshape(h_eta, [-1])
            e_lambda = h_eta - tf.linalg.matvec(H, eta_bar)
            S_lambda = lambda_k * (H @ P_pred @ tf.transpose(H)) + R_full
            S_reg = (S_lambda + tf.transpose(S_lambda)) / 2.0 + tf.eye(
                tf.shape(S_lambda)[0], dtype=tf.float32) * 1e-6
            try:
                S_inv = tf.linalg.inv(S_reg)
            except Exception:
                S_inv = tf.linalg.pinv(S_reg)
            A = -0.5 * P_pred @ tf.transpose(H) @ S_inv @ H
            I = tf.eye(self.state_dim, dtype=tf.float32)
            I_plus_lambda_A = I + lambda_k * A
            I_plus_2lambda_A = I + 2.0 * lambda_k * A
            z_minus_e = z_meas - e_lambda
            if hasattr(self.ssm, 'meas_per_landmark') and self.ssm.meas_per_landmark == 2:
                if tf.shape(z_minus_e)[0] > 1:
                    z_minus_e_reshaped = tf.reshape(z_minus_e, [-1, 2])
                    bearings = z_minus_e_reshaped[:, 1]
                    bearings_wrapped = tf.math.atan2(
                        tf.sin(bearings), tf.cos(bearings))
                    z_minus_e_reshaped = tf.concat([
                        z_minus_e_reshaped[:, 0:1],
                        bearings_wrapped[:, tf.newaxis]], axis=1)
                    z_minus_e = tf.reshape(z_minus_e_reshaped, [-1])
            z_minus_e_col = z_minus_e[:, tf.newaxis]
            temp = R_inv @ z_minus_e_col
            temp = tf.transpose(H) @ temp
            temp = P_pred @ temp
            term1 = tf.reshape(I_plus_lambda_A @ temp, [-1])
            term2 = tf.linalg.matvec(A, eta_bar)
            b = tf.linalg.matvec(I_plus_2lambda_A, term1 + term2)
            J_step = I + epsilon_j * A
            current_log_det = log_det_jacobians[0]
            is_already_failed = current_log_det < -1e9
            if not is_already_failed:
                sign_step, log_det_step = tf.linalg.slogdet(J_step)
                is_finite_step = tf.math.is_finite(log_det_step)
                is_negative_step = sign_step < 0.0
                if is_negative_step:
                    log_det_jacobians.assign(
                        tf.ones(self.num_particles, dtype=tf.float32) * (-1e10))
                elif is_finite_step:
                    new_log_det = current_log_det + tf.abs(log_det_step)
                    log_det_jacobians.assign(
                        tf.ones(self.num_particles, dtype=tf.float32) * new_log_det)
                else:
                    log_det_jacobians.assign(
                        tf.ones(self.num_particles, dtype=tf.float32) * (-1e10))
            velocities = (self.particles @ tf.transpose(A)) + b
            self.particles.assign(self.particles + epsilon_j * velocities)
            if self.state_dim > 2:
                angles = self.particles[:, 2]
                angles_wrapped = tf.math.atan2(
                    tf.sin(angles), tf.cos(angles))
                particles_new = tf.concat([
                    self.particles[:, :2],
                    angles_wrapped[:, tf.newaxis],
                    self.particles[:, 3:]
                    if self.state_dim > 3
                    else tf.zeros([self.num_particles, 0], dtype=tf.float32)
                ], axis=1)
                self.particles.assign(particles_new)
            velocity_bar = tf.linalg.matvec(A, eta_bar) + b
            eta_bar.assign(eta_bar + epsilon_j * velocity_bar)
            if self.state_dim > 2:
                angle = eta_bar[2]
                eta_bar_new = tf.concat([
                    eta_bar[:2],
                    [tf.math.atan2(tf.sin(angle), tf.cos(angle))],
                    eta_bar[3:]
                    if self.state_dim > 3
                    else tf.zeros([0], dtype=tf.float32)], axis=0)
                eta_bar.assign(eta_bar_new)
        return log_det_jacobians

    def _check_jacobian_consistency_edh(self, particles_before_flow,
                                        log_det_jacobians):
        """No-op consistency check for EDH."""
        pass

    def _compute_log_likelihood_batch_edh(self, particles, measurement,
                                          landmarks):
        """Vectorized log-likelihood (EDH)."""
        R_full = self.ssm.full_measurement_cov(tf.shape(landmarks)[0])
        R_inv = tf.linalg.inv(
            R_full + tf.eye(tf.shape(R_full)[0], dtype=tf.float32) * 1e-6)
        log_det_R = tf.linalg.slogdet(R_full)[1]
        z_preds = self.ssm.measurement_model(particles, landmarks)
        if len(z_preds.shape) > 2:
            z_preds = tf.reshape(z_preds, [tf.shape(particles)[0], -1])
        z_meas = tf.reshape(measurement, [1, -1])
        innovations = z_meas - z_preds
        if (hasattr(self.ssm, 'meas_per_landmark')
                and self.ssm.meas_per_landmark == 2
                and tf.shape(innovations)[1] > 1):
            innovations_reshaped = tf.reshape(innovations, [-1, 2])
            bearings = innovations_reshaped[:, 1]
            bearings_wrapped = tf.math.atan2(
                tf.sin(bearings), tf.cos(bearings))
            innovations_reshaped = tf.concat([
                innovations_reshaped[:, 0:1],
                bearings_wrapped[:, tf.newaxis]], axis=1)
            innovations = tf.reshape(
                innovations_reshaped, tf.shape(innovations))
        weighted_innovations = innovations @ R_inv
        mahalanobis_dists = tf.reduce_sum(
            weighted_innovations * innovations, axis=1)
        meas_dim = tf.cast(tf.shape(z_meas)[1], tf.float32)
        log_liks = -0.5 * (
            mahalanobis_dists + log_det_R
            + meas_dim * tf.math.log(2.0 * 3.14159))
        return log_liks

    def _compute_log_transition_batch_edh(self, x_currents, x_previouss):
        """Vectorized transition density (EDH)."""
        if self.control_prev is None:
            control = tf.zeros([1, 2], dtype=tf.float32)
        else:
            control = tf.reshape(self.control_prev, [1, -1])
        control_batch = tf.tile(
            control, [tf.shape(x_previouss)[0], 1])
        x_preds = self.ssm.motion_model(x_previouss, control_batch)
        if len(x_preds.shape) > 2:
            x_preds = tf.reshape(x_preds, tf.shape(x_previouss))
        diffs = x_currents - x_preds
        if tf.shape(diffs)[1] > 2:
            angles = diffs[:, 2]
            diffs = tf.concat([
                diffs[:, :2],
                tf.math.atan2(tf.sin(angles), tf.cos(angles))[:, tf.newaxis],
                diffs[:, 3:]
                if tf.shape(diffs)[1] > 3
                else tf.zeros([tf.shape(diffs)[0], 0], dtype=tf.float32)
            ], axis=1)
        weighted_diffs = diffs @ self.Q_inv
        mahalanobis_dists = tf.reduce_sum(weighted_diffs * diffs, axis=1)
        state_dim_f = tf.cast(self.state_dim, tf.float32)
        log_ps = -0.5 * (
            mahalanobis_dists + self.log_det_Q
            + state_dim_f * tf.math.log(2.0 * 3.14159))
        return log_ps

    def _normalize_weights_edh(self):
        """Normalize weights with regularization (EDH)."""
        finite_mask = tf.math.is_finite(self.log_weights)
        num_p = tf.cast(self.num_particles, tf.float32)
        uniform_weight = 1.0 / num_p
        if tf.reduce_any(finite_mask):
            max_log_weight = tf.reduce_max(tf.where(
                finite_mask, self.log_weights,
                tf.constant(-1e10, dtype=tf.float32)))
            self.log_weights.assign(self.log_weights - max_log_weight)
            weights_unnorm = tf.where(
                finite_mask, tf.exp(self.log_weights),
                tf.zeros_like(self.log_weights))
            weight_sum = tf.reduce_sum(weights_unnorm)
            if weight_sum > 1e-10:
                self.weights.assign(weights_unnorm / weight_sum)
                weight_sq_sum = tf.reduce_sum(self.weights ** 2)
                ess = (tf.reduce_sum(self.weights) ** 2) / (weight_sq_sum + 1e-15)
                alpha = tf.cond(
                    ess < num_p / 10.0, lambda: 0.4,
                    lambda: tf.cond(
                        ess < num_p / 5.0, lambda: 0.3,
                        lambda: tf.cond(
                            ess < num_p / 3.0, lambda: 0.15,
                            lambda: tf.cond(
                                ess < num_p / 2.0, lambda: 0.05, lambda: 0.02))))
                self.weights.assign(
                    alpha * uniform_weight + (1.0 - alpha) * self.weights)
                self.log_weights.assign(tf.math.log(self.weights + 1e-10))
            else:
                self.weights.assign(
                    tf.ones(self.num_particles, dtype=tf.float32) * uniform_weight)
                self.log_weights.assign(tf.math.log(self.weights))
        else:
            self.weights.assign(
                tf.ones(self.num_particles, dtype=tf.float32) * uniform_weight)
            self.log_weights.assign(tf.math.log(self.weights))

    def _systematic_resample_edh(self):
        """Systematic resampling (EDH)."""
        N = self.num_particles
        cumsum = tf.cumsum(self.weights)
        u = tf.random.uniform([], dtype=tf.float32) / tf.cast(N, tf.float32)
        positions = u + tf.cast(tf.range(N), tf.float32) / tf.cast(N, tf.float32)
        indices = tf.searchsorted(cumsum, positions, side='right')
        indices = tf.minimum(indices, N - 1)
        return indices

    def _resample_edh(self):
        """Resample when ESS is low; add jitter (EDH)."""
        weight_sum = tf.reduce_sum(self.weights)
        weight_sq_sum = tf.reduce_sum(self.weights ** 2)
        ess = (weight_sum ** 2) / (weight_sq_sum + 1e-15)
        self.ess_before_resample.assign(ess)
        num_p = tf.cast(self.num_particles, tf.float32)
        if ess < num_p * 0.7:
            indices = self._systematic_resample_edh()
            self.particles.assign(tf.gather(self.particles, indices))
            uniform_weight = 1.0 / num_p
            self.weights.assign(
                tf.ones(self.num_particles, dtype=tf.float32) * uniform_weight)
            self.log_weights.assign(tf.math.log(self.weights))
            particle_std = tf.math.reduce_std(self.particles, axis=0)
            jitter_scale = tf.maximum(particle_std * 0.02, 0.02)
            jitter_dist = tfd.Normal(
                loc=tf.zeros(self.state_dim, dtype=tf.float32),
                scale=jitter_scale)
            jitter = jitter_dist.sample(self.num_particles)
            self.particles.assign_add(jitter)
            if self.state_dim > 2:
                angles = self.particles[:, 2]
                angles_wrapped = tf.math.atan2(
                    tf.sin(angles), tf.cos(angles))
                particles_new = tf.concat([
                    self.particles[:, :2],
                    angles_wrapped[:, tf.newaxis],
                    self.particles[:, 3:]
                    if self.state_dim > 3
                    else tf.zeros([self.num_particles, 0], dtype=tf.float32)
                ], axis=1)
                self.particles.assign(particles_new)


__all__ = ['PFPFLEDHFilter', 'PFPFEDHFilter', '_to_tensor', '_get_shape_dim']
