"""
Neural OT resampling: training data generation, training loop, and
drop-in resampling function for differentiable particle filters.

Replaces the iterative Sinkhorn-based DET resampling with a single
forward pass through a trained conditional mGradNet.

Training modes
--------------
1. **Supervised** (default): Generate (particles, weights) pairs from
   diverse sources and use Sinkhorn solutions as regression targets.
2. **Monge-Ampere**: Directly minimise the Monge-Ampere residual using
   the mGradNet's log-det-Jacobian (no Sinkhorn targets needed, but
   requires source and target density evaluation).

Usage
-----
::

    from src.filters.bonus.neural_ot_resampling import (
        NeuralOTTrainer,
        neural_ot_resample,
        build_context_scalars,
        mask_context_columns,
    )

    # 1. Train
    trainer = NeuralOTTrainer(state_dim=1, num_particles=50)
    model = trainer.train(y_obs, sinkhorn_epsilon=2.0, epochs=2000)

    # 2. Use as drop-in replacement
    x_new, w_new = neural_ot_resample(
        particles, log_w, model,
        theta=theta, t=t, y_t=y_t, epsilon=eps,
    )
"""

from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

try:
    from tqdm import tqdm, trange
except ImportError:
    tqdm = lambda x, **kwargs: x
    trange = range

from src.filters.dpf.resampling import det_resample
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet

tfd = tfp.distributions

_EPS = 1e-6


def mask_context_columns(
    ctx: tf.Tensor, zero_dims: Optional[List[int]]
) -> tf.Tensor:
    """Return a copy of *ctx* with listed column indices set to zero.

    Used for context ablations when *ctx* is a ``tf.Tensor`` of shape
    ``(M, n_ctx)`` (no NumPy in-place assignment).
    """
    if not zero_dims:
        return tf.identity(ctx)
    ctx = tf.cast(ctx, tf.float32)
    for d in sorted(set(zero_dims), reverse=True):
        zcol = tf.zeros([tf.shape(ctx)[0], 1], dtype=ctx.dtype)
        ctx = tf.concat([ctx[:, :d], zcol, ctx[:, d + 1 :]], axis=1)
    return ctx


# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------

def build_context_scalars(
    theta: tf.Tensor,
    t: float,
    y_t: float,
    ess: float,
    epsilon: float,
    T_max: float = 50.0,
) -> tf.Tensor:
    """Build the scalar context vector for the conditional mGradNet.

    Returns shape ``(6,)``:  [log_sv2, log_sw2, t/T, y_t, ESS, epsilon].
    """
    log_sv2 = theta[0] if len(tf.shape(theta)) >= 1 else theta
    log_sw2 = theta[1] if len(tf.shape(theta)) >= 1 else theta
    return tf.stack([
        tf.cast(log_sv2, tf.float32),
        tf.cast(log_sw2, tf.float32),
        tf.cast(t / T_max, tf.float32),
        tf.cast(y_t, tf.float32),
        tf.cast(ess, tf.float32),
        tf.cast(epsilon, tf.float32),
    ])


def _compute_ess(weights: tf.Tensor) -> tf.Tensor:
    """ESS = 1 / sum(w_i^2).  *weights* must be normalised."""
    return 1.0 / tf.reduce_sum(weights ** 2)


# ---------------------------------------------------------------------------
# Drop-in resampling function
# ---------------------------------------------------------------------------

def neural_ot_resample(
    x: tf.Tensor,
    log_w: tf.Tensor,
    model: ConditionalMGradNet,
    *,
    theta: tf.Tensor,
    t: float,
    y_t: float,
    epsilon: float = 2.0,
    T_max: float = 50.0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Neural OT resampling — single forward pass, no Sinkhorn iterations.

    Drop-in replacement for :func:`src.filters.dpf.resampling.det_resample`.

    Parameters
    ----------
    x      : (N, d)  particle positions.
    log_w  : (N,)    log-weights.
    model  : trained :class:`ConditionalMGradNet`.
    theta  : (2,)    current model parameters [log_sv2, log_sw2].
    t      : float   current timestep.
    y_t    : float   current observation.
    epsilon: float   Sinkhorn ε the model was trained for.
    T_max  : float   time horizon (for normalising t).

    Returns
    -------
    x_resampled : (N, d)
    w_uniform   : (N,)   uniform weights 1/N.
    """
    N = tf.shape(x)[0]
    w = tf.nn.softmax(log_w, axis=0)
    ess = _compute_ess(w)

    # Normalise particles (scale equivariance)
    if len(x.shape) == 1:
        x_2d = x[:, tf.newaxis]
    else:
        x_2d = x

    p_mean = tf.reduce_mean(x_2d, axis=0, keepdims=True)
    p_std = tf.math.reduce_std(x_2d, axis=0, keepdims=True) + _EPS
    x_norm = (x_2d - p_mean) / p_std

    ctx = build_context_scalars(theta, t, y_t, ess, epsilon, T_max)

    # Single forward pass
    x_resampled_norm = model(x_norm, w, ctx)

    # Handle shape: model may return (N,) for d=1 input
    if len(x_resampled_norm.shape) == 1:
        x_resampled_norm = x_resampled_norm[:, tf.newaxis]

    x_resampled = x_resampled_norm * p_std + p_mean

    if len(x.shape) == 1:
        x_resampled = x_resampled[:, 0]

    w_uniform = tf.fill([N], 1.0 / tf.cast(N, tf.float32))
    return x_resampled, w_uniform


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for neural OT training data generation."""
    num_particles: int = 50
    state_dim: int = 1
    sinkhorn_epsilon: float = 2.0
    sinkhorn_iters: int = 100        # high iters for accurate targets
    T_max: float = 50.0
    # -- Synthetic data generation --
    n_theta_samples: int = 200       # different theta values
    n_seeds_per_theta: int = 3       # random seeds per theta
    # -- theta sampling ranges (log-space) --
    log_sv2_range: Tuple[float, float] = (math.log(0.5), math.log(50.0))
    log_sw2_range: Tuple[float, float] = (math.log(0.1), math.log(10.0))


def _generate_filter_data(
    cfg: TrainingConfig,
    y_obs: tf.Tensor,
) -> dict:
    """Generate training data by running the 1-D LEDH filter with Sinkhorn.

    Collects (particles, weights) before resampling and the Sinkhorn
    solution at every timestep, for many random theta values.

    Returns dict with keys (each value is a ``tf.Tensor``):
        particles_norm  (M, N)   normalised particle positions
        weights         (M, N)   normalised weights
        targets_norm    (M, N)   Sinkhorn-resampled positions (normalised)
        context_scalars (M, 6)   [log_sv2, log_sw2, t/T, y_t, ESS, eps]
    """
    N = cfg.num_particles
    T = int(y_obs.shape[0])
    if len(y_obs.shape) == 1:
        y_obs_2d = y_obs[:, tf.newaxis]
    else:
        y_obs_2d = y_obs
    eps = cfg.sinkhorn_epsilon

    tf.random.set_seed(42)
    log_sv2_vals = tf.random.uniform(
        [cfg.n_theta_samples],
        float(cfg.log_sv2_range[0]),
        float(cfg.log_sv2_range[1]),
        dtype=tf.float32,
    )
    log_sw2_vals = tf.random.uniform(
        [cfg.n_theta_samples],
        float(cfg.log_sw2_range[0]),
        float(cfg.log_sw2_range[1]),
        dtype=tf.float32,
    )

    all_particles, all_weights, all_targets, all_ctx = [], [], [], []

    for i_theta in trange(cfg.n_theta_samples, desc="  Filter Data"):
        Q_val = tf.maximum(
            tf.exp(log_sv2_vals[i_theta]), tf.cast(_EPS, tf.float32)
        )
        R_val = tf.maximum(
            tf.exp(log_sw2_vals[i_theta]), tf.cast(_EPS, tf.float32)
        )
        R_inv = 1.0 / R_val
        log_det_R = tf.math.log(R_val)
        log_norm_c = -0.5 * (log_det_R + tf.math.log(2.0 * 3.141592653589793))
        init_var = tf.constant(5.0, tf.float32)

        for i_seed in range(cfg.n_seeds_per_theta):
            seed = 1000 * i_theta + i_seed
            tf.random.set_seed(seed)

            particles = tf.random.normal([N]) * tf.sqrt(init_var)
            P = tf.fill([N], init_var)
            log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

            for t_int in range(1, T + 1):
                t_f = tf.cast(t_int, tf.float32)

                # -- Predict --
                if t_int >= 2:
                    x_det = (0.5 * particles
                             + 25.0 * particles / (1.0 + particles ** 2)
                             + 8.0 * tf.cos(1.2 * t_f))
                    particles = x_det + tf.random.normal([N]) * tf.sqrt(Q_val)
                    F = (0.5 + 25.0 * (1.0 - particles ** 2)
                         / tf.maximum((1.0 + particles ** 2) ** 2, _EPS))
                    P = tf.clip_by_value(F ** 2 * P + Q_val, _EPS, 1e4)

                # -- Simplified LEDH flow (5 steps, geometric schedule) --
                n_lam = 5
                q_geo = 1.2
                eps1 = (1.0 - q_geo) / (1.0 - q_geo ** n_lam)
                lam_cum = 0.0
                for j in range(n_lam):
                    eps_j = eps1 * q_geo ** j
                    lam_k = lam_cum + eps_j / 2.0
                    lam_cum += eps_j
                    H = particles / 10.0
                    h_eta = particles ** 2 / 20.0
                    e_lam = h_eta - H * particles
                    S = tf.maximum(lam_k * H ** 2 * P + R_val, _EPS)
                    A = tf.clip_by_value(-0.5 * P * H ** 2 / S, -10.0, 0.0)
                    I_lam_A = 1.0 + lam_k * A
                    I_2lam_A = 1.0 + 2.0 * lam_k * A
                    innov = tf.clip_by_value(
                        y_obs_2d[t_int - 1, 0] - e_lam, -100.0, 100.0
                    )
                    b_vec = I_2lam_A * (
                        I_lam_A * P * H * R_inv * innov + A * particles
                    )
                    b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)
                    vel = tf.clip_by_value(A * particles + b_vec, -50.0, 50.0)
                    particles = particles + eps_j * vel
                    particles = tf.where(
                        tf.math.is_finite(particles), particles,
                        tf.zeros_like(particles),
                    )
                    particles = tf.clip_by_value(particles, -1e4, 1e4)

                # -- Weight --
                y_pred = particles ** 2 / 20.0
                resid = y_obs_2d[t_int - 1, 0] - y_pred
                log_lik = -0.5 * R_inv * resid ** 2 + log_norm_c
                log_lik = tf.where(
                    tf.math.is_finite(log_lik), log_lik,
                    tf.constant(-100.0, tf.float32),
                )
                log_w = log_w + log_lik
                log_w = log_w - tf.reduce_logsumexp(log_w)

                # -- Record pre-resample state --
                w_norm = tf.nn.softmax(log_w, axis=0)
                ess = _compute_ess(w_norm)

                p_2d = particles[:, tf.newaxis]
                p_mean = tf.reduce_mean(p_2d, axis=0, keepdims=True)
                p_std = tf.math.reduce_std(p_2d, axis=0, keepdims=True) + _EPS
                p_norm_2d = (p_2d - p_mean) / p_std

                # -- Sinkhorn target --
                target_norm, _ = det_resample(
                    p_norm_2d, log_w,
                    epsilon=eps,
                    n_iters=cfg.sinkhorn_iters,
                )
                target_norm = tf.cast(tf.math.real(target_norm), tf.float32)

                all_particles.append(p_norm_2d[:, 0])
                all_weights.append(w_norm)
                all_targets.append(target_norm[:, 0])
                all_ctx.append(
                    tf.stack(
                        [
                            log_sv2_vals[i_theta],
                            log_sw2_vals[i_theta],
                            tf.cast(t_int / cfg.T_max, tf.float32),
                            y_obs_2d[t_int - 1, 0],
                            ess,
                            tf.cast(eps, tf.float32),
                        ]
                    )
                )

                # -- Sinkhorn resample for continuing the filter --
                p_resampled_norm, w_u = det_resample(
                    p_norm_2d, log_w, epsilon=eps, n_iters=30,
                )
                p_resampled_norm = tf.cast(
                    tf.math.real(p_resampled_norm), tf.float32
                )
                particles = (p_resampled_norm * p_std + p_mean)[:, 0]
                log_w = tf.math.log(
                    tf.cast(tf.math.real(w_u), tf.float32) + 1e-20
                )
                P_mean = tf.reduce_mean(P)
                P = tf.fill([N], P_mean)

    return {
        "particles_norm":  tf.stack(all_particles),
        "weights":         tf.stack(all_weights),
        "targets_norm":    tf.stack(all_targets),
        "context_scalars": tf.stack(all_ctx),
    }


def _generate_synthetic_data(cfg: TrainingConfig, n_examples: int = 5000):
    """Generate synthetic (random) training data without running LEDH.

    Each example is a random weighted particle set with its Sinkhorn
    resampled output as the target.

    Returns tensors (no NumPy) under keys ``particles_norm``, ``weights``,
    ``targets_norm``, ``context_scalars``.
    """
    N = cfg.num_particles
    eps = cfg.sinkhorn_epsilon

    all_p, all_w, all_t, all_c = [], [], [], []

    for i in trange(n_examples, desc="  Synthetic Data"):
        tf.random.set_seed(i)
        # Random particles: mixture of 1–3 Gaussians (TF + TFP)
        n_comp = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        mus = tf.random.normal([n_comp]) * 3.0
        sigs = -tf.math.log(tf.random.uniform([n_comp], dtype=tf.float32)) * 1.5 + 0.3
        if n_comp == 1:
            mix = tf.ones([1], dtype=tf.float32)
        else:
            mix = tfd.Dirichlet(tf.ones([n_comp], dtype=tf.float32)).sample()
        comps = tfd.Categorical(probs=mix).sample(N)
        mus_c = tf.gather(mus, comps)
        sigs_c = tf.gather(sigs, comps)
        particles = mus_c + tf.random.normal([N]) * sigs_c

        alpha_dir = -tf.math.log(tf.random.uniform([], dtype=tf.float32)) * 2.0 + 0.5
        weights = tfd.Dirichlet(tf.fill([N], alpha_dir)).sample()
        log_w = tf.math.log(weights + 1e-20)

        # Normalise particles
        p_2d = particles[:, tf.newaxis]
        p_mean = tf.reduce_mean(p_2d, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(p_2d, axis=0, keepdims=True) + _EPS
        p_norm_2d = (p_2d - p_mean) / p_std

        # Sinkhorn target
        target_norm, _ = det_resample(
            p_norm_2d, log_w, epsilon=eps, n_iters=cfg.sinkhorn_iters,
        )
        target_norm = tf.cast(tf.math.real(target_norm), tf.float32)

        ess = _compute_ess(weights)
        ctx = tf.stack(
            [
                tf.random.uniform(
                    [],
                    float(cfg.log_sv2_range[0]),
                    float(cfg.log_sv2_range[1]),
                    dtype=tf.float32,
                ),
                tf.random.uniform(
                    [],
                    float(cfg.log_sw2_range[0]),
                    float(cfg.log_sw2_range[1]),
                    dtype=tf.float32,
                ),
                tf.random.uniform([], 0.0, 1.0, dtype=tf.float32),
                tf.random.normal([]) * 5.0,
                ess,
                tf.cast(eps, tf.float32),
            ]
        )

        all_p.append(p_norm_2d[:, 0])
        all_w.append(weights)
        all_t.append(target_norm[:, 0])
        all_c.append(ctx)

    return {
        "particles_norm":  tf.stack(all_p),
        "weights":         tf.stack(all_w),
        "targets_norm":    tf.stack(all_t),
        "context_scalars": tf.stack(all_c),
    }


# ---------------------------------------------------------------------------
# Data generation without Sinkhorn targets (for Monge-Ampère loss)
# ---------------------------------------------------------------------------

def _generate_filter_data_ma(
    cfg: TrainingConfig,
    y_obs: tf.Tensor,
) -> dict:
    """Generate training data for Monge-Ampère loss — no Sinkhorn targets.

    Identical filter loop to :func:`_generate_filter_data` but skips all
    Sinkhorn calls.  Resampling between timesteps uses fast multinomial
    sampling instead, so data generation is roughly ``sinkhorn_iters`` times
    faster.

    Returns dict with keys (each value is a ``tf.Tensor``):
        particles_norm  (M, N)   normalised particle positions (pre-resample)
        weights         (M, N)   normalised weights (pre-resample)
        context_scalars (M, 6)   [log_sv2, log_sw2, t/T, y_t, ESS, eps]
    """
    N = cfg.num_particles
    T = int(y_obs.shape[0])
    if len(y_obs.shape) == 1:
        y_obs_2d = y_obs[:, tf.newaxis]
    else:
        y_obs_2d = y_obs
    eps = cfg.sinkhorn_epsilon

    tf.random.set_seed(42)
    log_sv2_vals = tf.random.uniform(
        [cfg.n_theta_samples],
        float(cfg.log_sv2_range[0]),
        float(cfg.log_sv2_range[1]),
        dtype=tf.float32,
    )
    log_sw2_vals = tf.random.uniform(
        [cfg.n_theta_samples],
        float(cfg.log_sw2_range[0]),
        float(cfg.log_sw2_range[1]),
        dtype=tf.float32,
    )

    all_particles, all_weights, all_ctx = [], [], []

    for i_theta in trange(cfg.n_theta_samples, desc="  Filter Data (MA, no Sinkhorn)"):
        Q_val = tf.maximum(
            tf.exp(log_sv2_vals[i_theta]), tf.cast(_EPS, tf.float32)
        )
        R_val = tf.maximum(
            tf.exp(log_sw2_vals[i_theta]), tf.cast(_EPS, tf.float32)
        )
        R_inv = 1.0 / R_val
        log_det_R = tf.math.log(R_val)
        log_norm_c = -0.5 * (log_det_R + tf.math.log(2.0 * 3.141592653589793))
        init_var = tf.constant(5.0, tf.float32)

        for i_seed in range(cfg.n_seeds_per_theta):
            seed = 1000 * i_theta + i_seed
            tf.random.set_seed(seed)

            particles = tf.random.normal([N]) * tf.sqrt(init_var)
            P = tf.fill([N], init_var)
            log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

            for t_int in range(1, T + 1):
                t_f = tf.cast(t_int, tf.float32)

                # -- Predict --
                if t_int >= 2:
                    x_det = (0.5 * particles
                             + 25.0 * particles / (1.0 + particles ** 2)
                             + 8.0 * tf.cos(1.2 * t_f))
                    particles = x_det + tf.random.normal([N]) * tf.sqrt(Q_val)
                    F = (0.5 + 25.0 * (1.0 - particles ** 2)
                         / tf.maximum((1.0 + particles ** 2) ** 2, _EPS))
                    P = tf.clip_by_value(F ** 2 * P + Q_val, _EPS, 1e4)

                # -- Simplified LEDH flow (5 steps, geometric schedule) --
                n_lam = 5
                q_geo = 1.2
                eps1 = (1.0 - q_geo) / (1.0 - q_geo ** n_lam)
                lam_cum = 0.0
                for j in range(n_lam):
                    eps_j = eps1 * q_geo ** j
                    lam_k = lam_cum + eps_j / 2.0
                    lam_cum += eps_j
                    H = particles / 10.0
                    h_eta = particles ** 2 / 20.0
                    e_lam = h_eta - H * particles
                    S = tf.maximum(lam_k * H ** 2 * P + R_val, _EPS)
                    A = tf.clip_by_value(-0.5 * P * H ** 2 / S, -10.0, 0.0)
                    I_lam_A = 1.0 + lam_k * A
                    I_2lam_A = 1.0 + 2.0 * lam_k * A
                    innov = tf.clip_by_value(
                        y_obs_2d[t_int - 1, 0] - e_lam, -100.0, 100.0
                    )
                    b_vec = I_2lam_A * (
                        I_lam_A * P * H * R_inv * innov + A * particles
                    )
                    b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)
                    vel = tf.clip_by_value(A * particles + b_vec, -50.0, 50.0)
                    particles = particles + eps_j * vel
                    particles = tf.where(
                        tf.math.is_finite(particles), particles,
                        tf.zeros_like(particles),
                    )
                    particles = tf.clip_by_value(particles, -1e4, 1e4)

                # -- Weight --
                y_pred = particles ** 2 / 20.0
                resid = y_obs_2d[t_int - 1, 0] - y_pred
                log_lik = -0.5 * R_inv * resid ** 2 + log_norm_c
                log_lik = tf.where(
                    tf.math.is_finite(log_lik), log_lik,
                    tf.constant(-100.0, tf.float32),
                )
                log_w = log_w + log_lik
                log_w = log_w - tf.reduce_logsumexp(log_w)

                # -- Record pre-resample state --
                w_norm = tf.nn.softmax(log_w, axis=0)
                ess = _compute_ess(w_norm)

                p_2d = particles[:, tf.newaxis]
                p_mean = tf.reduce_mean(p_2d, axis=0, keepdims=True)
                p_std = tf.math.reduce_std(p_2d, axis=0, keepdims=True) + _EPS
                p_norm_2d = (p_2d - p_mean) / p_std

                all_particles.append(p_norm_2d[:, 0])
                all_weights.append(w_norm)
                all_ctx.append(
                    tf.stack(
                        [
                            log_sv2_vals[i_theta],
                            log_sw2_vals[i_theta],
                            tf.cast(t_int / cfg.T_max, tf.float32),
                            y_obs_2d[t_int - 1, 0],
                            ess,
                            tf.cast(eps, tf.float32),
                        ]
                    )
                )

                # -- Multinomial resampling (fast, no Sinkhorn) --
                norm_log_w = log_w - tf.reduce_logsumexp(log_w)
                indices = tf.random.categorical(norm_log_w[tf.newaxis, :], N)[0]
                particles = tf.gather(particles, tf.cast(indices, tf.int32))
                log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))
                P_mean = tf.reduce_mean(P)
                P = tf.fill([N], P_mean)

    return {
        "particles_norm":  tf.stack(all_particles),
        "weights":         tf.stack(all_weights),
        "context_scalars": tf.stack(all_ctx),
    }


def _generate_synthetic_data_ma(cfg: TrainingConfig, n_examples: int = 5000):
    """Synthetic data for Monge-Ampère loss — no Sinkhorn targets.

    Same random particle / weight generation as :func:`_generate_synthetic_data`
    but skips the Sinkhorn target computation entirely.

    Returns dict with keys (``tf.Tensor`` values):
        particles_norm  (M, N)
        weights         (M, N)
        context_scalars (M, 6)
    """
    N = cfg.num_particles
    eps = cfg.sinkhorn_epsilon

    all_p, all_w, all_c = [], [], []

    for i in trange(n_examples, desc="  Synthetic Data (MA, no Sinkhorn)"):
        tf.random.set_seed(i)
        n_comp = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        mus = tf.random.normal([n_comp]) * 3.0
        sigs = -tf.math.log(tf.random.uniform([n_comp], dtype=tf.float32)) * 1.5 + 0.3
        if n_comp == 1:
            mix = tf.ones([1], dtype=tf.float32)
        else:
            mix = tfd.Dirichlet(tf.ones([n_comp], dtype=tf.float32)).sample()
        comps = tfd.Categorical(probs=mix).sample(N)
        mus_c = tf.gather(mus, comps)
        sigs_c = tf.gather(sigs, comps)
        particles = mus_c + tf.random.normal([N]) * sigs_c

        alpha_dir = -tf.math.log(tf.random.uniform([], dtype=tf.float32)) * 2.0 + 0.5
        weights = tfd.Dirichlet(tf.fill([N], alpha_dir)).sample()

        p_2d = particles[:, tf.newaxis]
        p_mean = tf.reduce_mean(p_2d, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(p_2d, axis=0, keepdims=True) + _EPS
        p_norm_2d = (p_2d - p_mean) / p_std

        ess = _compute_ess(weights)
        ctx = tf.stack(
            [
                tf.random.uniform(
                    [],
                    float(cfg.log_sv2_range[0]),
                    float(cfg.log_sv2_range[1]),
                    dtype=tf.float32,
                ),
                tf.random.uniform(
                    [],
                    float(cfg.log_sw2_range[0]),
                    float(cfg.log_sw2_range[1]),
                    dtype=tf.float32,
                ),
                tf.random.uniform([], 0.0, 1.0, dtype=tf.float32),
                tf.random.normal([]) * 5.0,
                ess,
                tf.cast(eps, tf.float32),
            ]
        )

        all_p.append(p_norm_2d[:, 0])
        all_w.append(weights)
        all_c.append(ctx)

    return {
        "particles_norm":  tf.stack(all_p),
        "weights":         tf.stack(all_w),
        "context_scalars": tf.stack(all_c),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class NeuralOTDiagnostics:
    """Diagnostics from neural OT training."""
    final_loss: float
    best_loss: float
    epochs_trained: int
    training_examples: int
    sinkhorn_calls_saved_per_mcmc_iter: int  # T timesteps, each 1 Sinkhorn
    loss_mode: str = "supervised"            # "supervised", "monge_ampere", or "mixed"


class NeuralOTTrainer:
    """End-to-end trainer for the conditional mGradNet.

    Parameters
    ----------
    state_dim      : int   particle dimensionality (1 for Kitagawa SSM).
    num_particles  : int   number of particles N.
    n_ridges       : int   mGradNet width (number of ridge functions).
    d_set, d_scalar: int   latent widths for set / scalar encoders.
    """

    def __init__(
        self,
        state_dim: int = 1,
        num_particles: int = 50,
        n_ridges: int = 128,
        d_set: int = 64,
        d_scalar: int = 64,
    ):
        """Initialise trainer; see class docstring for parameter details."""
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.model = ConditionalMGradNet(
            state_dim=state_dim,
            n_ridges=n_ridges,
            d_set=d_set,
            d_scalar=d_scalar,
            n_scalar_ctx=6,
        )

    def train(
        self,
        y_obs: tf.Tensor,
        *,
        sinkhorn_epsilon: float = 2.0,
        epochs: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        data_mode: str = "filter",
        loss_mode: str = "supervised",
        ma_weight: float = 0.5,
        n_synthetic: int = 5000,
        n_theta_samples: int = 100,
        n_seeds_per_theta: int = 2,
        verbose: bool = True,
        shuffle_seed: int = 42,
    ) -> Tuple[ConditionalMGradNet, NeuralOTDiagnostics]:
        """Train the conditional mGradNet.

        Parameters
        ----------
        y_obs      : (T,) or (T,1) observations for filter-based data.
        data_mode  : ``"filter"`` (run LEDH to collect data) or
                     ``"synthetic"`` (random particles/weights).
        loss_mode  : Training objective:

                     ``"supervised"`` (default, Option A)
                         MSE against Sinkhorn solutions.  Requires Sinkhorn
                         targets; data generation is slower but the training
                         signal is unambiguous.

                     ``"monge_ampere"`` (Option B)
                         Self-supervised Monge-Ampère residual loss.
                         No Sinkhorn targets needed — data generation is
                         ~``sinkhorn_iters`` times faster.  The loss enforces
                         the change-of-variables identity::

                             log|det J_T(x_i)| = log(N · w_i)

                         which is the discrete form of the Monge-Ampère
                         equation for mapping weighted particles to a uniform
                         distribution.

                     ``"mixed"`` (Option A + B)
                         Weighted combination::

                             L = (1 - ma_weight) * L_supervised
                               +      ma_weight  * L_monge_ampere

                         Requires Sinkhorn targets (same data as
                         ``"supervised"``).  Use for fine-tuning a pre-trained
                         supervised model to improve OT accuracy.

        ma_weight  : Weight of the Monge-Ampère term in ``"mixed"`` mode.
                     Ignored for other modes.  Default 0.5.
        epochs     : Training epochs.
        shuffle_seed
            Passed to ``tf.data.Dataset.shuffle`` for reproducible batch order
            (uses ``reshuffle_each_iteration=False``).

        Returns
        -------
        model, diagnostics
        """
        if loss_mode not in ("supervised", "monge_ampere", "mixed"):
            raise ValueError(
                f"loss_mode must be 'supervised', 'monge_ampere', or 'mixed'; "
                f"got '{loss_mode}'"
            )

        cfg = TrainingConfig(
            num_particles=self.num_particles,
            state_dim=self.state_dim,
            sinkhorn_epsilon=sinkhorn_epsilon,
            n_theta_samples=n_theta_samples,
            n_seeds_per_theta=n_seeds_per_theta,
        )

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Neural OT Training (mGradNet)")
            print(f"  data={data_mode}  loss={loss_mode}  N={self.num_particles}  "
                  f"K={self.model.n_ridges}  eps={sinkhorn_epsilon}")
            if loss_mode == "mixed":
                print(f"  ma_weight={ma_weight:.2f}  "
                      f"supervised_weight={1.0 - ma_weight:.2f}")
            print(f"{'=' * 60}")

        # ---- Generate training data ----
        # Monge-Ampère mode skips Sinkhorn entirely — much faster.
        # Supervised and mixed modes need Sinkhorn targets.
        needs_targets = (loss_mode in ("supervised", "mixed"))

        if verbose:
            if needs_targets:
                print("  Generating training data (with Sinkhorn targets) ...")
            else:
                print("  Generating training data (no Sinkhorn — fast) ...")

        if needs_targets:
            if data_mode == "filter":
                data = _generate_filter_data(cfg, y_obs)
            else:
                data = _generate_synthetic_data(cfg, n_examples=n_synthetic)
        else:
            if data_mode == "filter":
                data = _generate_filter_data_ma(cfg, y_obs)
            else:
                data = _generate_synthetic_data_ma(cfg, n_examples=n_synthetic)

        pn = data["particles_norm"]
        M = int(pn.shape[0]) if pn.shape[0] is not None else int(
            tf.shape(pn)[0].numpy()
        )
        if verbose:
            print(f"  Generated {M} training examples")

        # ---- Build dataset ----
        _buf = min(M, 10000)
        _shuffle_kw = dict(seed=shuffle_seed, reshuffle_each_iteration=False)
        if needs_targets:
            ds = tf.data.Dataset.from_tensor_slices((
                data["particles_norm"],   # (M, N)
                data["weights"],          # (M, N)
                data["targets_norm"],     # (M, N)
                data["context_scalars"],  # (M, 6)
            )).shuffle(_buf, **_shuffle_kw).batch(batch_size).prefetch(2)
        else:
            ds = tf.data.Dataset.from_tensor_slices((
                data["particles_norm"],   # (M, N)
                data["weights"],          # (M, N)
                data["context_scalars"],  # (M, 6)
            )).shuffle(_buf, **_shuffle_kw).batch(batch_size).prefetch(2)

        # ---- Training step functions ----
        optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

        @tf.function(jit_compile=True)
        def _train_step_supervised(p_batch, w_batch, c_batch, t_batch):
            """Option A: MSE against Sinkhorn targets."""
            with tf.GradientTape() as tape:
                pred = self.model(p_batch, w_batch, c_batch)
                loss = tf.reduce_mean((pred - t_batch) ** 2)
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads = [tf.clip_by_norm(g, 5.0) if g is not None else g
                     for g in grads]
            optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        @tf.function(jit_compile=True)
        def _train_step_ma(p_batch, w_batch, c_batch):
            """Option B: Monge-Ampère residual loss (no targets needed).

            Enforces the discrete change-of-variables identity:
                log|det J_T(x_i)| = log(N · w_i)

            The expectation E_{x~p}[...] is approximated by the weighted sum
            Σ_i w_i (residual_i)^2, which is the unbiased estimator under the
            source measure p = Σ_i w_i δ(x_i).
            """
            with tf.GradientTape() as tape:
                # log|det J_T(x_i)| for each particle i, shape (B, N)
                log_det = self.model.log_det_jacobian(p_batch, w_batch, c_batch)
                # Target: map must expand volume by factor N·w_i
                # (converts weight w_i to uniform 1/N)
                N_f = tf.cast(tf.shape(p_batch)[1], tf.float32)
                log_N_w = tf.math.log(N_f * w_batch + 1e-20)  # (B, N)
                residual = log_det - log_N_w                   # (B, N)
                # Weighted MSE under the source measure
                loss = tf.reduce_mean(
                    tf.reduce_sum(w_batch * tf.square(residual), axis=1)
                )
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads = [tf.clip_by_norm(g, 5.0) if g is not None else g
                     for g in grads]
            optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        # Capture ma_weight as a TF constant so the mixed step is JIT-safe.
        _ma_w = tf.constant(ma_weight, dtype=tf.float32)

        @tf.function(jit_compile=True)
        def _train_step_mixed(p_batch, w_batch, c_batch, t_batch):
            """Option A + B: weighted combination of supervised and MA losses.

            The context encoding is computed twice (once inside model() and
            once inside log_det_jacobian()).  This is a known trade-off; an
            optimised implementation would fuse the two forward passes.
            """
            with tf.GradientTape() as tape:
                # Option A term
                pred = self.model(p_batch, w_batch, c_batch)
                loss_a = tf.reduce_mean((pred - t_batch) ** 2)
                # Option B term
                log_det = self.model.log_det_jacobian(p_batch, w_batch, c_batch)
                N_f = tf.cast(tf.shape(p_batch)[1], tf.float32)
                log_N_w = tf.math.log(N_f * w_batch + 1e-20)
                residual = log_det - log_N_w
                loss_b = tf.reduce_mean(
                    tf.reduce_sum(w_batch * tf.square(residual), axis=1)
                )
                loss = (1.0 - _ma_w) * loss_a + _ma_w * loss_b
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads = [tf.clip_by_norm(g, 5.0) if g is not None else g
                     for g in grads]
            optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss, loss_a, loss_b

        # ---- Training loop ----
        best_loss = float("inf")
        final_loss = float("inf")

        if verbose:
            try:
                from tqdm import trange
            except ImportError:
                trange = range
            pbar = trange(epochs)
            if hasattr(pbar, "set_description"):
                pbar.set_description(f"  Training ({loss_mode})")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            epoch_loss = 0.0
            n_batches = 0

            if loss_mode == "monge_ampere":
                for p_batch, w_batch, c_batch in ds:
                    loss = _train_step_ma(p_batch, w_batch, c_batch)
                    epoch_loss += float(loss)
                    n_batches += 1
            elif loss_mode == "mixed":
                for p_batch, w_batch, t_batch, c_batch in ds:
                    loss, _, _ = _train_step_mixed(
                        p_batch, w_batch, c_batch, t_batch
                    )
                    epoch_loss += float(loss)
                    n_batches += 1
            else:  # supervised
                for p_batch, w_batch, t_batch, c_batch in ds:
                    loss = _train_step_supervised(
                        p_batch, w_batch, c_batch, t_batch
                    )
                    epoch_loss += float(loss)
                    n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss
            if avg_loss < best_loss:
                best_loss = avg_loss

            if verbose:
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(loss=f"{avg_loss:.6f}", best=f"{best_loss:.6f}")
                elif epoch % 200 == 0 or epoch == 1:
                    print(f"    epoch {epoch:>5d}/{epochs}  "
                          f"loss={avg_loss:.6f}  best={best_loss:.6f}")

        T_obs = int(y_obs.shape[0]) if y_obs is not None else 50
        diag = NeuralOTDiagnostics(
            final_loss=final_loss,
            best_loss=best_loss,
            epochs_trained=epochs,
            training_examples=M,
            sinkhorn_calls_saved_per_mcmc_iter=T_obs,
            loss_mode=loss_mode,
        )

        if verbose:
            loss_label = {"supervised": "MSE", "monge_ampere": "MA residual",
                          "mixed": "combined"}[loss_mode]
            print(f"\n  Training done.  Best {loss_label} = {best_loss:.6f}")
            if loss_mode != "supervised":
                print(f"  (Sinkhorn targets not needed — faster data generation)")
            print(f"  Each MCMC iteration saves ~{T_obs} Sinkhorn calls")
            print(f"{'=' * 60}\n")

        return self.model, diag
