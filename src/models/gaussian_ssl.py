"""
Gaussian State-Space LSTM (SSL) Model — Zheng et al. (2017), Example 1.

Combines LSTM transition dynamics with Gaussian emission:

    Transition:  s_t = LSTM(s_{t-1}, z_{t-1})
                 z_t ~ N(g_mu(s_t), g_sigma(s_t))

    Emission:    x_t ~ N(C z_t + b, R)

The LSTM learns nonlinear, non-Markovian dynamics in latent space,
while the emission remains linear Gaussian for tractable forward messages.

This module provides:
  - GaussianSSL: The full model (LSTM + emission) with trainable parameters
  - GaussianSSLasSSM: Adapter wrapping GaussianSSL into the SSM interface
    required by DifferentiableLEDHLogLikelihood and the HMC/PMMH code
  - Data generation utilities

Particle Gibbs inference lives in ``src.filters.bonus.ssl_particle_gibbs``.

References
----------
- Zheng et al. "State Space LSTM Models with Particle MCMC Inference",
  arXiv:1711.11179, 2017
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import tensorflow as tf


# =========================================================================
# Gaussian SSL Model
# =========================================================================

class GaussianSSL(tf.keras.Model):
    """Gaussian State-Space LSTM model.

    Parameters
    ----------
    state_dim : int
        Latent state dimension (z_t).
    obs_dim : int
        Observation dimension (x_t).
    lstm_units : int
        Number of LSTM hidden units.
    """

    def __init__(
        self,
        state_dim: int = 2,
        obs_dim: int = 2,
        lstm_units: int = 32,
        **kwargs,
    ):
        """Construct LSTM transition, g_mu / g_log_sigma heads, and emission (C, b, R_diag)."""
        super().__init__(**kwargs)
        self.state_dim_val = state_dim
        self.obs_dim_val = obs_dim
        self.lstm_units = lstm_units

        # LSTM transition in latent space
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_units)

        # g_mu, g_sigma: LSTM state -> transition distribution params
        self.g_mu = tf.keras.layers.Dense(state_dim, name="g_mu")
        self.g_log_sigma = tf.keras.layers.Dense(
            state_dim, name="g_log_sigma",
            bias_initializer=tf.keras.initializers.Constant(-1.0),
        )

        # Emission: x_t = C z_t + b + noise(R)
        self.C = tf.Variable(
            tf.eye(obs_dim, state_dim, dtype=tf.float32),
            name="C", trainable=True,
        )
        self.b = tf.Variable(
            tf.zeros([obs_dim], dtype=tf.float32),
            name="b_emission", trainable=True,
        )
        # Log-diagonal of R (emission noise covariance, diagonal)
        self.log_R_diag = tf.Variable(
            tf.zeros([obs_dim], dtype=tf.float32),
            name="log_R_diag", trainable=True,
        )

    @property
    def R_matrix(self) -> tf.Tensor:
        """Emission noise covariance (diagonal)."""
        return tf.linalg.diag(tf.exp(self.log_R_diag))

    @property
    def R_diag(self) -> tf.Tensor:
        """Diagonal entries of R, shape ``(obs_dim,)`` (``exp(log_R_diag)``)."""
        return tf.exp(self.log_R_diag)

    def get_initial_lstm_state(self, batch_size: int = 1):
        """Return zero initial LSTM state."""
        return self.lstm_cell.get_initial_state(batch_size=batch_size)

    def transition_params(
        self, lstm_state, z_prev: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tuple]:
        """One-step LSTM transition: compute p(z_t | z_{1:t-1}).

        Args
        ----
        lstm_state : tuple of (h, c) from LSTMCell
        z_prev     : (batch, state_dim) previous latent state

        Returns
        -------
        mu    : (batch, state_dim) transition mean
        sigma : (batch, state_dim) transition std (positive)
        new_lstm_state : updated (h, c)
        """
        lstm_out, new_state = self.lstm_cell(z_prev, lstm_state)
        mu = self.g_mu(lstm_out)
        log_sigma = self.g_log_sigma(lstm_out)
        sigma = tf.exp(tf.clip_by_value(log_sigma, -5.0, 3.0))
        return mu, sigma, new_state

    def emission_log_prob(self, z: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """log p(x_t | z_t) = log N(x_t; C z_t + b, R).

        Args
        ----
        z : (batch, state_dim) or (state_dim,)
        x : (batch, obs_dim) or (obs_dim,)

        Returns
        -------
        (batch,) log-probabilities
        """
        if len(z.shape) == 1:
            z = z[tf.newaxis, :]
        if len(x.shape) == 1:
            x = x[tf.newaxis, :]

        mean = tf.matmul(z, self.C, transpose_b=True) + self.b  # (B, obs_dim)
        R_diag = self.R_diag  # (obs_dim,)

        # log N(x; mean, diag(R))
        diff = x - mean
        log_prob = -0.5 * tf.reduce_sum(
            diff ** 2 / R_diag + tf.math.log(R_diag) + tf.math.log(2.0 * math.pi),
            axis=-1,
        )
        return log_prob  # (B,)

    def transition_log_prob(
        self, z: tf.Tensor, mu: tf.Tensor, sigma: tf.Tensor
    ) -> tf.Tensor:
        """log p(z_t | z_{1:t-1}) = log N(z_t; mu, diag(sigma^2))."""
        diff = z - mu
        log_prob = -0.5 * tf.reduce_sum(
            diff ** 2 / (sigma ** 2) + 2.0 * tf.math.log(sigma)
            + tf.math.log(2.0 * math.pi),
            axis=-1,
        )
        return log_prob

    def forward_messages(
        self, lstm_state, z_prev: tf.Tensor, x_t: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tuple]:
        """Compute forward messages alpha_t and gamma_t (Zheng(17) Eq. 21).

        For Gaussian SSL with linear emission x = Cz + b + eps:
            alpha_t = N(x_t; C g_mu(s_t) + b, R + C diag(g_sigma^2) C^T)
            gamma_t = N(z_t; V(C^T R^{-1}(x-b) + Sigma^{-1} mu), V)
            where V = (Sigma^{-1} + C^T R^{-1} C)^{-1}

        Returns
        -------
        alpha     : (batch,) log marginal weight
        gamma_mu  : (batch, state_dim) posterior mean for z_t
        gamma_var : (batch, state_dim, state_dim) posterior covariance
        new_state : updated LSTM state
        """
        mu, sigma, new_state = self.transition_params(lstm_state, z_prev)
        # mu: (B, d_z), sigma: (B, d_z)

        d_z = self.state_dim_val
        d_x = self.obs_dim_val
        B = tf.shape(mu)[0]

        # Transition covariance Sigma = diag(sigma^2)
        Sigma_diag = sigma ** 2  # (B, d_z)

        C = self.C  # (d_x, d_z)
        b = self.b  # (d_x,)
        R_diag = self.R_diag  # (d_x,)

        # alpha_t: marginal likelihood p(x_t | z_{1:t-1})
        # = N(x_t; C mu + b, R + C Sigma C^T)
        pred_mean = tf.matmul(mu, C, transpose_b=True) + b  # (B, d_x)

        # Marginal covariance: R + C Sigma C^T (full matrix)
        # C Sigma C^T where Sigma is diagonal
        C_scaled = C[tf.newaxis, :, :] * tf.sqrt(Sigma_diag)[:, tf.newaxis, :]  # (B, d_x, d_z)
        marg_cov = tf.linalg.diag(
            tf.broadcast_to(R_diag[tf.newaxis, :], [B, d_x])
        ) + tf.matmul(C_scaled, C_scaled, transpose_b=True)  # (B, d_x, d_x)

        # log alpha = log N(x_t; pred_mean, marg_cov)
        diff_x = x_t - pred_mean  # (B, d_x)
        marg_cov_reg = marg_cov + tf.eye(d_x) * 1e-6
        _, log_det = tf.linalg.slogdet(marg_cov_reg)
        marg_inv = tf.linalg.inv(marg_cov_reg)
        mahal = tf.einsum("bi,bij,bj->b", diff_x, marg_inv, diff_x)
        alpha = -0.5 * (mahal + log_det + d_x * tf.math.log(2.0 * math.pi))

        # gamma_t: posterior p(z_t | z_{1:t-1}, x_t)
        # V = (Sigma^{-1} + C^T R^{-1} C)^{-1}
        Sigma_inv_diag = 1.0 / (Sigma_diag + 1e-8)  # (B, d_z)
        R_inv_diag = 1.0 / (R_diag + 1e-8)  # (d_x,)

        # C^T R^{-1} C: (d_z, d_z)
        CtRinvC = tf.matmul(
            C * R_inv_diag[tf.newaxis, :], C, transpose_a=True
        )  # (d_z, d_z), same for all batch

        # V^{-1} = diag(Sigma_inv) + C^T R^{-1} C
        V_inv = tf.linalg.diag(Sigma_inv_diag) + CtRinvC[tf.newaxis, :, :]  # (B, d_z, d_z)
        V_inv_reg = V_inv + tf.eye(d_z) * 1e-6
        V = tf.linalg.inv(V_inv_reg)  # (B, d_z, d_z)

        # gamma_mu = V (Sigma^{-1} mu + C^T R^{-1} (x - b))
        info_prior = Sigma_inv_diag * mu  # (B, d_z)
        info_lik = tf.matmul(
            (x_t - b) * R_inv_diag, C
        )  # (B, d_z)
        gamma_mu = tf.einsum("bij,bj->bi", V, info_prior + info_lik)  # (B, d_z)

        return alpha, gamma_mu, V, new_state


# =========================================================================
# SSM Adapter — makes GaussianSSL compatible with the filter interface
# =========================================================================

class GaussianSSLasSSM:
    """Wraps a GaussianSSL so it looks like PMCMCNonlinearSSM.

    The LEDH filter calls:
      - ssm.motion_model(state, control)     -> next state mean
      - ssm.measurement_model(state)         -> predicted observation
      - ssm.Q, ssm.R                         -> noise covariances
      - ssm.motion_jacobian(state, control)
      - ssm.measurement_jacobian(state)

    For the SSL, the transition is non-Markovian (depends on LSTM state).
    We handle this by running the LSTM forward in a pre-pass and storing
    per-timestep transition parameters (mu_t, sigma_t) that the filter
    uses as if they were time-varying but Markovian.

    This is a necessary approximation: the LEDH filter assumes Markovian
    dynamics, so we "freeze" the LSTM's predictions at each timestep.

    Parameters
    ----------
    ssl : GaussianSSL
        The trained SSL model.
    z_trajectory : tf.Tensor
        (T, state_dim) — reference latent trajectory from which LSTM
        states are computed. Typically from a previous PG or EM iteration.
    """

    def __init__(
        self,
        ssl: GaussianSSL,
        z_trajectory: tf.Tensor,
    ):
        """Unroll the LSTM on *z_trajectory* and cache per-time mus/sigmas; see class docstring."""
        self.ssl = ssl
        self.state_dim = ssl.state_dim_val
        self.meas_per_landmark = ssl.obs_dim_val

        # Run LSTM forward to get per-timestep transition params
        T = z_trajectory.shape[0]
        lstm_state = ssl.get_initial_lstm_state(batch_size=1)

        mus, sigmas = [], []
        z_prev = tf.zeros([1, self.state_dim])
        for t in range(T):
            mu, sigma, lstm_state = ssl.transition_params(lstm_state, z_prev)
            mus.append(mu[0])
            sigmas.append(sigma[0])
            z_prev = z_trajectory[t:t+1]

        self._mus = tf.stack(mus)      # (T, state_dim)
        self._sigmas = tf.stack(sigmas)  # (T, state_dim)

        # Use transition variance as process noise (time-varying, use mean)
        mean_var = tf.reduce_mean(self._sigmas ** 2, axis=0)
        self.Q = tf.linalg.diag(mean_var)
        self.R = ssl.R_matrix
        self.initial_var = 5.0

    def full_measurement_cov(self, num_landmarks=1):
        """Emission covariance R; *num_landmarks* ignored (single fixed R for SSL)."""
        return self.R

    def motion_model(self, state, control):
        """Uses pre-computed LSTM transition means."""
        state = tf.cast(state, tf.float32)
        control = tf.cast(control, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        if len(control.shape) == 1:
            control = control[tf.newaxis, :]

        # control encodes time index
        t_idx = tf.cast(tf.round(control[:, 0]), tf.int32)
        t_idx = tf.clip_by_value(t_idx, 0, tf.shape(self._mus)[0] - 1)

        # For LEDH: motion_model gives the deterministic prediction
        # The SSL transition mean at time t is f(z_{t-1}) ≈ mu_t
        # (the LSTM encodes the nonlinear mapping)
        mu_t = tf.gather(self._mus, t_idx)  # (batch, state_dim)
        return mu_t

    def measurement_model(self, state, landmarks=None):
        """Linear emission: x = C z + b."""
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        return tf.matmul(state, self.ssl.C, transpose_b=True) + self.ssl.b

    def motion_jacobian(self, state, control):
        """The LSTM transition doesn't have a simple Jacobian w.r.t. state.
        Return zero (transition mean is pre-computed, independent of input state)."""
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        B = tf.shape(state)[0]
        d = self.state_dim
        # Since motion_model returns a pre-computed mu (not a function of state),
        # the Jacobian is zero. LEDH uses this for linearization.
        return tf.zeros([B, d, d])

    def measurement_jacobian(self, state, landmarks=None):
        """d(Cz+b)/dz = C, constant."""
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        B = tf.shape(state)[0]
        C = self.ssl.C  # (obs_dim, state_dim)
        return tf.broadcast_to(C[tf.newaxis, :, :], [B, self.ssl.obs_dim_val, self.state_dim])


# =========================================================================
# Data Generation
# =========================================================================

def generate_ssl_data(
    T: int = 100,
    state_dim: int = 2,
    obs_dim: int = 2,
    dynamics: str = "sine",
    noise_std: float = 0.3,
    seed: int = 42,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Generate synthetic data for Gaussian SSL.

    Dynamics options matching Zheng(17) Section 5.1:
      - "line": straight line in 2D
      - "sine": sine wave in 2D
      - "circle": circular trajectory
      - "swiss_roll": swiss roll curve

    Returns
    -------
    z_true : (T, state_dim) true latent trajectory
    x_obs  : (T, obs_dim) noisy observations
    """
    tf.random.set_seed(seed)
    PI = math.pi
    t = tf.linspace(0.0, 4.0 * PI, T)

    if dynamics == "line":
        z = tf.stack([t / (4.0 * PI), t / (4.0 * PI)], axis=-1)
    elif dynamics == "sine":
        z = tf.stack([t / (4.0 * PI), tf.math.sin(t)], axis=-1)
    elif dynamics == "circle":
        z = tf.stack([tf.math.cos(t), tf.math.sin(t)], axis=-1)
    elif dynamics == "swiss_roll":
        r = t / (4.0 * PI) + 0.5
        z = tf.stack([r * tf.math.cos(t), r * tf.math.sin(t)], axis=-1)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics}")

    # Truncate or pad to state_dim
    if state_dim > 2:
        z = tf.concat([z, tf.zeros([T, state_dim - 2])], axis=-1)
    elif state_dim < 2:
        z = z[:, :state_dim]

    # Observations: linear emission with noise
    # C = I (identity), b = 0
    if obs_dim == state_dim:
        x = z + tf.random.normal([T, obs_dim]) * noise_std
    else:
        C_true = tf.random.normal([obs_dim, state_dim]) * 0.5
        min_dim = min(obs_dim, state_dim)
        # Set top-left block to identity
        eye_block = tf.eye(min_dim)
        # Pad to full shape and add to C_true
        eye_padded = tf.pad(eye_block, [
            [0, obs_dim - min_dim], [0, state_dim - min_dim]
        ])
        # Zero out the top-left block in C_true, replace with identity
        mask = tf.pad(tf.ones([min_dim, min_dim]), [
            [0, obs_dim - min_dim], [0, state_dim - min_dim]
        ])
        C_true = C_true * (1.0 - mask) + eye_padded
        x = tf.matmul(z, C_true, transpose_b=True) + tf.random.normal([T, obs_dim]) * noise_std

    return (
        tf.cast(z, tf.float32),
        tf.cast(x, tf.float32),
    )
