"""
Conditional Monotone Gradient Network for Neural Optimal Transport.

Implements the conditional mGradNet architecture for amortised neural OT
resampling in differentiable particle filters.  Based on:

  - GradNetOT (arXiv:2507.13191): learning OT maps via mGradNets
  - Gradient Networks (Chaudhari et al., IEEE TSP 2025): mGradNet-M arch.
  - Brenier's theorem: OT map = gradient of convex function => monotone

The network directly parameterises a monotone gradient transport map
T = nabla phi (phi convex), conditioned on (theta, t, y_t, ESS, epsilon)
so that a single trained model generalises across all MCMC iterations and
particle-filter timesteps.

Architecture
------------
1. **DeepSetEncoder** — permutation-invariant summary of (particles, weights).
2. **ScalarContextEncoder** — MLP over (theta, t, y_t, ESS, epsilon).
3. **ConditionalMGradNet** — mGradNet-M whose ridge parameters are modulated
   by FiLM layers (Feature-wise Linear Modulation) driven by the joint
   context.  The PSD Jacobian guarantee is preserved because:
     * softplus constrains all scale factors to be non-negative,
     * the mGradNet-M residual form ensures J_T = aI + sum PSD.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf


# ---------------------------------------------------------------------------
# DeepSet encoder
# ---------------------------------------------------------------------------

class DeepSetEncoder(tf.keras.layers.Layer):
    """Permutation-invariant encoder for weighted particle sets.

    Uses the DeepSet architecture:  rho( sum_i w_i * phi(x_i, w_i) )
    to produce a fixed-size summary regardless of particle ordering.
    """

    def __init__(
        self,
        state_dim: int = 1,
        d_hidden: int = 64,
        d_output: int = 64,
        **kwargs,
    ):
        """Initialise phi and rho sub-networks for the DeepSet encoder."""
        super().__init__(**kwargs)
        self.state_dim = state_dim

        # phi: per-particle transform
        self.phi_net = tf.keras.Sequential([
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(d_hidden),
        ])

        # rho: post-aggregation transform
        self.rho_net = tf.keras.Sequential([
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(d_output),
        ])

    def call(self, particles: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """
        Args
        ----
        particles : (B, N, d) or (N, d)
        weights   : (B, N) or (N,)

        Returns
        -------
        (B, d_output) or (d_output,)
        """
        single = len(weights.shape) == 1
        if single:
            particles = particles[tf.newaxis, ...]
            weights = weights[tf.newaxis, :]

        if len(particles.shape) == 2:
            particles = particles[:, :, tf.newaxis]

        # Concat position and weight: (B, N, d+1)
        pw = tf.concat([particles, weights[:, :, tf.newaxis]], axis=-1)

        h = self.phi_net(pw)  # (B, N, d_hidden)

        # Weighted-sum pooling
        h_weighted = h * weights[:, :, tf.newaxis]
        pooled = tf.reduce_sum(h_weighted, axis=1)  # (B, d_hidden)

        out = self.rho_net(pooled)  # (B, d_output)
        return out[0] if single else out


# ---------------------------------------------------------------------------
# Conditional mGradNet-M
# ---------------------------------------------------------------------------

class ConditionalMGradNet(tf.keras.Model):
    r"""Conditional Monotone Gradient Network for neural OT resampling.

    Implements::

        T(x | c) = a(c)*x + offset(c)
                   + sum_k  splus(alpha_k) * splus(gamma_k(c))
                            * sigmoid(w_k^T x + b_k + delta_k(c)) * w_k

    where ``splus`` = softplus, ensuring  dT/dx >= a(c) > 0  (monotone).

    Parameters
    ----------
    state_dim : int
        Particle dimensionality (1 for the Kitagawa SSM).
    n_ridges : int
        Number of ridge functions *K* in the mGradNet-M layer.
    d_set, d_scalar : int
        Latent widths for set / scalar context encoders.
    n_scalar_ctx : int
        Length of the scalar context vector [theta, t, y_t, ESS, eps].
    """

    def __init__(
        self,
        state_dim: int = 1,
        n_ridges: int = 128,
        d_set: int = 64,
        d_scalar: int = 64,
        n_scalar_ctx: int = 6,
        **kwargs,
    ):
        """Initialise ridge-function network with context encoders."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.n_ridges = n_ridges

        # ---- encoders ----
        self.set_encoder = DeepSetEncoder(state_dim, d_hidden=d_set, d_output=d_set)

        self.scalar_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(d_scalar, activation="gelu"),
            tf.keras.layers.Dense(d_scalar, activation="gelu"),
        ])

        d_joint_in = d_set + d_scalar
        self.joint_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="gelu"),
            tf.keras.layers.Dense(128, activation="gelu"),
        ])

        # ---- mGradNet-M learnable parameters ----
        self.ridge_w = self.add_weight(
            name="ridge_w", shape=(n_ridges, state_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.5),
        )
        self.ridge_b = self.add_weight(
            name="ridge_b", shape=(n_ridges,),
            initializer="zeros",
        )
        self.alpha_raw = self.add_weight(
            name="alpha_raw", shape=(n_ridges,),
            initializer=tf.keras.initializers.Constant(0.1),
        )

        # ---- FiLM generators from joint context ----
        self.film_gamma = tf.keras.layers.Dense(n_ridges, name="film_gamma")
        self.film_delta = tf.keras.layers.Dense(n_ridges, name="film_delta")
        self.film_a_raw = tf.keras.layers.Dense(state_dim, name="film_a")
        self.film_offset = tf.keras.layers.Dense(state_dim, name="film_offset")

    # ------------------------------------------------------------------ #
    #  Forward pass                                                       #
    # ------------------------------------------------------------------ #

    def call(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        context_scalars: tf.Tensor,
    ) -> tf.Tensor:
        """
        Args
        ----
        particles       : (B, N, d) | (N, d) | (B, N) | (N,) for d=1
        weights         : (B, N) | (N,)  — normalised (sum=1)
        context_scalars : (B, n_ctx) | (n_ctx,)

        Returns
        -------
        Resampled particles, same leading shape as *particles*.
        """
        single = len(weights.shape) == 1
        if single:
            particles = particles[tf.newaxis, ...]
            weights = weights[tf.newaxis, :]
            context_scalars = context_scalars[tf.newaxis, :]

        # Ensure (B, N, d)
        squeeze_last = False
        if len(particles.shape) == 2:
            particles = particles[:, :, tf.newaxis]
            squeeze_last = True

        # ---- context encoding ----
        set_ctx = self.set_encoder(particles, weights)     # (B, d_set)
        scl_ctx = self.scalar_encoder(context_scalars)     # (B, d_scalar)
        joint = self.joint_encoder(
            tf.concat([set_ctx, scl_ctx], axis=-1)
        )  # (B, 128)

        # ---- FiLM parameters ----
        gamma = tf.nn.softplus(self.film_gamma(joint))     # (B, K) >= 0
        delta = self.film_delta(joint)                     # (B, K)
        a = tf.nn.softplus(self.film_a_raw(joint))         # (B, d)  > 0
        offset = self.film_offset(joint)                   # (B, d)

        alpha = tf.nn.softplus(self.alpha_raw)             # (K,)

        # ---- mGradNet-M forward (vectorised over B, N) ----
        # wx = <w_k, x_n> for all k, n:  (B, N, K)
        w = self.ridge_w                                   # (K, d)
        wx = tf.einsum("kd,bnd->bnk", w, particles)       # (B, N, K)
        bias = self.ridge_b[tf.newaxis, tf.newaxis, :]     # (1, 1, K)
        delta_3d = delta[:, tf.newaxis, :]                 # (B, 1, K)

        h = tf.sigmoid(wx + bias + delta_3d)               # (B, N, K)

        # coeff[b, k] = alpha[k] * gamma[b, k]
        coeff = alpha[tf.newaxis, :] * gamma               # (B, K)

        # residual_n = sum_k coeff[b,k] * h[b,n,k] * w_k  -> (B, N, d)
        weighted_h = coeff[:, tf.newaxis, :] * h           # (B, N, K)
        residual = tf.einsum("bnk,kd->bnd", weighted_h, w)  # (B, N, d)

        # T(x) = diag(a)*x + offset + residual
        output = (particles * a[:, tf.newaxis, :] +
                  offset[:, tf.newaxis, :] +
                  residual)

        if squeeze_last:
            output = output[:, :, 0]
        if single:
            output = output[0]
        return output

    # ------------------------------------------------------------------ #
    #  Log-|det Jacobian| (for Monge-Ampère loss or importance correction)#
    # ------------------------------------------------------------------ #

    def log_det_jacobian(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        context_scalars: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute log |det J_T(x)| for each particle.

        For d=1 this is ``log(dT/dx)``; for general d the full Jacobian
        determinant is computed.

        Returns shape (B, N) or (N,).
        """
        single = len(weights.shape) == 1
        if single:
            particles = particles[tf.newaxis, ...]
            weights = weights[tf.newaxis, :]
            context_scalars = context_scalars[tf.newaxis, :]

        squeeze_last = False
        if len(particles.shape) == 2:
            particles = particles[:, :, tf.newaxis]
            squeeze_last = True

        # ---- recompute context (cached in practice) ----
        set_ctx = self.set_encoder(particles, weights)
        scl_ctx = self.scalar_encoder(context_scalars)
        joint = self.joint_encoder(
            tf.concat([set_ctx, scl_ctx], axis=-1)
        )

        gamma = tf.nn.softplus(self.film_gamma(joint))
        delta = self.film_delta(joint)
        a = tf.nn.softplus(self.film_a_raw(joint))
        alpha = tf.nn.softplus(self.alpha_raw)

        w = self.ridge_w  # (K, d)
        wx = tf.einsum("kd,bnd->bnk", w, particles)
        bias = self.ridge_b[tf.newaxis, tf.newaxis, :]
        delta_3d = delta[:, tf.newaxis, :]
        h = tf.sigmoid(wx + bias + delta_3d)  # (B, N, K)

        if self.state_dim == 1:
            # dT/dx = a + sum_k alpha_k gamma_k h_k(1-h_k) w_k^2
            h_prime = h * (1.0 - h)                              # sigmoid'
            w_sq = tf.reduce_sum(w ** 2, axis=-1)                # (K,)
            coeff = alpha[tf.newaxis, :] * gamma                 # (B, K)
            deriv = a[:, 0:1] + tf.einsum(
                "bk,bnk,k->bn", coeff, h_prime, w_sq
            )
            return tf.squeeze(tf.math.log(tf.maximum(deriv, 1e-20)),
                              axis=0) if single else tf.math.log(
                tf.maximum(deriv, 1e-20)
            )

        # General d: J_T = diag(a) + sum_k alpha_k gamma_k h'_k w_k w_k^T
        # This is a rank-K update of a diagonal; use matrix_determinant_lemma
        # for efficiency if K < d, otherwise direct.
        d = self.state_dim
        B = tf.shape(particles)[0]
        N = tf.shape(particles)[1]
        h_prime = h * (1.0 - h)  # (B, N, K)
        coeff = alpha[tf.newaxis, :] * gamma  # (B, K)

        # Build J per (b, n): (B, N, d, d)
        diag_a = tf.linalg.diag(a)  # (B, d, d)
        diag_a = diag_a[:, tf.newaxis, :, :]  # (B, 1, d, d)

        # Outer products w_k w_k^T scaled by coeff * h'
        # scaled_w[b,n,k] = sqrt(coeff[b,k] * h'[b,n,k]) * w[k,:]
        scale = tf.sqrt(tf.maximum(coeff[:, tf.newaxis, :] * h_prime, 1e-20))
        scaled_w = scale[:, :, :, tf.newaxis] * w[tf.newaxis, tf.newaxis, :, :]
        # (B, N, K, d); J = diag(a) + sum_k sw sw^T = diag(a) + SW SW^T
        # log det via Cholesky
        J = diag_a + tf.einsum("bnki,bnkj->bnij", scaled_w, scaled_w)
        _, logdet = tf.linalg.slogdet(J)
        logdet = tf.cast(tf.math.real(logdet), tf.float32)
        return logdet[0] if single else logdet
