"""
Hyper-DeepONet: DeepONet with Branch-Produced Phase Modulation.

Addresses the expressiveness gap between DeepONet and FiLM-conditioned
mGradNet by having the branch net produce **bias shifts** delta_k for
the trunk's ridge activations — recovering FiLM's phase modulation
within the DeepONet framework.

Standard DeepONet (amplitude only):
    T(x) = a*x + offset + sum_k  splus(beta_k(p,c)) * sigma(w_k^T x + b_k) * w_k

Hyper-DeepONet (amplitude + phase):
    T(x) = a*x + offset + sum_k  splus(beta_k(p,c)) * sigma(w_k^T x + b_k + delta_k(p,c)) * w_k

The delta_k shifts are "hypernetwork" outputs: the branch net generates
parameters (biases) for the trunk net.  This lets the trunk's basis
functions adapt their activation locations to the input measure, not
just their amplitudes.

Why this matters
----------------
- In standard DeepONet, trunk basis functions sigma(w_k^T x + b_k) are
  fixed across all input measures — only the linear combination changes.
- FiLM (mGradNet) has delta_k(context) that shifts where each sigmoid
  activates: different inputs produce different basis shapes.
- Hyper-DeepONet recovers this expressiveness while maintaining the
  branch-trunk conceptual decomposition and PSD Jacobian guarantee.

PSD Jacobian is preserved because:
    J_T = diag(a) + sum_k splus(beta_k) * sigma'(w_k^T x + b_k + delta_k) * w_k w_k^T
    = diag(a) + sum of rank-1 PSD terms  (since splus >= 0, sigma' >= 0)

References
----------
- Lu et al. "Learning nonlinear operators via DeepONet", Nature MI, 2021
- Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf


# ---------------------------------------------------------------------------
# Branch Net with phase output
# ---------------------------------------------------------------------------

class HyperBranchNet(tf.keras.layers.Layer):
    """Branch net that produces both amplitude coefficients AND phase shifts.

    Outputs:
        beta_raw : (B, n_basis)  — amplitude coefficients (-> softplus for PSD)
        delta    : (B, n_basis)  — phase shifts for trunk activations

    This is the "hyper" part: the branch generates parameters for the trunk.
    """

    def __init__(
        self,
        d_hidden: int = 128,
        n_basis: int = 128,
        n_scalar_ctx: int = 6,
        **kwargs,
    ):
        """Initialise per-particle and post-pooling MLPs for hyper-branch."""
        super().__init__(**kwargs)
        self.n_basis = n_basis

        # phi: per-particle feature extractor (DeepSet inner)
        self.phi_net = tf.keras.Sequential([
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(d_hidden),
        ], name="branch_phi")

        # scalar context encoder
        self.scalar_enc = tf.keras.Sequential([
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
        ], name="branch_scalar_enc")

        # rho: post-pooling MLP producing joint encoding
        self.rho_net = tf.keras.Sequential([
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
        ], name="branch_rho")

        # Separate heads for amplitude and phase
        self.beta_head = tf.keras.layers.Dense(
            n_basis, name="beta_head"
        )
        self.delta_head = tf.keras.layers.Dense(
            n_basis, name="delta_head",
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        context_scalars: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns
        -------
        beta_raw : (B, n_basis)  raw amplitude coefficients
        delta    : (B, n_basis)  phase shifts for trunk biases
        """
        pw = tf.concat([particles, weights[:, :, tf.newaxis]], axis=-1)
        h = self.phi_net(pw)

        h_weighted = h * weights[:, :, tf.newaxis]
        pooled = tf.reduce_sum(h_weighted, axis=1)

        scl = self.scalar_enc(context_scalars)
        joint = tf.concat([pooled, scl], axis=-1)
        encoding = self.rho_net(joint)

        beta_raw = self.beta_head(encoding)
        delta = self.delta_head(encoding)

        return beta_raw, delta


# ---------------------------------------------------------------------------
# Trunk Net (same as standard DeepONet — ridge structure)
# ---------------------------------------------------------------------------

class TrunkNet(tf.keras.layers.Layer):
    """Ridge-function trunk with externally supplied bias shifts.

    Evaluates: h_k(x; delta_k) = sigma(w_k^T x + b_k + delta_k)
    """

    def __init__(
        self,
        state_dim: int = 1,
        n_basis: int = 128,
        **kwargs,
    ):
        """Initialise ridge-function trunk with external bias shift support."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.n_basis = n_basis

        self.ridge_w = self.add_weight(
            name="ridge_w", shape=(n_basis, state_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.5),
        )
        self.ridge_b = self.add_weight(
            name="ridge_b", shape=(n_basis,),
            initializer="zeros",
        )

    def call(
        self, x: tf.Tensor, delta: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args
        ----
        x     : (B, N, d) query positions
        delta : (B, n_basis) phase shifts from branch

        Returns
        -------
        h       : (B, N, n_basis) sigmoid activations with phase shift
        ridge_w : (n_basis, d)    ridge directions
        """
        wx = tf.einsum("kd,bnd->bnk", self.ridge_w, x)
        bias = self.ridge_b[tf.newaxis, tf.newaxis, :]
        delta_3d = delta[:, tf.newaxis, :]  # (B, 1, n_basis)

        h = tf.sigmoid(wx + bias + delta_3d)
        return h, self.ridge_w


# ---------------------------------------------------------------------------
# Hyper-DeepONet Monotone OT Model
# ---------------------------------------------------------------------------

class HyperDeepONetMonotoneOT(tf.keras.Model):
    r"""Hyper-DeepONet with amplitude + phase modulation for monotone OT.

    .. math::

        T(x | p, c) = a(p,c) \cdot x + \text{offset}(p,c)
            + \sum_{k=1}^{K} \underbrace{\text{splus}(\beta_k(p,c))}_{\text{amplitude}}
              \cdot \underbrace{\sigma(w_k^\top x + b_k
                + \delta_k(p,c))}_{\text{phase-shifted trunk}}
              \cdot w_k

    Compared to standard DeepONet which has:
        sigma(w_k^T x + b_k)           -- fixed basis functions

    Hyper-DeepONet has:
        sigma(w_k^T x + b_k + delta_k) -- input-adaptive basis functions

    The delta_k shifts where each sigmoid "turns on", allowing the
    trunk basis to adapt to different input measures. This is equivalent
    to FiLM conditioning in mGradNet, but framed as a hypernetwork
    within the DeepONet architecture.

    PSD Jacobian guarantee is preserved because:
        J_T = diag(a) + sum_k splus(beta_k) * sigma'(...) * w_k w_k^T >= 0
    (sigma' >= 0 for sigmoid, splus(beta_k) >= 0)
    """

    def __init__(
        self,
        state_dim: int = 1,
        n_basis: int = 128,
        d_branch: int = 128,
        d_trunk: int = 128,
        n_scalar_ctx: int = 6,
        **kwargs,
    ):
        """Initialise hyper-branch, trunk, and affine head; see class docstring."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.n_basis = n_basis
        self.n_ridges = n_basis  # compatibility alias

        self.branch = HyperBranchNet(
            d_hidden=d_branch,
            n_basis=n_basis,
            n_scalar_ctx=n_scalar_ctx,
        )

        self.trunk = TrunkNet(
            state_dim=state_dim,
            n_basis=n_basis,
        )

        # Affine head (from branch encoding -> a, offset)
        self.affine_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(2 * state_dim),
        ], name="affine_head")

    def call(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        context_scalars: tf.Tensor,
    ) -> tf.Tensor:
        """Forward pass: map weighted particles to transported particles."""
        single = len(weights.shape) == 1
        if single:
            particles = particles[tf.newaxis, ...]
            weights = weights[tf.newaxis, :]
            context_scalars = context_scalars[tf.newaxis, :]

        squeeze_last = False
        if len(particles.shape) == 2:
            particles = particles[:, :, tf.newaxis]
            squeeze_last = True

        # Branch: amplitude coefficients + phase shifts
        beta_raw, delta = self.branch(particles, weights, context_scalars)

        # Affine parameters
        affine = self.affine_head(beta_raw)
        d = self.state_dim
        a = tf.nn.softplus(affine[:, :d])
        offset = affine[:, d:]

        # PSD amplitude coefficients
        beta = tf.nn.softplus(beta_raw)

        # Trunk: phase-shifted basis functions
        h, w = self.trunk(particles, delta)

        # Monotone output
        weighted_h = beta[:, tf.newaxis, :] * h
        residual = tf.einsum("bnk,kd->bnd", weighted_h, w)

        output = (particles * a[:, tf.newaxis, :] +
                  offset[:, tf.newaxis, :] +
                  residual)

        if squeeze_last:
            output = output[:, :, 0]
        if single:
            output = output[0]
        return output

    def log_det_jacobian(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        context_scalars: tf.Tensor,
    ) -> tf.Tensor:
        """log |det J_T(x)| — same formula as DeepONet but with delta shifts."""
        single = len(weights.shape) == 1
        if single:
            particles = particles[tf.newaxis, ...]
            weights = weights[tf.newaxis, :]
            context_scalars = context_scalars[tf.newaxis, :]

        squeeze_last = False
        if len(particles.shape) == 2:
            particles = particles[:, :, tf.newaxis]
            squeeze_last = True

        beta_raw, delta = self.branch(particles, weights, context_scalars)
        affine = self.affine_head(beta_raw)
        d = self.state_dim
        a = tf.nn.softplus(affine[:, :d])
        beta = tf.nn.softplus(beta_raw)

        h, w = self.trunk(particles, delta)
        h_prime = h * (1.0 - h)

        if self.state_dim == 1:
            w_sq = tf.reduce_sum(w ** 2, axis=-1)
            deriv = a[:, 0:1] + tf.einsum(
                "bk,bnk,k->bn", beta, h_prime, w_sq
            )
            logdet = tf.math.log(tf.maximum(deriv, 1e-20))
            return logdet[0] if single else logdet

        diag_a = tf.linalg.diag(a)[:, tf.newaxis, :, :]
        scale = tf.sqrt(
            tf.maximum(beta[:, tf.newaxis, :] * h_prime, 1e-20)
        )
        scaled_w = scale[:, :, :, tf.newaxis] * w[tf.newaxis, tf.newaxis, :, :]
        J = diag_a + tf.einsum("bnki,bnkj->bnij", scaled_w, scaled_w)
        _, logdet = tf.linalg.slogdet(J)
        logdet = tf.cast(tf.math.real(logdet), tf.float32)
        return logdet[0] if single else logdet
