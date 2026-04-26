"""
DeepONet-based Neural Operator for Optimal Transport Resampling.

Applies neural operator theory (Lu et al. 2021; Kovachki et al. 2023)
to learn the OT solution operator:

    G: (source measure p, scalar context c) -> T   (transport map)

Instead of the ad-hoc DeepSet+FiLM conditioning in mgradnet_ot.py, this
uses the principled DeepONet (Deep Operator Network) architecture:

  - **Branch net**: encodes the input function (weighted particle cloud)
    into a finite basis coefficient vector.
  - **Trunk net**: evaluates the output function (transport map) at
    query positions using learned basis functions.
  - **Output**: T(x) = sum_p b_p(particles, weights, ctx) * t_p(x)

The PSD Jacobian guarantee from mGradNet-M is preserved by decomposing
the output as:

    T(x) = a * x + offset + sum_k alpha_k * sigma(w_k^T x + bias_k) * w_k

where the trunk net produces the ridge directions w_k and biases, and the
branch net produces the non-negative coefficients alpha_k (via softplus).

This gives the best of both worlds:
  - DeepONet's universal approximation of continuous operators (UAT)
  - mGradNet's architectural PSD Jacobian guarantee (Brenier map)
  - Mesh-invariance: branch net can handle variable particle counts N

References
----------
- Lu et al. "Learning nonlinear operators via DeepONet", Nature MI, 2021
- Kovachki et al. "Neural Operator: Learning Maps Between Function Spaces
  with Applications to PDEs", JMLR, 2023
- Chaudhari et al. "GradNetOT", arXiv:2507.13191

Architecture
------------
1. **BranchNet** -- permutation-invariant encoder for (particles, weights,
   scalar context).  Uses DeepSet pooling over particles, concatenated with
   scalar context, then MLP to produce branch coefficients.
2. **TrunkNet** -- per-query-point MLP that produces basis functions
   evaluated at x.  Incorporates mGradNet-M ridge structure for PSD
   guarantee.
3. **DeepONetMonotoneOT** -- full model combining branch + trunk with
   monotone output structure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf


# ---------------------------------------------------------------------------
# Branch Net: encodes the source measure + scalar context
# ---------------------------------------------------------------------------

class BranchNet(tf.keras.layers.Layer):
    """Encodes (particles, weights, scalar_context) into branch coefficients.

    Uses DeepSet pooling for permutation invariance over particles,
    then concatenates scalar context and produces p basis coefficients.

    This is the "sensor" side of DeepONet: it reads the input function
    (the weighted particle cloud, discretised on N points) and compresses
    it into a finite-dimensional representation.

    Parameters
    ----------
    d_hidden : int
        Hidden dimension for per-particle and post-pooling MLPs.
    n_basis : int
        Number of basis functions p (= number of branch outputs).
        This controls the expressiveness of the operator.
    n_scalar_ctx : int
        Dimension of the scalar context vector.
    """

    def __init__(
        self,
        d_hidden: int = 128,
        n_basis: int = 128,
        n_scalar_ctx: int = 6,
        **kwargs,
    ):
        """Initialise per-particle and post-pooling MLPs for the branch net."""
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

        # rho: post-pooling MLP producing branch coefficients
        self.rho_net = tf.keras.Sequential([
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(d_hidden, activation="gelu"),
            tf.keras.layers.Dense(n_basis),
        ], name="branch_rho")

    def call(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        context_scalars: tf.Tensor,
    ) -> tf.Tensor:
        """
        Args
        ----
        particles       : (B, N, d)
        weights         : (B, N)
        context_scalars : (B, n_ctx)

        Returns
        -------
        branch_coeffs : (B, n_basis)  — the b_p coefficients
        """
        # Concat position and weight for per-particle features
        pw = tf.concat([particles, weights[:, :, tf.newaxis]], axis=-1)
        h = self.phi_net(pw)  # (B, N, d_hidden)

        # Weighted-sum pooling (DeepSet aggregation)
        h_weighted = h * weights[:, :, tf.newaxis]
        pooled = tf.reduce_sum(h_weighted, axis=1)  # (B, d_hidden)

        # Encode scalar context
        scl = self.scalar_enc(context_scalars)  # (B, d_hidden)

        # Joint encoding -> branch coefficients
        joint = tf.concat([pooled, scl], axis=-1)  # (B, 2*d_hidden)
        return self.rho_net(joint)  # (B, n_basis)


# ---------------------------------------------------------------------------
# Trunk Net: evaluates basis functions at query points
# ---------------------------------------------------------------------------

class TrunkNet(tf.keras.layers.Layer):
    """Evaluates learned basis functions at query positions x.

    This is the "evaluation" side of DeepONet: given a query point x,
    it produces p basis function values [t_1(x), ..., t_p(x)].

    For the standard DeepONet this would be a generic MLP.  Here we
    use a structure that enables the monotone (PSD Jacobian) guarantee:
    the basis functions are ridge functions sigma(w_k^T x + b_k) scaled
    by their ridge directions w_k.

    Parameters
    ----------
    state_dim : int
        Particle dimensionality.
    n_basis : int
        Number of basis functions p (must match BranchNet.n_basis).
    d_hidden : int
        Hidden dimension for the trunk MLP that produces extra features.
    """

    def __init__(
        self,
        state_dim: int = 1,
        n_basis: int = 128,
        d_hidden: int = 128,
        **kwargs,
    ):
        """Initialise ridge-function trunk with basis dimension *n_basis*."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.n_basis = n_basis

        # Ridge directions and biases (mGradNet-M structure)
        self.ridge_w = self.add_weight(
            name="ridge_w", shape=(n_basis, state_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.5),
        )
        self.ridge_b = self.add_weight(
            name="ridge_b", shape=(n_basis,),
            initializer="zeros",
        )

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args
        ----
        x : (B, N, d)  query positions

        Returns
        -------
        h       : (B, N, n_basis)  sigmoid activations at each query point
        ridge_w : (n_basis, d)     ridge directions (for PSD output)
        """
        # wx = <w_k, x_n>: (B, N, n_basis)
        wx = tf.einsum("kd,bnd->bnk", self.ridge_w, x)
        bias = self.ridge_b[tf.newaxis, tf.newaxis, :]
        h = tf.sigmoid(wx + bias)  # (B, N, n_basis)
        return h, self.ridge_w


# ---------------------------------------------------------------------------
# DeepONet Monotone OT Model
# ---------------------------------------------------------------------------

class DeepONetMonotoneOT(tf.keras.Model):
    r"""DeepONet-based neural operator for monotone OT maps.

    Combines the branch net (source measure encoder) and trunk net
    (query-point basis evaluator) with a monotone output structure:

    .. math::

        T(x | p, c) = a(p,c) \cdot x + \text{offset}(p,c)
            + \sum_{k=1}^{K} \underbrace{\text{splus}(\beta_k(p,c))}_{\text{branch}}
              \cdot \underbrace{\sigma(w_k^\top x + b_k)}_{\text{trunk}}
              \cdot w_k

    The Jacobian is:

    .. math::

        J_T = a \cdot I + \sum_k \text{splus}(\beta_k) \cdot
              \sigma'(\cdot) \cdot w_k w_k^\top \succeq 0

    which is PSD by construction (diagonal + sum of rank-1 PSD terms).

    **Key difference from ConditionalMGradNet**: the branch-trunk
    decomposition gives a principled operator-learning structure with
    universal approximation guarantees (Chen & Chen 1995, Lu et al. 2021),
    while preserving the mGradNet monotonicity constraint.

    Parameters
    ----------
    state_dim    : int   particle dimensionality
    n_basis      : int   number of basis functions (= n_ridges equivalent)
    d_branch     : int   hidden width of branch net
    d_trunk      : int   hidden width of trunk net
    n_scalar_ctx : int   scalar context vector length
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
        """Initialise branch, trunk, and affine head; see class docstring."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.n_basis = n_basis
        # Alias for compatibility with NeuralOTTrainer
        self.n_ridges = n_basis

        # Branch: (particles, weights, ctx) -> coefficients
        self.branch = BranchNet(
            d_hidden=d_branch,
            n_basis=n_basis,
            n_scalar_ctx=n_scalar_ctx,
        )

        # Trunk: x -> basis function values (ridge structure)
        self.trunk = TrunkNet(
            state_dim=state_dim,
            n_basis=n_basis,
            d_hidden=d_trunk,
        )

        # Affine parameters from branch (a and offset)
        self.affine_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(2 * state_dim),  # [a_raw, offset]
        ], name="affine_head")

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
        weights         : (B, N) | (N,)
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

        squeeze_last = False
        if len(particles.shape) == 2:
            particles = particles[:, :, tf.newaxis]
            squeeze_last = True

        # ---- Branch: encode source measure ----
        branch_coeffs = self.branch(particles, weights, context_scalars)
        # (B, n_basis)

        # ---- Affine parameters from branch encoding ----
        affine = self.affine_head(branch_coeffs)  # (B, 2*d)
        d = self.state_dim
        a = tf.nn.softplus(affine[:, :d])       # (B, d), > 0
        offset = affine[:, d:]                    # (B, d)

        # ---- Branch coefficients through softplus for PSD guarantee ----
        beta = tf.nn.softplus(branch_coeffs)  # (B, n_basis), >= 0

        # ---- Trunk: evaluate basis at query points ----
        h, w = self.trunk(particles)
        # h: (B, N, n_basis), w: (n_basis, d)

        # ---- Monotone output: T(x) = a*x + offset + sum_k beta_k * h_k * w_k
        weighted_h = beta[:, tf.newaxis, :] * h  # (B, N, n_basis)
        residual = tf.einsum("bnk,kd->bnd", weighted_h, w)  # (B, N, d)

        output = (particles * a[:, tf.newaxis, :] +
                  offset[:, tf.newaxis, :] +
                  residual)

        if squeeze_last:
            output = output[:, :, 0]
        if single:
            output = output[0]
        return output

    # ------------------------------------------------------------------ #
    #  Log-|det Jacobian| (for Monge-Ampère loss)                        #
    # ------------------------------------------------------------------ #

    def log_det_jacobian(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        context_scalars: tf.Tensor,
    ) -> tf.Tensor:
        """Compute log |det J_T(x)| for each particle.

        The Jacobian is:
            J_T = diag(a) + sum_k beta_k * sigma'(w_k^T x + b_k) * w_k w_k^T

        For d=1: dT/dx = a + sum_k beta_k * h_k * (1-h_k) * w_k^2
        For d>1: full slogdet

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

        # ---- Branch ----
        branch_coeffs = self.branch(particles, weights, context_scalars)
        affine = self.affine_head(branch_coeffs)
        d = self.state_dim
        a = tf.nn.softplus(affine[:, :d])  # (B, d)
        beta = tf.nn.softplus(branch_coeffs)  # (B, n_basis)

        # ---- Trunk activations ----
        h, w = self.trunk(particles)  # h: (B,N,K), w: (K,d)
        h_prime = h * (1.0 - h)  # sigmoid derivative

        if self.state_dim == 1:
            # dT/dx = a + sum_k beta_k * h'_k * w_k^2
            w_sq = tf.reduce_sum(w ** 2, axis=-1)  # (K,)
            deriv = a[:, 0:1] + tf.einsum(
                "bk,bnk,k->bn", beta, h_prime, w_sq
            )  # (B, N)
            logdet = tf.math.log(tf.maximum(deriv, 1e-20))
            return logdet[0] if single else logdet

        # General d: build full Jacobian and slogdet
        diag_a = tf.linalg.diag(a)[:, tf.newaxis, :, :]  # (B,1,d,d)

        scale = tf.sqrt(
            tf.maximum(beta[:, tf.newaxis, :] * h_prime, 1e-20)
        )  # (B,N,K)
        scaled_w = scale[:, :, :, tf.newaxis] * w[tf.newaxis, tf.newaxis, :, :]
        # (B, N, K, d)

        J = diag_a + tf.einsum("bnki,bnkj->bnij", scaled_w, scaled_w)
        _, logdet = tf.linalg.slogdet(J)
        logdet = tf.cast(tf.math.real(logdet), tf.float32)
        return logdet[0] if single else logdet
