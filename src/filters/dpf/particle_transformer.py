"""
Particle Transformer for Differentiable Resampling.

Implements the architecture from:
  "Towards Differentiable Resampling" (Zhu et al., 2020, arXiv:2004.11938)

A learned, permutation-invariant, scale-equivariant neural network that
replaces traditional resampling in particle filters.  The architecture is
based on the Set Transformer with weighted multi-head attention.
"""

from __future__ import annotations

import math
from typing import Tuple

import tensorflow as tf


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class WeightedMultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention with per-key weight modulation.

    Standard MHA computes:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    We generalise to weighted attention (Eq. in §IV-A of arXiv:2004.11938):
        WeightedAttn(Q, K, V, w) =
            sum_i [ w_i * exp(q·k_i/sqrt(d)) * v_i ]
            / sum_j [ w_j * exp(q·k_j/sqrt(d)) ]

    The weights w_i are broadcast across heads.
    """

    def __init__(self, d_model: int, n_heads: int, **kwargs):
        """Initialise projection layers for *n_heads* attention heads."""
        super().__init__(**kwargs)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wo = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value, weights=None):
        """
        Args:
            query:   (B, n_q, d_model)
            key:     (B, n_kv, d_model)
            value:   (B, n_kv, d_model)
            weights: (B, n_kv) or None – per-key particle weights.
        Returns:
            (B, n_q, d_model)
        """
        B = tf.shape(query)[0]

        Q = self._split_heads(self.wq(query), B)   # (B, H, n_q, d_k)
        K = self._split_heads(self.wk(key), B)      # (B, H, n_kv, d_k)
        V = self._split_heads(self.wv(value), B)    # (B, H, n_kv, d_k)

        # Scaled dot-product scores  (B, H, n_q, n_kv)
        scores = tf.matmul(Q, K, transpose_b=True) / math.sqrt(self.d_k)

        if weights is not None:
            # weights: (B, n_kv) -> (B, 1, 1, n_kv)
            log_w = tf.math.log(weights + 1e-20)
            log_w = log_w[:, tf.newaxis, tf.newaxis, :]
            scores = scores + log_w  # log(w * exp(score)) = log_w + score

        attn = tf.nn.softmax(scores, axis=-1)       # (B, H, n_q, n_kv)
        out = tf.matmul(attn, V)                     # (B, H, n_q, d_k)

        # Merge heads
        out = tf.transpose(out, perm=[0, 2, 1, 3])   # (B, n_q, H, d_k)
        out = tf.reshape(out, [B, -1, self.d_model])
        return self.wo(out)

    def _split_heads(self, x, batch_size):
        """(B, seq, d_model) -> (B, H, seq, d_k)"""
        x = tf.reshape(x, [batch_size, -1, self.n_heads, self.d_k])
        return tf.transpose(x, perm=[0, 2, 1, 3])


class FeedForward(tf.keras.layers.Layer):
    """Pointwise two-layer FFN with ReLU."""

    def __init__(self, d_model: int, d_ff: int, **kwargs):
        """Initialise two-layer FFN with hidden dimension *d_ff*."""
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.fc2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        """Forward pass through the two-layer FFN."""
        return self.fc2(self.fc1(x))


class TransformerBlock(tf.keras.layers.Layer):
    """Attention + FFN + residual + layer norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, **kwargs):
        """Initialise attention, FFN, and layer-norm sub-layers."""
        super().__init__(**kwargs)
        self.attn = WeightedMultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x, context=None, weights=None):
        """
        If context is None: self-attention on x.
        If context is given: cross-attention (x=query, context=key/value).
        """
        kv = context if context is not None else x
        h = self.ln1(x + self.attn(x, kv, kv, weights=weights))
        return self.ln2(h + self.ffn(h))


# ---------------------------------------------------------------------------
# Particle Transformer
# ---------------------------------------------------------------------------

class ParticleTransformer(tf.keras.Model):
    """
    Full Particle Transformer (Zhu et al., arXiv:2004.11938).

    Maps N weighted particles {x_i, w_i} to N equally-weighted particles.
    N is determined dynamically at call time (not fixed at construction),
    so a single trained model can serve any particle count.

    Architecture (adapted for low-dimensional state, e.g. 1D Kitagawa):
      Encoder: rescale → linear → 2× weighted-MHSA blocks
      Decoder: learned seed bank → select/interpolate N seeds →
               2× (self-attn + weighted cross-attn) → linear → unscale
    """

    def __init__(
        self,
        state_dim: int = 1,
        n_seed_bank: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        **kwargs,
    ):
        """Initialise encoder, decoder, seed bank, and projection layers."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.n_seed_bank = n_seed_bank
        self.d_model = d_model

        # Encoder
        self.input_proj = tf.keras.layers.Dense(d_model)
        self.enc_block1 = TransformerBlock(d_model, n_heads, d_ff)
        self.enc_block2 = TransformerBlock(d_model, n_heads, d_ff)

        # Decoder -- seed bank of fixed size; we interpolate to match N at runtime
        self.seed_bank = self.add_weight(
            name="seed_bank",
            shape=(n_seed_bank, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.dec_self1 = TransformerBlock(d_model, n_heads, d_ff)
        self.dec_cross1 = TransformerBlock(d_model, n_heads, d_ff)
        self.dec_self2 = TransformerBlock(d_model, n_heads, d_ff)
        self.dec_cross2 = TransformerBlock(d_model, n_heads, d_ff)

        # Output projection back to state_dim
        self.output_proj = tf.keras.layers.Dense(state_dim)

    def _get_seeds(self, N):
        """Interpolate the fixed-size seed bank to produce exactly N seeds.

        When N == n_seed_bank this is an identity.  Otherwise we treat the
        seed bank as a 1-D "sequence" and use linear interpolation along that
        axis so the operation stays differentiable w.r.t. seed_bank weights.
        """
        S = self.n_seed_bank
        if N == S:
            return self.seed_bank[tf.newaxis, :, :]  # (1, N, d)

        bank = self.seed_bank[tf.newaxis, :, :]  # (1, S, d)
        # tf.image.resize does bilinear interp on (H, W) -- treat as (1, S, d_model)
        # Reshape to (1, S, 1, d) so the "width" dim is 1 and height is S
        bank_4d = bank[:, :, tf.newaxis, :]       # (1, S, 1, d)
        resized = tf.image.resize(bank_4d, [N, 1], method='bilinear')  # (1, N, 1, d)
        return resized[:, :, 0, :]                # (1, N, d)

    def call(self, particles, weights):
        """
        Args:
            particles: (B, N, state_dim) or (N, state_dim) — particle positions.
            weights:   (B, N) or (N,) — normalised particle weights.
        Returns:
            new_particles: same shape as input particles — equally-weighted.
        """
        single = False
        if len(particles.shape) == 2:
            single = True
            particles = particles[tf.newaxis, :]  # (1, N, d)
            weights = weights[tf.newaxis, :]        # (1, N)

        B = tf.shape(particles)[0]
        N = tf.shape(particles)[1]

        # --- Scale equivariance: normalise per-dim to [-1, 1] ---
        x_min = tf.reduce_min(particles, axis=1, keepdims=True)   # (B,1,d)
        x_max = tf.reduce_max(particles, axis=1, keepdims=True)   # (B,1,d)
        x_range = x_max - x_min + 1e-8
        x_norm = 2.0 * (particles - x_min) / x_range - 1.0       # (B,N,d)

        # --- Encoder ---
        h = self.input_proj(x_norm)                                # (B,N,d_model)
        h = self.enc_block1(h, weights=weights)
        h = self.enc_block2(h, weights=weights)

        # --- Decoder ---
        seeds = self._get_seeds(N)                                 # (1,N,d_model)
        z = tf.tile(seeds, [B, 1, 1])                             # (B,N,d_model)
        z = self.dec_self1(z)
        z = self.dec_cross1(z, context=h, weights=weights)
        z = self.dec_self2(z)
        z = self.dec_cross2(z, context=h, weights=weights)

        # --- Output projection + un-scale ---
        out_norm = self.output_proj(z)                             # (B,N,d)
        out = (out_norm + 1.0) / 2.0 * x_range + x_min            # undo scaling

        if single:
            out = out[0]
        return out
