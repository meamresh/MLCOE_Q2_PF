"""Standalone verification: expm_2x2_batch vs tf.linalg.expm.

Sweeps random 2x2 matrices across magnitudes (||A||_F in {0.1, 1, 5, 10, 50})
plus the boundary case xi -> 0 (skew-symmetric-ish near-identity), and
checks max absolute and relative error.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    expm_2x2_batch,
)


def case(name: str, A: tf.Tensor) -> None:
    ref = tf.linalg.expm(A)
    got = expm_2x2_batch(A)
    abs_err = tf.reduce_max(tf.abs(ref - got)).numpy()
    denom = tf.maximum(tf.reduce_max(tf.abs(ref)), tf.constant(1e-12))
    rel_err = (abs_err / denom).numpy()
    print(f"{name:36s} max|abs|={abs_err:.3e}  max|rel|={rel_err:.3e}")


def main() -> None:
    tf.random.set_seed(0)

    n = 1024
    for mag in (0.1, 1.0, 5.0, 10.0):
        A = mag * tf.random.normal([n, 2, 2], dtype=tf.float32)
        case(f"random N={n} ||A||~{mag}", A)

    # LEDH-flow-like A: clipped to [-10, 10] (the filter clips A).
    A = tf.clip_by_value(20.0 * tf.random.normal([n, 2, 2]), -10.0, 10.0)
    case("clipped to [-10,10]", A)

    # Boundary: xi exactly zero (skew-symmetric off-diagonals cancel diag^2)
    diag = tf.random.normal([n], dtype=tf.float32)
    A = tf.zeros([n, 2, 2], dtype=tf.float32)
    # Build A with a=d (half_diff=0), b=c=0 -> xi=0
    A = tf.stack([
        tf.stack([diag, tf.zeros([n])], axis=-1),
        tf.stack([tf.zeros([n]), diag], axis=-1),
    ], axis=-2)
    case("xi=0 (diagonal a=d, b=c=0)", A)

    # xi very small: nearly-real-degenerate pair
    a = 0.5 + 1e-7 * tf.random.normal([n])
    d_ = 0.5 - 1e-7 * tf.random.normal([n])
    b = 1e-7 * tf.random.normal([n])
    c = 1e-7 * tf.random.normal([n])
    A = tf.stack([
        tf.stack([a, b], axis=-1),
        tf.stack([c, d_], axis=-1),
    ], axis=-2)
    case("xi ~ 1e-14 (near-degenerate)", A)

    # xi very negative: rotation-heavy
    a = tf.random.normal([n])
    b = 5.0 * tf.random.normal([n])
    c = -5.0 * tf.random.normal([n])  # b*c < 0 typically
    A = tf.stack([
        tf.stack([a, b], axis=-1),
        tf.stack([c, a], axis=-1),  # half_diff=0 + bc<0 -> xi<0
    ], axis=-2)
    case("xi < 0 (rotation-heavy)", A)

    # XLA path: confirm jit_compile=True actually compiles
    @tf.function(jit_compile=True)
    def jitted_expm(X):
        return expm_2x2_batch(X)

    A = tf.random.normal([n, 2, 2])
    got_xla = jitted_expm(A)
    ref = tf.linalg.expm(A)
    abs_err = tf.reduce_max(tf.abs(ref - got_xla)).numpy()
    print(f"{'XLA jit_compile=True':36s} max|abs|={abs_err:.3e}  "
          f"(compiles + matches eager expm)")


if __name__ == "__main__":
    main()
