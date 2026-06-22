"""Verification: expm_pade_batch vs tf.linalg.expm for general d.

Checks, per d in {3, 4, 8}:
  1. Accuracy across the LEDH-relevant norm range (A clipped entrywise
     to [-10, 10], scaled by eps in [0.04, 0.2]).
  2. Accuracy at the clip extreme (worst case the flow can produce).
  3. XLA compilation under @tf.function(jit_compile=True).
  4. Gradient flow: d/dA of a scalar functional matches between Pade
     and stock expm.
Also re-checks d=2 against expm_2x2_batch for consistency.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    expm_pade_batch, pade_scaling_for_dim, expm_2x2_batch,
)


def report(name, ref, got):
    abs_err = float(tf.reduce_max(tf.abs(ref - got)))
    denom = float(tf.maximum(tf.reduce_max(tf.abs(ref)), 1e-12))
    print(f"  {name:<42s} max|abs|={abs_err:.3e}  max|rel|={abs_err/denom:.3e}")
    return abs_err / denom


def main():
    tf.random.set_seed(0)
    n = 256
    all_ok = True

    for d in (3, 4, 8):
        s = pade_scaling_for_dim(d)
        print(f"\n=== d={d}  (static s={s}) ===")

        # 1. LEDH-realistic: A entries ~ clipped N(0, 3), times eps in range
        A_raw = tf.clip_by_value(3.0 * tf.random.normal([n, d, d]), -10.0, 10.0)
        for eps in (0.04, 0.2):
            A = A_raw * eps
            ref = tf.linalg.expm(tf.cast(A, tf.float64))
            got = expm_pade_batch(tf.cast(A, tf.float64), s=s)
            rel = report(f"realistic eps={eps} (f64)", ref, got)
            all_ok &= rel < 1e-10
            # float32, as used in the filter
            ref32 = tf.linalg.expm(A)
            got32 = expm_pade_batch(A, s=s)
            rel = report(f"realistic eps={eps} (f32)", ref32, got32)
            all_ok &= rel < 1e-4

        # 2. Clip extreme: all entries at +-10, eps = 0.2 (worst case)
        signs = tf.sign(tf.random.normal([n, d, d]))
        A_ext = signs * 10.0 * 0.2
        ref = tf.linalg.expm(tf.cast(A_ext, tf.float64))
        got = expm_pade_batch(tf.cast(A_ext, tf.float64), s=s)
        rel = report("clip extreme +-10*0.2 (f64)", ref, got)
        all_ok &= rel < 1e-9

        # 3. XLA compile
        @tf.function(jit_compile=True)
        def jitted(X):
            return expm_pade_batch(X, s=s)

        A = A_raw * 0.2
        got_xla = jitted(A)
        ref = tf.linalg.expm(A)
        rel = report("XLA jit_compile=True (f32)", ref, got_xla)
        all_ok &= rel < 1e-4

        # 4. Gradient check: grad of sum(expm(A) @ v) wrt A
        v = tf.random.normal([n, d, 1])
        A_var = tf.Variable(A_raw * 0.1)
        with tf.GradientTape() as t1:
            y1 = tf.reduce_sum(tf.matmul(tf.linalg.expm(A_var), v))
        g_ref = t1.gradient(y1, A_var)
        with tf.GradientTape() as t2:
            y2 = tf.reduce_sum(tf.matmul(expm_pade_batch(A_var, s=s), v))
        g_got = t2.gradient(y2, A_var)
        rel = report("gradient d/dA (f32)", g_ref, g_got)
        all_ok &= rel < 1e-3

    # 5. Consistency with the d=2 closed form
    print(f"\n=== d=2 cross-check vs expm_2x2_batch ===")
    s2 = pade_scaling_for_dim(2)
    A = tf.clip_by_value(3.0 * tf.random.normal([n, 2, 2]), -10.0, 10.0) * 0.2
    got_ch = expm_2x2_batch(A)
    got_pade = expm_pade_batch(A, s=s2)
    rel = report(f"Cayley-Hamilton vs Pade (s={s2})", got_ch, got_pade)
    all_ok &= rel < 1e-4

    print(f"\n{'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
