"""Sanity check XLA-enabled full-Phi filter.

1. Verify call_mat_phi at zero off-diagonal Phi matches call_full at the
   same diagonal Phi + Sigma_eta (regression check after the expm swap).
2. Time per-call wall: eager (jit_compile=False) vs XLA (jit_compile=True).
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
)


def gen_obs(T: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu = np.array([0.0, -0.3], dtype=np.float32)[:d]
    phi = np.array([0.95, 0.85], dtype=np.float32)[:d]
    sig = np.array([0.3, 0.4], dtype=np.float32)[:d]
    h = np.zeros((T, d), dtype=np.float32)
    h[0] = mu + sig * rng.standard_normal(d) / np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        h[t] = mu + phi * (h[t - 1] - mu) + sig * rng.standard_normal(d)
    y = np.exp(h / 2) * rng.standard_normal((T, d)).astype(np.float32)
    return tf.constant(y, dtype=tf.float32), mu, phi, sig


def main() -> None:
    tf.random.set_seed(0)

    T, d, N = 50, 2, 64
    y, mu_np, phi_np, sig_np = gen_obs(T, d, seed=0)

    mu = tf.constant(mu_np, dtype=tf.float32)
    phi_diag = tf.constant(phi_np, dtype=tf.float32)
    sigma_eta_sq = tf.constant(sig_np ** 2, dtype=tf.float32)

    # Build (d,d) Phi from diag, (d,d) Cholesky from sigma_eta_sq.
    Phi_mat = tf.linalg.diag(phi_diag)
    L_eta = tf.linalg.diag(tf.sqrt(sigma_eta_sq))

    for use_jit in (False, True):
        f = DifferentiableLEDHLogLikelihoodSVSSMmulti(
            state_dim=d, num_particles=N, n_lambda=10,
            jit_compile=use_jit, init_type="stationary",
        )

        # call_full = diagonal Phi + full Sigma_eta path
        ll_full = float(f.call_full(mu, phi_diag, L_eta, y))
        # call_mat_phi = full matrix Phi path; at Phi=diag should match
        ll_mat = float(f.call_mat_phi(mu, Phi_mat, L_eta, y))
        diff = abs(ll_full - ll_mat)
        tag = "XLA" if use_jit else "eager"
        print(f"[{tag:5s}]  call_full={ll_full:+.4f}  "
              f"call_mat_phi={ll_mat:+.4f}  |Δ|={diff:.4f}")

        # Time call_mat_phi: warm + 3 calls.
        f.call_mat_phi(mu, Phi_mat, L_eta, y)  # warmup compile
        t0 = time.time()
        for _ in range(3):
            _ = f.call_mat_phi(mu, Phi_mat, L_eta, y)
        wall = (time.time() - t0) / 3
        print(f"[{tag:5s}]  call_mat_phi  forward wall: {wall*1000:7.1f} ms / call")

        # Now time with gradient: this is what HMC actually runs (value + grad).
        mu_v = tf.Variable(mu_np, dtype=tf.float32)
        phi_v = tf.Variable(tf.linalg.diag(phi_np), dtype=tf.float32)
        L_v = tf.Variable(tf.linalg.diag(np.sqrt(sig_np ** 2)), dtype=tf.float32)

        def value_and_grad():
            with tf.GradientTape() as tape:
                ll = f.call_mat_phi(mu_v, phi_v, L_v, y)
            return ll, tape.gradient(ll, [mu_v, phi_v, L_v])

        value_and_grad()  # warmup
        t0 = time.time()
        for _ in range(3):
            _ = value_and_grad()
        wall_g = (time.time() - t0) / 3
        print(f"[{tag:5s}]  call_mat_phi value+grad wall: {wall_g*1000:7.1f} ms / call")


if __name__ == "__main__":
    main()
