"""Confirm call_mat_phi == call_full at Phi=diag with matched RNG seeds."""

from __future__ import annotations
import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
)


def gen_obs(T=50, d=2, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.array([0.0, -0.3], dtype=np.float32)
    phi = np.array([0.95, 0.85], dtype=np.float32)
    sig = np.array([0.3, 0.4], dtype=np.float32)
    h = np.zeros((T, d), dtype=np.float32)
    h[0] = mu + sig * rng.standard_normal(d) / np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        h[t] = mu + phi * (h[t-1] - mu) + sig * rng.standard_normal(d)
    y = np.exp(h / 2) * rng.standard_normal((T, d)).astype(np.float32)
    return tf.constant(y, dtype=tf.float32), mu, phi, sig


def main():
    T, d, N = 50, 2, 64
    y, mu_np, phi_np, sig_np = gen_obs(T, d)
    mu = tf.constant(mu_np, dtype=tf.float32)
    phi_diag = tf.constant(phi_np, dtype=tf.float32)
    L_eta = tf.linalg.diag(tf.constant(sig_np, dtype=tf.float32))
    Phi_mat = tf.linalg.diag(phi_diag)

    for use_jit in (False, True):
        tag = "XLA" if use_jit else "eager"
        # Fresh filter + identical seed reset for each call
        f = DifferentiableLEDHLogLikelihoodSVSSMmulti(
            state_dim=d, num_particles=N, n_lambda=10,
            jit_compile=use_jit, init_type="stationary",
        )
        tf.random.set_seed(42)
        ll1 = float(f.call_full(mu, phi_diag, L_eta, y))
        tf.random.set_seed(42)
        ll2 = float(f.call_mat_phi(mu, Phi_mat, L_eta, y))
        print(f"[{tag:5s}] call_full={ll1:+.4f}  call_mat_phi={ll2:+.4f}  "
              f"|Δ|={abs(ll1-ll2):.4f}")


if __name__ == "__main__":
    main()
