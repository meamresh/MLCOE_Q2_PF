"""
Smoke test for the multivariate V1 SVSSM filter.

Three checks:

1. **It runs**: filter call produces a finite scalar log p at d=2.

2. **Gradient is finite**: forward+grad through (mu, phi_diag, sigma_eta_diag_sq).
   Required for HMC.

3. **Reduces to d=1 sum under diagonal Sigma_eta**: at d=2 with diagonal
   Phi, diagonal Sigma_eta and component-wise observation, the
   log-likelihood should approximately equal the sum of the two
   univariate filters' log p (modulo Monte-Carlo noise from the
   resampling step interacting differently in 2-D vs 1-D).

Run from repo root:
    PYTHONPATH=. python scripts/exp/smoketest_svssm_multivariate.py
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    DifferentiableLEDHLogLikelihoodSVSSM,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
)


def gen_svssm_data(T, mu_vec, phi_vec, sigma_eta_vec, seed=42):
    """Generate (T, d) SVSSM data with diagonal Phi + diagonal Sigma_eta.

    Returns y_obs as a (T, d) tf.float32 tensor.
    """
    tf.random.set_seed(seed)
    d = len(mu_vec)
    mu = tf.constant(mu_vec, tf.float32)
    phi = tf.constant(phi_vec, tf.float32)
    sigma_eta = tf.constant(sigma_eta_vec, tf.float32)

    h = tf.identity(mu)  # initialise at the long-run mean
    ys = []
    for _ in range(T):
        eta = tf.random.normal([d])
        h = mu + phi * (h - mu) + sigma_eta * eta
        eps = tf.random.normal([d])
        y = tf.exp(h / 2.0) * eps
        ys.append(y)
    return tf.stack(ys, axis=0)  # (T, d)


def main():
    print(f"TF {tf.__version__}\n")
    T = 20
    N = 64
    d = 2

    # Two components with different persistence and volatility.
    mu_truth        = np.array([0.0, -0.3],  dtype=np.float32)
    phi_truth       = np.array([0.95, 0.85], dtype=np.float32)
    sigma_eta_truth = np.array([0.3,  0.4],  dtype=np.float32)
    sigma_eta_sq_truth = sigma_eta_truth ** 2

    y_obs_2d = gen_svssm_data(T, mu_truth, phi_truth, sigma_eta_truth, seed=42)
    print(f"y_obs shape: {y_obs_2d.shape}  range: "
          f"[{float(tf.reduce_min(y_obs_2d)):.3f}, {float(tf.reduce_max(y_obs_2d)):.3f}]")

    # ----- 1) basic filter call -----
    print("\n[1] basic filter call at truth ...")
    ll_nd = DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=d, num_particles=N, n_lambda=10,
        sinkhorn_epsilon=1.0, sinkhorn_iters=10,
        grad_window=4, jit_compile=True,
    )
    mu_t  = tf.constant(mu_truth, tf.float32)
    phi_t = tf.constant(phi_truth, tf.float32)
    sig_t = tf.constant(sigma_eta_sq_truth, tf.float32)

    tf.random.set_seed(123)
    v_nd = float(ll_nd(mu_t, phi_t, sig_t, y_obs_2d).numpy())
    print(f"    log p (multivariate) = {v_nd:.4f}")
    if not np.isfinite(v_nd):
        raise RuntimeError("multivariate log p is not finite")

    # ----- 2) gradient check -----
    print("\n[2] forward+grad through parameters ...")
    theta_raw = tf.constant(
        np.concatenate([
            mu_truth,                      # (d,)
            np.arctanh(phi_truth),         # (d,) — unconstrained phi
            np.log(sigma_eta_sq_truth),    # (d,)
        ]).astype(np.float32)
    )
    with tf.GradientTape() as tape:
        tape.watch(theta_raw)
        mu_p   = theta_raw[:d]
        phi_p  = tf.tanh(theta_raw[d:2*d])
        sig_p  = tf.exp(theta_raw[2*d:3*d])
        v = ll_nd(mu_p, phi_p, sig_p, y_obs_2d)
    g = tape.gradient(v, theta_raw).numpy()
    print(f"    grad = {g}")
    if not np.all(np.isfinite(g)):
        raise RuntimeError(f"non-finite gradient: {g}")
    print(f"    log p (multivariate, gradient pass) = {float(v.numpy()):.4f}")

    # ----- 3) compare to sum of two d=1 filters on per-component data -----
    print("\n[3] compare to sum of two d=1 filter log p's ...")
    ll_1d = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N, n_lambda=10, sinkhorn_epsilon=1.0, sinkhorn_iters=10,
        grad_window=4, jit_compile=True, integrator="exp",
    )
    # The 2-D filter shares one set of N particles across both components, so
    # the per-step coupling differs from independent 1-D filters (resample is
    # joint in 2-D, separate in 1-D). We expect AGREEMENT within MC noise but
    # not exact equality. MC noise at T=20 is roughly 0.3 per component.
    per_comp_logps = []
    for i in range(d):
        y_i = y_obs_2d[:, i]
        mu_i  = tf.constant(mu_truth[i], tf.float32)
        phi_i = tf.constant(phi_truth[i], tf.float32)
        sig_i = tf.constant(sigma_eta_sq_truth[i], tf.float32)
        tf.random.set_seed(123 + i)
        v_i = float(ll_1d(mu_i, phi_i, sig_i, y_i).numpy())
        per_comp_logps.append(v_i)
        print(f"    component {i}: log p (d=1) = {v_i:.4f}")
    sum_1d = sum(per_comp_logps)
    print(f"    sum (d=1 components) = {sum_1d:.4f}")
    print(f"    multivariate         = {v_nd:.4f}")
    diff = abs(sum_1d - v_nd)
    print(f"    |Δ| = {diff:.4f}  "
          f"(expected ~ MC noise of LEDH-OT, roughly 0.3 per component)")

    # Rough sanity bound: within 5x typical MC noise (somewhat generous; the
    # joint Sinkhorn resample in d=2 genuinely differs from per-component).
    bound = 5.0 * d * 0.3
    if diff > bound:
        print(f"    WARNING: |Δ| > {bound:.2f} — multivariate and"
              f" sum-of-d=1 disagree more than MC noise should allow.")
    else:
        print(f"    PASS: |Δ| within {bound:.2f}.")

    print("\nSMOKE TEST PASSED" if diff <= bound else "\nSMOKE TEST SOFT FAIL (see warning)")
    return 0 if diff <= bound else 1


if __name__ == "__main__":
    raise SystemExit(main())
