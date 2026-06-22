"""
Multivariate V2 SVSSM identifiability demo (section3 §11 multivariate
empirical, the analogue of exp_v2_identifiability_demo.py for 1D).

Setup: d_y = d_h = 2, diagonal Phi, diagonal Sigma_eta, diagonal Sigma_eps,
FULL 2x2 matrix A. Model:

    h_t = mu + Phi @ (h_{t-1} - mu) + L_eta @ eta_t,    eta_t ~ N(0, I_2)
    y_t = A @ h_t + L_eps @ eps_t,                       eps_t ~ N(0, I_2)

where Phi = diag(phi_1, phi_2), L_eta = diag(sigma_eta_1, sigma_eta_2),
L_eps = diag(sigma_eps_1, sigma_eps_2), and A is unrestricted in the FREE
run vs. fixed at A = I_2 in the RESTRICTED run.

The diagonal Phi case has invariance T = diag(c_1, c_2) per latent
component (§11.4). The FREE posterior should walk a 2-dim ridge; the
A = I restriction should collapse the ridge entirely.

Predicted ridge collapse:
  - mu_1 marginal sd should drop substantially (it scales with c_1)
  - mu_2 marginal sd should drop substantially (it scales with c_2)
  - sigma_eta_1^2 marginal sd should drop substantially (it scales with c_1^2)
  - sigma_eta_2^2 marginal sd should drop substantially (it scales with c_2^2)
  - phi_1, phi_2, sigma_eps_1^2, sigma_eps_2^2 should be invariant
    (off the ridge, same posterior in both runs)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfm = tfp.mcmc

LOG_2PI = math.log(2.0 * math.pi)
D_H = 2
D_Y = 2


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def gen_v2_mv_data(T, mu, phi, sigma_eta, A, sigma_eps, seed=42):
    """Generate multivariate V2 data.
    mu, phi, sigma_eta, sigma_eps: length-d_h arrays (mu, sigma_eps also d_y).
    A: (d_y, d_h) array.
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    sigma_eta = np.asarray(sigma_eta, dtype=np.float64)
    sigma_eps = np.asarray(sigma_eps, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)

    # Stationary init: h_0 ~ N(mu, diag(sigma_eta^2 / (1 - phi^2)))
    h = mu + (sigma_eta / np.sqrt(1.0 - phi ** 2)) * rng.standard_normal(D_H)
    ys = np.zeros((T, D_Y), dtype=np.float32)
    hs = np.zeros((T, D_H), dtype=np.float32)
    for t in range(T):
        h = mu + phi * (h - mu) + sigma_eta * rng.standard_normal(D_H)
        hs[t] = h.astype(np.float32)
        ys[t] = (A @ h + sigma_eps * rng.standard_normal(D_Y)).astype(np.float32)
    return tf.constant(ys, tf.float32), hs


# ---------------------------------------------------------------------------
# Multivariate Kalman log-likelihood (diagonal Phi, Sigma_eta, Sigma_eps)
# ---------------------------------------------------------------------------

def kalman_v2_mv_loglik(mu, phi, sigma_eta_sq, A, sigma_eps_sq, y):
    """Multivariate Kalman log-likelihood for d_y = d_h = 2.

    mu, phi, sigma_eta_sq, sigma_eps_sq : (2,) tensors (diagonal entries).
    A : (2, 2) tensor.
    y : (T, 2) tensor of observations.
    """
    one_minus_phi_sq = tf.maximum(1.0 - phi * phi, 1e-6)
    # Stationary cov: diag(sigma_eta^2 / (1 - phi^2))
    P0 = tf.linalg.diag(sigma_eta_sq / one_minus_phi_sq)
    m0 = mu  # (2,)

    Phi_mat = tf.linalg.diag(phi)
    Q = tf.linalg.diag(sigma_eta_sq)       # process noise covariance
    R = tf.linalg.diag(sigma_eps_sq)       # observation noise covariance
    I_dh = tf.eye(D_H, dtype=tf.float32)

    T = tf.shape(y)[0]
    ll0 = tf.constant(0.0, tf.float32)
    i0 = tf.constant(0, tf.int32)

    def cond(i, m, P, ll):
        return i < T

    def body(i, m, P, ll):
        y_t = y[i]  # (2,)
        # Predict
        m_pred = mu + phi * (m - mu)                 # (2,)
        P_pred = Phi_mat @ P @ tf.transpose(Phi_mat) + Q   # (2,2)
        # Innovation
        innov = y_t - tf.linalg.matvec(A, m_pred)    # (2,)
        # Innovation cov
        S = A @ P_pred @ tf.transpose(A) + R         # (2,2)
        # Cholesky for log-det and inverse-application
        L = tf.linalg.cholesky(S)                    # (2,2)
        # Solve L z = innov  ->  z = L^{-1} innov
        z = tf.linalg.triangular_solve(L, innov[:, None], lower=True)[:, 0]
        # log|S| = 2 * sum(log(diag(L)))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        # Kalman gain K = P_pred @ A^T @ S^{-1}
        # We compute K via Cholesky solve: solve S X = A P_pred ^T then K = X^T
        # Easier: K = P_pred @ A^T @ S^{-1}; we apply S^{-1} via solve.
        AP = A @ P_pred                              # (d_y, d_h) — = A P_pred when P_pred (d_h,d_h)
        # Actually we want P_pred @ A^T, shape (d_h, d_y)
        Pat = P_pred @ tf.transpose(A)               # (d_h, d_y)
        # K = Pat @ S^{-1} ; equivalently  S^T @ K^T = Pat^T -> use solve
        # Since S is symmetric: solve S X = Pat^T then K = X^T
        X = tf.linalg.cholesky_solve(L, tf.transpose(Pat))   # (d_y, d_h)
        K = tf.transpose(X)                           # (d_h, d_y)
        # Update
        m_new = m_pred + tf.linalg.matvec(K, innov)
        P_new = (I_dh - K @ A) @ P_pred
        # ll contribution: -0.5 * (d_y * log(2pi) + log|S| + z^T z)
        ll_contrib = -0.5 * (float(D_Y) * LOG_2PI + log_det
                             + tf.reduce_sum(z * z))
        return i + 1, m_new, P_new, ll + ll_contrib

    _, _, _, log_lik = tf.while_loop(cond, body, [i0, m0, P0, ll0],
                                     maximum_iterations=T)
    return log_lik


# ---------------------------------------------------------------------------
# Target factories
# ---------------------------------------------------------------------------

def make_target_free_A(y_obs):
    """12-param FREE A:
        theta_raw = [mu1, mu2,
                     phi1_raw, phi2_raw,
                     log_sig_eta1_sq, log_sig_eta2_sq,
                     A11, A12, A21, A22,
                     log_sig_eps1_sq, log_sig_eps2_sq]
    """
    # Wide priors. On A entries we use Normal(0, 5) — not constrained to
    # positive — so the chain can explore the full A-space and the ridge.
    p_mu = tfd.Normal(0.0, 5.0)
    p_phi_raw = tfd.Normal(0.0, 2.0)
    p_log_s2 = tfd.Normal(-2.0, 2.0)
    p_A = tfd.Normal(0.0, 5.0)
    p_log_se2 = tfd.Normal(-2.0, 2.0)

    @tf.function
    def target(theta_raw):
        mu = theta_raw[0:2]
        phi_raw = theta_raw[2:4]
        log_s2 = theta_raw[4:6]
        A_flat = theta_raw[6:10]
        log_se2 = theta_raw[10:12]
        phi = tf.tanh(phi_raw)
        sigma_eta_sq = tf.exp(log_s2)
        sigma_eps_sq = tf.exp(log_se2)
        A = tf.reshape(A_flat, [D_Y, D_H])

        lp = (tf.reduce_sum(p_mu.log_prob(mu))
              + tf.reduce_sum(p_phi_raw.log_prob(phi_raw))
              + tf.reduce_sum(p_log_s2.log_prob(log_s2))
              + tf.reduce_sum(p_A.log_prob(A_flat))
              + tf.reduce_sum(p_log_se2.log_prob(log_se2)))
        ll = kalman_v2_mv_loglik(mu, phi, sigma_eta_sq, A, sigma_eps_sq, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float32))
        return lp + ll

    return target


def make_target_fixed_A_identity(y_obs):
    """8-param FIXED A=I_2:
        theta_raw = [mu1, mu2, phi1_raw, phi2_raw,
                     log_sig_eta1_sq, log_sig_eta2_sq,
                     log_sig_eps1_sq, log_sig_eps2_sq]
    """
    p_mu = tfd.Normal(0.0, 5.0)
    p_phi_raw = tfd.Normal(0.0, 2.0)
    p_log_s2 = tfd.Normal(-2.0, 2.0)
    p_log_se2 = tfd.Normal(-2.0, 2.0)
    A_I = tf.eye(D_H, dtype=tf.float32)

    @tf.function
    def target(theta_raw):
        mu = theta_raw[0:2]
        phi_raw = theta_raw[2:4]
        log_s2 = theta_raw[4:6]
        log_se2 = theta_raw[6:8]
        phi = tf.tanh(phi_raw)
        sigma_eta_sq = tf.exp(log_s2)
        sigma_eps_sq = tf.exp(log_se2)

        lp = (tf.reduce_sum(p_mu.log_prob(mu))
              + tf.reduce_sum(p_phi_raw.log_prob(phi_raw))
              + tf.reduce_sum(p_log_s2.log_prob(log_s2))
              + tf.reduce_sum(p_log_se2.log_prob(log_se2)))
        ll = kalman_v2_mv_loglik(mu, phi, sigma_eta_sq, A_I, sigma_eps_sq, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float32))
        return lp + ll

    return target


# ---------------------------------------------------------------------------
# HMC runner
# ---------------------------------------------------------------------------

def run_hmc(target_fn, init_raw, num_burn, num_samp, step_size, L, seed):
    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_fn,
        step_size=tf.constant(step_size, tf.float32),
        num_leapfrog_steps=int(L),
    )
    kernel = tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(num_burn),
        target_accept_prob=tf.constant(0.65, tf.float32),
    )

    def trace_fn(_, kr):
        inner = kr.inner_results if hasattr(kr, "inner_results") else kr
        acc = tf.cast(inner.is_accepted, tf.bool) \
            if hasattr(inner, "is_accepted") else tf.constant(False)
        return acc

    init = tf.constant(np.asarray(init_raw, dtype=np.float32))
    samples, accepted = tfm.sample_chain(
        num_results=int(num_samp),
        num_burnin_steps=int(num_burn),
        current_state=init,
        kernel=kernel,
        trace_fn=trace_fn,
        seed=int(seed),
    )
    return samples.numpy(), accepted.numpy()


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(samples, names, truth_dict=None):
    flat = samples.reshape(-1, samples.shape[-1])
    rows = []
    for j, n in enumerate(names):
        col = flat[:, j]
        row = {
            "param": n,
            "median": float(np.median(col)),
            "mean": float(np.mean(col)),
            "sd": float(np.std(col)),
            "q025": float(np.percentile(col, 2.5)),
            "q975": float(np.percentile(col, 97.5)),
        }
        if truth_dict is not None and n in truth_dict:
            row["truth"] = float(truth_dict[n])
            row["covered"] = bool(row["q025"] <= truth_dict[n] <= row["q975"])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, nargs=2, default=[1.0, 1.0])
    p.add_argument("--phi", type=float, nargs=2, default=[0.9, 0.85])
    p.add_argument("--sigma_eta", type=float, nargs=2, default=[0.5, 0.6])
    # Full 2x2 A: [[A11, A12], [A21, A22]]
    p.add_argument("--A", type=float, nargs=4, default=[2.0, 0.0, 0.0, 2.0],
                   help="Flat A in row-major: A11 A12 A21 A22")
    p.add_argument("--sigma_eps", type=float, nargs=2, default=[0.3, 0.3])
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--num_chains", type=int, default=2)
    p.add_argument("--num_burnin", type=int, default=600)
    p.add_argument("--num_results", type=int, default=1500)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--step_size", type=float, default=0.03)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--dispersion", type=float, default=0.15)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "v2_multivariate_demo")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_true = np.asarray(args.mu)
    phi_true = np.asarray(args.phi)
    sigma_eta_true = np.asarray(args.sigma_eta)
    A_true = np.asarray(args.A).reshape(2, 2)
    sigma_eps_true = np.asarray(args.sigma_eps)

    print(f"[v2-mv-demo] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  truth:")
    print(f"    mu        = {mu_true.tolist()}")
    print(f"    phi       = {phi_true.tolist()}")
    print(f"    sigma_eta = {sigma_eta_true.tolist()}")
    print(f"    A         = {A_true.tolist()}")
    print(f"    sigma_eps = {sigma_eps_true.tolist()}")

    # ---- data ----
    y_obs, hs = gen_v2_mv_data(args.T, mu_true, phi_true, sigma_eta_true,
                                A_true, sigma_eps_true, seed=args.data_seed)
    print(f"  y_obs shape: {y_obs.shape}, range: "
          f"[{float(tf.reduce_min(y_obs)):.3f}, {float(tf.reduce_max(y_obs)):.3f}]")

    rng = np.random.default_rng(args.base_seed)

    # ---- Predicted ridge under A=I restriction ----
    # The invariance T = diag(c_1, c_2) under FREE A means under A=I we
    # converge to the c values that solve A_true @ T^{-1} = I, i.e.
    # T = A_true (if A_true is diagonal). For A_true = diag(2, 2):
    # c_i = 2 for both, so:
    #   mu*_i        = 2 * mu_true_i
    #   sigma_eta*_i = 2 * sigma_eta_true_i
    #   A*           = A_true @ diag(1/c) = I  (if A_true diagonal)
    # For non-diagonal A_true the mapping is more complex.
    if abs(A_true[0, 1]) < 1e-6 and abs(A_true[1, 0]) < 1e-6:
        c_vec = np.array([A_true[0, 0], A_true[1, 1]])
        ridge_AI = {
            "mu_1": c_vec[0] * mu_true[0], "mu_2": c_vec[1] * mu_true[1],
            "phi_1": phi_true[0], "phi_2": phi_true[1],
            "sigma_eta_1_sq": (c_vec[0] * sigma_eta_true[0]) ** 2,
            "sigma_eta_2_sq": (c_vec[1] * sigma_eta_true[1]) ** 2,
            "sigma_eps_1_sq": sigma_eps_true[0] ** 2,
            "sigma_eps_2_sq": sigma_eps_true[1] ** 2,
        }
    else:
        ridge_AI = None
        print("  [warning] A_true is non-diagonal; ridge prediction not computed")
    print(f"  predicted A=I ridge: {ridge_AI}")

    # ============ Run 1: FREE A (12 params) ============
    print("\n========== FREE A (12 params) ==========")
    target_free = make_target_free_A(y_obs)
    # Init near truth in raw space
    truth_raw_free = np.concatenate([
        mu_true,
        np.arctanh(np.clip(phi_true, -0.999, 0.999)),
        np.log(np.maximum(sigma_eta_true ** 2, 1e-8)),
        A_true.flatten(),
        np.log(np.maximum(sigma_eps_true ** 2, 1e-8)),
    ]).astype(np.float32)
    print(f"  truth_raw_free: {truth_raw_free.tolist()}")
    samples_free = []
    accs_free = []
    t0 = time.perf_counter()
    for c in range(args.num_chains):
        init = truth_raw_free + args.dispersion * rng.standard_normal(12).astype(np.float32)
        seed = args.base_seed + 1009 * (c + 1)
        s, a = run_hmc(target_free, init, args.num_burnin, args.num_results,
                       args.step_size, args.L, seed)
        print(f"  [FREE chain {c+1}/{args.num_chains}] accept={a.mean():.3f} "
              f"elapsed={time.perf_counter()-t0:.1f}s")
        samples_free.append(s)
        accs_free.append(a)
    samples_free = np.stack(samples_free, axis=0)  # (chains, draws, 12)

    # Constrained: [mu1, mu2, phi1, phi2, s2_1, s2_2, A11, A12, A21, A22, se2_1, se2_2]
    free_const = np.stack([
        samples_free[..., 0], samples_free[..., 1],
        np.tanh(samples_free[..., 2]), np.tanh(samples_free[..., 3]),
        np.exp(samples_free[..., 4]), np.exp(samples_free[..., 5]),
        samples_free[..., 6], samples_free[..., 7],
        samples_free[..., 8], samples_free[..., 9],
        np.exp(samples_free[..., 10]), np.exp(samples_free[..., 11]),
    ], axis=-1)
    free_names = ["mu_1", "mu_2", "phi_1", "phi_2",
                  "sigma_eta_1_sq", "sigma_eta_2_sq",
                  "A_11", "A_12", "A_21", "A_22",
                  "sigma_eps_1_sq", "sigma_eps_2_sq"]
    free_truth = {
        "mu_1": mu_true[0], "mu_2": mu_true[1],
        "phi_1": phi_true[0], "phi_2": phi_true[1],
        "sigma_eta_1_sq": sigma_eta_true[0] ** 2,
        "sigma_eta_2_sq": sigma_eta_true[1] ** 2,
        "A_11": A_true[0, 0], "A_12": A_true[0, 1],
        "A_21": A_true[1, 0], "A_22": A_true[1, 1],
        "sigma_eps_1_sq": sigma_eps_true[0] ** 2,
        "sigma_eps_2_sq": sigma_eps_true[1] ** 2,
    }
    free_summary = summarize(free_const, free_names, free_truth)

    print("\n=== FREE A: marginals ===")
    for r in free_summary:
        tval = r.get("truth", float("nan"))
        ck = "OK" if r.get("covered", False) else "NOT-cov"
        print(f"  {r['param']:>15}  truth={tval:>8.4f}  "
              f"med={r['median']:>8.4f}  sd={r['sd']:>8.4f}  "
              f"CI=[{r['q025']:>8.4f}, {r['q975']:>8.4f}]  {ck}")

    # ============ Run 2: FIXED A=I_2 (8 params) ============
    print("\n========== FIXED A=I (8 params) ==========")
    target_fixed = make_target_fixed_A_identity(y_obs)
    # Init near the ridge prediction
    if ridge_AI is not None:
        truth_raw_fixed = np.asarray([
            ridge_AI["mu_1"], ridge_AI["mu_2"],
            np.arctanh(np.clip(phi_true[0], -0.999, 0.999)),
            np.arctanh(np.clip(phi_true[1], -0.999, 0.999)),
            np.log(max(ridge_AI["sigma_eta_1_sq"], 1e-8)),
            np.log(max(ridge_AI["sigma_eta_2_sq"], 1e-8)),
            np.log(max(ridge_AI["sigma_eps_1_sq"], 1e-8)),
            np.log(max(ridge_AI["sigma_eps_2_sq"], 1e-8)),
        ], dtype=np.float32)
    else:
        truth_raw_fixed = np.zeros(8, dtype=np.float32)
    print(f"  truth_raw_fixed: {truth_raw_fixed.tolist()}")
    samples_fixed = []
    accs_fixed = []
    t0 = time.perf_counter()
    for c in range(args.num_chains):
        init = truth_raw_fixed + args.dispersion * rng.standard_normal(8).astype(np.float32)
        seed = args.base_seed + 1009 * (c + 1) + 7
        s, a = run_hmc(target_fixed, init, args.num_burnin, args.num_results,
                       args.step_size, args.L, seed)
        print(f"  [FIXED chain {c+1}/{args.num_chains}] accept={a.mean():.3f} "
              f"elapsed={time.perf_counter()-t0:.1f}s")
        samples_fixed.append(s)
        accs_fixed.append(a)
    samples_fixed = np.stack(samples_fixed, axis=0)  # (chains, draws, 8)

    # Constrained: [mu1, mu2, phi1, phi2, s2_1, s2_2, se2_1, se2_2]
    fixed_const = np.stack([
        samples_fixed[..., 0], samples_fixed[..., 1],
        np.tanh(samples_fixed[..., 2]), np.tanh(samples_fixed[..., 3]),
        np.exp(samples_fixed[..., 4]), np.exp(samples_fixed[..., 5]),
        np.exp(samples_fixed[..., 6]), np.exp(samples_fixed[..., 7]),
    ], axis=-1)
    fixed_names = ["mu_1", "mu_2", "phi_1", "phi_2",
                   "sigma_eta_1_sq", "sigma_eta_2_sq",
                   "sigma_eps_1_sq", "sigma_eps_2_sq"]
    fixed_truth = {
        "mu_1": ridge_AI["mu_1"] if ridge_AI else float("nan"),
        "mu_2": ridge_AI["mu_2"] if ridge_AI else float("nan"),
        "phi_1": phi_true[0], "phi_2": phi_true[1],
        "sigma_eta_1_sq": ridge_AI["sigma_eta_1_sq"] if ridge_AI else float("nan"),
        "sigma_eta_2_sq": ridge_AI["sigma_eta_2_sq"] if ridge_AI else float("nan"),
        "sigma_eps_1_sq": sigma_eps_true[0] ** 2,
        "sigma_eps_2_sq": sigma_eps_true[1] ** 2,
    }
    fixed_summary = summarize(fixed_const, fixed_names, fixed_truth)

    print("\n=== FIXED A=I: marginals ===")
    for r in fixed_summary:
        tval = r.get("truth", float("nan"))
        ck = "OK" if r.get("covered", False) else "NOT-cov"
        print(f"  {r['param']:>15}  truth={tval:>8.4f}  "
              f"med={r['median']:>8.4f}  sd={r['sd']:>8.4f}  "
              f"CI=[{r['q025']:>8.4f}, {r['q975']:>8.4f}]  {ck}")

    # ============ Ridge collapse table ============
    print("\n=== Ridge collapse: sd(FREE) / sd(FIXED) ===")
    for p in ["mu_1", "mu_2", "phi_1", "phi_2",
              "sigma_eta_1_sq", "sigma_eta_2_sq",
              "sigma_eps_1_sq", "sigma_eps_2_sq"]:
        free_sd = next(r["sd"] for r in free_summary if r["param"] == p)
        fixed_sd = next(r["sd"] for r in fixed_summary if r["param"] == p)
        ratio = free_sd / fixed_sd if fixed_sd > 0 else float("nan")
        print(f"  {p:>20}  free_sd={free_sd:>8.4f}  fixed_sd={fixed_sd:>8.4f}  "
              f"ratio={ratio:>6.2f}x")

    # ============ Save outputs ============
    np.savez_compressed(out_dir / "v2_mv_samples.npz",
                        free_const=free_const, free_raw=samples_free,
                        fixed_const=fixed_const, fixed_raw=samples_fixed,
                        free_accept=np.stack(accs_free),
                        fixed_accept=np.stack(accs_fixed),
                        y_obs=y_obs.numpy(), hs_true=hs)
    result = {
        "tf_version": tf.__version__,
        "tfp_version": tfp.__version__,
        "truth": {
            "mu": mu_true.tolist(), "phi": phi_true.tolist(),
            "sigma_eta": sigma_eta_true.tolist(),
            "A": A_true.tolist(), "sigma_eps": sigma_eps_true.tolist(),
        },
        "ridge_AI_prediction": ridge_AI,
        "config": {
            "d_h": D_H, "d_y": D_Y,
            "T": args.T, "num_chains": args.num_chains,
            "num_burnin": args.num_burnin, "num_results": args.num_results,
            "L": args.L, "step_size": args.step_size,
            "data_seed": args.data_seed, "base_seed": args.base_seed,
        },
        "free_summary": free_summary,
        "fixed_summary": fixed_summary,
        "accept_rates": {
            "free": float(np.mean([a.mean() for a in accs_free])),
            "fixed": float(np.mean([a.mean() for a in accs_fixed])),
        },
    }
    with open(out_dir / "v2_mv_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote: {out_dir}/v2_mv_result.json")
    print(f"       {out_dir}/v2_mv_samples.npz")


if __name__ == "__main__":
    raise SystemExit(main())
