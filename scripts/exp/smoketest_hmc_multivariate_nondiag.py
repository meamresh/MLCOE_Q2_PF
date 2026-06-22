"""
Smoke HMC at d=2 with non-diagonal Σ_η.

Parameter vector at d=2 (7 unconstrained reals):
    theta_raw = [ mu_0, mu_1,
                  phi_raw_0, phi_raw_1,
                  log_L00, L_10, log_L11 ]
    phi_diag = tanh(phi_raw)
    L_eta = [[ exp(log_L00),        0          ],
             [ L_10,                exp(log_L11) ]]
    Sigma_eta = L_eta @ L_eta^T  (PSD by construction)

Truth: μ=(0, -0.3), φ=(0.95, 0.85), σ_eta diag (0.3, 0.4) with corr ρ=0.5.
That is Σ_η = [[0.09, 0.06], [0.06, 0.16]],
which gives L_eta = [[0.3, 0], [0.2, 0.346]].

We run a short HMC chain (laptop, 1 chain × 100 burnin + 100 samples)
just to verify mechanics + recovery is reasonable.
"""

from __future__ import annotations

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
)

tfd = tfp.distributions
tfm = tfp.mcmc


def gen_data_corr(T, mu, phi, L_eta, seed=42):
    tf.random.set_seed(seed)
    d = len(mu)
    h = tf.identity(mu)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + tf.linalg.matvec(L_eta, tf.random.normal([d]))
        ys.append(tf.exp(h / 2.0) * tf.random.normal([d]))
    return tf.stack(ys, axis=0)


def theta_to_chol(theta_raw, d):
    """Map (3,) → (2, 2) lower-triangular Cholesky factor (d=2 specific)."""
    # theta_raw = [log_L00, L_10, log_L11]
    L00 = tf.exp(theta_raw[0])
    L10 = theta_raw[1]
    L11 = tf.exp(theta_raw[2])
    L = tf.stack([
        tf.stack([L00, tf.zeros_like(L00)]),
        tf.stack([L10, L11]),
    ])
    return L


def main():
    T, N, d = 20, 64, 2
    print(f"TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  d={d}  T={T}  N={N}")

    mu_truth = np.array([0.0, -0.3], dtype=np.float32)
    phi_truth = np.array([0.95, 0.85], dtype=np.float32)
    rho_truth = 0.5
    sigma_eta_diag_truth = np.array([0.3, 0.4], dtype=np.float32)
    Sigma_eta_truth = np.array([
        [sigma_eta_diag_truth[0]**2, rho_truth * sigma_eta_diag_truth[0] * sigma_eta_diag_truth[1]],
        [rho_truth * sigma_eta_diag_truth[0] * sigma_eta_diag_truth[1], sigma_eta_diag_truth[1]**2],
    ], dtype=np.float32)
    L_eta_truth = np.linalg.cholesky(Sigma_eta_truth)
    print(f"  truth: mu={mu_truth}, phi={phi_truth}, rho={rho_truth}")
    print(f"  Sigma_eta_truth:\n{Sigma_eta_truth}")
    print(f"  L_eta_truth:\n{L_eta_truth}")

    # Build truth_raw (7-vector)
    truth_raw = np.array([
        mu_truth[0], mu_truth[1],
        np.arctanh(phi_truth[0]), np.arctanh(phi_truth[1]),
        np.log(L_eta_truth[0, 0]),  # log L_00
        L_eta_truth[1, 0],            # L_10 (free)
        np.log(L_eta_truth[1, 1]),  # log L_11
    ], dtype=np.float32)
    print(f"  truth_raw (7-vec): {truth_raw}")

    # Generate data
    y_obs = gen_data_corr(T, tf.constant(mu_truth), tf.constant(phi_truth),
                          tf.constant(L_eta_truth), seed=42)
    print(f"  y_obs shape: {tuple(y_obs.shape)}  range: "
          f"[{float(tf.reduce_min(y_obs)):.3f}, {float(tf.reduce_max(y_obs)):.3f}]")

    # Filter
    ll = DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=d, num_particles=N, n_lambda=10,
        sinkhorn_epsilon=1.0, sinkhorn_iters=10,
        grad_window=4, jit_compile=True,   # diagonal path is JIT'd; full path runs eager (expm)
    )

    # Sanity at truth
    L_truth_tf = tf.constant(L_eta_truth, tf.float32)
    mu_t = tf.constant(mu_truth, tf.float32)
    phi_t = tf.constant(phi_truth, tf.float32)
    tf.random.set_seed(123)
    v0 = float(ll.call_full(mu_t, phi_t, L_truth_tf, y_obs).numpy())
    print(f"  sanity: log p at truth = {v0:.4f}")

    # Priors (per-component analogues of the univariate setup)
    prior_mu = tfd.Normal(loc=0.0, scale=1.0)
    prior_phi_raw = tfd.Normal(loc=2.0, scale=0.5)
    # Cholesky params: log_L00 ~ N(-1.2, 1), L_10 ~ N(0, 0.3), log_L11 ~ N(-0.95, 1)
    # Centered roughly on truth in unconstrained space.
    prior_log_L00 = tfd.Normal(loc=-1.2, scale=1.0)
    prior_L10     = tfd.Normal(loc=0.0,  scale=0.3)
    prior_log_L11 = tfd.Normal(loc=-0.95, scale=1.0)

    crn_seed = 300

    def target_log_prob(theta_raw):
        mu              = theta_raw[:2]
        phi_raw         = theta_raw[2:4]
        chol_raw        = theta_raw[4:7]
        phi_diag = tf.tanh(phi_raw)
        L = theta_to_chol(chol_raw, d)

        log_prior = (
            tf.reduce_sum(prior_mu.log_prob(mu))
            + tf.reduce_sum(prior_phi_raw.log_prob(phi_raw))
            + prior_log_L00.log_prob(chol_raw[0])
            + prior_L10.log_prob(chol_raw[1])
            + prior_log_L11.log_prob(chol_raw[2])
        )
        tf.random.set_seed(crn_seed)
        log_lik = ll.call_full(mu, phi_diag, L, y_obs)
        return log_prior + log_lik

    # Test target_log_prob at truth and a perturbed point
    print(f"  target_log_prob(truth) = {float(target_log_prob(tf.constant(truth_raw)).numpy()):.4f}")

    # HMC
    print("\n  Running HMC: 1 chain × 100 burnin + 100 results ...")
    inner = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=0.05, num_leapfrog_steps=5,
    )
    kernel = tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=inner, num_adaptation_steps=100,
    )

    init_raw = truth_raw + 0.10 * np.random.default_rng(300).standard_normal(7).astype(np.float32)
    state = tf.constant(init_raw)
    results = kernel.bootstrap_results(state)

    samples = []
    accs = []
    t0 = time.perf_counter()
    for s in range(200):
        state, results = kernel.one_step(state, results)
        if s >= 100:
            samples.append(state.numpy().copy())
        acc = float(np.exp(min(0.0, float(results.inner_results.log_accept_ratio.numpy()))))
        accs.append(acc)
        if (s + 1) % 25 == 0:
            phase = "burn" if s < 100 else "samp"
            ss = float(results.new_step_size.numpy()) if hasattr(results, "new_step_size") \
                else float(results.inner_results.accepted_results.step_size.numpy())
            print(f"    [{phase}] step {s+1}/200  eps={ss:.5f}  acc={acc:.1f}", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  chain done in {elapsed:.1f}s, overall accept rate = {np.mean(accs):.3f}")

    # Posterior summary on 7 parameters
    samples = np.asarray(samples, dtype=np.float32)
    constrained = np.zeros_like(samples)
    constrained[:, 0:2] = samples[:, 0:2]              # mu
    constrained[:, 2:4] = np.tanh(samples[:, 2:4])     # phi
    constrained[:, 4] = np.exp(samples[:, 4])          # L_00
    constrained[:, 5] = samples[:, 5]                   # L_10
    constrained[:, 6] = np.exp(samples[:, 6])          # L_11

    truth_constrained = np.array([
        mu_truth[0], mu_truth[1],
        phi_truth[0], phi_truth[1],
        L_eta_truth[0, 0], L_eta_truth[1, 0], L_eta_truth[1, 1],
    ], dtype=np.float32)

    names = ["mu_0", "mu_1", "phi_0", "phi_1", "L_00", "L_10", "L_11"]
    print(f"\n  {'param':<10s} {'truth':>10s} {'mean':>10s} {'std':>10s} {'median':>10s} "
          f"{'2.5%':>10s} {'97.5%':>10s}  cov?")
    n_covered = 0
    for i, name in enumerate(names):
        samp = constrained[:, i]
        tval = float(truth_constrained[i])
        mean = float(samp.mean())
        std = float(samp.std())
        med = float(np.median(samp))
        q025 = float(np.quantile(samp, 0.025))
        q975 = float(np.quantile(samp, 0.975))
        covered = q025 <= tval <= q975
        if covered: n_covered += 1
        print(f"  {name:<10s} {tval:>10.4f} {mean:>10.4f} {std:>10.4f} {med:>10.4f} "
              f"{q025:>10.4f} {q975:>10.4f}  {'OK' if covered else 'OUT'}")

    # Implied correlation rho posterior
    rho_post = constrained[:, 5] / np.sqrt(constrained[:, 4]**2 + constrained[:, 5]**2 + 1e-8)
    # Wait this is wrong — the correlation is L_10 / sqrt(L_00^2 + L_10^2)?
    # Actually: Sigma_eta[0,1] = L_00 * L_10 (since L_00 * L_10 + 0 * L_11 = L_00 L_10)
    # Sigma_eta[0,0] = L_00^2
    # Sigma_eta[1,1] = L_10^2 + L_11^2
    # rho = Sigma_eta[0,1] / sqrt(Sigma_eta[0,0] * Sigma_eta[1,1])
    #     = L_00 L_10 / sqrt(L_00^2 (L_10^2 + L_11^2))
    #     = L_10 / sqrt(L_10^2 + L_11^2)
    rho_post = constrained[:, 5] / np.sqrt(constrained[:, 5]**2 + constrained[:, 6]**2 + 1e-12)
    print(f"\n  Implied correlation rho:")
    print(f"    truth = {rho_truth:.4f}")
    print(f"    mean  = {rho_post.mean():.4f}, median = {np.median(rho_post):.4f}, "
          f"95% CI = [{np.quantile(rho_post, 0.025):.4f}, {np.quantile(rho_post, 0.975):.4f}]")

    print(f"\n  Coverage: {n_covered}/{len(names)} parameters covered at 95% CI")
    print(f"  Total wall: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
