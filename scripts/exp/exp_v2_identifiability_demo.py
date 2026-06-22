"""
V2 SVSSM identifiability demo (section3 §11 empirical).

Model:
    h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,   eta_t ~ N(0,1)
    y_t = A * h_t + sigma_eps * eps_t,                     eps_t ~ N(0,1)

The 1D scale invariance: for any c > 0, the transformation
    (mu, sigma_eta, A) -> (c*mu, c*sigma_eta, A/c)
leaves the observed distribution of y_t unchanged. sigma_eps is invariant.

The demo runs two HMC chains on the SAME data:

  (1) FREE A: estimate (mu, phi, sigma_eta, A, sigma_eps) -> chain should
      slide along the ridge; mu, sigma_eta, A marginally wide; products
      A*mu and A^2*sigma_eta^2 identified to within MC noise.

  (2) FIXED A=1: estimate (mu, phi, sigma_eta, sigma_eps) -> all four
      parameters identified at the parametric rate.

The Kalman filter is exact for this linear-Gaussian model -- no particle
filter needed. JIT-compiled for HMC.
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


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def gen_v2_data(T: int, mu: float, phi: float, sigma_eta: float,
                A: float, sigma_eps: float, seed: int = 42):
    """Generate V2 1D data from the canonical model. Numpy for clarity."""
    rng = np.random.default_rng(seed)
    h = mu + (sigma_eta / math.sqrt(1.0 - phi ** 2)) * rng.standard_normal()
    ys = np.zeros(T, dtype=np.float32)
    hs = np.zeros(T, dtype=np.float32)
    for t in range(T):
        h = mu + phi * (h - mu) + sigma_eta * rng.standard_normal()
        hs[t] = h
        ys[t] = A * h + sigma_eps * rng.standard_normal()
    return tf.constant(ys, tf.float32), hs


# ---------------------------------------------------------------------------
# Exact Kalman log-likelihood for V2 1D
# ---------------------------------------------------------------------------

def kalman_v2_loglik(mu, phi, sigma_eta_sq, A, sigma_eps_sq, y):
    """Exact log p(y_{1:T} | theta) for the V2 1D linear-Gaussian SSM.

    Stationary init: h_0 ~ N(mu, sigma_eta^2 / (1 - phi^2)).
    Closed-form via Kalman forward recursion (no particle approximation).
    Uses tf.while_loop with scalar state so the whole call is XLA-compilable.
    """
    one_minus_phi_sq = tf.maximum(1.0 - phi * phi, 1e-6)
    T = tf.shape(y)[0]
    m0 = mu
    P0 = sigma_eta_sq / one_minus_phi_sq
    ll0 = tf.constant(0.0, tf.float32)
    i0 = tf.constant(0, tf.int32)

    def cond(i, m, P, ll):
        return i < T

    def body(i, m, P, ll):
        y_t = y[i]
        # Predict
        m_pred = mu + phi * (m - mu)
        P_pred = phi * phi * P + sigma_eta_sq
        # Innovation + variance
        innov = y_t - A * m_pred
        S = tf.maximum(A * A * P_pred + sigma_eps_sq, 1e-6)
        # Kalman gain + update
        K = A * P_pred / S
        m_new = m_pred + K * innov
        P_new = P_pred * (1.0 - A * K)
        # Log-likelihood contribution
        ll_contrib = -0.5 * (tf.math.log(S) + LOG_2PI + innov * innov / S)
        return i + 1, m_new, P_new, ll + ll_contrib

    _, _, _, log_lik = tf.while_loop(cond, body, [i0, m0, P0, ll0],
                                     maximum_iterations=T)
    return log_lik


# ---------------------------------------------------------------------------
# Target log-prob factories
# ---------------------------------------------------------------------------

def make_target_free_A(y_obs):
    """Five-parameter target: theta_raw = [mu, phi_raw, log_sigma_eta_sq,
                                          log_A, log_sigma_eps_sq].
    A = exp(log_A), so A > 0. Wide priors on every component.
    """
    p_mu      = tfd.Normal(0.0, 5.0)
    p_phi_raw = tfd.Normal(0.0, 2.0)
    p_log_s2  = tfd.Normal(-2.0, 2.0)
    p_log_A   = tfd.Normal(0.0, 5.0)   # very wide -> chain free to walk the ridge
    p_log_se2 = tfd.Normal(-2.0, 2.0)

    @tf.function
    def target(theta_raw):
        mu = theta_raw[0]
        phi_raw = theta_raw[1]
        log_s2 = theta_raw[2]
        log_A = theta_raw[3]
        log_se2 = theta_raw[4]

        phi = tf.tanh(phi_raw)
        sigma_eta_sq = tf.exp(log_s2)
        A = tf.exp(log_A)
        sigma_eps_sq = tf.exp(log_se2)

        lp = (p_mu.log_prob(mu)
              + p_phi_raw.log_prob(phi_raw)
              + p_log_s2.log_prob(log_s2)
              + p_log_A.log_prob(log_A)
              + p_log_se2.log_prob(log_se2))
        ll = kalman_v2_loglik(mu, phi, sigma_eta_sq, A, sigma_eps_sq, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float32))
        return lp + ll

    return target


def make_target_fixed_A1(y_obs):
    """Four-parameter target with A == 1 fixed.
    theta_raw = [mu, phi_raw, log_sigma_eta_sq, log_sigma_eps_sq].
    """
    p_mu      = tfd.Normal(0.0, 5.0)
    p_phi_raw = tfd.Normal(0.0, 2.0)
    p_log_s2  = tfd.Normal(-2.0, 2.0)
    p_log_se2 = tfd.Normal(-2.0, 2.0)
    A_one = tf.constant(1.0, tf.float32)

    @tf.function
    def target(theta_raw):
        mu = theta_raw[0]
        phi_raw = theta_raw[1]
        log_s2 = theta_raw[2]
        log_se2 = theta_raw[3]

        phi = tf.tanh(phi_raw)
        sigma_eta_sq = tf.exp(log_s2)
        sigma_eps_sq = tf.exp(log_se2)

        lp = (p_mu.log_prob(mu)
              + p_phi_raw.log_prob(phi_raw)
              + p_log_s2.log_prob(log_s2)
              + p_log_se2.log_prob(log_se2))
        ll = kalman_v2_loglik(mu, phi, sigma_eta_sq, A_one,
                              sigma_eps_sq, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float32))
        return lp + ll

    return target


# ---------------------------------------------------------------------------
# HMC runner
# ---------------------------------------------------------------------------

def run_hmc(target_fn, init_raw, num_burn, num_samp, step_size, L, seed):
    """Vanilla HMC with dual-averaging step-size adaptation."""
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
# Constrain helpers
# ---------------------------------------------------------------------------

def constrain_free(theta_raw):
    """theta_raw [mu, phi_raw, log_s2, log_A, log_se2]
       -> constrained [mu, phi, sigma_eta_sq, A, sigma_eps_sq]."""
    th = np.asarray(theta_raw)
    return np.stack([
        th[..., 0],
        np.tanh(th[..., 1]),
        np.exp(th[..., 2]),
        np.exp(th[..., 3]),
        np.exp(th[..., 4]),
    ], axis=-1)


def constrain_fixed(theta_raw):
    """theta_raw [mu, phi_raw, log_s2, log_se2]
       -> constrained [mu, phi, sigma_eta_sq, sigma_eps_sq] (A is fixed=1)."""
    th = np.asarray(theta_raw)
    return np.stack([
        th[..., 0],
        np.tanh(th[..., 1]),
        np.exp(th[..., 2]),
        np.exp(th[..., 3]),
    ], axis=-1)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_marginals(samples_constrained, names, truth=None):
    """samples_constrained: (chains, draws, dim).
       Return list of dicts per param: median, mean, sd, 95% CI, R-hat."""
    out = []
    flat = samples_constrained.reshape(-1, samples_constrained.shape[-1])
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
        if truth is not None and n in truth:
            row["truth"] = float(truth[n])
            row["covered"] = bool(row["q025"] <= truth[n] <= row["q975"])
        out.append(row)
    return out


def summarize_products(samples_constrained_free, truth):
    """Compute A*mu and A^2*sigma_eta_sq posteriors from the FREE samples.

    samples_constrained_free: (chains, draws, 5) with cols
        [mu, phi, sigma_eta_sq, A, sigma_eps_sq]
    """
    flat = samples_constrained_free.reshape(-1, 5)
    mu = flat[:, 0]
    sig_eta_sq = flat[:, 2]
    A = flat[:, 3]
    A_mu = A * mu
    A2_sig_eta_sq = A * A * sig_eta_sq
    rows = [
        {"name": "A*mu", "median": float(np.median(A_mu)),
         "sd": float(np.std(A_mu)),
         "q025": float(np.percentile(A_mu, 2.5)),
         "q975": float(np.percentile(A_mu, 97.5)),
         "truth": float(truth["A"] * truth["mu"])},
        {"name": "A^2*sigma_eta_sq", "median": float(np.median(A2_sig_eta_sq)),
         "sd": float(np.std(A2_sig_eta_sq)),
         "q025": float(np.percentile(A2_sig_eta_sq, 2.5)),
         "q975": float(np.percentile(A2_sig_eta_sq, 97.5)),
         "truth": float(truth["A"] ** 2 * truth["sigma_eta"] ** 2)},
    ]
    return rows


# ---------------------------------------------------------------------------
# Plotting (saved to disk)
# ---------------------------------------------------------------------------

def make_plots(out_dir: Path,
               samples_free: np.ndarray,
               samples_fixed: np.ndarray,
               truth: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1) Ridge plot: posterior on (log A, log sigma_eta_sq) and
    #         (log A, mu) for the FREE run ----
    free_flat = samples_free.reshape(-1, 5)
    log_A_free = np.log(np.clip(free_flat[:, 3], 1e-8, None))
    log_s2_free = np.log(np.clip(free_flat[:, 2], 1e-8, None))
    mu_free = free_flat[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(log_A_free, log_s2_free, s=2, alpha=0.3)
    axes[0].axvline(np.log(truth["A"]), color="r", ls="--", label="truth log A")
    axes[0].axhline(np.log(truth["sigma_eta"] ** 2), color="r", ls=":",
                    label="truth log sigma_eta_sq")
    # The ridge prediction: log_s2 + 2*log_A = constant
    c = np.log(truth["sigma_eta"] ** 2 * truth["A"] ** 2)
    log_A_range = np.linspace(log_A_free.min(), log_A_free.max(), 100)
    axes[0].plot(log_A_range, c - 2 * log_A_range,
                 color="green", ls="-", label="predicted ridge: "
                                             r"$\log\sigma^2 + 2\log A = c$")
    axes[0].set_xlabel("log A")
    axes[0].set_ylabel("log sigma_eta_sq")
    axes[0].set_title("FREE A: log sigma_eta_sq vs log A (ridge view)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(log_A_free, mu_free, s=2, alpha=0.3)
    axes[1].axvline(np.log(truth["A"]), color="r", ls="--", label="truth log A")
    axes[1].axhline(truth["mu"], color="r", ls=":", label="truth mu")
    # Ridge: mu - log_A*c0 ... actually mu * A = const ->
    # for log scale: mu = const_mu / A => mu * exp(log_A) = const_mu
    # so on (log_A, mu) the ridge is mu = const_mu * exp(-log_A)
    const_mu = truth["A"] * truth["mu"]
    if abs(const_mu) > 1e-6:
        mu_pred = const_mu * np.exp(-log_A_range)
        axes[1].plot(log_A_range, mu_pred, color="green",
                     label=r"predicted ridge: $A \mu = c$")
    axes[1].set_xlabel("log A")
    axes[1].set_ylabel("mu")
    axes[1].set_title("FREE A: mu vs log A (ridge view)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "v2_free_ridge.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- 2) Marginals comparison: FREE vs FIXED, on shared params ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fixed_flat = samples_fixed.reshape(-1, 4)
    free_flat = samples_free.reshape(-1, 5)

    # mu, phi, sigma_eta_sq, sigma_eps_sq in FREE columns 0, 1, 2, 4
    free_cols = [free_flat[:, 0], free_flat[:, 1], free_flat[:, 2], free_flat[:, 4]]
    fixed_cols = [fixed_flat[:, 0], fixed_flat[:, 1], fixed_flat[:, 2], fixed_flat[:, 3]]
    names = ["mu", "phi", "sigma_eta_sq", "sigma_eps_sq"]
    truth_vals = [truth["mu"], truth["phi"],
                  truth["sigma_eta"] ** 2, truth["sigma_eps"] ** 2]

    for ax, free_c, fixed_c, name, tval in zip(
            axes, free_cols, fixed_cols, names, truth_vals):
        ax.hist(free_c, bins=40, density=True, alpha=0.45,
                label="FREE A", color="C0")
        ax.hist(fixed_c, bins=40, density=True, alpha=0.45,
                label="A=1 fixed", color="C2")
        ax.axvline(tval, color="r", ls="--", label="truth")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v2_marginals_compare.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 3) Identified product posteriors from FREE ----
    A_mu = free_flat[:, 3] * free_flat[:, 0]
    A2_s2 = free_flat[:, 3] ** 2 * free_flat[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(A_mu, bins=40, density=True, alpha=0.7, color="C0")
    axes[0].axvline(truth["A"] * truth["mu"], color="r", ls="--", label="truth A*mu")
    axes[0].set_title("FREE A: posterior on identified A*mu")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(A2_s2, bins=40, density=True, alpha=0.7, color="C0")
    axes[1].axvline(truth["A"] ** 2 * truth["sigma_eta"] ** 2,
                    color="r", ls="--", label=r"truth $A^2 \sigma_\eta^2$")
    axes[1].set_title(r"FREE A: posterior on identified $A^2 \sigma_\eta^2$")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v2_identified_products.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--phi", type=float, default=0.9)
    p.add_argument("--sigma_eta", type=float, default=0.5)
    p.add_argument("--A", type=float, default=2.0,
                   help="True A. Set to 1 if you want truth on the canonical "
                        "restriction; set != 1 to make the ridge visible.")
    p.add_argument("--sigma_eps", type=float, default=0.3)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--num_chains", type=int, default=2)
    p.add_argument("--num_burnin", type=int, default=400)
    p.add_argument("--num_results", type=int, default=1000)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--step_size", type=float, default=0.05)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--dispersion", type=float, default=0.2)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "v2_identifiability_demo")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = dict(mu=args.mu, phi=args.phi, sigma_eta=args.sigma_eta,
                 A=args.A, sigma_eps=args.sigma_eps)
    print(f"[v2-id-demo] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  truth: {truth}")
    print(f"  T={args.T}  chains={args.num_chains}  "
          f"burn={args.num_burnin}  samp={args.num_results}")
    print(f"  out_dir={out_dir}")

    # ---------------- data ----------------
    y_obs, hs = gen_v2_data(args.T, args.mu, args.phi, args.sigma_eta,
                             args.A, args.sigma_eps, seed=args.data_seed)
    print(f"  y_obs range: [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")

    # ---------------- init states ----------------
    # Initialise near truth in unconstrained space, with small dispersion.
    truth_raw_free = np.asarray([
        args.mu,
        math.atanh(min(max(args.phi, -0.999), 0.999)),
        math.log(max(args.sigma_eta ** 2, 1e-8)),
        math.log(max(args.A, 1e-8)),
        math.log(max(args.sigma_eps ** 2, 1e-8)),
    ], dtype=np.float32)
    truth_raw_fixed = np.asarray([
        args.mu,
        math.atanh(min(max(args.phi, -0.999), 0.999)),
        # For A=1 mode the data-consistent sigma_eta is c*sigma_eta_true
        # where c = A_true. So initialise at log(A * sigma_eta)^2.
        math.log(max((args.A * args.sigma_eta) ** 2, 1e-8)),
        math.log(max(args.sigma_eps ** 2, 1e-8)),
    ], dtype=np.float32)
    truth_raw_fixed_mu_init = args.A * args.mu  # the identified A*mu

    rng = np.random.default_rng(args.base_seed)
    init_raws_free = [truth_raw_free + args.dispersion *
                      rng.standard_normal(5).astype(np.float32)
                      for _ in range(args.num_chains)]
    # For fixed mode, the data-consistent mu is A_true*mu_true (since A=1)
    truth_raw_fixed_seeded = truth_raw_fixed.copy()
    truth_raw_fixed_seeded[0] = truth_raw_fixed_mu_init
    init_raws_fixed = [truth_raw_fixed_seeded + args.dispersion *
                       rng.standard_normal(4).astype(np.float32)
                       for _ in range(args.num_chains)]

    print(f"  init_raw_free (per chain): {[ir.tolist() for ir in init_raws_free]}")
    print(f"  init_raw_fixed (per chain): {[ir.tolist() for ir in init_raws_fixed]}")

    # ============== RUN 1: FREE A ==============
    print("\n========== FREE A ==========")
    target_free = make_target_free_A(y_obs)
    free_samples = []
    free_accs = []
    t0 = time.perf_counter()
    for c in range(args.num_chains):
        seed = args.base_seed + 1009 * (c + 1)
        s, a = run_hmc(target_free, init_raws_free[c],
                       args.num_burnin, args.num_results,
                       args.step_size, args.L, seed)
        print(f"  [chain {c+1}/{args.num_chains}] accept={a.mean():.3f} "
              f"elapsed={time.perf_counter()-t0:.1f}s")
        free_samples.append(s)
        free_accs.append(a)
    free_samples = np.stack(free_samples, axis=0)  # (chains, draws, 5)
    free_constrained = constrain_free(free_samples)

    free_names = ["mu", "phi", "sigma_eta_sq", "A", "sigma_eps_sq"]
    truth_for_free = dict(mu=args.mu, phi=args.phi,
                          sigma_eta_sq=args.sigma_eta ** 2,
                          A=args.A, sigma_eps_sq=args.sigma_eps ** 2)
    free_summary = summarize_marginals(free_constrained, free_names,
                                       truth=truth_for_free)
    free_product_summary = summarize_products(free_constrained, truth)

    print("\n=== FREE A: marginals ===")
    for r in free_summary:
        ck = "OK" if r.get("covered", False) else "NOT-cov"
        print(f"  {r['param']:>15}  truth={r.get('truth', float('nan')):>8.4f}  "
              f"med={r['median']:>8.4f}  "
              f"CI=[{r['q025']:>8.4f}, {r['q975']:>8.4f}]  {ck}")
    print("\n=== FREE A: identified products ===")
    for r in free_product_summary:
        print(f"  {r['name']:>20}  truth={r['truth']:>8.4f}  "
              f"med={r['median']:>8.4f}  "
              f"CI=[{r['q025']:>8.4f}, {r['q975']:>8.4f}]")

    # ============== RUN 2: FIXED A=1 ==============
    print("\n========== FIXED A=1 ==========")
    target_fixed = make_target_fixed_A1(y_obs)
    fixed_samples = []
    fixed_accs = []
    t0 = time.perf_counter()
    for c in range(args.num_chains):
        seed = args.base_seed + 1009 * (c + 1) + 7
        s, a = run_hmc(target_fixed, init_raws_fixed[c],
                       args.num_burnin, args.num_results,
                       args.step_size, args.L, seed)
        print(f"  [chain {c+1}/{args.num_chains}] accept={a.mean():.3f} "
              f"elapsed={time.perf_counter()-t0:.1f}s")
        fixed_samples.append(s)
        fixed_accs.append(a)
    fixed_samples = np.stack(fixed_samples, axis=0)  # (chains, draws, 4)
    fixed_constrained = constrain_fixed(fixed_samples)

    fixed_names = ["mu", "phi", "sigma_eta_sq", "sigma_eps_sq"]
    # Under fixed A=1, the data-consistent truth on the identifying axis is:
    truth_for_fixed = dict(
        mu=args.A * args.mu,                  # since A_true was 2 -> A_fit=1, mu_fit=A_true*mu_true
        phi=args.phi,                         # phi invariant
        sigma_eta_sq=(args.A ** 2) * (args.sigma_eta ** 2),  # ridge-scaled
        sigma_eps_sq=args.sigma_eps ** 2,     # sigma_eps invariant
    )
    fixed_summary = summarize_marginals(fixed_constrained, fixed_names,
                                        truth=truth_for_fixed)
    print("\n=== FIXED A=1: marginals (truth = data-consistent ridge values) ===")
    for r in fixed_summary:
        ck = "OK" if r.get("covered", False) else "NOT-cov"
        print(f"  {r['param']:>15}  truth={r.get('truth', float('nan')):>8.4f}  "
              f"med={r['median']:>8.4f}  "
              f"CI=[{r['q025']:>8.4f}, {r['q975']:>8.4f}]  {ck}")

    # ============== PLOTS + SAVE ==============
    make_plots(out_dir, free_constrained, fixed_constrained, truth)
    np.savez_compressed(out_dir / "v2_samples.npz",
                        free_constrained=free_constrained,
                        free_raw=free_samples,
                        fixed_constrained=fixed_constrained,
                        fixed_raw=fixed_samples,
                        free_accept=np.stack(free_accs),
                        fixed_accept=np.stack(fixed_accs),
                        y_obs=y_obs.numpy(),
                        hs_true=hs)

    result = {
        "tf_version": tf.__version__,
        "tfp_version": tfp.__version__,
        "truth": truth,
        "truth_data_consistent_fixed": truth_for_fixed,
        "config": {
            "T": args.T, "num_chains": args.num_chains,
            "num_burnin": args.num_burnin, "num_results": args.num_results,
            "L": args.L, "step_size": args.step_size,
            "dispersion": args.dispersion,
            "data_seed": args.data_seed, "base_seed": args.base_seed,
        },
        "free": {
            "summary": free_summary,
            "products": free_product_summary,
            "accept_rate": float(np.mean([a.mean() for a in free_accs])),
        },
        "fixed": {
            "summary": fixed_summary,
            "accept_rate": float(np.mean([a.mean() for a in fixed_accs])),
        },
    }
    with open(out_dir / "v2_id_demo_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote: {out_dir}/v2_id_demo_result.json")
    print(f"       {out_dir}/v2_samples.npz")
    print(f"       {out_dir}/v2_free_ridge.png")
    print(f"       {out_dir}/v2_marginals_compare.png")
    print(f"       {out_dir}/v2_identified_products.png")


if __name__ == "__main__":
    raise SystemExit(main())
