"""
PMMH-only baseline for Bonus 1b (Andrieu et al. 2010 nonlinear SSM).

This script runs the same Particle-Marginal Metropolis–Hastings setup as
``exp_part3_bonus1b_hmc_vs_pmmh.py`` when that module is invoked with
``--first_part`` — bootstrap particle filter likelihood, random-walk MH on
``(log sigma_v^2, log sigma_w^2)``, multi-chain sampling for R-hat / ESS /
coverage — but **does not** run HMC-LEDH or L-HNN HMC.

Outputs (separate from the 3-way comparison so nothing is overwritten)::

    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/pmmh_only/comparison/results.txt
    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/pmmh_only/comparison/trace_posterior.png
    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/pmmh_only/comparison/acf.png

Usage::

    python -m src.experiments.exp_part3_bonus1b_pmmh_only
    python -m src.experiments.exp_part3_bonus1b_pmmh_only --num_chains 4 --samples_per_chain 1000 --burn_per_chain 1000

Keep ``T``, ``N_bpf``, priors, ``init_state``, and PMMH hyper-parameters in
sync with the PMMH block in ``exp_part3_bonus1b_hmc_vs_pmmh.py`` when you
want numerically comparable baselines.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.hmc_pf import disperse_initial_states
from src.filters.bonus.pmmh import (
    MultiChainPMMHResult,
    PMMHResult,
    bootstrap_pf_log_likelihood,
    run_pmmh,
    run_pmmh_multi_chain,
)
from src.models.ssm_katigawa import PMCMCNonlinearSSM
from src.utils.mcmc_diagnostics import diagnostics_summary, format_diagnostics_table

tfd = tfp.distributions

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def generate_data(
    T: int = 100,
    sigma_v_sq: float = 10.0,
    sigma_w_sq: float = 1.0,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Same simulator as ``exp_part3_bonus1b_hmc_vs_pmmh.generate_data``."""
    tf.random.set_seed(seed)
    sv = tf.sqrt(tf.cast(sigma_v_sq, tf.float32))
    sw = tf.sqrt(tf.cast(sigma_w_sq, tf.float32))

    x = tf.random.normal([]) * tf.sqrt(tf.constant(5.0, tf.float32))
    xs, ys = [x], [x**2 / 20.0 + sw * tf.random.normal([])]

    for t in range(2, T + 1):
        t_f = tf.cast(t, tf.float32)
        x = (
            0.5 * x
            + 25.0 * x / (1.0 + x**2)
            + 8.0 * tf.cos(1.2 * t_f)
            + sv * tf.random.normal([])
        )
        ys.append(x**2 / 20.0 + sw * tf.random.normal([]))
        xs.append(x)

    return tf.stack(xs), tf.stack(ys)


def make_target_log_prob(y_obs: tf.Tensor, pf_log_lik_fn) -> callable:
    """Same posterior as ``exp_part3_bonus1b_hmc_vs_pmmh.make_target_log_prob``."""
    prior_v = tfd.InverseGamma(concentration=0.01, scale=0.01)
    prior_w = tfd.InverseGamma(concentration=0.01, scale=0.01)

    def target(theta: tf.Tensor) -> tf.Tensor:
        log_sv2, log_sw2 = theta[0], theta[1]
        sv2, sw2 = tf.exp(log_sv2), tf.exp(log_sw2)
        lp_prior = prior_v.log_prob(sv2) + prior_w.log_prob(sw2)
        jacobian = log_sv2 + log_sw2
        ssm = PMCMCNonlinearSSM(sigma_v_sq=sv2, sigma_w_sq=sw2)
        ll = pf_log_lik_fn(ssm, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-1e6, tf.float32))
        result = lp_prior + jacobian + ll
        result = tf.math.real(result)
        result = tf.cast(result, tf.float32)
        return tf.where(tf.math.is_finite(result), result, tf.constant(-1e6, tf.float32))

    return target


def compute_acf(chain: tf.Tensor, max_lag: int = 50) -> np.ndarray:
    n = chain.shape[0]
    c = chain - tf.reduce_mean(chain)
    var = tf.maximum(tf.reduce_sum(c**2) / tf.cast(n, tf.float32), 1e-12)
    acf = [1.0]
    for k in range(1, max_lag):
        cov_k = tf.reduce_sum(c[:-k] * c[k:]) / tf.cast(n, tf.float32)
        acf.append(float((cov_k / var).numpy()))
    return np.array(acf)


def rmse(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.sqrt(tf.reduce_mean((samples - true_val) ** 2)).numpy())


def mae(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.reduce_mean(tf.abs(samples - true_val)).numpy())


def per_step_cost(total_time: float, n_samples: int, n_burnin: int) -> float:
    return total_time / (n_samples + n_burnin)


def summarise_chain(
    samples: tf.Tensor,
    name: str,
    param_names: list,
    true_values: tf.Tensor,
) -> None:
    ess = tfp.mcmc.effective_sample_size(samples)
    for j, pname in enumerate(param_names):
        chain_j = samples[:, j]
        tf.print(
            f"  {name} {pname}:",
            "  mean =", tf.reduce_mean(chain_j),
            "  std =", tf.math.reduce_std(chain_j),
            "  ESS =", float(ess[j].numpy()),
            "  true =", true_values[j],
        )


def plot_pmmh_diagnostics(
    pmmh_samples: tf.Tensor,
    true_values: tf.Tensor,
    param_names: list,
    out_dir: Path,
) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = len(param_names)
    max_lag = 50

    fig, axes = plt.subplots(n_params, 2, figsize=(12, 4 * n_params))
    if n_params == 1:
        axes = axes[None, :]

    for j in range(n_params):
        pm_j = pmmh_samples[:, j].numpy()
        tv_j = float(true_values[j].numpy())

        ax = axes[j, 0]
        ax.plot(pm_j, alpha=0.65, lw=0.8, label="PMMH (pooled chains)")
        ax.axhline(tv_j, color="k", ls="--", lw=1.2, label="true")
        ax.set_ylabel(param_names[j])
        ax.set_xlabel("iteration (pooled)")
        ax.set_title(f"Trace — {param_names[j]}", fontweight="bold")
        ax.legend(fontsize=8)

        ax = axes[j, 1]
        ax.hist(pm_j, bins=40, density=True, alpha=0.7, label="PMMH")
        ax.axvline(tv_j, color="k", ls="--", lw=1.2, label="true")
        ax.set_title(f"Posterior — {param_names[j]}", fontweight="bold")
        ax.set_xlabel(param_names[j])
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison" / "trace_posterior.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_params, 1, figsize=(8, 3.5 * n_params))
    if n_params == 1:
        axes = [axes]
    for j in range(n_params):
        chain_arr = pmmh_samples[:, j].numpy()
        acf_vals = compute_acf(tf.constant(chain_arr), max_lag)
        ci = 1.96 / np.sqrt(len(chain_arr))
        ax = axes[j]
        ax.bar(range(max_lag), acf_vals, width=0.6, alpha=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(ci, color="r", ls="--", lw=0.8, label="±1.96/√n")
        ax.axhline(-ci, color="r", ls="--", lw=0.8)
        ax.set_title(f"ACF — PMMH — {param_names[j]}", fontweight="bold")
        ax.set_xlabel("lag")
        ax.set_ylabel("autocorrelation")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison" / "acf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pmmh_report(
    pmmh_result: PMMHResult,
    pmmh_samples: tf.Tensor,
    pmmh_samples_chains: tf.Tensor,
    t_pmmh: float,
    n_samp_total: int,
    n_burn_total: int,
    num_chains: int,
    samples_per_chain: int,
    burn_per_chain: int,
    true_values: tf.Tensor,
    param_names: list,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ess_pm = tfp.mcmc.effective_sample_size(pmmh_samples)
    mean_ess = float(tf.reduce_mean(ess_pm).numpy())
    acc_pm = float(pmmh_result.accept_rate.numpy())
    cost_pm = per_step_cost(t_pmmh, n_samp_total, n_burn_total)

    col_w = 40
    lines = [
        "=" * 85,
        "PMMH only — Bootstrap PF + Random-Walk MH (Bonus 1b baseline)",
        "Model: Andrieu et al. (2010), Eq 14-15  |  Prior: InvGamma(0.01, 0.01)",
        f"Chains: {num_chains}  |  samples/chain: {samples_per_chain}  |  burn/chain: {burn_per_chain}",
        "=" * 85,
        f"{'Metric':<{col_w}} {'PMMH':>20}",
        "-" * 85,
        f"{'Acceptance rate':<{col_w}} {acc_pm:>20.3f}",
    ]
    for j, pname in enumerate(param_names):
        e_pm = float(ess_pm[j].numpy())
        lines.append(f"{'ESS (' + pname + ')':<{col_w}} {e_pm:>20.1f}")
    lines += [
        f"{'Mean ESS':<{col_w}} {mean_ess:>20.1f}",
        f"{'Runtime (s)':<{col_w}} {t_pmmh:>20.1f}",
        f"{'ESS/s':<{col_w}} {mean_ess / max(t_pmmh, 1e-9):>20.3f}",
        f"{'Cost per step (s/proposal)':<{col_w}} {cost_pm:>20.4f}",
        "-" * 85,
    ]
    for j, pname in enumerate(param_names):
        m_pm = float(tf.reduce_mean(pmmh_samples[:, j]).numpy())
        tv = float(true_values[j].numpy())
        lines.append(f"{'Mean ' + pname:<{col_w}} {m_pm:>20.3f}  (true={tv:.3f})")
    for j, pname in enumerate(param_names):
        s_pm = float(tf.math.reduce_std(pmmh_samples[:, j]).numpy())
        lines.append(f"{'Std ' + pname:<{col_w}} {s_pm:>20.3f}")
    lines.append("-" * 85)
    lines.append("Quality vs ground truth:")
    for j, pname in enumerate(param_names):
        tv = float(true_values[j].numpy())
        lines.append(
            f"  RMSE ({pname:<16}) {rmse(pmmh_samples[:, j], tv):.4f}"
        )
        lines.append(
            f"  MAE  ({pname:<16}) {mae(pmmh_samples[:, j], tv):.4f}"
        )
    lines.append("=" * 85)

    truth_cpu = tf.cast(true_values, tf.float32)
    summary = diagnostics_summary(pmmh_samples_chains, param_names, truth=truth_cpu)
    lines += [
        "",
        "Multi-chain convergence diagnostics  (Vehtari et al., 2021)",
        "  bulk-ESS = ESS of rank-normalised split chains (mixing of the mean).",
        "  tail-ESS = ESS at the 5th / 95th percentile indicators (worst-case tail).",
        "  splitR^  = classic Gelman-Rubin R-hat with half-chain split.",
        "  rankR^   = max(rank-normalised, folded rank-normalised) split R-hat.",
        "  Pass criteria: rankR^ < 1.01 AND bulk-ESS > 400 AND tail-ESS > 400 per param.",
        "",
        format_diagnostics_table("PMMH (Bootstrap PF)", summary),
    ]

    report = "\n".join(lines)
    path = out_dir / "comparison" / "results.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved results to {path}")


def _chain_save_path(out_dir: Path, chain_id: int) -> Path:
    return out_dir / "chains" / f"chain_{chain_id}.npz"


def _save_single_chain(
    result: PMMHResult,
    chain_id: int,
    out_dir: Path,
    runtime_s: float = 0.0,
) -> Path:
    path = _chain_save_path(out_dir, chain_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        samples=result.samples.numpy(),
        is_accepted=result.is_accepted.numpy(),
        accept_rate=float(result.accept_rate.numpy()),
        target_log_probs=result.target_log_probs.numpy(),
        runtime_s=float(runtime_s),
    )
    return path


def _load_all_chains(
    out_dir: Path, num_chains: int
) -> tuple[MultiChainPMMHResult, float]:
    """Load saved chains. Returns (result, total_runtime_s).

    Total runtime is summed from the per-chain ``runtime_s`` field when
    available. For backward compatibility with chain files saved before
    that field existed, the runtime is estimated from filesystem mtimes:
    total ≈ (mtime[last] - mtime[first]) * N / (N - 1).
    """
    samples_list, accepted_list, ar_list, lp_list = [], [], [], []
    runtimes = []
    mtimes = []
    for c in range(num_chains):
        path = _chain_save_path(out_dir, c)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing chain file: {path}. Run --chain_id {c} first."
            )
        z = np.load(path)
        samples_list.append(tf.constant(z["samples"], tf.float32))
        accepted_list.append(tf.constant(z["is_accepted"], tf.bool))
        ar_list.append(tf.constant(float(z["accept_rate"]), tf.float32))
        lp_list.append(tf.constant(z["target_log_probs"], tf.float32))
        if "runtime_s" in z.files:
            runtimes.append(float(z["runtime_s"]))
        mtimes.append(path.stat().st_mtime)

    if len(runtimes) == num_chains and sum(runtimes) > 0:
        total_runtime = sum(runtimes)
    elif num_chains >= 2:
        delta = max(mtimes) - min(mtimes)
        total_runtime = delta * num_chains / (num_chains - 1)
    else:
        total_runtime = 0.0

    return (
        MultiChainPMMHResult(
            samples=tf.stack(samples_list, axis=0),
            is_accepted=tf.stack(accepted_list, axis=0),
            accept_rate=tf.stack(ar_list, axis=0),
            target_log_probs=tf.stack(lp_list, axis=0),
        ),
        total_runtime,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PMMH-only multi-chain baseline (same setup as bonus1b --first_part PMMH block).",
    )
    p.add_argument(
        "--num_chains",
        type=int,
        default=4,
        help="Independent PMMH chains (default 4).",
    )
    p.add_argument(
        "--samples_per_chain",
        type=int,
        default=2000,
        help="Post-burnin samples per chain (default 250).",
    )
    p.add_argument(
        "--burn_per_chain",
        type=int,
        default=1000,
        help="Burn-in iterations per chain (default 200).",
    )
    p.add_argument(
        "--chain_id",
        type=int,
        default=-1,
        help=(
            "If >= 0, run ONLY this chain (0..num_chains-1) and save to "
            "<out_dir>/chains/chain_{id}.npz. Used by the isolated-chain "
            "driver to avoid TFP eager memory leaks across chains."
        ),
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help=(
            "Load saved per-chain .npz files from <out_dir>/chains/, build "
            "diagnostics + plots, and write the final report. Use after "
            "running --chain_id 0..N-1 separately."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    tf.random.set_seed(123)

    T = 50
    true_sv2 = 10.0
    true_sw2 = 1.0
    N_bpf = 1000
    num_chains = int(args.num_chains)
    n_samp = int(args.samples_per_chain)
    n_burn = int(args.burn_per_chain)
    chain_id = int(args.chain_id)
    aggregate = bool(args.aggregate)
    param_names = ["sigma_v^2", "sigma_w^2"]
    true_values = tf.constant([true_sv2, true_sw2], tf.float32)
    out_dir = Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/pmmh_only")

    base_seed = 200
    step_size = 0.08
    dispersion_scale = 0.1

    if chain_id >= 0 and aggregate:
        raise SystemExit("--chain_id and --aggregate are mutually exclusive.")
    if chain_id >= num_chains:
        raise SystemExit(f"--chain_id {chain_id} must be < --num_chains {num_chains}.")

    print("=" * 70)
    print("  PMMH only (Bootstrap PF + RW-MH) — Bonus 1b baseline")
    print(f"  T={T}  N_bpf={N_bpf}")
    print(f"  samples/chain={n_samp}  burn/chain={n_burn}  chains={num_chains}")
    print(f"  True: sigma_v^2={true_sv2}  sigma_w^2={true_sw2}")
    if chain_id >= 0:
        print(f"  Mode: ISOLATED CHAIN (chain_id={chain_id}) — saves .npz only")
    elif aggregate:
        print("  Mode: AGGREGATE — load saved chains, build report")
    else:
        print("  Mode: standard multi-chain (all chains in one process)")
    print("  (Independent CRN per chain — same policy as full experiment.)")
    print("=" * 70)

    # ---- aggregate mode: load saved chains, run diagnostics, exit ----------
    if aggregate:
        pmmh_mc, t_pmmh = _load_all_chains(out_dir, num_chains)
        pmmh_samples_chains = tf.exp(pmmh_mc.samples)
        pmmh_samples = tf.reshape(pmmh_samples_chains, [num_chains * n_samp, 2])
        pmmh_result = PMMHResult(
            samples=pmmh_mc.samples[0],
            is_accepted=pmmh_mc.is_accepted[0],
            accept_rate=tf.reduce_mean(pmmh_mc.accept_rate),
            target_log_probs=pmmh_mc.target_log_probs[0],
        )
        print(f"\n  Aggregate: total runtime across chains = {t_pmmh:.1f}s")
        summarise_chain(pmmh_samples, "PMMH (chains pooled)", param_names, true_values)
        save_pmmh_report(
            pmmh_result=pmmh_result,
            pmmh_samples=pmmh_samples,
            pmmh_samples_chains=pmmh_samples_chains,
            t_pmmh=t_pmmh,
            n_samp_total=num_chains * n_samp,
            n_burn_total=num_chains * n_burn,
            num_chains=num_chains,
            samples_per_chain=n_samp,
            burn_per_chain=n_burn,
            true_values=true_values,
            param_names=param_names,
            out_dir=out_dir,
        )
        plot_pmmh_diagnostics(pmmh_samples, true_values, param_names, out_dir)
        return 0

    # ---- shared setup for chain modes --------------------------------------
    _, y_obs = generate_data(T, true_sv2, true_sw2, seed=42)
    init_state = tf.stack([tf.math.log(8.0), tf.math.log(1.5)])

    def bpf_ll(ssm, y):
        return bootstrap_pf_log_likelihood(ssm, y, num_particles=N_bpf)

    target_pmmh = make_target_log_prob(y_obs, bpf_ll)
    pmmh_inits = disperse_initial_states(
        init_state, num_chains=num_chains, scale=dispersion_scale, seed=base_seed
    )

    # ---- isolated chain mode: run one chain, save, exit --------------------
    if chain_id >= 0:
        chain_seed = base_seed + 1009 * (chain_id + 1)
        print(f"\n[isolated chain {chain_id}/{num_chains}]  seed={chain_seed}"
              f"  init={pmmh_inits[chain_id].numpy().tolist()}")
        t0 = time.time()
        result = run_pmmh(
            target_log_prob_fn=target_pmmh,
            initial_state=pmmh_inits[chain_id],
            num_results=n_samp,
            num_burnin=n_burn,
            step_size=step_size,
            seed=chain_seed,
            verbose=True,
        )
        t_chain = time.time() - t0
        path = _save_single_chain(result, chain_id, out_dir, runtime_s=t_chain)
        print(
            f"\n  Chain {chain_id} done in {t_chain:.1f}s  "
            f"accept={float(result.accept_rate.numpy()):.3f}  "
            f"saved -> {path}"
        )
        return 0

    # ---- default mode: all chains in one process (legacy) ------------------
    print("\n" + "=" * 70)
    print(f"  Running PMMH — {num_chains} chains")
    print("=" * 70)
    t0 = time.time()
    pmmh_mc = run_pmmh_multi_chain(
        target_log_prob_fn=target_pmmh,
        initial_states=pmmh_inits,
        num_results=n_samp,
        num_burnin=n_burn,
        step_size=step_size,
        seed=base_seed,
        verbose=True,
    )
    t_pmmh = time.time() - t0

    pmmh_result = PMMHResult(
        samples=pmmh_mc.samples[0],
        is_accepted=pmmh_mc.is_accepted[0],
        accept_rate=tf.reduce_mean(pmmh_mc.accept_rate),
        target_log_probs=pmmh_mc.target_log_probs[0],
    )
    pmmh_samples_chains = tf.exp(pmmh_mc.samples)
    pmmh_samples = tf.reshape(pmmh_samples_chains, [num_chains * n_samp, 2])

    print(
        f"\n  PMMH ({num_chains} chains) done in {t_pmmh:.1f}s  "
        f"mean_accept={float(tf.reduce_mean(pmmh_mc.accept_rate)):.3f}"
    )
    summarise_chain(pmmh_samples, "PMMH (chains pooled)", param_names, true_values)

    save_pmmh_report(
        pmmh_result=pmmh_result,
        pmmh_samples=pmmh_samples,
        pmmh_samples_chains=pmmh_samples_chains,
        t_pmmh=t_pmmh,
        n_samp_total=num_chains * n_samp,
        n_burn_total=num_chains * n_burn,
        num_chains=num_chains,
        samples_per_chain=n_samp,
        burn_per_chain=n_burn,
        true_values=true_values,
        param_names=param_names,
        out_dir=out_dir,
    )
    plot_pmmh_diagnostics(pmmh_samples, true_values, param_names, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
