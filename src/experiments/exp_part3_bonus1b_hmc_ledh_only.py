"""
HMC-LEDH-only baseline for Bonus 1b — per-chain process isolation.

Mirrors ``exp_part3_bonus1b_pmmh_only.py`` but for the HMC-LEDH sampler:
4 dispersed chains run as fresh Python processes via
``scripts/run_hmc_ledh_isolated.sh`` so memory leaks reset between chains.

Outputs::

    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/hmc_ledh_only/comparison/results.txt
    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/hmc_ledh_only/chains/chain_{id}.npz

The experiment matches ``--first_part`` HMC-LEDH config in
``exp_part3_bonus1b_hmc_vs_pmmh.py``: ``T=50``, ``N_ledh=200``, ``L=5``,
``step_size=0.005``, dispersion ``scale=0.15``, base seed ``300``,
``crn_offset = base_seed`` (shared CRN across chains, §4.5 of the
addendum).

Usage::

    # Single-shot (legacy, runs all chains in one process):
    python -m src.experiments.exp_part3_bonus1b_hmc_ledh_only \\
        --num_chains 4 --samples_per_chain 1000 --burn_per_chain 500

    # Isolated per-chain (recommended for long runs):
    python -m src.experiments.exp_part3_bonus1b_hmc_ledh_only \\
        --chain_id 0 --num_chains 4 --samples_per_chain 2000 --burn_per_chain 1000
    # ...repeat for chain_id 1..3, then:
    python -m src.experiments.exp_part3_bonus1b_hmc_ledh_only \\
        --aggregate --num_chains 4 --samples_per_chain 2000 --burn_per_chain 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.filters.bonus.hmc_pf import (
    HMCResult,
    MultiChainHMCResult,
    disperse_initial_states,
    run_hmc,
    run_hmc_multi_chain,
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


# =====================================================================
# Same data + target as the 3-way experiment
# =====================================================================

def generate_data(
    T: int = 100,
    sigma_v_sq: float = 10.0,
    sigma_w_sq: float = 1.0,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor]:
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


def make_target_log_prob(y_obs: tf.Tensor, ll_fn) -> callable:
    prior_v = tfd.InverseGamma(concentration=0.01, scale=0.01)
    prior_w = tfd.InverseGamma(concentration=0.01, scale=0.01)

    def target(theta: tf.Tensor) -> tf.Tensor:
        log_sv2, log_sw2 = theta[0], theta[1]
        sv2, sw2 = tf.exp(log_sv2), tf.exp(log_sw2)
        lp_prior = prior_v.log_prob(sv2) + prior_w.log_prob(sw2)
        jacobian = log_sv2 + log_sw2
        ssm = PMCMCNonlinearSSM(sigma_v_sq=sv2, sigma_w_sq=sw2)
        ll = ll_fn(ssm, y_obs)
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-1e6, tf.float32))
        result = lp_prior + jacobian + ll
        result = tf.math.real(result)
        result = tf.cast(result, tf.float32)
        return tf.where(tf.math.is_finite(result), result, tf.constant(-1e6, tf.float32))

    return target


# =====================================================================
# Per-chain save / load
# =====================================================================

def _chain_save_path(out_dir: Path, chain_id: int) -> Path:
    return out_dir / "chains" / f"chain_{chain_id}.npz"


def _save_single_chain(
    result: HMCResult,
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
        step_sizes=result.step_sizes.numpy(),
        runtime_s=float(runtime_s),
    )
    return path


def _load_all_chains(out_dir: Path, num_chains: int):
    samples_list, accepted_list, ar_list, lp_list, ss_list = [], [], [], [], []
    runtimes, mtimes = [], []
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
        ss_list.append(tf.constant(z["step_sizes"], tf.float32))
        if "runtime_s" in z.files:
            runtimes.append(float(z["runtime_s"]))
        mtimes.append(path.stat().st_mtime)

    if len(runtimes) == num_chains and sum(runtimes) > 0:
        total_runtime = sum(runtimes)
    elif num_chains >= 2:
        total_runtime = (max(mtimes) - min(mtimes)) * num_chains / (num_chains - 1)
    else:
        total_runtime = 0.0

    mc = MultiChainHMCResult(
        samples=tf.stack(samples_list, axis=0),
        is_accepted=tf.stack(accepted_list, axis=0),
        accept_rate=tf.stack(ar_list, axis=0),
        target_log_probs=tf.stack(lp_list, axis=0),
        step_sizes=tf.stack(ss_list, axis=0),
    )
    return mc, total_runtime


# =====================================================================
# Reporting
# =====================================================================

def rmse(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.sqrt(tf.reduce_mean((samples - true_val) ** 2)).numpy())


def mae(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.reduce_mean(tf.abs(samples - true_val)).numpy())


def save_hmc_report(
    samples: tf.Tensor,
    samples_chains: tf.Tensor,
    accept_rate: float,
    t_total: float,
    num_chains: int,
    samples_per_chain: int,
    burn_per_chain: int,
    true_values: tf.Tensor,
    param_names: list,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ess = tfp.mcmc.effective_sample_size(samples)
    mean_ess = float(tf.reduce_mean(ess).numpy())
    n_total = num_chains * samples_per_chain
    n_burn_total = num_chains * burn_per_chain
    cost_per_step = t_total / max(n_total + n_burn_total, 1)

    col_w = 40
    lines = [
        "=" * 85,
        "HMC-LEDH only — Stan-style 3-window warmup, shared CRN across chains",
        "Model: Andrieu et al. (2010), Eq 14-15  |  Prior: InvGamma(0.01, 0.01)",
        f"Chains: {num_chains}  |  samples/chain: {samples_per_chain}  |  burn/chain: {burn_per_chain}",
        "=" * 85,
        f"{'Metric':<{col_w}} {'HMC-LEDH':>20}",
        "-" * 85,
        f"{'Acceptance rate':<{col_w}} {accept_rate:>20.3f}",
    ]
    for j, pname in enumerate(param_names):
        lines.append(f"{'ESS (' + pname + ')':<{col_w}} {float(ess[j].numpy()):>20.1f}")
    lines += [
        f"{'Mean ESS':<{col_w}} {mean_ess:>20.1f}",
        f"{'Runtime (s)':<{col_w}} {t_total:>20.1f}",
        f"{'ESS/s':<{col_w}} {mean_ess / max(t_total, 1e-9):>20.4f}",
        f"{'Cost per step (s/proposal)':<{col_w}} {cost_per_step:>20.4f}",
        "-" * 85,
    ]
    for j, pname in enumerate(param_names):
        m = float(tf.reduce_mean(samples[:, j]).numpy())
        tv = float(true_values[j].numpy())
        lines.append(f"{'Mean ' + pname:<{col_w}} {m:>20.3f}  (true={tv:.3f})")
    for j, pname in enumerate(param_names):
        s = float(tf.math.reduce_std(samples[:, j]).numpy())
        lines.append(f"{'Std ' + pname:<{col_w}} {s:>20.3f}")
    lines.append("-" * 85)
    lines.append("Quality vs ground truth:")
    for j, pname in enumerate(param_names):
        tv = float(true_values[j].numpy())
        lines.append(f"  RMSE ({pname:<16}) {rmse(samples[:, j], tv):.4f}")
        lines.append(f"  MAE  ({pname:<16}) {mae(samples[:, j], tv):.4f}")
    lines.append("=" * 85)

    truth_cpu = tf.cast(true_values, tf.float32)
    summary = diagnostics_summary(samples_chains, param_names, truth=truth_cpu)
    lines += [
        "",
        "Multi-chain convergence diagnostics  (Vehtari et al., 2021)",
        "  bulk-ESS = ESS of rank-normalised split chains.",
        "  tail-ESS = ESS at the 5/95 percentile indicators.",
        "  splitR^  = classic Gelman-Rubin R-hat with half-chain split.",
        "  rankR^   = max(rank-normalised, folded rank-normalised) split R-hat.",
        "  Pass: rankR^ < 1.01 AND bulk-ESS > 400 AND tail-ESS > 400 per param.",
        "",
        format_diagnostics_table("HMC-LEDH (adapted M, shared CRN)", summary),
    ]

    report = "\n".join(lines)
    path = out_dir / "comparison" / "results.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved results to {path}")


def plot_hmc_diagnostics(
    samples: tf.Tensor,
    true_values: tf.Tensor,
    param_names: list,
    out_dir: Path,
) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 4 * n_params))
    if n_params == 1:
        axes = axes[None, :]
    for j in range(n_params):
        s_j = samples[:, j].numpy()
        tv = float(true_values[j].numpy())
        ax = axes[j, 0]
        ax.plot(s_j, alpha=0.65, lw=0.8, label="HMC-LEDH (pooled)")
        ax.axhline(tv, color="k", ls="--", lw=1.2, label="true")
        ax.set_ylabel(param_names[j])
        ax.set_xlabel("iteration (pooled)")
        ax.set_title(f"Trace — {param_names[j]}", fontweight="bold")
        ax.legend(fontsize=8)
        ax = axes[j, 1]
        ax.hist(s_j, bins=40, density=True, alpha=0.7, label="HMC-LEDH")
        ax.axvline(tv, color="k", ls="--", lw=1.2, label="true")
        ax.set_title(f"Posterior — {param_names[j]}", fontweight="bold")
        ax.set_xlabel(param_names[j])
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison" / "trace_posterior.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# CLI
# =====================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMC-LEDH-only with chain isolation.")
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--samples_per_chain", type=int, default=1000)
    p.add_argument("--burn_per_chain", type=int, default=500)
    p.add_argument(
        "--chain_id",
        type=int,
        default=-1,
        help="If >= 0, run ONLY this chain and save to chains/chain_{id}.npz.",
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Load saved chains, build diagnostics + plots, write report.",
    )
    p.add_argument("--no_adapt_mass_matrix", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    tf.random.set_seed(123)

    T = 50
    true_sv2 = 10.0
    true_sw2 = 1.0
    N_ledh = 200
    L = 5
    num_chains = int(args.num_chains)
    n_samp = int(args.samples_per_chain)
    n_burn = int(args.burn_per_chain)
    chain_id = int(args.chain_id)
    aggregate = bool(args.aggregate)
    adapt_mass = not bool(args.no_adapt_mass_matrix)
    param_names = ["sigma_v^2", "sigma_w^2"]
    true_values = tf.constant([true_sv2, true_sw2], tf.float32)
    out_dir = Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/hmc_ledh_only")

    base_seed = 300
    step_size = 0.005
    dispersion_scale = 0.15

    if chain_id >= 0 and aggregate:
        raise SystemExit("--chain_id and --aggregate are mutually exclusive.")
    if chain_id >= num_chains:
        raise SystemExit(f"--chain_id {chain_id} must be < --num_chains {num_chains}.")

    print("=" * 70)
    print("  HMC-LEDH only — Bonus 1b (chain-isolated)")
    print(f"  T={T}  N_ledh={N_ledh}  L={L}  adapt_mass_matrix={adapt_mass}")
    print(f"  samples/chain={n_samp}  burn/chain={n_burn}  chains={num_chains}")
    if chain_id >= 0:
        print(f"  Mode: ISOLATED CHAIN (chain_id={chain_id})")
    elif aggregate:
        print("  Mode: AGGREGATE")
    else:
        print("  Mode: standard multi-chain (one process)")
    print("  CRN: shared across chains (crn_offset=base_seed)")
    print("=" * 70)

    if aggregate:
        mc, t_total = _load_all_chains(out_dir, num_chains)
        samples_chains = tf.exp(mc.samples)
        samples_flat = tf.reshape(samples_chains, [num_chains * n_samp, 2])
        accept_rate = float(tf.reduce_mean(mc.accept_rate).numpy())
        print(f"\n  Aggregate: total runtime across chains = {t_total:.1f}s  "
              f"mean_accept={accept_rate:.3f}")
        save_hmc_report(
            samples=samples_flat,
            samples_chains=samples_chains,
            accept_rate=accept_rate,
            t_total=t_total,
            num_chains=num_chains,
            samples_per_chain=n_samp,
            burn_per_chain=n_burn,
            true_values=true_values,
            param_names=param_names,
            out_dir=out_dir,
        )
        plot_hmc_diagnostics(samples_flat, true_values, param_names, out_dir)
        return 0

    # Shared setup
    _, y_obs = generate_data(T, true_sv2, true_sw2, seed=42)
    init_state = tf.stack([tf.math.log(8.0), tf.math.log(1.5)])

    ledh_filter = DifferentiableLEDHLogLikelihood(
        num_particles=N_ledh,
        n_lambda=5,
        sinkhorn_epsilon=2.0,
        sinkhorn_iters=20,
        resample_threshold=0.5,
        grad_window=1,
        jit_compile=True,
    )

    def ledh_ll(ssm, y):
        return ledh_filter(ssm, y)

    target_hmc = make_target_log_prob(y_obs, ledh_ll)
    hmc_inits = disperse_initial_states(
        init_state, num_chains=num_chains, scale=dispersion_scale, seed=base_seed
    )

    if chain_id >= 0:
        chain_seed = base_seed + 1009 * (chain_id + 1)
        print(f"\n[isolated HMC-LEDH chain {chain_id}/{num_chains}]  seed={chain_seed}"
              f"  init={hmc_inits[chain_id].numpy().tolist()}")
        t0 = time.time()
        result = run_hmc(
            target_log_prob_fn=target_hmc,
            initial_state=hmc_inits[chain_id],
            num_results=n_samp,
            num_burnin=n_burn,
            step_size=step_size,
            num_leapfrog_steps=L,
            target_accept_prob=0.65,
            seed=chain_seed,
            verbose=True,
            adapt_step_size=True,
            adapt_mass_matrix=adapt_mass,
            crn_offset=base_seed,  # shared CRN across chains (§4.5)
        )
        t_chain = time.time() - t0
        path = _save_single_chain(result, chain_id, out_dir, runtime_s=t_chain)
        print(
            f"\n  Chain {chain_id} done in {t_chain:.1f}s  "
            f"accept={float(result.accept_rate.numpy()):.3f}  saved -> {path}"
        )
        return 0

    # Default: legacy single-process multi-chain
    print("\n" + "=" * 70)
    print(f"  Running HMC-LEDH ({num_chains} chains in one process)")
    print("=" * 70)
    t0 = time.time()
    mc = run_hmc_multi_chain(
        target_log_prob_fn=target_hmc,
        initial_states=hmc_inits,
        num_results=n_samp,
        num_burnin=n_burn,
        step_size=step_size,
        num_leapfrog_steps=L,
        target_accept_prob=0.65,
        seed=base_seed,
        verbose=True,
        adapt_step_size=True,
        adapt_mass_matrix=adapt_mass,
    )
    t_total = time.time() - t0

    samples_chains = tf.exp(mc.samples)
    samples_flat = tf.reshape(samples_chains, [num_chains * n_samp, 2])
    accept_rate = float(tf.reduce_mean(mc.accept_rate).numpy())

    print(f"\n  HMC-LEDH ({num_chains} chains) done in {t_total:.1f}s  "
          f"mean_accept={accept_rate:.3f}")
    save_hmc_report(
        samples=samples_flat,
        samples_chains=samples_chains,
        accept_rate=accept_rate,
        t_total=t_total,
        num_chains=num_chains,
        samples_per_chain=n_samp,
        burn_per_chain=n_burn,
        true_values=true_values,
        param_names=param_names,
        out_dir=out_dir,
    )
    plot_hmc_diagnostics(samples_flat, true_values, param_names, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
