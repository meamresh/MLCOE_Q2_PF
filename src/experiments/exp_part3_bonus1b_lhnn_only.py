"""
L-HNN HMC-only baseline for Bonus 1b — per-chain process isolation
+ dispersed-pilot training (fixes the wrong-mode issue from a single
pilot starting point).

Three modes:

1. ``--train`` — generate pilot trajectories from ``num_chains`` dispersed
   starting points (the same dispersion used for the actual chains),
   train the L-HNN, save the network weights and the pilot-grad count.
2. ``--chain_id N`` — load the trained L-HNN, run only chain N, save
   samples to ``chains/chain_{N}.npz``.
3. ``--aggregate`` — load saved chains, build diagnostics + plots, write
   the report.

The dispersed-pilot fix is the principled remediation for the wrong-mode
problem documented in §5.3 of ``Report_II_Addendum_Diagnostics.md``:
the original training generated all pilot trajectories from a single
``init_state``, so the surrogate Hamiltonian saw only one basin of the
σ_v²/σ_w² posterior trade-off and pushed chains there at sample time.

Outputs::

    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/lhnn_only/comparison/results.txt
    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/lhnn_only/chains/chain_{id}.npz
    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/lhnn_only/lhnn_model/  (Keras dir)
    reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/lhnn_only/lhnn_model/training_meta.npz

Usage::

    python -m src.experiments.exp_part3_bonus1b_lhnn_only --train
    python -m src.experiments.exp_part3_bonus1b_lhnn_only --chain_id 0
    # repeat 1..3 …
    python -m src.experiments.exp_part3_bonus1b_lhnn_only --aggregate
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
)
from src.filters.bonus.lhnn_hmc_pf import (
    LHNNConfig,
    LatentHNN,
    generate_training_data,
    run_lhnn_hmc,
    train_lhnn,
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
# L-HNN model save / load (Keras weights + meta)
# =====================================================================

def _model_dir(out_dir: Path) -> Path:
    return out_dir / "lhnn_model"


def _save_trained_lhnn(
    lhnn: LatentHNN,
    cfg: LHNNConfig,
    training_grad_evals: int,
    pilot_runtime_s: float,
    train_runtime_s: float,
    out_dir: Path,
) -> Path:
    md = _model_dir(out_dir)
    md.mkdir(parents=True, exist_ok=True)
    weights_path = md / "lhnn.weights.h5"
    lhnn.save_weights(str(weights_path))
    np.savez(
        md / "training_meta.npz",
        training_grad_evals=int(training_grad_evals),
        pilot_runtime_s=float(pilot_runtime_s),
        train_runtime_s=float(train_runtime_s),
        hidden_units=int(cfg.hidden_units),
        num_hidden=int(cfg.num_hidden),
        epochs=int(cfg.epochs),
        num_pilot_trajectories=int(cfg.num_pilot_trajectories),
        pilot_steps_per_trajectory=int(cfg.pilot_steps_per_trajectory),
        error_threshold=float(cfg.error_threshold),
        cooldown_steps=int(cfg.cooldown_steps),
    )
    return weights_path


def _load_trained_lhnn(out_dir: Path, d: int) -> tuple[LatentHNN, LHNNConfig, dict]:
    md = _model_dir(out_dir)
    meta_path = md / "training_meta.npz"
    weights_path = md / "lhnn.weights.h5"
    if not meta_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Trained L-HNN not found at {md}. Run --train first."
        )
    meta = np.load(meta_path)
    cfg = LHNNConfig(
        hidden_units=int(meta["hidden_units"]),
        num_hidden=int(meta["num_hidden"]),
        epochs=int(meta["epochs"]),
        num_pilot_trajectories=int(meta["num_pilot_trajectories"]),
        pilot_steps_per_trajectory=int(meta["pilot_steps_per_trajectory"]),
        error_threshold=float(meta["error_threshold"]),
        cooldown_steps=int(meta["cooldown_steps"]),
    )
    lhnn = LatentHNN(d, cfg.hidden_units, cfg.num_hidden)
    _ = lhnn(tf.zeros([1, 2 * d]))  # build weights
    lhnn.load_weights(str(weights_path))
    info = {
        "training_grad_evals": int(meta["training_grad_evals"]),
        "pilot_runtime_s": float(meta["pilot_runtime_s"]),
        "train_runtime_s": float(meta["train_runtime_s"]),
    }
    return lhnn, cfg, info


# =====================================================================
# Per-chain save / load (mirrors HMC-LEDH file)
# =====================================================================

def _chain_save_path(out_dir: Path, chain_id: int) -> Path:
    return out_dir / "chains" / f"chain_{chain_id}.npz"


def _save_single_chain(
    result: HMCResult,
    chain_id: int,
    out_dir: Path,
    runtime_s: float,
    sampling_grad_evals: int,
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
        sampling_grad_evals=int(sampling_grad_evals),
    )
    return path


def _load_all_chains(out_dir: Path, num_chains: int):
    samples_list, accepted_list, ar_list, lp_list, ss_list = [], [], [], [], []
    runtimes, mtimes, grad_evals = [], [], []
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
        if "sampling_grad_evals" in z.files:
            grad_evals.append(int(z["sampling_grad_evals"]))
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
    return mc, total_runtime, sum(grad_evals)


# =====================================================================
# Reporting
# =====================================================================

def rmse(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.sqrt(tf.reduce_mean((samples - true_val) ** 2)).numpy())


def mae(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.reduce_mean(tf.abs(samples - true_val)).numpy())


def save_lhnn_report(
    samples: tf.Tensor,
    samples_chains: tf.Tensor,
    accept_rate: float,
    t_total: float,
    num_chains: int,
    samples_per_chain: int,
    burn_per_chain: int,
    L: int,
    training_grad_evals: int,
    sampling_grad_evals: int,
    true_values: tf.Tensor,
    param_names: list,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ess = tfp.mcmc.effective_sample_size(samples)
    mean_ess = float(tf.reduce_mean(ess).numpy())
    n_total = num_chains * (samples_per_chain + burn_per_chain)
    cost_per_step = t_total / max(n_total, 1)

    total_lhnn_grads = training_grad_evals + sampling_grad_evals
    traditional_hmc_grads = (L + 1) * num_chains * (samples_per_chain + burn_per_chain)
    saved = traditional_hmc_grads - total_lhnn_grads
    saved_pct = 100.0 * saved / max(traditional_hmc_grads, 1)
    fallback_intensity = sampling_grad_evals / max(n_total, 1)
    breakeven = training_grad_evals / max(L + 1, 1)
    ess_per_grad = mean_ess / max(total_lhnn_grads, 1)

    col_w = 40
    lines = [
        "=" * 85,
        "L-HNN HMC only — dispersed-pilot training, shared CRN across chains",
        "Model: Andrieu et al. (2010), Eq 14-15  |  Prior: InvGamma(0.01, 0.01)",
        f"Chains: {num_chains}  |  samples/chain: {samples_per_chain}  |  burn/chain: {burn_per_chain}",
        "=" * 85,
        f"{'Metric':<{col_w}} {'L-HNN HMC':>20}",
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
    lines.append("-" * 85)
    lines += [
        "L-HNN gradient cost breakdown (Dhulipala et al. 2022, Table 1):",
        f"  Training (pilot) gradient evals : {training_grad_evals}",
        f"  Sampling fallback gradient evals : {sampling_grad_evals}",
        f"  Total L-HNN gradient evals       : {total_lhnn_grads}",
        f"  Traditional HMC gradient evals   : {traditional_hmc_grads}  ((L+1)×iters)",
        f"  Gradient evals saved             : {saved} ({saved_pct:.1f}%)",
        f"  Fallback intensity (∇ / step)    : {fallback_intensity:.3f}",
        f"  Break-even iters (train÷(L+1))   : {breakeven:.0f}",
        "-" * 85,
        f"ESS per gradient evaluation       : {ess_per_grad:.4e}",
        "=" * 85,
    ]

    truth_cpu = tf.cast(true_values, tf.float32)
    summary = diagnostics_summary(samples_chains, param_names, truth=truth_cpu)
    lines += [
        "",
        "Multi-chain convergence diagnostics  (Vehtari et al., 2021)",
        "  Pass: rankR^ < 1.01 AND bulk-ESS > 400 AND tail-ESS > 400 per param.",
        "",
        format_diagnostics_table("L-HNN HMC (dispersed pilots, shared CRN)", summary),
    ]

    report = "\n".join(lines)
    path = out_dir / "comparison" / "results.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved results to {path}")


def plot_lhnn_diagnostics(
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
        ax.plot(s_j, alpha=0.65, lw=0.8, label="L-HNN HMC (pooled)")
        ax.axhline(tv, color="k", ls="--", lw=1.2, label="true")
        ax.set_ylabel(param_names[j])
        ax.set_xlabel("iteration (pooled)")
        ax.set_title(f"Trace — {param_names[j]}", fontweight="bold")
        ax.legend(fontsize=8)
        ax = axes[j, 1]
        ax.hist(s_j, bins=40, density=True, alpha=0.7, label="L-HNN HMC")
        ax.axvline(tv, color="k", ls="--", lw=1.2, label="true")
        ax.set_title(f"Posterior — {param_names[j]}", fontweight="bold")
        ax.set_xlabel(param_names[j])
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison" / "trace_posterior.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Dispersed-pilot training
# =====================================================================

def train_lhnn_with_dispersed_pilots(
    target_log_prob_fn,
    init_state: tf.Tensor,
    num_chains: int,
    dispersion_scale: float,
    base_seed: int,
    cfg: LHNNConfig,
    step_size: float,
    out_dir: Path,
    verbose: bool = True,
) -> tuple[LatentHNN, int, float, float]:
    """Generate pilots from `num_chains` dispersed starts; concatenate;
    train L-HNN. Returns (lhnn, total_grad_evals, pilot_time_s, train_time_s).

    Why this differs from the legacy `run_lhnn_hmc` training: the original
    chained all `num_pilot_trajectories` from a single point, so the network
    only ever saw one basin of the bimodal σ_v²/σ_w² posterior. Splitting
    the budget across the same dispersion used at sample time ensures the
    network's training distribution covers the modes the chains will visit.
    """
    inits = disperse_initial_states(
        init_state, num_chains=num_chains, scale=dispersion_scale, seed=base_seed
    )

    # Split pilot trajectories evenly across dispersed starts.
    base_count = cfg.num_pilot_trajectories // num_chains
    remainder = cfg.num_pilot_trajectories - base_count * num_chains
    per_start = [base_count + (1 if i < remainder else 0) for i in range(num_chains)]

    if verbose:
        print(f"[L-HNN train] dispersed-pilot generation: "
              f"{cfg.num_pilot_trajectories} traj split across {num_chains} starts "
              f"(scale={dispersion_scale}); per-start counts: {per_start}")

    q_all, p_all, dq_all, dp_all = [], [], [], []
    total_grad = 0
    t_pilot0 = time.time()
    for i in range(num_chains):
        if per_start[i] == 0:
            continue
        if verbose:
            print(f"\n[L-HNN train] start {i + 1}/{num_chains}  "
                  f"init={inits[i].numpy().tolist()}  trajectories={per_start[i]}")
        q_i, p_i, dq_i, dp_i, grad_i = generate_training_data(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=inits[i],
            num_trajectories=per_start[i],
            steps_per_trajectory=cfg.pilot_steps_per_trajectory,
            step_size=step_size,
            seed=base_seed + 7919 * (i + 1),
            verbose=verbose,
        )
        q_all.append(q_i)
        p_all.append(p_i)
        dq_all.append(dq_i)
        dp_all.append(dp_i)
        total_grad += int(grad_i)
    t_pilot = time.time() - t_pilot0

    q_data = tf.concat(q_all, axis=0)
    p_data = tf.concat(p_all, axis=0)
    dq_data = tf.concat(dq_all, axis=0)
    dp_data = tf.concat(dp_all, axis=0)

    if verbose:
        print(f"\n[L-HNN train] dispersed pilot data: {int(q_data.shape[0])} points "
              f"({total_grad} gradient evals, {t_pilot:.1f}s)")

    d = int(init_state.shape[0])
    lhnn = LatentHNN(d, cfg.hidden_units, cfg.num_hidden)
    _ = lhnn(tf.zeros([1, 2 * d]))

    t_train0 = time.time()
    train_lhnn(
        lhnn=lhnn,
        q_data=q_data,
        p_data=p_data,
        dq_dt=dq_data,
        dp_dt=dp_data,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        verbose=verbose,
    )
    t_train = time.time() - t_train0

    return lhnn, total_grad, t_pilot, t_train


# =====================================================================
# CLI
# =====================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L-HNN HMC-only with chain isolation + dispersed-pilot training.")
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--samples_per_chain", type=int, default=1000)
    p.add_argument("--burn_per_chain", type=int, default=500)
    p.add_argument("--chain_id", type=int, default=-1)
    p.add_argument("--train", action="store_true",
                   help="Train L-HNN with dispersed pilots and save weights.")
    p.add_argument("--aggregate", action="store_true")
    # L-HNN hyper-parameters (match exp_part3_bonus1b_hmc_vs_pmmh.py defaults)
    p.add_argument("--lhnn_hidden", type=int, default=256)
    p.add_argument("--lhnn_layers", type=int, default=3)
    p.add_argument("--lhnn_epochs", type=int, default=3000)
    p.add_argument("--lhnn_trajectories", type=int, default=20)
    p.add_argument("--lhnn_pilot_steps", type=int, default=30)
    p.add_argument("--lhnn_error_thresh", type=float, default=10.0)
    p.add_argument("--lhnn_cooldown", type=int, default=10)
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
    do_train = bool(args.train)
    param_names = ["sigma_v^2", "sigma_w^2"]
    true_values = tf.constant([true_sv2, true_sw2], tf.float32)
    out_dir = Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/lhnn_only")

    base_seed = 400
    step_size = 0.005
    dispersion_scale = 0.1

    # mode validation
    modes = sum([do_train, chain_id >= 0, aggregate])
    if modes > 1:
        raise SystemExit("--train, --chain_id, --aggregate are mutually exclusive.")
    if chain_id >= num_chains:
        raise SystemExit(f"--chain_id {chain_id} must be < --num_chains {num_chains}.")

    cfg = LHNNConfig(
        hidden_units=args.lhnn_hidden,
        num_hidden=args.lhnn_layers,
        epochs=args.lhnn_epochs,
        num_pilot_trajectories=args.lhnn_trajectories,
        pilot_steps_per_trajectory=args.lhnn_pilot_steps,
        error_threshold=args.lhnn_error_thresh,
        cooldown_steps=args.lhnn_cooldown,
    )

    print("=" * 70)
    print("  L-HNN HMC only — Bonus 1b (chain-isolated, dispersed pilots)")
    print(f"  T={T}  N_ledh={N_ledh}  L={L}")
    print(f"  samples/chain={n_samp}  burn/chain={n_burn}  chains={num_chains}")
    print(f"  L-HNN: hidden={cfg.hidden_units}×{cfg.num_hidden} epochs={cfg.epochs}")
    print(f"         pilots={cfg.num_pilot_trajectories} × {cfg.pilot_steps_per_trajectory} steps")
    print(f"  CRN: shared across chains (crn_offset=base_seed)")
    if do_train:
        print("  Mode: TRAIN (dispersed pilots)")
    elif chain_id >= 0:
        print(f"  Mode: ISOLATED CHAIN (chain_id={chain_id})")
    elif aggregate:
        print("  Mode: AGGREGATE")
    else:
        print("  Mode: DEFAULT — train and run all chains in one process")
    print("=" * 70)

    # ---- aggregate mode ----------------------------------------------------
    if aggregate:
        mc, t_total, sampling_grad_evals = _load_all_chains(out_dir, num_chains)
        # also need training_grad_evals from saved meta
        try:
            meta = np.load(_model_dir(out_dir) / "training_meta.npz")
            training_grad_evals = int(meta["training_grad_evals"])
            t_train_pilot = float(meta["pilot_runtime_s"]) + float(meta["train_runtime_s"])
        except FileNotFoundError:
            training_grad_evals = 0
            t_train_pilot = 0.0
            print("  WARNING: no training_meta.npz found; gradient breakdown will be zero.")

        samples_chains = tf.exp(mc.samples)
        samples_flat = tf.reshape(samples_chains, [num_chains * n_samp, 2])
        accept_rate = float(tf.reduce_mean(mc.accept_rate).numpy())
        t_grand = t_total + t_train_pilot
        print(f"\n  Aggregate: chain runtime = {t_total:.1f}s  "
              f"+ training {t_train_pilot:.1f}s = {t_grand:.1f}s  "
              f"mean_accept={accept_rate:.3f}")
        save_lhnn_report(
            samples=samples_flat,
            samples_chains=samples_chains,
            accept_rate=accept_rate,
            t_total=t_grand,
            num_chains=num_chains,
            samples_per_chain=n_samp,
            burn_per_chain=n_burn,
            L=L,
            training_grad_evals=training_grad_evals,
            sampling_grad_evals=sampling_grad_evals,
            true_values=true_values,
            param_names=param_names,
            out_dir=out_dir,
        )
        plot_lhnn_diagnostics(samples_flat, true_values, param_names, out_dir)
        return 0

    # ---- shared setup for train / chain modes ------------------------------
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

    # ---- train mode --------------------------------------------------------
    if do_train:
        lhnn, training_grad_evals, t_pilot, t_train = train_lhnn_with_dispersed_pilots(
            target_log_prob_fn=target_hmc,
            init_state=init_state,
            num_chains=num_chains,
            dispersion_scale=dispersion_scale,
            base_seed=base_seed,
            cfg=cfg,
            step_size=step_size,
            out_dir=out_dir,
            verbose=True,
        )
        weights_path = _save_trained_lhnn(
            lhnn=lhnn,
            cfg=cfg,
            training_grad_evals=training_grad_evals,
            pilot_runtime_s=t_pilot,
            train_runtime_s=t_train,
            out_dir=out_dir,
        )
        print(f"\n  Training done: pilot_grads={training_grad_evals}  "
              f"pilot_time={t_pilot:.1f}s  train_time={t_train:.1f}s")
        print(f"  Saved L-HNN -> {weights_path}")
        return 0

    # ---- isolated chain mode -----------------------------------------------
    if chain_id >= 0:
        d = int(init_state.shape[0])
        lhnn, cfg_loaded, info = _load_trained_lhnn(out_dir, d=d)
        chain_seed = base_seed + 1009 * (chain_id + 1)
        lhnn_inits = disperse_initial_states(
            init_state, num_chains=num_chains, scale=dispersion_scale, seed=base_seed
        )
        print(f"\n[isolated L-HNN chain {chain_id}/{num_chains}]  seed={chain_seed}"
              f"  init={lhnn_inits[chain_id].numpy().tolist()}")
        t0 = time.time()
        result, _, diag = run_lhnn_hmc(
            target_log_prob_fn=target_hmc,
            initial_state=lhnn_inits[chain_id],
            num_results=n_samp,
            num_burnin=n_burn,
            step_size=step_size,
            num_leapfrog_steps=L,
            target_accept_prob=0.45,
            seed=chain_seed,
            verbose=True,
            adapt_step_size=True,
            lhnn_config=cfg_loaded,
            pretrained_lhnn=lhnn,
            crn_offset=base_seed,  # shared CRN across chains (§4.5)
        )
        t_chain = time.time() - t0
        path = _save_single_chain(
            result=result,
            chain_id=chain_id,
            out_dir=out_dir,
            runtime_s=t_chain,
            sampling_grad_evals=int(diag.sampling_real_gradient_evals),
        )
        print(
            f"\n  Chain {chain_id} done in {t_chain:.1f}s  "
            f"accept={float(result.accept_rate.numpy()):.3f}  "
            f"sampling_grads={int(diag.sampling_real_gradient_evals)}  saved -> {path}"
        )
        return 0

    # ---- default: train + run all chains in one process --------------------
    print("\nDefault mode runs train + all chains in one process. Use the "
          "isolated driver (scripts/run_lhnn_isolated.sh) for long runs.")
    lhnn, training_grad_evals, t_pilot, t_train = train_lhnn_with_dispersed_pilots(
        target_log_prob_fn=target_hmc,
        init_state=init_state,
        num_chains=num_chains,
        dispersion_scale=dispersion_scale,
        base_seed=base_seed,
        cfg=cfg,
        step_size=step_size,
        out_dir=out_dir,
        verbose=True,
    )
    _save_trained_lhnn(
        lhnn=lhnn,
        cfg=cfg,
        training_grad_evals=training_grad_evals,
        pilot_runtime_s=t_pilot,
        train_runtime_s=t_train,
        out_dir=out_dir,
    )
    lhnn_inits = disperse_initial_states(
        init_state, num_chains=num_chains, scale=dispersion_scale, seed=base_seed
    )
    samples_per_chain_list = []
    accepted_per_chain = []
    accept_rates = []
    target_lps = []
    step_sizes_list = []
    sampling_grad_evals_total = 0
    t_chain_total = 0.0
    for c in range(num_chains):
        chain_seed = base_seed + 1009 * (c + 1)
        print(f"\n[chain {c + 1}/{num_chains}] seed={chain_seed}")
        t0 = time.time()
        result, _, diag = run_lhnn_hmc(
            target_log_prob_fn=target_hmc,
            initial_state=lhnn_inits[c],
            num_results=n_samp,
            num_burnin=n_burn,
            step_size=step_size,
            num_leapfrog_steps=L,
            target_accept_prob=0.45,
            seed=chain_seed,
            verbose=True,
            adapt_step_size=True,
            lhnn_config=cfg,
            pretrained_lhnn=lhnn,
            crn_offset=base_seed,
        )
        t_chain_total += time.time() - t0
        samples_per_chain_list.append(result.samples)
        accepted_per_chain.append(result.is_accepted)
        accept_rates.append(result.accept_rate)
        target_lps.append(result.target_log_probs)
        step_sizes_list.append(result.step_sizes)
        sampling_grad_evals_total += int(diag.sampling_real_gradient_evals)

    mc = MultiChainHMCResult(
        samples=tf.stack(samples_per_chain_list, axis=0),
        is_accepted=tf.stack(accepted_per_chain, axis=0),
        accept_rate=tf.stack(accept_rates, axis=0),
        target_log_probs=tf.stack(target_lps, axis=0),
        step_sizes=tf.stack(step_sizes_list, axis=0),
    )
    samples_chains = tf.exp(mc.samples)
    samples_flat = tf.reshape(samples_chains, [num_chains * n_samp, 2])
    accept_rate = float(tf.reduce_mean(mc.accept_rate).numpy())
    t_grand = t_chain_total + t_pilot + t_train

    print(f"\n  L-HNN HMC ({num_chains} chains) sampling done in {t_chain_total:.1f}s  "
          f"mean_accept={accept_rate:.3f}")
    save_lhnn_report(
        samples=samples_flat,
        samples_chains=samples_chains,
        accept_rate=accept_rate,
        t_total=t_grand,
        num_chains=num_chains,
        samples_per_chain=n_samp,
        burn_per_chain=n_burn,
        L=L,
        training_grad_evals=training_grad_evals,
        sampling_grad_evals=sampling_grad_evals_total,
        true_values=true_values,
        param_names=param_names,
        out_dir=out_dir,
    )
    plot_lhnn_diagnostics(samples_flat, true_values, param_names, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
