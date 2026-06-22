"""
Aggregate SVSSM HMC chains saved by ``scripts/exp/exp_hmc_svssm.py``.

Reads ``svssm_hmc_samples.npz`` (joint save of all chains).  Optionally reads
``svssm_hmc_summary.json`` next to it for burn-in length and elapsed seconds.

Writes Kitagawa-style aggregate artefacts (same layout as
``exp_part3_bonus1b_hmc_ledh_only_old.py --aggregate``):

  ``<out_dir>/comparison/results.txt``
  ``<out_dir>/comparison/trace_posterior.png``

Truth in the npz is stored as ``[mu, phi, sigma_eta]`` (standard deviation).
Diagnostics use constrained draws ``[mu, phi, sigma_eta_sq]`` and compare to
``sigma_eta ** 2`` for the third component.

``trash/trace_posterior.png`` (plots only) shows ``mu``, ``phi``, ``sigma_eta``
(sqrt of stored variance), and derived stationary variance
``Var(h)_stat = sigma_eta_sq / (1 - phi^2)``.

Usage::

    python -m src.experiments.exp_svssm_hmc_aggregate \\
        --samples_npz reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc/svssm_hmc_samples.npz

    # If JSON is missing, supply burn + runtime for acceptance/cost lines::
    python -m src.experiments.exp_svssm_hmc_aggregate \\
        --samples_npz path/to/svssm_hmc_samples.npz \\
        --burn_per_chain 200 --runtime_total 3600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.utils.mcmc_diagnostics import diagnostics_summary, format_diagnostics_table

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def rmse(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.sqrt(tf.reduce_mean((samples - true_val) ** 2)).numpy())


def mae(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.reduce_mean(tf.abs(samples - true_val)).numpy())


def _try_load_sidecar_meta(npz_path: Path) -> tuple[int | None, int | None, float | None]:
    """Return (num_burnin, num_results, elapsed_s) from ``svssm_hmc_summary.json`` if present."""
    summary_path = npz_path.with_name("svssm_hmc_summary.json")
    if not summary_path.is_file():
        return None, None, None
    try:
        meta = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = meta.get("config") or {}
        burn = cfg.get("num_burnin")
        nres = cfg.get("num_results")
        elapsed = meta.get("elapsed_s")
        return (
            int(burn) if burn is not None else None,
            int(nres) if nres is not None else None,
            float(elapsed) if elapsed is not None else None,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None, None, None


def save_svssm_aggregate_report(
    *,
    samples_chains: tf.Tensor,
    samples_flat: tf.Tensor,
    accept_rate: float,
    t_total: float,
    num_chains: int,
    samples_per_chain: int,
    burn_per_chain: int,
    true_values: tf.Tensor,
    param_names: list[str],
    out_dir: Path,
    title_line: str,
) -> None:
    """Human-readable report + Vehtari-style table (mirrors Kitagawa aggregate)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ess = tfp.mcmc.effective_sample_size(samples_flat)
    mean_ess = float(tf.reduce_mean(ess).numpy())
    n_total = num_chains * samples_per_chain
    n_burn_total = num_chains * burn_per_chain
    cost_denom = n_total + n_burn_total
    cost_per_step = t_total / max(cost_denom, 1)

    col_w = 40
    lines = [
        "=" * 85,
        title_line,
        "SVSSM (Gaussian quasi-likelihood LEDH). Priors: see original HMC script.",
        f"Chains: {num_chains}  |  samples/chain: {samples_per_chain}  |  burn/chain: {burn_per_chain}",
        "=" * 85,
        f"{'Metric':<{col_w}} {'SVSSM HMC':>20}",
        "-" * 85,
        f"{'Acceptance rate (post-burn-in est.)':<{col_w}} {accept_rate:>20.3f}",
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
        m = float(tf.reduce_mean(samples_flat[:, j]).numpy())
        tv = float(true_values[j].numpy())
        lines.append(f"{'Mean ' + pname:<{col_w}} {m:>20.4f}  (true={tv:.4f})")
    for j, pname in enumerate(param_names):
        s = float(tf.math.reduce_std(samples_flat[:, j]).numpy())
        lines.append(f"{'Std ' + pname:<{col_w}} {s:>20.4f}")
    lines.append("-" * 85)
    lines.append("Quality vs ground truth (pooled posterior):")
    for j, pname in enumerate(param_names):
        tv = float(true_values[j].numpy())
        lines.append(f"  RMSE ({pname:<18}) {rmse(samples_flat[:, j], tv):.6f}")
        lines.append(f"  MAE  ({pname:<18}) {mae(samples_flat[:, j], tv):.6f}")
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
        format_diagnostics_table("SVSSM HMC (reloaded chains)", summary),
        "",
        "Notes:",
        " - Harvey-Ruiz-Shephard quasi-likelihood often biases sigma_eta upward;",
        "   sigma_eta_sq posterior vs truth is informative but may miss coverage.",
        "=" * 85,
    ]

    report = "\n".join(lines)
    path = out_dir / "trash" / "results.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved results to {path}")


def plot_svssm_diagnostics(
    samples_flat: tf.Tensor,
    out_dir: Path,
    *,
    mu_t: float,
    phi_t: float,
    sigma_eta_t: float,
) -> None:
    """Trace + posterior plots (pooled chains).

    Row 3 uses ``sigma_eta`` (not ``sigma_eta_sq``). Row 4 is the derived
    stationary-state variance ``sigma_eta_sq / (1 - phi^2)`` per draw.
    """
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = samples_flat.numpy()
    eps = 1e-8
    phi = flat[:, 1]
    sig_sq = flat[:, 2]
    var_stat_true = sigma_eta_t ** 2 / max(1.0 - phi_t ** 2, eps)
    plot_specs = [
        ("mu", flat[:, 0], mu_t),
        ("phi", phi, phi_t),
        ("sigma_eta", np.sqrt(np.maximum(sig_sq, eps)), sigma_eta_t),
        ("Var(h)_stat", sig_sq / np.maximum(1.0 - phi ** 2, eps), var_stat_true),
    ]
    n_rows = len(plot_specs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for j, (label, s_j, tv) in enumerate(plot_specs):
        ax = axes[j, 0]
        ax.plot(s_j, alpha=0.65, lw=0.8, label="SVSSM HMC (pooled)")
        ax.axhline(tv, color="k", ls="--", lw=1.2, label="true")
        ax.set_ylabel(label)
        ax.set_xlabel("iteration (pooled)")
        ax.set_title(f"Trace — {label}", fontweight="bold")
        ax.legend(fontsize=8)
        ax = axes[j, 1]
        ax.hist(s_j, bins=40, density=True, alpha=0.7, label="SVSSM HMC")
        ax.axvline(tv, color="k", ls="--", lw=1.2, label="true")
        ax.set_title(f"Posterior — {label}", fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / "trash" / "trace_posterior.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate SVSSM HMC samples.npz like Kitagawa --aggregate.")
    p.add_argument(
        "--samples_npz",
        type=str,
        default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc/svssm_hmc_samples.npz",
        help="Path to svssm_hmc_samples.npz from exp_hmc_svssm.py.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory for comparison/ output (default: parent directory of npz).",
    )
    p.add_argument(
        "--burn_per_chain",
        type=int,
        default=-1,
        help="Burn-in per chain for acceptance + cost lines. -1 = JSON, else infer from accept trace length minus draws, else 0.",
    )
    p.add_argument(
        "--runtime_total",
        type=float,
        default=-1.0,
        help="Total wall seconds across chains. -1 = infer from JSON if present else 0.",
    )
    args = p.parse_args()

    npz_path = Path(args.samples_npz).resolve()
    if not npz_path.is_file():
        raise SystemExit(f"Missing samples file: {npz_path}")

    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else npz_path.parent

    z = np.load(npz_path)
    if "samples_constrained" not in z.files:
        raise SystemExit(f"{npz_path} missing 'samples_constrained' array.")

    samples_np = z["samples_constrained"].astype(np.float32)
    if samples_np.ndim != 3:
        raise SystemExit(f"Expected samples_constrained rank 3 (chains, draws, dim); got {samples_np.shape}.")

    num_chains, samples_per_chain, dim = samples_np.shape
    if dim != 3:
        raise SystemExit(f"Expected 3 constrained parameters; got dim={dim}.")

    json_burn, json_results, json_elapsed = _try_load_sidecar_meta(npz_path)

    acc_for_infer = np.asarray(z["accept"]) if "accept" in z.files else None
    inferred_burn = 0
    if (
        acc_for_infer is not None
        and acc_for_infer.ndim == 2
        and json_burn is None
        and int(args.burn_per_chain) < 0
    ):
        diff = int(acc_for_infer.shape[1]) - int(samples_per_chain)
        if diff > 0:
            inferred_burn = diff

    if int(args.burn_per_chain) >= 0:
        burn_per_chain = int(args.burn_per_chain)
    elif json_burn is not None:
        burn_per_chain = int(json_burn)
    elif inferred_burn > 0:
        burn_per_chain = inferred_burn
        print(f"Inferred burn_per_chain={burn_per_chain} from accept trace vs saved samples.")
    else:
        burn_per_chain = 0

    t_total = float(args.runtime_total)
    if t_total < 0:
        t_total = float(json_elapsed) if json_elapsed is not None else 0.0

    if json_results is not None and int(json_results) != samples_per_chain:
        print(
            f"Warning: JSON num_results={json_results} != samples_per_chain={samples_per_chain} "
            "(npz wins for shapes)."
        )

    # Truth: npz stores [mu, phi, sigma_eta]; diagnostics on sigma_eta_sq.
    if "truth" in z.files:
        tnp = np.asarray(z["truth"], dtype=np.float64).reshape(-1)
        mu_t, phi_t, sigma_eta_t = float(tnp[0]), float(tnp[1]), float(tnp[2])
    else:
        raise SystemExit(f"{npz_path} missing 'truth' vector.")

    true_values = tf.constant([mu_t, phi_t, sigma_eta_t ** 2], tf.float32)
    param_names = ["mu", "phi", "sigma_eta_sq"]

    accept = z["accept"] if "accept" in z.files else None
    accept_rate = float("nan")
    if accept is not None:
        acc = np.asarray(accept)
        if acc.ndim == 2 and burn_per_chain > 0 and acc.shape[1] > burn_per_chain:
            acc_mean = np.mean(acc[:, burn_per_chain:].astype(np.float64))
        else:
            acc_mean = np.mean(acc.astype(np.float64))
        accept_rate = float(acc_mean)

    samples_chains = tf.constant(samples_np, dtype=tf.float32)
    samples_flat = tf.reshape(samples_chains, [num_chains * samples_per_chain, dim])

    title_line = f"SVSSM HMC aggregate (reloaded)  |  source={npz_path.name}"

    save_svssm_aggregate_report(
        samples_chains=samples_chains,
        samples_flat=samples_flat,
        accept_rate=accept_rate,
        t_total=t_total,
        num_chains=num_chains,
        samples_per_chain=samples_per_chain,
        burn_per_chain=burn_per_chain,
        true_values=true_values,
        param_names=param_names,
        out_dir=out_dir,
        title_line=title_line,
    )
    plot_svssm_diagnostics(
        samples_flat,
        out_dir,
        mu_t=mu_t,
        phi_t=phi_t,
        sigma_eta_t=sigma_eta_t,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
