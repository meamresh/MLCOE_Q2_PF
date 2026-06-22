"""
Aggregate multivariate SVSSM HMC chains saved by
``scripts/exp/exp_hmc_svssm_multivariate.py``.

This is the multivariate sibling of
``src.experiments.exp_svssm_hmc_aggregate``. The difference is the
saved-npz shape:

- Univariate driver saves ``samples_constrained`` as ``(chains, draws, 3)``
  with ``truth = [mu, phi, sigma_eta]`` (length 3).
- Multivariate driver saves ``samples_constrained`` as
  ``(chains, draws, 3, d)`` with ``truth = (3, d)`` --- one row per
  parameter type (mu / phi / sigma_eta_sq), one column per component.

We flatten the trailing ``(3, d)`` into ``3d`` named parameters and run
the same Vehtari-style diagnostics (rank-R$\\hat{}$, bulk-ESS, tail-ESS,
coverage) that the univariate aggregator uses.

The third row of ``truth`` is ``sigma_eta_sq`` directly (the multivariate
driver saves sigma_eta_sq, not sigma_eta), so no squaring needed --- a
small but important difference from the univariate npz.

Usage::

    python -m src.experiments.exp_svssm_hmc_multivariate_aggregate \\
        --samples_npz reports/.../svssm_hmc_multivariate_d2_long/svssm_hmc_multi_samples.npz

Outputs (in ``<out_dir>/trash/`` for parity with the univariate
aggregator)::

    results.txt                        # human-readable report with Vehtari table
    trace_posterior_multivariate.png   # 4d rows: mu, phi, sigma_eta (sqrt),
                                       # plus derived Var(h)_stat_i per component
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


_PARAM_TYPE_NAMES = ["mu", "phi", "sigma_eta_sq"]


def rmse(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.sqrt(tf.reduce_mean((samples - true_val) ** 2)).numpy())


def mae(samples: tf.Tensor, true_val: float) -> float:
    return float(tf.reduce_mean(tf.abs(samples - true_val)).numpy())


def _try_load_sidecar_meta(npz_path: Path):
    """Return (num_burnin, num_results, elapsed_s) from the summary JSON if present.

    The multivariate driver writes ``svssm_hmc_multi_summary.json`` next
    to the npz; we also fall back to the univariate-style
    ``svssm_hmc_summary.json`` for compatibility.
    """
    for sidecar in ("svssm_hmc_multi_summary.json", "svssm_hmc_summary.json"):
        path = npz_path.with_name(sidecar)
        if not path.is_file():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
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
            continue
    return None, None, None


def _build_param_names(d: int) -> list[str]:
    """Flatten (mu, phi, sigma_eta_sq) × d components to a 3d-name list.

    Order: mu_0, mu_1, ..., mu_{d-1}, phi_0, ..., phi_{d-1}, sigma_eta_sq_0, ...
    """
    return [f"{ptype}_{c}" for ptype in _PARAM_TYPE_NAMES for c in range(d)]


def save_multi_aggregate_report(
    *,
    samples_chains: tf.Tensor,   # (chains, draws, 3d)  -- flattened
    samples_flat: tf.Tensor,     # (chains*draws, 3d)
    accept_rate: float,
    t_total: float,
    num_chains: int,
    samples_per_chain: int,
    burn_per_chain: int,
    true_values: tf.Tensor,      # (3d,) -- flattened
    param_names: list[str],      # length 3d
    d: int,
    out_dir: Path,
    title_line: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ess = tfp.mcmc.effective_sample_size(samples_flat)
    mean_ess = float(tf.reduce_mean(ess).numpy())
    n_total = num_chains * samples_per_chain
    n_burn_total = num_chains * burn_per_chain
    cost_denom = n_total + n_burn_total
    cost_per_step = t_total / max(cost_denom, 1)

    col_w = 40
    lines = [
        "=" * 90,
        title_line,
        f"Multivariate SVSSM (d={d}, diagonal Phi + diagonal Sigma_eta).",
        f"Chains: {num_chains}  |  samples/chain: {samples_per_chain}  |  "
        f"burn/chain: {burn_per_chain}",
        "=" * 90,
        f"{'Metric':<{col_w}} {'multivariate HMC':>20}",
        "-" * 90,
        f"{'Acceptance rate (post-burn-in est.)':<{col_w}} {accept_rate:>20.3f}",
    ]
    for j, pname in enumerate(param_names):
        lines.append(f"{'ESS (' + pname + ')':<{col_w}} {float(ess[j].numpy()):>20.1f}")
    lines += [
        f"{'Mean ESS':<{col_w}} {mean_ess:>20.1f}",
        f"{'Runtime (s)':<{col_w}} {t_total:>20.1f}",
        f"{'ESS/s':<{col_w}} {mean_ess / max(t_total, 1e-9):>20.4f}",
        f"{'Cost per step (s/proposal)':<{col_w}} {cost_per_step:>20.4f}",
        "-" * 90,
    ]
    for j, pname in enumerate(param_names):
        m = float(tf.reduce_mean(samples_flat[:, j]).numpy())
        tv = float(true_values[j].numpy())
        lines.append(f"{'Mean ' + pname:<{col_w}} {m:>20.4f}  (true={tv:.4f})")
    for j, pname in enumerate(param_names):
        s = float(tf.math.reduce_std(samples_flat[:, j]).numpy())
        lines.append(f"{'Std ' + pname:<{col_w}} {s:>20.4f}")
    lines.append("-" * 90)
    lines.append("Quality vs ground truth (pooled posterior):")
    for j, pname in enumerate(param_names):
        tv = float(true_values[j].numpy())
        lines.append(f"  RMSE ({pname:<18}) {rmse(samples_flat[:, j], tv):.6f}")
        lines.append(f"  MAE  ({pname:<18}) {mae(samples_flat[:, j], tv):.6f}")
    lines.append("=" * 90)

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
        format_diagnostics_table(
            f"Multivariate SVSSM HMC, d={d} (reloaded chains)",
            summary,
        ),
        "",
        "Notes:",
        " - Harvey-Ruiz-Shephard quasi-likelihood often biases sigma_eta upward;",
        "   sigma_eta_sq posteriors vs truth are informative but may miss coverage.",
        " - For non-diagonal Sigma_eta, expect rank-deficient boundary modes",
        "   (L_11 -> 0, implied rho -> 1) at small sample budgets; see Phase 15.",
        "=" * 90,
    ]

    report = "\n".join(lines)
    path = out_dir / "trash" / "results.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved results to {path}")


def plot_multi_diagnostics(
    samples_flat: tf.Tensor,
    out_dir: Path,
    *,
    d: int,
    truth_np: np.ndarray,
) -> None:
    """Trace + posterior plots (pooled chains).

    Rows ``3d``: ``mu_i``, ``phi_i``, ``sigma_eta_i`` (sqrt of stored variance).
    Rows ``d``: derived stationary variance ``Var(h)_stat_i = sigma_eta_sq_i / (1 - phi_i^2)``.
    """
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = samples_flat.numpy()
    eps = 1e-8
    plot_specs: list[tuple[str, np.ndarray, float]] = []

    for c in range(d):
        plot_specs.append((f"mu_{c}", flat[:, c], float(truth_np[0, c])))
    for c in range(d):
        plot_specs.append((f"phi_{c}", flat[:, d + c], float(truth_np[1, c])))
    for c in range(d):
        sig_sq = flat[:, 2 * d + c]
        plot_specs.append((
            f"sigma_eta_{c}",
            np.sqrt(np.maximum(sig_sq, eps)),
            float(np.sqrt(max(truth_np[2, c], eps))),
        ))
    for c in range(d):
        phi_c = flat[:, d + c]
        sig_sq_c = flat[:, 2 * d + c]
        plot_specs.append((
            f"Var(h)_stat_{c}",
            sig_sq_c / np.maximum(1.0 - phi_c ** 2, eps),
            float(truth_np[2, c] / max(1.0 - truth_np[1, c] ** 2, eps)),
        ))

    n_rows = len(plot_specs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for j, (label, s_j, tv) in enumerate(plot_specs):
        ax = axes[j, 0]
        ax.plot(s_j, alpha=0.6, lw=0.7, label="multi HMC (pooled)")
        ax.axhline(tv, color="k", ls="--", lw=1.2, label="true")
        ax.set_ylabel(label)
        ax.set_xlabel("iteration (pooled)")
        ax.set_title(f"Trace — {label}", fontweight="bold")
        ax.legend(fontsize=8)
        ax = axes[j, 1]
        ax.hist(s_j, bins=40, density=True, alpha=0.7, label="posterior")
        ax.axvline(tv, color="k", ls="--", lw=1.2, label="true")
        ax.set_title(f"Posterior — {label}", fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / "trash" / "trace_posterior_multivariate.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Aggregate multivariate SVSSM HMC samples.npz with Vehtari diagnostics.",
    )
    p.add_argument(
        "--samples_npz",
        type=str,
        required=True,
        help="Path to svssm_hmc_multi_samples.npz from exp_hmc_svssm_multivariate.py.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory for results.txt + trace_posterior.png (default: npz parent).",
    )
    p.add_argument("--burn_per_chain", type=int, default=-1,
                   help="-1 = read from sidecar JSON, else infer from accept "
                        "trace, else 0.")
    p.add_argument("--runtime_total", type=float, default=-1.0,
                   help="-1 = read from sidecar JSON if present, else 0.")
    args = p.parse_args()

    npz_path = Path(args.samples_npz).resolve()
    if not npz_path.is_file():
        raise SystemExit(f"Missing samples file: {npz_path}")
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else npz_path.parent

    z = np.load(npz_path)
    if "samples_constrained" not in z.files:
        raise SystemExit(f"{npz_path} missing 'samples_constrained' array.")

    sc_np = z["samples_constrained"].astype(np.float32)
    if sc_np.ndim != 4 or sc_np.shape[2] != 3:
        raise SystemExit(
            f"Expected samples_constrained rank-4 with shape "
            f"(chains, draws, 3, d); got {sc_np.shape}. "
            f"For univariate use exp_svssm_hmc_aggregate instead."
        )
    num_chains, samples_per_chain, _, d = sc_np.shape

    # Flatten the (3, d) trailing pair into 3d params: ordering matches
    # _build_param_names = [mu_0, mu_1, ..., phi_0, phi_1, ..., sig_0, sig_1, ...].
    # In sc_np[ chain, draw, param_type, component ], so transpose (param_type, component)
    # to interleave by parameter type first, then component.
    sc_flat = sc_np.transpose(0, 1, 2, 3).reshape(num_chains, samples_per_chain, 3 * d)
    # That preserves ordering [mu_0, mu_1, ..., phi_0, phi_1, ..., sig_0, sig_1, ...]
    # because numpy reshape uses row-major and param_type varies slower than component.

    param_names = _build_param_names(d)

    if "truth" not in z.files:
        raise SystemExit(f"{npz_path} missing 'truth' array.")
    truth_np = np.asarray(z["truth"], dtype=np.float32)
    if truth_np.shape != (3, d):
        raise SystemExit(
            f"Expected truth shape (3, d) for multivariate; got {truth_np.shape}."
        )
    truth_flat = truth_np.reshape(3 * d)  # same ordering as param_names
    true_values = tf.constant(truth_flat, tf.float32)

    # Sidecar JSON for burn-in / runtime.
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
        print(f"Inferred burn_per_chain={burn_per_chain} from accept trace.")
    else:
        burn_per_chain = 0

    t_total = float(args.runtime_total)
    if t_total < 0:
        t_total = float(json_elapsed) if json_elapsed is not None else 0.0

    if json_results is not None and int(json_results) != samples_per_chain:
        print(
            f"Warning: JSON num_results={json_results} != samples_per_chain="
            f"{samples_per_chain} (npz wins for shapes)."
        )

    accept = z["accept"] if "accept" in z.files else None
    accept_rate = float("nan")
    if accept is not None:
        acc = np.asarray(accept)
        if acc.ndim == 2 and burn_per_chain > 0 and acc.shape[1] > burn_per_chain:
            acc_mean = np.mean(acc[:, burn_per_chain:].astype(np.float64))
        else:
            acc_mean = np.mean(acc.astype(np.float64))
        accept_rate = float(acc_mean)

    samples_chains = tf.constant(sc_flat, dtype=tf.float32)
    samples_flat = tf.reshape(samples_chains,
                               [num_chains * samples_per_chain, 3 * d])

    title_line = (
        f"Multivariate SVSSM HMC aggregate (reloaded)  |  source={npz_path.name}  "
        f"d={d}"
    )

    save_multi_aggregate_report(
        samples_chains=samples_chains,
        samples_flat=samples_flat,
        accept_rate=accept_rate,
        t_total=t_total,
        num_chains=num_chains,
        samples_per_chain=samples_per_chain,
        burn_per_chain=burn_per_chain,
        true_values=true_values,
        param_names=param_names,
        d=d,
        out_dir=out_dir,
        title_line=title_line,
    )
    plot_multi_diagnostics(samples_flat, out_dir, d=d, truth_np=truth_np)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
