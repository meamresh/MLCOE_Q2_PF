"""
Diagnose gradient stability of differentiable LEDH likelihood estimators.

The HMC experiments are only as reliable as the gradient of the differentiable
particle-filter likelihood.  This script probes that gradient directly by
evaluating repeated Common-Random-Number (CRN) realisations at fixed parameter
values.

For each method and theta = (log sigma_v^2, log sigma_w^2), it reports:

- finite log-likelihood and gradient rates,
- mean/std/CV of log-likelihood values,
- mean/std/CV of gradient norms,
- MAD of log10 gradient norms,
- component-wise gradient SNR = |mean grad_j| / std grad_j,
- vector gradient SNR = ||E[g]|| / sqrt(trace Cov[g]),
- cosine similarity to the mean gradient direction.

This is a diagnostic experiment, not a sampler.  It is intentionally small by
default so it can be used before launching expensive HMC runs.

Usage:
    python -m src.experiments.exp_part3_bonus1b_ledh_gradient_stability
    python -m src.experiments.exp_part3_bonus1b_ledh_gradient_stability --methods generic_pfpf,legacy_ledh --n_seeds 20
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.filters.bonus.differentiable_pfpf_ledh import (
    DifferentiablePFPFLEDHLogLikelihood,
    KitagawaPFPFLEDHLogLikelihood,
)
from src.models.ssm_katigawa import PMCMCNonlinearSSM

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for one likelihood implementation."""

    name: str
    label: str
    make_loglik: Callable[[], Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]


def generate_data(
    T: int = 50,
    sigma_v_sq: float = 10.0,
    sigma_w_sq: float = 1.0,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Generate Kitagawa/Andrieu nonlinear SSM data."""
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
        xs.append(x)
        ys.append(x**2 / 20.0 + sw * tf.random.normal([]))
    return tf.stack(xs), tf.stack(ys)


def _finite_float(x) -> float:
    """Convert scalar-like value to float, preserving non-finite values."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _safe_cv(mean: float, std: float) -> float:
    """Coefficient of variation with NaN for near-zero denominator."""
    return float(std / abs(mean)) if math.isfinite(mean) and abs(mean) > 1e-12 else float("nan")


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation."""
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _cosine_to_mean(grads: np.ndarray) -> tuple[float, float]:
    """Mean/std cosine similarity between each gradient and the mean gradient."""
    if grads.shape[0] < 2:
        return float("nan"), float("nan")
    mean_g = np.mean(grads, axis=0)
    mean_norm = np.linalg.norm(mean_g)
    if mean_norm <= 1e-12:
        return float("nan"), float("nan")
    norms = np.linalg.norm(grads, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        return float("nan"), float("nan")
    cos = grads[valid] @ mean_g / (norms[valid] * mean_norm)
    return float(np.mean(cos)), float(np.std(cos))


def evaluate_gradient_replicates(
    loglik_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    theta: tf.Tensor,
    y_obs: tf.Tensor,
    seeds: list[int],
    grad_clip_norm: float | None = None,
) -> list[dict]:
    """Evaluate log-likelihood and gradient across CRN seeds."""
    rows: list[dict] = []
    for seed in seeds:
        tf.random.set_seed(seed)
        theta_t = tf.identity(tf.cast(theta, tf.float32))
        t0 = time.time()
        try:
            with tf.GradientTape() as tape:
                tape.watch(theta_t)
                ll = loglik_fn(theta_t, y_obs)
                ll = tf.cast(tf.math.real(ll), tf.float32)
            grad = tape.gradient(ll, theta_t)
            if grad is None:
                grad = tf.fill(tf.shape(theta_t), tf.constant(float("nan"), tf.float32))
            grad = tf.cast(tf.math.real(grad), tf.float32)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                grad = tf.clip_by_norm(grad, grad_clip_norm)
            elapsed = time.time() - t0

            g_np = grad.numpy().astype(np.float64)
            ll_val = _finite_float(ll.numpy())
            finite_ll = math.isfinite(ll_val)
            finite_grad = bool(np.all(np.isfinite(g_np)))
            grad_norm = float(np.linalg.norm(g_np)) if finite_grad else float("nan")
            rows.append(
                {
                    "seed": seed,
                    "loglik": ll_val,
                    "grad_log_sigma_v2": float(g_np[0]) if g_np.size > 0 else float("nan"),
                    "grad_log_sigma_w2": float(g_np[1]) if g_np.size > 1 else float("nan"),
                    "grad_norm": grad_norm,
                    "finite_loglik": int(finite_ll),
                    "finite_grad": int(finite_grad),
                    "elapsed_s": elapsed,
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic should keep going.
            elapsed = time.time() - t0
            rows.append(
                {
                    "seed": seed,
                    "loglik": float("nan"),
                    "grad_log_sigma_v2": float("nan"),
                    "grad_log_sigma_w2": float("nan"),
                    "grad_norm": float("nan"),
                    "finite_loglik": 0,
                    "finite_grad": 0,
                    "elapsed_s": elapsed,
                    "error": repr(exc),
                }
            )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    """Aggregate replicate rows into stability metrics."""
    n = len(rows)
    finite_ll_rows = [r for r in rows if r["finite_loglik"]]
    finite_grad_rows = [r for r in rows if r["finite_grad"]]

    ll_vals = np.asarray([r["loglik"] for r in finite_ll_rows], dtype=np.float64)
    grads = np.asarray(
        [
            [r["grad_log_sigma_v2"], r["grad_log_sigma_w2"]]
            for r in finite_grad_rows
        ],
        dtype=np.float64,
    )
    grad_norms = np.asarray([r["grad_norm"] for r in finite_grad_rows], dtype=np.float64)
    elapsed = np.asarray([r["elapsed_s"] for r in rows], dtype=np.float64)

    ll_mean = float(np.mean(ll_vals)) if ll_vals.size else float("nan")
    ll_std = float(np.std(ll_vals, ddof=1)) if ll_vals.size > 1 else float("nan")
    grad_norm_mean = float(np.mean(grad_norms)) if grad_norms.size else float("nan")
    grad_norm_std = float(np.std(grad_norms, ddof=1)) if grad_norms.size > 1 else float("nan")

    if grad_norms.size:
        log10_norms = np.log10(np.maximum(grad_norms, 1e-300))
        grad_log10_median = float(np.median(log10_norms))
        grad_log10_mad = _mad(log10_norms)
    else:
        grad_log10_median = float("nan")
        grad_log10_mad = float("nan")

    if grads.shape[0] > 0:
        grad_mean = np.mean(grads, axis=0)
        grad_std = np.std(grads, axis=0, ddof=1) if grads.shape[0] > 1 else np.full(2, np.nan)
        comp_snr = np.abs(grad_mean) / np.maximum(grad_std, 1e-12)
        vec_noise = math.sqrt(float(np.sum(grad_std**2))) if np.all(np.isfinite(grad_std)) else float("nan")
        vec_snr = float(np.linalg.norm(grad_mean) / vec_noise) if math.isfinite(vec_noise) and vec_noise > 0 else float("nan")
        cos_mean, cos_std = _cosine_to_mean(grads)
    else:
        grad_mean = np.full(2, np.nan)
        grad_std = np.full(2, np.nan)
        comp_snr = np.full(2, np.nan)
        vec_snr = cos_mean = cos_std = float("nan")

    errors = [r["error"] for r in rows if r["error"]]
    return {
        "n": n,
        "finite_loglik_rate": len(finite_ll_rows) / max(n, 1),
        "finite_grad_rate": len(finite_grad_rows) / max(n, 1),
        "loglik_mean": ll_mean,
        "loglik_std": ll_std,
        "loglik_cv": _safe_cv(ll_mean, ll_std),
        "grad_norm_mean": grad_norm_mean,
        "grad_norm_std": grad_norm_std,
        "grad_norm_cv": _safe_cv(grad_norm_mean, grad_norm_std),
        "grad_log10_median": grad_log10_median,
        "grad_log10_mad": grad_log10_mad,
        "grad_log_sigma_v2_mean": float(grad_mean[0]),
        "grad_log_sigma_v2_std": float(grad_std[0]),
        "grad_log_sigma_v2_snr": float(comp_snr[0]),
        "grad_log_sigma_w2_mean": float(grad_mean[1]),
        "grad_log_sigma_w2_std": float(grad_std[1]),
        "grad_log_sigma_w2_snr": float(comp_snr[1]),
        "grad_vector_snr": vec_snr,
        "grad_cosine_to_mean": cos_mean,
        "grad_cosine_to_mean_std": cos_std,
        "elapsed_mean_s": float(np.mean(elapsed)) if elapsed.size else float("nan"),
        "n_errors": len(errors),
        "first_error": errors[0][:180] if errors else "",
    }


def make_methods(
    y_obs: tf.Tensor,
    num_particles: int,
    n_lambda: int,
    sinkhorn_epsilon: float,
    sinkhorn_iters: int,
    resample_threshold: float,
) -> dict[str, MethodConfig]:
    """Create available likelihood methods."""

    def make_kitagawa_pfpf():
        ll = KitagawaPFPFLEDHLogLikelihood(
            num_particles=num_particles,
            n_lambda=n_lambda,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iters=sinkhorn_iters,
            resample_threshold=resample_threshold,
            clip_weight_terms=False,
        )

        def loglik(theta: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return ll(tf.exp(theta[0]), tf.exp(theta[1]), y)

        return loglik

    def make_generic_pfpf():
        ll = DifferentiablePFPFLEDHLogLikelihood(
            num_particles=num_particles,
            n_lambda=n_lambda,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iters=sinkhorn_iters,
            resample_threshold=resample_threshold,
            clip_weight_terms=True,
            velocity_clip=None,
        )

        def loglik(theta: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            ssm = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(theta[0]),
                sigma_w_sq=tf.exp(theta[1]),
                initial_var=5.0,
            )
            return ll(ssm, y)

        return loglik

    def make_legacy_ledh():
        ll = DifferentiableLEDHLogLikelihood(
            num_particles=num_particles,
            n_lambda=n_lambda,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iters=sinkhorn_iters,
            resample_threshold=resample_threshold,
            grad_window=4,
            jit_compile=True,
        )

        def loglik(theta: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            ssm = PMCMCNonlinearSSM(
                sigma_v_sq=tf.exp(theta[0]),
                sigma_w_sq=tf.exp(theta[1]),
                initial_var=5.0,
            )
            return ll(ssm, y)

        return loglik

    _ = y_obs  # kept so callers can build methods after data generation.
    return {
        "kitagawa_pfpf": MethodConfig(
            name="kitagawa_pfpf",
            label="Kitagawa specialized PFPF-LEDH",
            make_loglik=make_kitagawa_pfpf,
        ),
        "generic_pfpf": MethodConfig(
            name="generic_pfpf",
            label="Generic DifferentiablePFPFLEDHLogLikelihood",
            make_loglik=make_generic_pfpf,
        ),
        "legacy_ledh": MethodConfig(
            name="legacy_ledh",
            label="Legacy differentiable LEDH",
            make_loglik=make_legacy_ledh,
        ),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_report(path: Path, summary_rows: list[dict], config_lines: list[str]) -> None:
    """Write human-readable gradient stability report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 100,
        "Differentiable LEDH Gradient Stability Diagnostics",
        "=" * 100,
        *config_lines,
        "",
        (
            f"{'method':<16} {'theta':<19} {'fin_ll':>7} {'fin_g':>7} "
            f"{'ll_cv':>9} {'g_cv':>9} {'log10MAD':>9} {'vecSNR':>9} "
            f"{'cosMean':>9} {'gmean':>11} {'elapsed':>9}"
        ),
        "-" * 100,
    ]
    for row in summary_rows:
        theta_label = f"({row['log_sigma_v2']:.3f},{row['log_sigma_w2']:.3f})"
        lines.append(
            f"{row['method']:<16} {theta_label:<19} "
            f"{row['finite_loglik_rate']:>7.2f} {row['finite_grad_rate']:>7.2f} "
            f"{row['loglik_cv']:>9.3g} {row['grad_norm_cv']:>9.3g} "
            f"{row['grad_log10_mad']:>9.3g} {row['grad_vector_snr']:>9.3g} "
            f"{row['grad_cosine_to_mean']:>9.3g} {row['grad_norm_mean']:>11.3g} "
            f"{row['elapsed_mean_s']:>9.2f}"
        )
        if row["n_errors"]:
            lines.append(f"  first error: {row['first_error']}")

    lines += [
        "-" * 100,
        "Interpretation:",
        "- finite_grad_rate < 1 indicates crashes, NaNs, or disconnected gradients.",
        "- high grad_norm_cv means gradient magnitude changes strongly across CRN seeds.",
        "- high MAD(log10 ||grad||) means order-of-magnitude gradient instability.",
        "- low vector SNR means CRN noise dominates the mean gradient direction.",
        "- low cosine-to-mean means gradient directions disagree across seeds.",
        "=" * 100,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_diagnostics(out_dir: Path, raw_rows: list[dict], summary_rows: list[dict]) -> None:
    """Create simple diagnostic plots."""
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted({r["method"] for r in summary_rows})
    labels = [
        f"{r['method']}\n({r['log_sigma_v2']:.2f},{r['log_sigma_w2']:.2f})"
        for r in summary_rows
    ]
    grad_cv = [r["grad_norm_cv"] for r in summary_rows]
    log10_mad = [r["grad_log10_mad"] for r in summary_rows]
    vec_snr = [r["grad_vector_snr"] for r in summary_rows]
    finite_grad = [r["finite_grad_rate"] for r in summary_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    axes[0].bar(range(len(labels)), grad_cv)
    axes[0].set_title("Gradient Norm CV")
    axes[0].set_ylabel("std(||g||) / |mean(||g||)|")
    axes[1].bar(range(len(labels)), log10_mad)
    axes[1].set_title("MAD of log10 Gradient Norm")
    axes[1].set_ylabel("MAD(log10 ||g||)")
    axes[2].bar(range(len(labels)), vec_snr)
    axes[2].set_title("Vector Gradient SNR")
    axes[2].set_ylabel("||E[g]|| / sqrt(trace Cov[g])")
    axes[3].bar(range(len(labels)), finite_grad)
    axes[3].set_title("Finite Gradient Rate")
    axes[3].set_ylim(0.0, 1.05)

    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "gradient_stability_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for method in methods:
        rows_m = [r for r in raw_rows if r["method"] == method and r["finite_grad"]]
        if not rows_m:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        theta_labels = sorted({r["theta_label"] for r in rows_m})
        data = [
            [r["grad_norm"] for r in rows_m if r["theta_label"] == label]
            for label in theta_labels
        ]
        axes[0].boxplot(data, labels=theta_labels, showfliers=True)
        axes[0].set_title(f"{method}: gradient norm by theta")
        axes[0].set_ylabel("||gradient||")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].scatter(
            [r["grad_log_sigma_v2"] for r in rows_m],
            [r["grad_log_sigma_w2"] for r in rows_m],
            alpha=0.75,
        )
        axes[1].axhline(0.0, color="k", lw=0.8)
        axes[1].axvline(0.0, color="k", lw=0.8)
        axes[1].set_title(f"{method}: gradient components")
        axes[1].set_xlabel("d ll / d log sigma_v^2")
        axes[1].set_ylabel("d ll / d log sigma_w^2")

        axes[2].hist([r["loglik"] for r in rows_m if math.isfinite(r["loglik"])], bins=20)
        axes[2].set_title(f"{method}: log-likelihood spread")
        axes[2].set_xlabel("log likelihood")

        plt.tight_layout()
        plt.savefig(out_dir / f"{method}_gradient_replicates.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def parse_theta_grid(value: str) -> list[tf.Tensor]:
    """Parse semicolon-separated theta pairs, e.g. '10,1;8,1.5' in variance scale."""
    pairs = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        sv2_s, sw2_s = chunk.split(",")
        sv2 = float(sv2_s)
        sw2 = float(sw2_s)
        pairs.append(tf.math.log(tf.constant([sv2, sw2], dtype=tf.float32)))
    if not pairs:
        raise ValueError("theta grid must contain at least one 'sigma_v_sq,sigma_w_sq' pair")
    return pairs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose gradients of differentiable LEDH likelihoods.")
    p.add_argument("--methods", type=str, default="generic_pfpf,legacy_ledh")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--num_particles", type=int, default=500)
    p.add_argument("--n_lambda", type=int, default=15)
    p.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    p.add_argument("--sinkhorn_iters", type=int, default=10)
    p.add_argument("--resample_threshold", type=float, default=0.5)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument(
        "--theta_grid",
        type=str,
        default="10,1;8,1.5;12,0.8",
        help="Semicolon-separated variance-scale pairs: 'sigma_v_sq,sigma_w_sq;...'",
    )
    p.add_argument("--grad_clip_norm", type=float, default=0.0)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/gradient_stability"),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, y_obs = generate_data(T=args.T, seed=args.data_seed)
    methods_available = make_methods(
        y_obs=y_obs,
        num_particles=args.num_particles,
        n_lambda=args.n_lambda,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        resample_threshold=args.resample_threshold,
    )
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in method_names if m not in methods_available]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}. Available: {sorted(methods_available)}")

    theta_grid = parse_theta_grid(args.theta_grid)
    seeds = [args.base_seed + i * 7919 for i in range(args.n_seeds)]
    grad_clip = args.grad_clip_norm if args.grad_clip_norm > 0 else None

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    print("=" * 80)
    print("Differentiable LEDH gradient stability diagnostics")
    print(
        f"T={args.T} N={args.num_particles} n_lambda={args.n_lambda} "
        f"sinkhorn_iters={args.sinkhorn_iters} n_seeds={args.n_seeds}"
    )
    print(f"methods={method_names}")
    print("=" * 80)

    for method_name in method_names:
        method = methods_available[method_name]
        loglik_fn = method.make_loglik()
        for theta in theta_grid:
            sv2, sw2 = np.exp(theta.numpy())
            theta_label = f"sv2={sv2:.3g},sw2={sw2:.3g}"
            print(f"\n[{method.name}] theta {theta_label}")
            rows = evaluate_gradient_replicates(loglik_fn, theta, y_obs, seeds, grad_clip)
            for row in rows:
                row.update(
                    {
                        "method": method.name,
                        "method_label": method.label,
                        "theta_label": theta_label,
                        "log_sigma_v2": float(theta.numpy()[0]),
                        "log_sigma_w2": float(theta.numpy()[1]),
                        "sigma_v2": float(sv2),
                        "sigma_w2": float(sw2),
                    }
                )
            raw_rows.extend(rows)

            summary = summarize_rows(rows)
            summary.update(
                {
                    "method": method.name,
                    "method_label": method.label,
                    "theta_label": theta_label,
                    "log_sigma_v2": float(theta.numpy()[0]),
                    "log_sigma_w2": float(theta.numpy()[1]),
                    "sigma_v2": float(sv2),
                    "sigma_w2": float(sw2),
                }
            )
            summary_rows.append(summary)
            print(
                f"  finite_grad={summary['finite_grad_rate']:.2f} "
                f"grad_cv={summary['grad_norm_cv']:.3g} "
                f"log10_mad={summary['grad_log10_mad']:.3g} "
                f"vec_snr={summary['grad_vector_snr']:.3g} "
                f"elapsed={summary['elapsed_mean_s']:.2f}s"
            )

    raw_fields = [
        "method",
        "method_label",
        "theta_label",
        "log_sigma_v2",
        "log_sigma_w2",
        "sigma_v2",
        "sigma_w2",
        "seed",
        "loglik",
        "grad_log_sigma_v2",
        "grad_log_sigma_w2",
        "grad_norm",
        "finite_loglik",
        "finite_grad",
        "elapsed_s",
        "error",
    ]
    summary_fields = [
        "method",
        "method_label",
        "theta_label",
        "log_sigma_v2",
        "log_sigma_w2",
        "sigma_v2",
        "sigma_w2",
        "n",
        "finite_loglik_rate",
        "finite_grad_rate",
        "loglik_mean",
        "loglik_std",
        "loglik_cv",
        "grad_norm_mean",
        "grad_norm_std",
        "grad_norm_cv",
        "grad_log10_median",
        "grad_log10_mad",
        "grad_log_sigma_v2_mean",
        "grad_log_sigma_v2_std",
        "grad_log_sigma_v2_snr",
        "grad_log_sigma_w2_mean",
        "grad_log_sigma_w2_std",
        "grad_log_sigma_w2_snr",
        "grad_vector_snr",
        "grad_cosine_to_mean",
        "grad_cosine_to_mean_std",
        "elapsed_mean_s",
        "n_errors",
        "first_error",
    ]
    write_csv(args.out_dir / "gradient_replicates.csv", raw_rows, raw_fields)
    write_csv(args.out_dir / "gradient_summary.csv", summary_rows, summary_fields)
    config_lines = [
        f"T={args.T}",
        f"num_particles={args.num_particles}",
        f"n_lambda={args.n_lambda}",
        f"sinkhorn_epsilon={args.sinkhorn_epsilon}",
        f"sinkhorn_iters={args.sinkhorn_iters}",
        f"resample_threshold={args.resample_threshold}",
        f"n_seeds={args.n_seeds}",
        f"theta_grid={args.theta_grid}",
        f"grad_clip_norm={args.grad_clip_norm}",
    ]
    write_summary_report(args.out_dir / "gradient_stability_report.txt", summary_rows, config_lines)
    plot_diagnostics(args.out_dir, raw_rows, summary_rows)

    print("\nSaved:")
    print(f"  {args.out_dir / 'gradient_replicates.csv'}")
    print(f"  {args.out_dir / 'gradient_summary.csv'}")
    print(f"  {args.out_dir / 'gradient_stability_report.txt'}")
    if plt is not None:
        print(f"  {args.out_dir / 'gradient_stability_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
