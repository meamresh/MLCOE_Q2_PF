"""
Gradient-source ablation for differentiable PFPF-LEDH-OT.

This experiment answers a specific question:

    If HMC gradients are unstable, is the instability caused by Sinkhorn/OT
    resampling, or by the rest of the PFPF-LEDH likelihood?

It keeps the same 1-D Kitagawa model and selectively blocks or removes gradient
paths through major components:

    full                  all gradients enabled
    stop_ot_grad          OT resampling forward pass kept, backward pass stopped
    no_ot                 no OT resampling
    stop_logdet_grad      log |det J| value kept, gradient stopped
    stop_transition_grad  transition-ratio value kept, gradient stopped
    stop_flow_grad        flowed particles kept, gradient through flow stopped
    measurement_only      only log p(y_t | x_t) used in the weight increment

If ``stop_ot_grad`` or ``no_ot`` greatly improves gradient metrics, neural OT may
help gradient stability.  If not, neural OT mainly addresses speed, and the
instability is in the LEDH flow / determinant / transition-ratio correction.

Usage:
    python -m src.experiments.exp_part3_bonus1b_pfpf_gradient_ablation
    python -m src.experiments.exp_part3_bonus1b_pfpf_gradient_ablation --variants full,stop_ot_grad,no_ot
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

from src.filters.dpf.resampling import det_resample

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

_EPS = 1e-6
_LOG_FLOOR = -1e6


@dataclass(frozen=True)
class Variant:
    """Gradient ablation switchboard."""

    name: str
    stop_ot_grad: bool = False
    disable_ot: bool = False
    stop_logdet_grad: bool = False
    stop_transition_grad: bool = False
    stop_flow_grad: bool = False
    measurement_only: bool = False


VARIANTS: dict[str, Variant] = {
    "full": Variant("full"),
    "stop_ot_grad": Variant("stop_ot_grad", stop_ot_grad=True),
    "no_ot": Variant("no_ot", disable_ot=True),
    "stop_logdet_grad": Variant("stop_logdet_grad", stop_logdet_grad=True),
    "stop_transition_grad": Variant("stop_transition_grad", stop_transition_grad=True),
    "stop_flow_grad": Variant("stop_flow_grad", stop_flow_grad=True),
    "measurement_only": Variant("measurement_only", measurement_only=True),
}


def generate_data(
    T: int = 30,
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


def parse_theta_grid(value: str) -> list[tf.Tensor]:
    """Parse semicolon-separated variance-scale pairs."""
    out = []
    for chunk in value.split(";"):
        if not chunk.strip():
            continue
        sv2_s, sw2_s = chunk.split(",")
        out.append(tf.math.log(tf.constant([float(sv2_s), float(sw2_s)], tf.float32)))
    if not out:
        raise ValueError("theta grid must contain at least one sigma_v_sq,sigma_w_sq pair")
    return out


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _safe_cv(mean: float, std: float) -> float:
    return float(std / abs(mean)) if math.isfinite(mean) and abs(mean) > 1e-12 else float("nan")


def _cosine_to_mean(grads: np.ndarray) -> tuple[float, float]:
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


class AblatedKitagawaPFPFLEDH:
    """1-D Kitagawa PFPF-LEDH likelihood with component-wise gradient switches."""

    def __init__(
        self,
        variant: Variant,
        num_particles: int = 50,
        n_lambda: int = 15,
        sinkhorn_epsilon: float = 1.0,
        sinkhorn_iters: int = 10,
        resample_threshold: float = 0.5,
        initial_var: float = 5.0,
        clip_weight_terms: bool = True,
    ) -> None:
        self.variant = variant
        self.num_particles = int(num_particles)
        self.n_lambda = int(n_lambda)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.resample_threshold = float(resample_threshold)
        self.initial_var = float(initial_var)
        self.clip_weight_terms = bool(clip_weight_terms)

        q = 1.2
        eps1 = (1.0 - q) / (1.0 - q ** self.n_lambda)
        self.epsilons = [eps1 * q**j for j in range(self.n_lambda)]

    def __call__(self, theta: tf.Tensor, y_obs: tf.Tensor) -> tf.Tensor:
        theta = tf.cast(theta, tf.float32)
        y_obs = tf.cast(tf.reshape(y_obs, [-1]), tf.float32)
        Q_val = tf.maximum(tf.exp(theta[0]), _EPS)
        R_val = tf.maximum(tf.exp(theta[1]), _EPS)

        N = self.num_particles
        N_f = tf.cast(N, tf.float32)
        particles = tf.random.normal([N], dtype=tf.float32) * tf.sqrt(
            tf.constant(self.initial_var, tf.float32)
        )
        particle_vars = tf.fill([N], tf.constant(self.initial_var, tf.float32))
        log_w = tf.fill([N], -tf.math.log(N_f))
        log_ml = tf.constant(0.0, tf.float32)

        T = int(y_obs.shape[0])
        for t_int in range(1, T + 1):
            t_f = tf.constant(float(t_int), tf.float32)
            particles_prev = particles
            particles, particle_vars = self._predict(particles, particle_vars, Q_val, t_f)
            particles_before_flow = particles
            z_t = y_obs[t_int - 1]

            particles, log_det_jac = self._flow(particles, particle_vars, z_t, R_val)
            if self.variant.stop_flow_grad:
                particles = tf.stop_gradient(particles)

            log_lik, transition_ratio, log_det_term = self._weight_terms(
                particles=particles,
                particles_before_flow=particles_before_flow,
                particles_prev=particles_prev,
                log_det_jac=log_det_jac,
                z_t=z_t,
                Q_val=Q_val,
                R_val=R_val,
                t_f=t_f,
            )

            if self.variant.measurement_only:
                log_incr = log_lik
            else:
                if self.variant.stop_transition_grad:
                    transition_ratio = tf.stop_gradient(transition_ratio)
                if self.variant.stop_logdet_grad:
                    log_det_term = tf.stop_gradient(log_det_term)
                log_incr = log_lik + transition_ratio + log_det_term

            log_incr = tf.where(tf.math.is_finite(log_incr), log_incr, _LOG_FLOOR)
            log_w_unnorm = log_w + log_incr
            prev_norm = tf.reduce_logsumexp(log_w)
            curr_norm = tf.reduce_logsumexp(log_w_unnorm)
            log_ml = log_ml + tf.where(
                tf.math.is_finite(curr_norm - prev_norm),
                curr_norm - prev_norm,
                tf.constant(-10.0, tf.float32),
            )
            log_w = log_w_unnorm - curr_norm

            particles, log_w, particle_vars = self._maybe_resample(particles, log_w, particle_vars)

        return tf.where(tf.math.is_finite(log_ml), log_ml, tf.constant(_LOG_FLOOR, tf.float32))

    def _predict(
        self,
        particles: tf.Tensor,
        particle_vars: tf.Tensor,
        Q_val: tf.Tensor,
        t_f: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        x_det = (
            0.5 * particles
            + 25.0 * particles / (1.0 + particles**2)
            + 8.0 * tf.cos(1.2 * t_f)
        )
        particles_pred = x_det + tf.random.normal([self.num_particles], dtype=tf.float32) * tf.sqrt(Q_val)
        F = 0.5 + 25.0 * (1.0 - particles_pred**2) / tf.square(1.0 + particles_pred**2)
        vars_pred = tf.maximum(F**2 * particle_vars + Q_val, _EPS)
        return particles_pred, vars_pred

    def _flow(
        self,
        particles: tf.Tensor,
        particle_vars: tf.Tensor,
        z_t: tf.Tensor,
        R_val: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        eta = tf.identity(particles)
        log_det_jac = tf.zeros_like(particles)
        R_inv = 1.0 / R_val
        lambda_cum = 0.0

        for eps_j in self.epsilons:
            lambda_k = lambda_cum + eps_j / 2.0
            lambda_cum += eps_j

            H = eta / 10.0
            h_eta = eta**2 / 20.0
            e_lambda = h_eta - H * eta

            S_lambda = tf.maximum(lambda_k * H**2 * particle_vars + R_val, _EPS)
            A = -0.5 * particle_vars * H**2 / S_lambda
            I_lam_A = 1.0 + lambda_k * A
            I_2lam_A = 1.0 + 2.0 * lambda_k * A
            b = I_2lam_A * (I_lam_A * particle_vars * H * R_inv * (z_t - e_lambda) + A * eta)

            particles = particles + eps_j * (A * particles + b)
            eta = eta + eps_j * (A * eta + b)

            J_step = 1.0 + eps_j * A
            log_det_jac = log_det_jac + tf.math.log(tf.maximum(tf.abs(J_step), _EPS))

        return particles, log_det_jac

    def _weight_terms(
        self,
        *,
        particles: tf.Tensor,
        particles_before_flow: tf.Tensor,
        particles_prev: tf.Tensor,
        log_det_jac: tf.Tensor,
        z_t: tf.Tensor,
        Q_val: tf.Tensor,
        R_val: tf.Tensor,
        t_f: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        log_norm_R = -0.5 * (tf.math.log(R_val) + tf.math.log(2.0 * math.pi))
        residual = z_t - particles**2 / 20.0
        log_lik = log_norm_R - 0.5 * residual**2 / R_val

        transition_mean = (
            0.5 * particles_prev
            + 25.0 * particles_prev / (1.0 + particles_prev**2)
            + 8.0 * tf.cos(1.2 * t_f)
        )
        log_norm_Q = -0.5 * (tf.math.log(Q_val) + tf.math.log(2.0 * math.pi))
        log_p_plus = log_norm_Q - 0.5 * (particles - transition_mean) ** 2 / Q_val
        log_p_minus = log_norm_Q - 0.5 * (particles_before_flow - transition_mean) ** 2 / Q_val
        transition_ratio = log_p_plus - log_p_minus
        log_det_term = log_det_jac

        if self.clip_weight_terms:
            log_lik = tf.clip_by_value(log_lik, -100.0, 100.0)
            transition_ratio = tf.clip_by_value(transition_ratio, -20.0, 20.0)
            log_det_term = tf.clip_by_value(log_det_term, -20.0, 20.0)
        return log_lik, transition_ratio, log_det_term

    def _maybe_resample(
        self,
        particles: tf.Tensor,
        log_w: tf.Tensor,
        particle_vars: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if self.variant.disable_ot:
            return particles, log_w, particle_vars

        N = self.num_particles
        N_f = tf.cast(N, tf.float32)
        weights = tf.nn.softmax(log_w, axis=0)
        ess = 1.0 / (tf.reduce_sum(weights**2) + 1e-15)
        do_resample = ess < N_f * self.resample_threshold

        def _resample():
            particles_2d = particles[:, tf.newaxis]
            p_mean = tf.reduce_mean(particles_2d, axis=0, keepdims=True)
            p_std = tf.math.reduce_std(particles_2d, axis=0, keepdims=True) + _EPS
            p_norm = (particles_2d - p_mean) / p_std
            p_re_norm, w_uni = det_resample(
                p_norm,
                log_w,
                epsilon=self.sinkhorn_epsilon,
                n_iters=self.sinkhorn_iters,
            )
            p_re = p_re_norm * p_std + p_mean
            var_re = tf.fill([N], tf.reduce_sum(weights * particle_vars))
            w_uni = tf.cast(tf.math.real(tf.cast(w_uni, tf.complex64)), tf.float32)
            w_uni = tf.maximum(w_uni, 1e-20)
            lw_new = tf.math.log(w_uni) - tf.reduce_logsumexp(tf.math.log(w_uni))
            p_out, lw_out, var_out = p_re[:, 0], lw_new, var_re
            if self.variant.stop_ot_grad:
                p_out = tf.stop_gradient(p_out)
                lw_out = tf.stop_gradient(lw_out)
                var_out = tf.stop_gradient(var_out)
            return p_out, lw_out, var_out

        return tf.cond(do_resample, _resample, lambda: (particles, log_w, particle_vars))


def evaluate_variant(
    variant: Variant,
    theta: tf.Tensor,
    y_obs: tf.Tensor,
    seeds: list[int],
    *,
    num_particles: int,
    n_lambda: int,
    sinkhorn_epsilon: float,
    sinkhorn_iters: int,
    resample_threshold: float,
    clip_weight_terms: bool,
) -> list[dict]:
    """Evaluate one variant across CRN seeds."""
    likelihood = AblatedKitagawaPFPFLEDH(
        variant=variant,
        num_particles=num_particles,
        n_lambda=n_lambda,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=sinkhorn_iters,
        resample_threshold=resample_threshold,
        clip_weight_terms=clip_weight_terms,
    )
    rows = []
    for seed in seeds:
        tf.random.set_seed(seed)
        theta_t = tf.identity(tf.cast(theta, tf.float32))
        t0 = time.time()
        try:
            with tf.GradientTape() as tape:
                tape.watch(theta_t)
                ll = likelihood(theta_t, y_obs)
            grad = tape.gradient(ll, theta_t)
            if grad is None:
                grad = tf.fill([2], tf.constant(float("nan"), tf.float32))
            elapsed = time.time() - t0
            ll_val = float(ll.numpy())
            g = grad.numpy().astype(np.float64)
            finite_ll = math.isfinite(ll_val)
            finite_grad = bool(np.all(np.isfinite(g)))
            rows.append(
                {
                    "variant": variant.name,
                    "seed": seed,
                    "loglik": ll_val,
                    "grad_log_sigma_v2": float(g[0]),
                    "grad_log_sigma_w2": float(g[1]),
                    "grad_norm": float(np.linalg.norm(g)) if finite_grad else float("nan"),
                    "finite_loglik": int(finite_ll),
                    "finite_grad": int(finite_grad),
                    "elapsed_s": elapsed,
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic must continue.
            rows.append(
                {
                    "variant": variant.name,
                    "seed": seed,
                    "loglik": float("nan"),
                    "grad_log_sigma_v2": float("nan"),
                    "grad_log_sigma_w2": float("nan"),
                    "grad_norm": float("nan"),
                    "finite_loglik": 0,
                    "finite_grad": 0,
                    "elapsed_s": time.time() - t0,
                    "error": repr(exc),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    """Summarize finite replicate rows."""
    n = len(rows)
    finite_ll = [r for r in rows if r["finite_loglik"]]
    finite_grad = [r for r in rows if r["finite_grad"]]
    ll = np.asarray([r["loglik"] for r in finite_ll], dtype=np.float64)
    grads = np.asarray(
        [[r["grad_log_sigma_v2"], r["grad_log_sigma_w2"]] for r in finite_grad],
        dtype=np.float64,
    )
    grad_norms = np.asarray([r["grad_norm"] for r in finite_grad], dtype=np.float64)
    elapsed = np.asarray([r["elapsed_s"] for r in rows], dtype=np.float64)

    ll_mean = float(np.mean(ll)) if ll.size else float("nan")
    ll_std = float(np.std(ll, ddof=1)) if ll.size > 1 else float("nan")
    g_mean = float(np.mean(grad_norms)) if grad_norms.size else float("nan")
    g_std = float(np.std(grad_norms, ddof=1)) if grad_norms.size > 1 else float("nan")
    if grad_norms.size:
        log10_norms = np.log10(np.maximum(grad_norms, 1e-300))
        log10_med = float(np.median(log10_norms))
        log10_mad = _mad(log10_norms)
    else:
        log10_med = log10_mad = float("nan")

    if grads.shape[0] > 0:
        mean_g = np.mean(grads, axis=0)
        std_g = np.std(grads, axis=0, ddof=1) if grads.shape[0] > 1 else np.full(2, np.nan)
        comp_snr = np.abs(mean_g) / np.maximum(std_g, 1e-12)
        noise = math.sqrt(float(np.sum(std_g**2))) if np.all(np.isfinite(std_g)) else float("nan")
        vec_snr = float(np.linalg.norm(mean_g) / noise) if math.isfinite(noise) and noise > 0 else float("nan")
        cos_mean, cos_std = _cosine_to_mean(grads)
    else:
        mean_g = std_g = comp_snr = np.full(2, np.nan)
        vec_snr = cos_mean = cos_std = float("nan")

    errors = [r["error"] for r in rows if r["error"]]
    return {
        "n": n,
        "finite_loglik_rate": len(finite_ll) / max(n, 1),
        "finite_grad_rate": len(finite_grad) / max(n, 1),
        "loglik_mean": ll_mean,
        "loglik_std": ll_std,
        "loglik_cv": _safe_cv(ll_mean, ll_std),
        "grad_norm_mean": g_mean,
        "grad_norm_std": g_std,
        "grad_norm_cv": _safe_cv(g_mean, g_std),
        "grad_log10_median": log10_med,
        "grad_log10_mad": log10_mad,
        "grad_log_sigma_v2_mean": float(mean_g[0]),
        "grad_log_sigma_v2_std": float(std_g[0]),
        "grad_log_sigma_v2_snr": float(comp_snr[0]),
        "grad_log_sigma_w2_mean": float(mean_g[1]),
        "grad_log_sigma_w2_std": float(std_g[1]),
        "grad_log_sigma_w2_snr": float(comp_snr[1]),
        "grad_vector_snr": vec_snr,
        "grad_cosine_to_mean": cos_mean,
        "grad_cosine_to_mean_std": cos_std,
        "elapsed_mean_s": float(np.mean(elapsed)) if elapsed.size else float("nan"),
        "n_errors": len(errors),
        "first_error": errors[0][:180] if errors else "",
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: list[dict], config_lines: list[str]) -> None:
    lines = [
        "=" * 112,
        "PFPF-LEDH Gradient Source Ablation",
        "=" * 112,
        *config_lines,
        "",
        (
            f"{'variant':<22} {'theta':<19} {'fin_g':>7} {'ll_cv':>9} "
            f"{'g_cv':>9} {'log10MAD':>9} {'vecSNR':>9} {'cosMean':>9} "
            f"{'gmean':>11} {'elapsed':>9}"
        ),
        "-" * 112,
    ]
    for row in summary_rows:
        theta = f"({row['log_sigma_v2']:.3f},{row['log_sigma_w2']:.3f})"
        lines.append(
            f"{row['variant']:<22} {theta:<19} {row['finite_grad_rate']:>7.2f} "
            f"{row['loglik_cv']:>9.3g} {row['grad_norm_cv']:>9.3g} "
            f"{row['grad_log10_mad']:>9.3g} {row['grad_vector_snr']:>9.3g} "
            f"{row['grad_cosine_to_mean']:>9.3g} {row['grad_norm_mean']:>11.3g} "
            f"{row['elapsed_mean_s']:>9.2f}"
        )
        if row["n_errors"]:
            lines.append(f"  first error: {row['first_error']}")
    lines += [
        "-" * 112,
        "Readout:",
        "- If stop_ot_grad/no_ot improves metrics, OT backward/resampling is a major source.",
        "- If stop_logdet_grad improves metrics, determinant differentiation is the major source.",
        "- If stop_transition_grad improves metrics, the full PFPF transition correction is the major source.",
        "- If stop_flow_grad improves metrics, differentiating through the LEDH flow map is the major source.",
        "=" * 112,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(out_dir: Path, summary_rows: list[dict]) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{r['variant']}\n({r['log_sigma_v2']:.2f},{r['log_sigma_w2']:.2f})"
        for r in summary_rows
    ]
    metrics = [
        ("grad_norm_cv", "Gradient Norm CV"),
        ("grad_log10_mad", "MAD(log10 ||grad||)"),
        ("grad_vector_snr", "Vector Gradient SNR"),
        ("finite_grad_rate", "Finite Gradient Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        ax.bar(range(len(labels)), [r[key] for r in summary_rows])
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "pfpf_gradient_ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate gradient sources in PFPF-LEDH-OT.")
    p.add_argument(
        "--variants",
        type=str,
        default="full,stop_ot_grad,no_ot,stop_logdet_grad,stop_transition_grad,stop_flow_grad,measurement_only",
    )
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--num_particles", type=int, default=50)
    p.add_argument("--n_lambda", type=int, default=15)
    p.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    p.add_argument("--sinkhorn_iters", type=int, default=10)
    p.add_argument("--resample_threshold", type=float, default=0.5)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--clip_weight_terms", action="store_true")
    p.add_argument(
        "--theta_grid",
        type=str,
        default="10,1;8,1.5;12,0.8",
        help="Semicolon-separated variance-scale pairs.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/pfpf_gradient_ablation"),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, y_obs = generate_data(T=args.T, seed=args.data_seed)
    theta_grid = parse_theta_grid(args.theta_grid)
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variant_names if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}. Available: {sorted(VARIANTS)}")
    seeds = [args.base_seed + i * 7919 for i in range(args.n_seeds)]

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    print("=" * 90)
    print("PFPF-LEDH gradient source ablation")
    print(
        f"T={args.T} N={args.num_particles} n_lambda={args.n_lambda} "
        f"sinkhorn_iters={args.sinkhorn_iters} n_seeds={args.n_seeds}"
    )
    print(f"variants={variant_names}")
    print("=" * 90)

    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        for theta in theta_grid:
            sv2, sw2 = np.exp(theta.numpy())
            theta_label = f"sv2={sv2:.3g},sw2={sw2:.3g}"
            print(f"\n[{variant.name}] theta {theta_label}")
            rows = evaluate_variant(
                variant,
                theta,
                y_obs,
                seeds,
                num_particles=args.num_particles,
                n_lambda=args.n_lambda,
                sinkhorn_epsilon=args.sinkhorn_epsilon,
                sinkhorn_iters=args.sinkhorn_iters,
                resample_threshold=args.resample_threshold,
                clip_weight_terms=args.clip_weight_terms,
            )
            for row in rows:
                row.update(
                    {
                        "theta_label": theta_label,
                        "log_sigma_v2": float(theta.numpy()[0]),
                        "log_sigma_w2": float(theta.numpy()[1]),
                        "sigma_v2": float(sv2),
                        "sigma_w2": float(sw2),
                    }
                )
            raw_rows.extend(rows)
            summary = summarize(rows)
            summary.update(
                {
                    "variant": variant.name,
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
                f"g_cv={summary['grad_norm_cv']:.3g} "
                f"log10_mad={summary['grad_log10_mad']:.3g} "
                f"vec_snr={summary['grad_vector_snr']:.3g} "
                f"cos={summary['grad_cosine_to_mean']:.3g}"
            )

    raw_fields = [
        "variant",
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
        "variant",
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
    write_csv(args.out_dir / "pfpf_gradient_ablation_replicates.csv", raw_rows, raw_fields)
    write_csv(args.out_dir / "pfpf_gradient_ablation_summary.csv", summary_rows, summary_fields)
    config_lines = [
        f"T={args.T}",
        f"num_particles={args.num_particles}",
        f"n_lambda={args.n_lambda}",
        f"sinkhorn_epsilon={args.sinkhorn_epsilon}",
        f"sinkhorn_iters={args.sinkhorn_iters}",
        f"resample_threshold={args.resample_threshold}",
        f"n_seeds={args.n_seeds}",
        f"theta_grid={args.theta_grid}",
        f"clip_weight_terms={args.clip_weight_terms}",
    ]
    write_report(args.out_dir / "pfpf_gradient_ablation_report.txt", summary_rows, config_lines)
    plot_summary(args.out_dir, summary_rows)
    print("\nSaved:")
    print(f"  {args.out_dir / 'pfpf_gradient_ablation_replicates.csv'}")
    print(f"  {args.out_dir / 'pfpf_gradient_ablation_summary.csv'}")
    print(f"  {args.out_dir / 'pfpf_gradient_ablation_report.txt'}")
    if plt is not None:
        print(f"  {args.out_dir / 'pfpf_gradient_ablation_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
