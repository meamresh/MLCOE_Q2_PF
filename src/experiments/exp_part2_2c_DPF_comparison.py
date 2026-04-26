"""
Definitive comparison of filters on the Kitagawa nonlinear growth model.
Quantifies accuracy (log-likelihood bias), differentiability (gradient SNR),
and efficiency.
"""

import math
import os
import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Patch
from tqdm import tqdm

try:
    import seaborn as sns
except Exception:
    sns = None

from src.filters.dpf import (
    BootstrapModel,
    DifferentiableParticleFilter,
    ParticleTransformer,
    ParticleTransformerFilter,
    SoftResamplingParticleFilter,
    StandardParticleFilter,
    StopGradientParticleFilter,
)
from src.filters.dpf.training import (
    end_to_end_train_particle_transformer,
    train_particle_transformer,
)
from src.models.ssm_katigawa import PMCMCNonlinearSSM


PT_TRAIN_SEED = 42


class KitagawaDPFWrapper:
    """Adapter to use PMCMCNonlinearSSM with BootstrapModel(sample_initial, step)."""

    def __init__(self, ssm: PMCMCNonlinearSSM, phi: Union[float, tf.Tensor, tf.Variable] = 1.0):
        self.ssm = ssm
        if isinstance(phi, tf.Variable):
            self.phi = phi
        else:
            self.phi = tf.Variable(phi, trainable=True, dtype=tf.float32)

    def sample_initial(self, N: int, y1: tf.Tensor):
        y1 = tf.cast(y1, tf.float32)
        x1 = tf.random.normal((N, 1), mean=0.0, stddev=tf.sqrt(self.ssm.initial_var), dtype=y1.dtype)
        log_w1 = self.log_obs_density(x1, y1)
        return x1, log_w1

    def step(self, t: int, x_prev: tf.Tensor, y_t: tf.Tensor):
        x_prev = tf.cast(x_prev, tf.float32)
        y_t = tf.cast(y_t, tf.float32)
        t_float = tf.cast(t + 1, x_prev.dtype)

        term1 = 0.5 * x_prev
        term2 = 25.0 * self.phi * (x_prev / (1.0 + x_prev**2))
        term3 = 8.0 * tf.cos(1.2 * t_float)
        mean = term1 + term2 + term3

        std_v = tf.sqrt(tf.cast(self.ssm.Q[0, 0], x_prev.dtype))
        v_t = tf.random.normal(tf.shape(x_prev), mean=0.0, stddev=std_v, dtype=x_prev.dtype)
        x_t = mean + v_t

        log_w_t = self.log_obs_density(x_t, y_t)
        return x_t, log_w_t

    def log_obs_density(self, x: tf.Tensor, y: tf.Tensor):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        mean_y = (x**2) / 20.0
        var_w = tf.cast(self.ssm.R[0, 0], x.dtype)
        diff = tf.expand_dims(y, axis=0) - mean_y
        loglik = -0.5 * (diff**2) / var_w - 0.5 * tf.math.log(2.0 * math.pi * var_w)
        return tf.reduce_sum(loglik, axis=-1)


def _build_bootstrap_model(phi: Union[float, tf.Tensor, tf.Variable] = 1.0) -> BootstrapModel:
    ssm = PMCMCNonlinearSSM(sigma_v_sq=10.0, sigma_w_sq=1.0, initial_var=10.0)
    wrapped = KitagawaDPFWrapper(ssm=ssm, phi=phi)
    return BootstrapModel(sample_initial=wrapped.sample_initial, transition=wrapped.step)


def _make_filter_and_phi(filter_cls, phi_init: float, N: int, extra_kwargs: dict):
    """Create filter and return the exact phi variable used inside step()."""
    phi_var = tf.Variable(phi_init, trainable=True, dtype=tf.float32)
    model = _build_bootstrap_model(phi=phi_var)
    pf = filter_cls(model, num_particles=N, **extra_kwargs)
    return pf, phi_var


def _load_or_train_pt(y_data: tf.Tensor, n_particles: int = 50):
    """Load cached PT weights if available; otherwise train once and cache."""
    repo_root = Path(__file__).resolve().parents[2]
    pt_weights_path = repo_root / "reports" / "5_Differential_PF_OT_Resampling" / "pt_filter.weights.h5"
    pt_weights_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    if os.path.exists(pt_weights_path):
        print(f"Loading cached PT weights from {pt_weights_path}...")
        tf.keras.utils.set_random_seed(PT_TRAIN_SEED)
        pt_model = ParticleTransformer(state_dim=1)
        dummy_x = tf.zeros((n_particles, 1), dtype=tf.float32)
        dummy_w = tf.fill((n_particles,), tf.constant(1.0 / n_particles, dtype=tf.float32))
        _ = pt_model(dummy_x, dummy_w)
        try:
            pt_model.load_weights(pt_weights_path)
            print(f"  Loaded PT weights. [{time.perf_counter() - t0:.1f}s]")
            return pt_model
        except Exception as exc:
            print(f"  Cached PT weights incompatible ({exc}); retraining...")

    print("Training Particle Transformer (cache miss)...")
    tf.keras.utils.set_random_seed(PT_TRAIN_SEED)
    pt_model = train_particle_transformer(y_data, n_particles=n_particles, n_epochs=20)
    pt_model = end_to_end_train_particle_transformer(
        y_data,
        pt_model,
        n_particles=n_particles,
        n_epochs=10,
        n_seeds_per_epoch=15,
        lr=1e-4,
        train_resampler=True,
    )
    pt_model.save_weights(pt_weights_path)
    print(f"  Saved PT weights to {pt_weights_path}. [{time.perf_counter() - t0:.1f}s]")
    return pt_model


def simulate_kitagawa_data(T: int = 50, seed: int = 42):
    """True generative simulation (independent of filter logic)."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    x_curr = tf.random.normal((1, 1), stddev=math.sqrt(10.0))
    y_list = []

    for t in range(T):
        t_float = float(t + 1)
        term1 = 0.5 * x_curr
        term2 = 25.0 * (x_curr / (1.0 + x_curr**2))
        term3 = 8.0 * math.cos(1.2 * t_float)
        mean = term1 + term2 + term3
        x_curr = mean + tf.random.normal((1, 1), stddev=math.sqrt(10.0))
        y_t = (x_curr**2 / 20.0) + tf.random.normal((1, 1))
        y_list.append(tf.squeeze(y_t))

    return tf.stack(y_list)


def compute_ground_truth(y_data, N_large=100000, n_seeds=5):
    """Approximate ground-truth log-likelihood with a very large PF."""
    print(f"Computing Ground Truth log-likelihood (N={N_large}, seeds={n_seeds})...")
    logliks = []
    for s in range(n_seeds):
        tf.random.set_seed(1000 + s)
        model = _build_bootstrap_model(phi=1.0)
        pf = StandardParticleFilter(model, num_particles=N_large, resample_threshold=2.0)
        loglik, _ = pf(y_data)
        logliks.append(float(loglik.numpy()))
    return sum(logliks) / len(logliks)


def benchmark_filter(filter_cls, y_data, N, n_runs=15, phi=1.0, **kwargs):
    """Standardized benchmark for a filter: pure inference timing."""
    accs = []
    latencies = []

    kwargs["resample_threshold"] = 2.0

    model = _build_bootstrap_model(phi=phi)
    pf = filter_cls(model, num_particles=N, **kwargs)
    _ = pf(y_data)  # warmup / graph tracing

    for run_id in range(n_runs):
        tf.random.set_seed(555 + run_id)
        start = time.time()
        loglik, _ = pf(y_data)
        elapsed = time.time() - start
        accs.append(float(loglik.numpy()))
        latencies.append(elapsed * 1000.0)

    mean_acc = sum(accs) / len(accs)
    mean_lat = sum(latencies) / len(latencies)
    acc_diffs = [(a - mean_acc) ** 2 for a in accs]
    std_acc = (sum(acc_diffs) / len(accs)) ** 0.5
    return mean_acc, std_acc, mean_lat


def measure_accuracy_and_efficiency(y_data, truth, n_particles_list=None, n_runs=20, pt_model=None):
    if n_particles_list is None:
        n_particles_list = [25, 50, 100, 200, 500]

    results = {
        "DPF (OT)": {"bias": [], "time": []},
        "Soft DPF": {"bias": [], "time": []},
        "DPF (SG)": {"bias": [], "time": []},
        "DPF (PT)": {"bias": [], "time": []},
    }

    if pt_model is None:
        N_train = 50
        pt_model = train_particle_transformer(y_data, n_particles=N_train, n_epochs=20)
        pt_model = end_to_end_train_particle_transformer(
            y_data,
            pt_model,
            n_particles=N_train,
            n_epochs=10,
            n_seeds_per_epoch=15,
            lr=1e-4,
            train_resampler=True,
        )

    for N in n_particles_list:
        print(f"Benchmarking N={N}...")

        m, s, t = benchmark_filter(
            DifferentiableParticleFilter,
            y_data,
            N,
            n_runs=n_runs,
            epsilon=0.25,
            sinkhorn_iters=20,
        )
        results["DPF (OT)"]["bias"].append((m - truth, s))
        results["DPF (OT)"]["time"].append(t)

        m, s, t = benchmark_filter(SoftResamplingParticleFilter, y_data, N, n_runs=n_runs, alpha=0.5)
        results["Soft DPF"]["bias"].append((m - truth, s))
        results["Soft DPF"]["time"].append(t)

        m, s, t = benchmark_filter(StopGradientParticleFilter, y_data, N, n_runs=n_runs)
        results["DPF (SG)"]["bias"].append((m - truth, s))
        results["DPF (SG)"]["time"].append(t)

        m, s, t = benchmark_filter(
            ParticleTransformerFilter,
            y_data,
            N,
            n_runs=n_runs,
            pt_model=pt_model,
        )
        results["DPF (PT)"]["bias"].append((m - truth, s))
        results["DPF (PT)"]["time"].append(t)

    return results, pt_model


def measure_gradient_snr(
    y_data: tf.Tensor,
    pt_model,
    N: int = 50,
    phi_true: float = 1.0,
    n_seeds: int = 30,
):
    """Compute gradient SNR = |E[grad]| / std(grad) for each filter type."""
    print(f"\nMeasuring Gradient SNR (N={N}, n_seeds={n_seeds})...")

    CLIP_NORM = 100.0
    GRAD_EXPLOSION_THRESHOLD = 1e6

    methods = [
        ("DPF (OT)", DifferentiableParticleFilter, {"epsilon": 0.25, "sinkhorn_iters": 20}),
        ("Soft DPF", SoftResamplingParticleFilter, {"alpha": 0.5}),
        ("DPF (SG)", StopGradientParticleFilter, {}),
        ("DPF (PT)", ParticleTransformerFilter, {"pt_model": pt_model}),
    ]

    snr_results = {}

    for name, filter_cls, extra_kwargs in methods:
        grads = []
        n_exploded = 0
        extra_kwargs = dict(extra_kwargs, resample_threshold=2.0)

        for seed_offset in tqdm(range(n_seeds), desc=name, leave=False):
            tf.random.set_seed(9000 + seed_offset)

            try:
                pf, phi_var = _make_filter_and_phi(filter_cls, phi_true, N, extra_kwargs)
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(phi_var)
                    loglik, _ = pf(y_data)
                grad = tape.gradient(loglik, phi_var)

                if grad is None:
                    grads.append(0.0)
                    continue

                grad_clipped = tf.clip_by_norm(grad, CLIP_NORM)
                g = float(grad_clipped.numpy())
                if abs(g) > GRAD_EXPLOSION_THRESHOLD or not math.isfinite(g):
                    n_exploded += 1
                    grads.append(float("nan"))
                else:
                    grads.append(g)

            except Exception as e:
                print(f"  Warning: seed {seed_offset} failed for {name}: {e}")
                grads.append(float("nan"))

        valid = [g for g in grads if not math.isnan(g)]

        if n_exploded > 0:
            print(
                f"  Warning: {name} had {n_exploded}/{n_seeds} exploded gradients "
                f"(|grad| > {GRAD_EXPLOSION_THRESHOLD:.0e}); excluded from stats."
            )

        if len(valid) == 0:
            mean_g = float("nan")
            std_g = float("nan")
            norm_g = float("nan")
            snr = float("nan")
        else:
            mean_g = sum(valid) / len(valid)
            std_g = (sum((g - mean_g) ** 2 for g in valid) / len(valid)) ** 0.5
            norm_g = sum(abs(g) for g in valid) / len(valid)
            snr = abs(mean_g) / (std_g + 1e-10)

        snr_results[name] = {
            "grads": valid,
            "mean_grad": mean_g,
            "std_grad": std_g,
            "grad_norm": norm_g,
            "snr": snr,
            "n_exploded": n_exploded,
        }

        print(
            f"  {name:<15}  mean={mean_g:+.4f}  std={std_g:.4f}  "
            f"|grad|={norm_g:.4f}  SNR={snr:.3f}  "
            f"(exploded: {n_exploded}/{n_seeds})"
        )

    return snr_results


def measure_differentiability(y_data, N=50, phi_range=None, n_seeds=10, pt_model=None):
    """
    If pt_model is provided, it is used for DPF (PT) and no training is done.
    If pt_model is None, PT is trained (individual + E2E) and then used.
    """
    print("Measuring Log-Likelihood surface across phi range...")
    diff_results = {
        "DPF (OT)": {"values": []},
        "Soft DPF": {"values": []},
        "DPF (SG)": {"values": []},
        "DPF (PT)": {"values": []},
    }

    if pt_model is None:
        pt_model = train_particle_transformer(y_data, n_particles=N, n_epochs=20)
        pt_model = end_to_end_train_particle_transformer(
            y_data,
            pt_model,
            n_particles=N,
            n_epochs=10,
            n_seeds_per_epoch=15,
            lr=1e-4,
            train_resampler=True,
        )

    if phi_range is None:
        phi_iter = [0.8 + i * (1.2 - 0.8) / (20 - 1) for i in range(20)]
    else:
        if isinstance(phi_range, tf.Tensor):
            phi_iter = [float(v) for v in phi_range.numpy()]
        else:
            phi_iter = phi_range

    methods = [
        ("DPF (OT)", DifferentiableParticleFilter, {"epsilon": 0.25, "sinkhorn_iters": 20, "resample_threshold": 2.0}),
        ("Soft DPF", SoftResamplingParticleFilter, {"alpha": 0.5, "resample_threshold": 2.0}),
        ("DPF (SG)", StopGradientParticleFilter, {"resample_threshold": 2.0}),
        ("DPF (PT)", ParticleTransformerFilter, {"pt_model": pt_model, "resample_threshold": 2.0}),
    ]

    for phi_val in tqdm(phi_iter):
        for name, filter_cls, kwargs in methods:
            run_vals = []
            for seed_offset in range(n_seeds):
                tf.random.set_seed(888 + seed_offset)
                model = _build_bootstrap_model(phi=phi_val)
                pf = filter_cls(model, num_particles=N, **kwargs)
                loglik, _ = pf(y_data)
                run_vals.append(float(loglik.numpy()))
            diff_results[name]["values"].append(sum(run_vals) / len(run_vals))

    return diff_results


def plot_results(acc_results, diff_results, snr_results, n_particles_list, phi_range, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    colors = {
        "DPF (OT)": "blue",
        "Soft DPF": "red",
        "DPF (SG)": "green",
        "DPF (PT)": "purple",
    }

    ax = axes[0, 0]
    for method, data in acc_results.items():
        biases = [x[0] for x in data["bias"]]
        stds = [x[1] for x in data["bias"]]
        ax.errorbar(n_particles_list, biases, yerr=stds, label=method, marker="o", color=colors[method], capsize=5)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Estimator Bias vs N\n(Closer to 0 is Better)")
    ax.set_xlabel("N Particles")
    ax.set_ylabel("Loglik Bias")
    ax.legend()

    ax = axes[0, 1]
    for method, data in acc_results.items():
        ax.plot(n_particles_list, data["time"], label=method, marker="s", color=colors[method])
    ax.set_title("Inference Efficiency vs N\n(Lower is Better)")
    ax.set_xlabel("N Particles")
    ax.set_ylabel("Time (ms)")
    ax.legend()

    ax = axes[1, 0]
    phi_vals = phi_range.numpy() if isinstance(phi_range, tf.Tensor) else phi_range
    for method, data in diff_results.items():
        vals = data["values"]
        ax.plot(phi_vals, vals, label=method, color=colors[method], linewidth=2)
        if len(vals) > 1:
            fd = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
            fd_extended = [fd[0]] + fd
            lower = [v - d for v, d in zip(vals, fd_extended)]
            upper = [v + d for v, d in zip(vals, fd_extended)]
            ax.fill_between(phi_vals, lower, upper, color=colors[method], alpha=0.15)
    ax.set_title("Log-Likelihood Surface\n(Shaded band = Local Jaggedness)")
    ax.set_xlabel("Parameter phi")
    ax.set_ylabel("E[Log p(y)]")
    ax.legend(fontsize="small")

    ax = axes[1, 1]
    methods_list = sorted(diff_results.keys())
    fd_means = []
    curv_means = []
    for method in methods_list:
        vals = diff_results[method]["values"]
        if len(vals) > 1:
            fd_sq = [(vals[i + 1] - vals[i]) ** 2 for i in range(len(vals) - 1)]
            fd_means.append(sum(fd_sq) / len(fd_sq))
        else:
            fd_means.append(1e-10)
        if len(vals) > 2:
            curv_sq = []
            for i in range(len(vals) - 2):
                d2 = vals[i + 2] - 2 * vals[i + 1] + vals[i]
                curv_sq.append(d2**2)
            curv_means.append(sum(curv_sq) / len(curv_sq))
        else:
            curv_means.append(1e-10)

    x = list(range(len(methods_list)))
    width = 0.35
    bar_colors = [colors[m] for m in methods_list]
    left_fd = [xi - width / 2 for xi in x]
    left_cur = [xi + width / 2 for xi in x]
    ax.bar(left_fd, fd_means, width, label="Mean FD²", color=bar_colors, alpha=0.4, hatch="//", edgecolor="black")
    ax.bar(left_cur, curv_means, width, label="Mean Curvature²", color=bar_colors, alpha=0.9, edgecolor="black")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_list)
    ax.set_title("Aggregate Surface Jaggedness\n(Lower is Better)")
    ax.set_ylabel("Metric Value (Log Scale)")
    metric_handles = [
        Patch(facecolor="gray", alpha=0.4, hatch="//", edgecolor="black", label="Local FD²"),
        Patch(facecolor="gray", alpha=0.9, edgecolor="black", label="Curvature²"),
    ]
    ax.legend(handles=metric_handles, loc="upper right", fontsize="x-small")

    ax = axes[2, 0]
    methods_ordered = ["DPF (OT)", "Soft DPF", "DPF (SG)", "DPF (PT)"]
    grad_data = [snr_results[m]["grads"] for m in methods_ordered]
    grad_means = [snr_results[m]["mean_grad"] for m in methods_ordered]
    n_exploded = [snr_results[m].get("n_exploded", 0) for m in methods_ordered]

    violin_positions = []
    violin_data = []
    for i, gd in enumerate(grad_data):
        if len(gd) >= 2:
            violin_positions.append(i)
            violin_data.append(gd)

    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_positions, showmeans=False, showmedians=False, showextrema=False)
        for pc, pos in zip(parts["bodies"], violin_positions):
            m = methods_ordered[pos]
            pc.set_facecolor(colors[m])
            pc.set_alpha(0.5)

    for i, (gd, gm, m, n_exp) in enumerate(zip(grad_data, grad_means, methods_ordered, n_exploded)):
        if len(gd) > 0:
            ax.scatter([i] * len(gd), gd, color=colors[m], s=20, alpha=0.6, zorder=3)
        if math.isfinite(gm):
            ax.axhline(
                gm,
                xmin=(i + 0.1) / len(methods_ordered),
                xmax=(i + 0.9) / len(methods_ordered),
                color=colors[m],
                linewidth=2,
                linestyle="--",
                zorder=4,
            )
        if n_exp > 0:
            ax.text(i, 0.98, f"{n_exp} exploded\n(excluded)", ha="center", va="top", fontsize=7, color=colors[m], style="italic", transform=ax.get_xaxis_transform())

    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_xticks(range(len(methods_ordered)))
    ax.set_xticklabels(methods_ordered)
    ax.set_title("Gradient Distribution (d log p(y) / d phi)\nDashed line = mean; dots = individual seeds")
    ax.set_ylabel("Gradient value")

    ax = axes[2, 1]
    snrs = [snr_results[m]["snr"] for m in methods_ordered]
    grad_norms = [snr_results[m]["grad_norm"] for m in methods_ordered]
    grad_stds = [snr_results[m]["std_grad"] for m in methods_ordered]
    n_exp_list = [snr_results[m].get("n_exploded", 0) for m in methods_ordered]

    def _safe(vals):
        return [v if math.isfinite(v) else 0.0 for v in vals]

    x = list(range(len(methods_ordered)))
    width = 0.28
    bc = [colors[m] for m in methods_ordered]
    left_snr = [xi - width for xi in x]
    center_pos = x
    right_std = [xi + width for xi in x]
    ax.bar(left_snr, _safe(snrs), width, color=bc, alpha=0.9, edgecolor="black")
    ax.bar(center_pos, _safe(grad_norms), width, color=bc, alpha=0.5, hatch="//", edgecolor="black")
    ax.bar(right_std, _safe(grad_stds), width, color=bc, alpha=0.3, hatch="xx", edgecolor="black")

    for i, (n_exp, m) in enumerate(zip(n_exp_list, methods_ordered)):
        if n_exp > 0:
            ax.text(x[i], 0.01, "all\nexploded", ha="center", va="bottom", fontsize=7, color=colors[m], style="italic", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(methods_ordered)
    all_bar_vals = _safe(snrs) + _safe(grad_norms) + _safe(grad_stds)
    if any(v > 0 for v in all_bar_vals):
        ax.set_yscale("log")
    ax.set_title("Gradient SNR & Statistics\n(Higher SNR = more useful gradient signal)")
    ax.set_ylabel("Value (Log Scale)")
    metric_handles = [
        Patch(facecolor="gray", alpha=0.9, edgecolor="black", label="SNR = |mean|/std"),
        Patch(facecolor="gray", alpha=0.5, hatch="//", edgecolor="black", label="Mean |grad|"),
        Patch(facecolor="gray", alpha=0.3, hatch="xx", edgecolor="black", label="Std(grad)"),
    ]
    ax.legend(handles=metric_handles, fontsize="x-small")

    plt.tight_layout()
    fig_path = out_dir / "dpf_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {fig_path}")


def write_summary(acc_results, diff_results, snr_results, truth, n_particles_list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DEFINITIVE KITAGAWA MODEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Ground Truth Log-Likelihood (N=100k): {truth:.4f}\n\n")

        f.write("1. ACCURACY & EFFICIENCY\n")
        f.write("-" * 40 + "\n")
        header = f"{'Method':<15} | {'N':<5} | {'Bias':<10} | {'Std':<10} | {'Time (ms)':<10}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for method, data in acc_results.items():
            for i, N in enumerate(n_particles_list):
                bias, std = data["bias"][i]
                t = data["time"][i]
                f.write(f"{method:<15} | {N:<5} | {bias:>10.4f} | {std:>10.4f} | {t:>10.2f}\n")
            f.write("-" * len(header) + "\n")

        f.write("\n2. SURFACE SMOOTHNESS (Finite-Difference Metrics)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Method':<15} | {'FD Variance':<12} | {'Curvature':<12}\n")
        f.write("-" * 45 + "\n")
        for method, data in diff_results.items():
            vals = data["values"]
            if len(vals) > 1:
                fd_sq = [(vals[i + 1] - vals[i]) ** 2 for i in range(len(vals) - 1)]
                fd_var = sum(fd_sq) / len(fd_sq)
            else:
                fd_var = 0.0
            if len(vals) > 2:
                curv_sq = []
                for i in range(len(vals) - 2):
                    d2 = vals[i + 2] - 2 * vals[i + 1] + vals[i]
                    curv_sq.append(d2**2)
                curvature = sum(curv_sq) / len(curv_sq)
            else:
                curvature = 0.0
            f.write(f"{method:<15} | {fd_var:>12.6f} | {curvature:>12.6f}\n")

        f.write("\n3. GRADIENT QUALITY (d log p(y) / d phi)\n")
        f.write("-" * 70 + "\n")
        f.write("SNR = |E[grad]| / std(grad). Higher is better.\n\n")
        f.write(
            f"{'Method':<15} | {'Mean grad':>10} | {'Std grad':>10} | "
            f"{'|Mean| grad':>12} | {'SNR':>8} | {'Exploded':>8}\n"
        )
        f.write("-" * 75 + "\n")
        for method, data in snr_results.items():
            f.write(
                f"{method:<15} | {data['mean_grad']:>10.5f} | {data['std_grad']:>10.5f} | "
                f"{data['grad_norm']:>12.5f} | {data['snr']:>8.3f} | "
                f"{data.get('n_exploded', 0):>8d}\n"
            )

    print(f"Saved summary: {summary_path}")


def main():
    print("=" * 80)
    print("Final Rigorous Kitagawa Model Comparison: Bias vs Gradient SNR vs Efficiency")
    print("=" * 80)

    y_data = simulate_kitagawa_data()

    t0 = time.perf_counter()
    truth = compute_ground_truth(y_data)
    print(f"Ground Truth Likelihood: {truth:.4f}  [{time.perf_counter() - t0:.1f}s]")

    n_particles_list = [25, 50, 100, 200, 300]
    phi_range = tf.linspace(0.8, 1.2, 25)

    t_pt_start = time.perf_counter()
    pt_model = _load_or_train_pt(y_data, n_particles=50)
    t_pt = time.perf_counter() - t_pt_start
    print(f"  PT ready (train/load): {t_pt:.1f}s\n")

    t0 = time.perf_counter()
    acc_results, _ = measure_accuracy_and_efficiency(
        y_data,
        truth,
        n_particles_list=n_particles_list,
        pt_model=pt_model,
    )
    print(f"  Accuracy & efficiency: {time.perf_counter() - t0:.1f}s\n")

    t0 = time.perf_counter()
    diff_results = measure_differentiability(
        y_data,
        phi_range=phi_range,
        n_seeds=5,
        pt_model=pt_model,
    )
    print(f"  Differentiability (LL surface): {time.perf_counter() - t0:.1f}s\n")

    t0 = time.perf_counter()
    snr_results = measure_gradient_snr(
        y_data,
        pt_model=pt_model,
        N=50,
        phi_true=1.0,
        n_seeds=30,
    )
    print(f"  Gradient SNR: {time.perf_counter() - t0:.1f}s\n")

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "reports" / "5_Differential_PF_OT_Resampling" / "dpf_comparison"

    t0 = time.perf_counter()
    plot_results(acc_results, diff_results, snr_results, n_particles_list, phi_range, out_dir=out_dir)
    write_summary(acc_results, diff_results, snr_results, truth, n_particles_list, out_dir=out_dir)
    print(f"  Plot + summary: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

