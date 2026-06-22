"""
Vehtari diagnostics + wall-clock speedup comparison between vanilla
HMC and LHNN HMC on SVSSM samples.

Inputs:
  --vanilla_dir   directory holding svssm_hmc_samples.npz + svssm_hmc_summary.json
  --lhnn_dir      directory holding svssm_lhnn_samples.npz + svssm_lhnn_summary.json

Outputs (to --out_dir):
  vehtari_comparison.txt    side-by-side table per parameter
  vehtari_comparison.json   machine-readable rows + headline numbers
  vehtari_marginals.png     overlapping LHNN vs vanilla histograms
                            (per-chain split shown above pooled)

Vehtari thresholds:
  rank-Rhat ≤ 1.01,  bulk-ESS ≥ 400,  tail-ESS ≥ 400.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    plt = None
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# Manual Vehtari implementations (TFP's are unreliable — see Phase v2_mv)
# ---------------------------------------------------------------------------

def _split_chains(chains: np.ndarray) -> np.ndarray:
    """Split each chain in half. (M, N) -> (2M, N//2)."""
    M, N = chains.shape
    half = N // 2
    return chains[:, :2 * half].reshape(M, 2, half).reshape(2 * M, half)


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Vehtari rank-normalization: replace pooled values by inverse-Phi of
    rank/(N+1). Yields approx-normal scores; Rhat/ESS on these is the
    rank-Rhat / rank-ESS Vehtari recommends."""
    flat = chains.reshape(-1)
    ranks = sp_stats.rankdata(flat, method="average")
    z = sp_stats.norm.ppf(ranks / (len(flat) + 1.0))
    return z.reshape(chains.shape)


def gelman_rubin(chains: np.ndarray) -> float:
    """Split-Rhat on (M, N). Returns sqrt(varhat/W)."""
    split = _split_chains(chains)
    M2, N2 = split.shape
    if N2 < 2:
        return float("nan")
    means = split.mean(axis=1)
    vars_ = split.var(axis=1, ddof=1)
    B = N2 * means.var(ddof=1)
    W = vars_.mean()
    if W <= 0:
        return float("nan")
    var_hat = ((N2 - 1) / N2) * W + B / N2
    return float(np.sqrt(var_hat / W))


def rank_rhat(chains: np.ndarray) -> float:
    """Vehtari rank-Rhat: split-Rhat on rank-normalised pooled samples,
    folded version added (using |x - median|), max of bulk-rank and
    folded-rank. Returns the max."""
    z = _rank_normalize(chains)
    rhat_bulk = gelman_rubin(z)
    pooled_median = np.median(chains)
    folded = np.abs(chains - pooled_median)
    z_folded = _rank_normalize(folded)
    rhat_tail = gelman_rubin(z_folded)
    return float(max(rhat_bulk, rhat_tail))


def autocorr_via_fft(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    N = len(x)
    f = np.fft.fft(x, n=2 * N)
    acf = np.fft.ifft(f * np.conj(f)).real[:N]
    if acf[0] == 0:
        return np.zeros_like(acf)
    acf /= acf[0]
    return acf


def ess_geyer(chains: np.ndarray) -> float:
    """ESS via Geyer's initial monotone sequence (Stan/Vehtari standard).
    chains: (M, N)."""
    M, N = chains.shape
    if N < 4:
        return float("nan")
    acfs = np.stack([autocorr_via_fft(chains[m]) for m in range(M)], axis=0)
    rho = acfs.mean(axis=0)
    K = (N - 1) // 2
    paired = rho[1:2 * K + 1].reshape(K, 2).sum(axis=1)
    neg_idx = np.where(paired <= 0)[0]
    cutoff = neg_idx[0] if len(neg_idx) > 0 else K
    tau = 1.0 + 2.0 * paired[:cutoff].sum() if cutoff > 0 else 1.0
    if tau <= 0:
        return float(M * N)
    return float(M * N / tau)


def bulk_ess(chains: np.ndarray) -> float:
    """Bulk-ESS = ESS on rank-normalised samples, split."""
    z = _rank_normalize(chains)
    return ess_geyer(_split_chains(z))


def tail_ess(chains: np.ndarray, q_lo: float = 0.05,
              q_hi: float = 0.95) -> float:
    """Tail-ESS = min(ESS(I{x<=q5%}), ESS(I{x>=q95%})). Vehtari 2021."""
    flat = chains.reshape(-1)
    qa = np.quantile(flat, q_lo)
    qb = np.quantile(flat, q_hi)
    ind_lo = (chains <= qa).astype(np.float64)
    ind_hi = (chains >= qb).astype(np.float64)
    ess_lo = ess_geyer(_split_chains(ind_lo))
    ess_hi = ess_geyer(_split_chains(ind_hi))
    return min(ess_lo, ess_hi)


def vehtari_row(chains: np.ndarray, name: str) -> dict:
    """Per-parameter Vehtari row. chains: (M, N)."""
    rhat = rank_rhat(chains)
    bulk = bulk_ess(chains)
    tail = tail_ess(chains)
    flat = chains.reshape(-1)
    return {
        "param": name,
        "median": float(np.median(flat)),
        "mean": float(np.mean(flat)),
        "sd": float(np.std(flat, ddof=1)),
        "q025": float(np.percentile(flat, 2.5)),
        "q975": float(np.percentile(flat, 97.5)),
        "rank_rhat": rhat,
        "bulk_ess": bulk,
        "tail_ess": tail,
        "rhat_ok": rhat <= 1.01,
        "bulk_ok": bulk >= 400,
        "tail_ok": tail >= 400,
        "vehtari_pass": (rhat <= 1.01) and (bulk >= 400) and (tail >= 400),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_samples(npz_path: Path) -> np.ndarray:
    """Load samples_constrained; (chains, draws, 3)."""
    d = np.load(npz_path)
    if "samples_constrained" not in d:
        raise KeyError(f"{npz_path} missing samples_constrained")
    return d["samples_constrained"]


def _load_summary(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Comparison + plots
# ---------------------------------------------------------------------------

def per_param_compare(rows_v: list[dict], rows_l: list[dict],
                       v_samples: np.ndarray, l_samples: np.ndarray,
                       names: list[str]) -> list[dict]:
    out = []
    for i, name in enumerate(names):
        rv = next(r for r in rows_v if r["param"] == name)
        rl = next(r for r in rows_l if r["param"] == name)
        x_v = v_samples[..., i].ravel()
        x_l = l_samples[..., i].ravel()
        ks_stat, ks_p = sp_stats.ks_2samp(x_v, x_l)
        out.append({
            "param": name,
            "vanilla": rv,
            "lhnn": rl,
            "median_abs_diff": abs(rv["median"] - rl["median"]),
            "sd_ratio_lhnn_over_vanilla": rl["sd"] / max(rv["sd"], 1e-12),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "ks_indistinguishable_at_p05": bool(ks_p > 0.05),
        })
    return out


def plot_overlay(out_path: Path, v_samples: np.ndarray, l_samples: np.ndarray,
                  truth_arr: np.ndarray, comparison: list[dict],
                  names: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    # Top row: pooled histograms
    for i, (ax, name, row) in enumerate(zip(axes[0], names, comparison)):
        x_v = v_samples[..., i].ravel()
        x_l = l_samples[..., i].ravel()
        lo = float(min(np.percentile(x_v, 0.5), np.percentile(x_l, 0.5)))
        hi = float(max(np.percentile(x_v, 99.5), np.percentile(x_l, 99.5)))
        bins = np.linspace(lo, hi, 60)
        ax.hist(x_v, bins=bins, density=True, alpha=0.55, color="steelblue",
                label=f"vanilla HMC\nR̂={row['vanilla']['rank_rhat']:.3f} "
                       f"ESS={row['vanilla']['bulk_ess']:.0f}")
        ax.hist(x_l, bins=bins, density=True, alpha=0.55, color="darkorange",
                label=f"LHNN HMC\nR̂={row['lhnn']['rank_rhat']:.3f} "
                       f"ESS={row['lhnn']['bulk_ess']:.0f}")
        ax.axvline(truth_arr[i], color="red", ls="--", lw=1.5,
                   label=f"truth={truth_arr[i]:.3f}")
        verdict = "AGREE" if row["ks_indistinguishable_at_p05"] else "DIFFER"
        ax.set_title(f"{name}  KS p={row['ks_p']:.3f}  [{verdict}]  "
                     f"(pooled)", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    # Bottom row: per-chain mean ± sd
    for i, (ax, name) in enumerate(zip(axes[1], names)):
        n_v = v_samples.shape[0]
        n_l = l_samples.shape[0]
        v_means = v_samples[..., i].mean(axis=1)
        v_sds = v_samples[..., i].std(axis=1, ddof=1)
        l_means = l_samples[..., i].mean(axis=1)
        l_sds = l_samples[..., i].std(axis=1, ddof=1)
        x_v_pos = np.arange(n_v) - 0.15
        x_l_pos = np.arange(n_l) + 0.15
        ax.errorbar(x_v_pos, v_means, yerr=v_sds, fmt="o", color="steelblue",
                     capsize=4, label="vanilla")
        ax.errorbar(x_l_pos, l_means, yerr=l_sds, fmt="s", color="darkorange",
                     capsize=4, label="LHNN")
        ax.axhline(truth_arr[i], color="red", ls="--", lw=1.2,
                   label=f"truth={truth_arr[i]:.3f}")
        ax.set_xticks(np.arange(max(n_v, n_l)))
        ax.set_xticklabels([f"c{c+1}" for c in range(max(n_v, n_l))])
        ax.set_xlabel("chain")
        ax.set_ylabel(name)
        ax.set_title(f"{name}: per-chain mean ± sd", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("LHNN HMC vs vanilla HMC — Vehtari + per-chain mixing",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_dir", required=True)
    p.add_argument("--lhnn_dir", required=True)
    p.add_argument("--out_dir", default=None,
                   help="Defaults to <lhnn_dir>/comparison")
    p.add_argument("--truth", nargs=3, type=float, default=(0.0, 0.95, 0.3),
                   help="(mu, phi, sigma_eta). sigma_eta_sq is squared.")
    args = p.parse_args()

    v_dir = Path(args.vanilla_dir)
    l_dir = Path(args.lhnn_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (l_dir / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    v_samples = _load_samples(v_dir / "svssm_hmc_samples.npz")
    l_samples = _load_samples(l_dir / "svssm_lhnn_samples.npz")
    v_summary = _load_summary(v_dir / "svssm_hmc_summary.json")
    l_summary = _load_summary(l_dir / "svssm_lhnn_summary.json")

    names = ["mu", "phi", "sigma_eta_sq"]
    truth_arr = np.asarray([args.truth[0], args.truth[1], args.truth[2] ** 2])

    print(f"vanilla samples: {v_samples.shape}, LHNN samples: {l_samples.shape}")

    # Per-parameter Vehtari rows
    rows_v = [vehtari_row(v_samples[..., i], names[i]) for i in range(3)]
    rows_l = [vehtari_row(l_samples[..., i], names[i]) for i in range(3)]
    comparison = per_param_compare(rows_v, rows_l, v_samples, l_samples, names)

    # Wall-clock + grad-eval comparison
    wall_v = float(v_summary.get("elapsed_s", float("nan")))
    wall_l = float(l_summary.get("elapsed_s", float("nan")))
    speedup_wall = wall_v / wall_l if (wall_v and wall_l and wall_l > 0) else float("nan")
    grad_evals_v = int(v_summary.get("config", {}).get("num_chains", 4)) \
        * (int(v_summary.get("config", {}).get("L", 5)) + 1) \
        * (int(v_summary.get("config", {}).get("num_burnin", 500))
           + int(v_summary.get("config", {}).get("num_results", 1500)))
    grad_evals_l_total = int(l_summary.get("total_lhnn_gradient_evals",
                                            l_summary.get("training_gradient_evals", 0)
                                            + l_summary.get("sampling_gradient_evals", 0)))
    grad_evals_l_pilot = int(l_summary.get("training_gradient_evals", 0))
    grad_evals_l_fallback = int(l_summary.get("sampling_gradient_evals", 0))
    grad_savings_pct = 100.0 * (grad_evals_v - grad_evals_l_total) / max(grad_evals_v, 1)

    # ---- Print summary ----
    bar = "=" * 110
    print("\n" + bar)
    print("Vehtari diagnostics (per parameter)")
    print(bar)
    hdr = (f"{'param':<14s} {'method':<10s} {'median':>10s} {'sd':>10s} "
           f"{'2.5%':>10s} {'97.5%':>10s} {'rank-R̂':>8s} {'bulkESS':>9s} "
           f"{'tailESS':>9s}  flags")
    print(hdr)
    print("-" * 110)
    for i, name in enumerate(names):
        for tag, r in [("vanilla", rows_v[i]), ("lhnn", rows_l[i])]:
            flags = []
            if not r["rhat_ok"]: flags.append("R̂")
            if not r["bulk_ok"]: flags.append("bulk")
            if not r["tail_ok"]: flags.append("tail")
            flag_str = ",".join(flags) if flags else "PASS"
            print(f"{name:<14s} {tag:<10s} {r['median']:>10.4f} "
                  f"{r['sd']:>10.4f} {r['q025']:>10.4f} {r['q975']:>10.4f} "
                  f"{r['rank_rhat']:>8.3f} {r['bulk_ess']:>9.1f} "
                  f"{r['tail_ess']:>9.1f}  {flag_str}")
    print(bar)
    print("Posterior agreement (KS p>0.05 ⇒ statistically indistinguishable)")
    print(bar)
    print(f"{'param':<14s} {'|Δmedian|':>12s} {'sd_ratio':>10s} "
          f"{'KS_stat':>9s} {'KS_p':>9s}  verdict")
    for row in comparison:
        v = "AGREE" if row["ks_indistinguishable_at_p05"] else "DIFFER"
        print(f"{row['param']:<14s} {row['median_abs_diff']:>12.4f} "
              f"{row['sd_ratio_lhnn_over_vanilla']:>10.3f} "
              f"{row['ks_stat']:>9.3f} {row['ks_p']:>9.4f}  {v}")
    print(bar)
    print("Wall-clock + gradient cost")
    print(bar)
    print(f"  Vanilla wall : {wall_v:>10.1f} s")
    print(f"  LHNN wall    : {wall_l:>10.1f} s")
    print(f"  Speedup (wall): {speedup_wall:.2f}×")
    print()
    print(f"  Vanilla real ∇log π evals : {grad_evals_v:>8d} "
          f"= chains * (L+1) * (burnin+results)")
    print(f"  LHNN pilot evals          : {grad_evals_l_pilot:>8d}")
    print(f"  LHNN sampling fallback    : {grad_evals_l_fallback:>8d}")
    print(f"  LHNN total                : {grad_evals_l_total:>8d}")
    print(f"  Grad-eval savings         : {grad_savings_pct:.1f}%")
    print(bar)

    # ---- Save ----
    headline = {
        "wall_vanilla_s": wall_v,
        "wall_lhnn_s": wall_l,
        "speedup_wall": speedup_wall,
        "vanilla_grad_evals": grad_evals_v,
        "lhnn_pilot_evals": grad_evals_l_pilot,
        "lhnn_sampling_fallback_evals": grad_evals_l_fallback,
        "lhnn_total_grad_evals": grad_evals_l_total,
        "grad_eval_savings_pct": grad_savings_pct,
        "vanilla_vehtari_all_pass": all(r["vehtari_pass"] for r in rows_v),
        "lhnn_vehtari_all_pass": all(r["vehtari_pass"] for r in rows_l),
        "ks_agree_all": all(c["ks_indistinguishable_at_p05"]
                             for c in comparison),
    }
    (out_dir / "vehtari_comparison.json").write_text(json.dumps({
        "headline": headline,
        "vanilla_rows": rows_v,
        "lhnn_rows": rows_l,
        "comparison": comparison,
        "truth": {"mu": args.truth[0], "phi": args.truth[1],
                   "sigma_eta": args.truth[2]},
    }, indent=2))
    print(f"\n[save] {out_dir / 'vehtari_comparison.json'}")

    # text report
    txt_lines = [
        bar, "Vehtari comparison — vanilla HMC vs LHNN HMC", bar,
        f"vanilla_dir : {v_dir}",
        f"lhnn_dir    : {l_dir}",
        f"truth       : mu={args.truth[0]}, phi={args.truth[1]}, "
        f"sigma_eta={args.truth[2]}", "",
        "Wall + cost:",
        f"  Vanilla wall                : {wall_v:.1f} s",
        f"  LHNN wall                   : {wall_l:.1f} s",
        f"  Speedup (wall)              : {speedup_wall:.2f}×",
        f"  Vanilla real ∇log π evals   : {grad_evals_v}",
        f"  LHNN total real ∇log π evals: {grad_evals_l_total} "
        f"({grad_evals_l_pilot} pilot + {grad_evals_l_fallback} fallback)",
        f"  Grad-eval savings           : {grad_savings_pct:.1f}%",
        "",
        "Vehtari per parameter:",
        hdr, "-" * 110,
    ]
    for i, name in enumerate(names):
        for tag, r in [("vanilla", rows_v[i]), ("lhnn", rows_l[i])]:
            flags = []
            if not r["rhat_ok"]: flags.append("R̂")
            if not r["bulk_ok"]: flags.append("bulk")
            if not r["tail_ok"]: flags.append("tail")
            flag_str = ",".join(flags) if flags else "PASS"
            txt_lines.append(
                f"{name:<14s} {tag:<10s} {r['median']:>10.4f} "
                f"{r['sd']:>10.4f} {r['q025']:>10.4f} {r['q975']:>10.4f} "
                f"{r['rank_rhat']:>8.3f} {r['bulk_ess']:>9.1f} "
                f"{r['tail_ess']:>9.1f}  {flag_str}"
            )
    txt_lines += [
        "", "Posterior agreement (KS p>0.05 ⇒ indistinguishable):",
        f"{'param':<14s} {'|Δmedian|':>12s} {'sd_ratio':>10s} "
        f"{'KS_stat':>9s} {'KS_p':>9s}  verdict",
    ]
    for row in comparison:
        v = "AGREE" if row["ks_indistinguishable_at_p05"] else "DIFFER"
        txt_lines.append(
            f"{row['param']:<14s} {row['median_abs_diff']:>12.4f} "
            f"{row['sd_ratio_lhnn_over_vanilla']:>10.3f} "
            f"{row['ks_stat']:>9.3f} {row['ks_p']:>9.4f}  {v}"
        )
    txt_lines += [bar]
    (out_dir / "vehtari_comparison.txt").write_text("\n".join(txt_lines))
    print(f"[save] {out_dir / 'vehtari_comparison.txt'}")

    # Plot
    if _HAVE_MPL:
        plot_overlay(out_dir / "vehtari_marginals.png", v_samples, l_samples,
                      truth_arr, comparison, names)
        print(f"[save] {out_dir / 'vehtari_marginals.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
