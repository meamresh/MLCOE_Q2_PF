"""
Triangle / corner plot comparing pmmh, HMC-LEDH, and LHNN_LEDH posteriors.

Loads per-chain ``.npz`` files saved by the three "*_only" experiments
(``pmmh_only``, ``hmc_ledh_only``, ``lhnn_only``), pools chains per
method, and produces an overlaid getdist triangle plot.

Robust to missing methods/chains — if a method has no saved chains it is
skipped with a warning, and partial L-HNN runs (e.g., only chains 0 + 1
saved so far) plot whatever is on disk.

Usage::

    python -m scripts.plot_corner_comparison
    python -m scripts.plot_corner_comparison --output reports/.../triangle.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from getdist import MCSamples, plots


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH"

# (legend label, chains-dir, color). Order matters for triangle_plot: each series is
# drawn on top of the previous, so put HMC-LEDH last or LHNN will obscure it when
# posteriors overlap.
METHOD_SPECS = [
    ("pmmh",       BASE / "pmmh_only"     / "chains", "#1f77b4"),
    ("LHNN_LEDH",  BASE / "lhnn_only"     / "chains", "#2ca02c"),
    ("HMC-LEDH",   BASE / "hmc_ledh_only" / "chains", "#d62728"),
]

PARAM_NAMES = ["sigma_v2", "sigma_w2"]
PARAM_LABELS = [r"\sigma_v^2", r"\sigma_w^2"]
TRUTH = {"sigma_v2": 10.0, "sigma_w2": 1.0}


def _load_method_samples(chains_dir: Path) -> np.ndarray | None:
    """Load all available chain_*.npz, exp the log-space samples, pool.

    Returns shape (num_total, 2) in (sigma_v^2, sigma_w^2) space, or
    None if no chain files exist.
    """
    if not chains_dir.exists():
        return None
    chain_files = sorted(chains_dir.glob("chain_*.npz"))
    if not chain_files:
        return None
    pooled = []
    for f in chain_files:
        z = np.load(f)
        # samples are stored in log space (log sigma_v^2, log sigma_w^2)
        s = np.exp(z["samples"])
        pooled.append(s)
    return np.concatenate(pooled, axis=0).astype(np.float64)


def _build_mcsamples(name: str, samples: np.ndarray) -> MCSamples:
    """Wrap pooled samples into a getdist MCSamples object."""
    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS,
        label=name,
        ranges={"sigma_v2": (0.0, None), "sigma_w2": (0.0, None)},
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Triangle plot comparing PMMH / HMC-LEDH / L-HNN posteriors.")
    p.add_argument(
        "--output",
        type=str,
        default=str(BASE / "comparison" / "triangle_compare.png"),
        help="Output PNG path.",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Down-sample each method to at most this many points (default 20000).",
    )
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mcs_list = []
    colors = []
    summary_rows = []

    for label, chains_dir, color in METHOD_SPECS:
        s = _load_method_samples(chains_dir)
        if s is None or len(s) == 0:
            print(f"  [skip] {label}: no chains found in {chains_dir}")
            continue
        n_chains = len(sorted(chains_dir.glob("chain_*.npz")))
        n_total = s.shape[0]
        # Down-sample if needed (keeps getdist density estimation snappy)
        if n_total > args.max_points:
            idx = np.random.default_rng(0).choice(n_total, size=args.max_points, replace=False)
            s_plot = s[idx]
        else:
            s_plot = s
        mcs_list.append(_build_mcsamples(label, s_plot))
        colors.append(color)

        m_v, m_w = float(s[:, 0].mean()), float(s[:, 1].mean())
        summary_rows.append(
            f"  {label:<12}  chains={n_chains:>2}  N_total={n_total:>6}  "
            f"plotted={s_plot.shape[0]:>5}  "
            f"mean(sigma_v^2)={m_v:6.3f}  mean(sigma_w^2)={m_w:6.3f}"
        )

    if not mcs_list:
        print("No method has saved chains. Nothing to plot.")
        return 1

    print("\nLoaded methods:")
    for row in summary_rows:
        print(row)
    print()

    g = plots.get_subplot_plotter(width_inch=8.0)
    g.settings.alpha_filled_add = 0.55
    g.settings.legend_fontsize = 11
    g.settings.axes_labelsize = 13
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 11
    # getdist's figure legend is often clipped when title_limit>0 (forces no_tight_layout).
    # We draw an explicit matplotlib legend after plotting instead.
    g.settings.line_labels = False

    g.triangle_plot(
        mcs_list,
        params=PARAM_NAMES,
        filled=True,
        contour_colors=colors,
        title_limit=None,
    )

    # Mark the true parameter values on every panel.
    for i, pname_y in enumerate(PARAM_NAMES):
        for j in range(i + 1):
            ax = g.subplots[i, j]
            if ax is None:
                continue
            if i == j:
                # Diagonal: 1D marginal of pname_y -> vertical line at truth_y
                ax.axvline(TRUTH[pname_y], color="k", ls="--", lw=1.0, alpha=0.8)
            else:
                pname_x = PARAM_NAMES[j]
                ax.axvline(TRUTH[pname_x], color="k", ls="--", lw=0.8, alpha=0.6)
                ax.axhline(TRUTH[pname_y], color="k", ls="--", lw=0.8, alpha=0.6)
                ax.plot(
                    [TRUTH[pname_x]], [TRUTH[pname_y]],
                    marker="*", markersize=14, color="black",
                    markerfacecolor="gold", markeredgewidth=1.0, zorder=10,
                )

    g.fig.suptitle(
        r"Posterior comparison — pmmh / HMC-LEDH / LHNN_LEDH  "
        r"(true $\sigma_v^2$=10, $\sigma_w^2$=1)",
        fontsize=12,
        y=1.02,
    )

    # Explicit legend (included in bbox_inches='tight' via bbox_extra_artists).
    # MCSamples.getName() is None here; use getLabel() for the user-supplied label=... .
    legend_names = [m.getLabel() for m in mcs_list]
    handles = [
        Line2D([0], [0], color=c, lw=10, linestyle="-", solid_capstyle="butt")
        for c in colors
    ]
    legend_art = g.fig.legend(
        handles,
        legend_names,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        bbox_transform=g.fig.transFigure,
        ncol=1,
        frameon=True,
        fancybox=True,
        fontsize=g.settings.legend_fontsize or 11,
    )

    bbox_extra = [legend_art]
    if g.fig._suptitle is not None:
        bbox_extra.append(g.fig._suptitle)

    g.fig.savefig(
        str(out_path),
        dpi=150,
        bbox_inches="tight",
        bbox_extra_artists=bbox_extra,
        pad_inches=0.15,
    )
    plt.close("all")
    print(f"Saved triangle plot -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
