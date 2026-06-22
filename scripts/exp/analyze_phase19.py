"""Phase 19 analysis: no-data baseline vs with-data + phi-grid recovery.

Two views:

(1) No-data vs with-data:
    Loads the no-data (prior-only) baseline and the matched with-data
    Phase 16 posterior. Tabulates median + 5/95% CI for both. Computes a
    "shift index" = (med_with_data - med_no_data) / sd_no_data — if the
    likelihood is informing, this should be substantially nonzero for the
    parameters the data can identify.

(2) phi-grid:
    Loads each cell at phi in {0.7, 0.85, 0.95} and tabulates the
    posterior median of phi against truth. The diagnostic question is
    whether the chain TRACKS each truth or PILES UP at the attractor.

Usage:
    python -m scripts.exp.analyze_phase19 \
        --no_data_dir reports/.../phase19_attractor/no_data_baseline \
        --with_data_dir reports/.../new/svssm_hmc_sweep_wide_T100 \
        --phi_grid_root reports/.../phase19_attractor/phi_grid
"""

import argparse
from pathlib import Path

import numpy as np


def load_samples(npz_path: Path):
    """Load (samples_constrained, samples_raw) from an svssm_hmc_samples.npz."""
    with np.load(npz_path) as data:
        sc = np.asarray(data["samples_constrained"])  # (chains, draws, 3)
        sr = np.asarray(data["samples_raw"])          # (chains, draws, 3)
    return sc, sr


def summarize_marginals(samples_constrained: np.ndarray, sigma_eta_sq: bool = False):
    """Return dict of {param: {'median': ..., 'q05': ..., 'q95': ..., 'sd': ...}}."""
    # samples_constrained: (chains, draws, 3) for [mu, phi, sigma_eta_sq]
    flat = samples_constrained.reshape(-1, samples_constrained.shape[-1])
    names = ["mu", "phi", "sigma_eta_sq" if sigma_eta_sq else "sigma_eta"]
    if not sigma_eta_sq:
        # convert sigma_eta_sq -> sigma_eta for last column
        flat = flat.copy()
        flat[:, 2] = np.sqrt(np.clip(flat[:, 2], 0.0, None))
    out = {}
    for j, n in enumerate(names):
        col = flat[:, j]
        out[n] = {
            "median": float(np.median(col)),
            "mean": float(np.mean(col)),
            "sd": float(np.std(col)),
            "q05": float(np.percentile(col, 5.0)),
            "q95": float(np.percentile(col, 95.0)),
        }
    return out


def shift_table(no_data: dict, with_data: dict, params=("mu", "phi", "sigma_eta")):
    rows = []
    for p in params:
        nd = no_data[p]
        wd = with_data[p]
        shift = (wd["median"] - nd["median"]) / max(nd["sd"], 1e-9)
        ci_nd = nd["q95"] - nd["q05"]
        ci_wd = wd["q95"] - wd["q05"]
        ci_shrink = (ci_nd - ci_wd) / max(ci_nd, 1e-9) * 100.0
        rows.append((p, nd["median"], wd["median"], shift,
                     ci_nd, ci_wd, ci_shrink))
    return rows


def print_shift_table(rows, header):
    print(f"\n========== {header} ==========")
    print(f"{'param':>14} | {'no-data med':>12} {'with-data med':>14} "
          f"{'shift /sd':>10} | {'no-data CI':>10} {'with-data CI':>12} "
          f"{'CI shrink %':>12}")
    print("-" * 100)
    for p, nd_med, wd_med, shift, ci_nd, ci_wd, ci_shrink in rows:
        print(f"{p:>14} | {nd_med:12.4f} {wd_med:14.4f} {shift:10.2f} | "
              f"{ci_nd:10.4f} {ci_wd:12.4f} {ci_shrink:12.1f}")
    print()
    print("Interpretation: |shift /sd| > 1 means the with-data posterior median")
    print("sits more than 1 prior-sd away from the no-data posterior median —")
    print("data is moving the chain. CI shrink > 0 means with-data is tighter")
    print("than prior-only.")


def phi_grid_table(phi_grid_root: Path):
    cells = sorted(phi_grid_root.glob("phi_*"))
    if not cells:
        print(f"\n(no phi-grid cells found at {phi_grid_root})")
        return
    print(f"\n========== phi-grid recovery ==========")
    print(f"{'truth_phi':>10} | {'med_mu':>10} {'med_phi':>10} {'med_sigma_eta':>14} "
          f"| {'phi_q05':>8} {'phi_q95':>8} | {'tracks?':>8}")
    print("-" * 90)
    for cell in cells:
        npz = cell / "svssm_hmc_samples.npz"
        if not npz.exists():
            print(f"{cell.name:>10} : MISSING samples.npz")
            continue
        truth_phi = float(cell.name.split("_")[-1])
        sc, _ = load_samples(npz)
        summ = summarize_marginals(sc)
        phi_med = summ["phi"]["median"]
        phi_q05 = summ["phi"]["q05"]
        phi_q95 = summ["phi"]["q95"]
        # "tracks" if truth is in the 90% CI AND median is within 0.1 of truth
        tracks = "YES" if (phi_q05 <= truth_phi <= phi_q95
                           and abs(phi_med - truth_phi) <= 0.1) else "no"
        print(f"{truth_phi:10.2f} | {summ['mu']['median']:10.3f} "
              f"{phi_med:10.3f} {summ['sigma_eta']['median']:14.3f} | "
              f"{phi_q05:8.3f} {phi_q95:8.3f} | {tracks:>8}")
    print()
    print("Interpretation: if YES at all 3 phis, filter is informing — data")
    print("discriminates the truth from the attractor. If 'no' at all 3 (and")
    print("all posteriors near 0.99), the attractor dominates.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no_data_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "phase19_attractor/no_data_baseline")
    p.add_argument("--with_data_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "new/svssm_hmc_sweep_wide_T100")
    p.add_argument("--phi_grid_root", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "phase19_attractor/phi_grid")
    args = p.parse_args()

    nd_dir = Path(args.no_data_dir)
    wd_dir = Path(args.with_data_dir)
    pg_root = Path(args.phi_grid_root)

    # ---- view (1): no-data vs with-data ----
    nd_npz = nd_dir / "svssm_hmc_samples.npz"
    wd_npz = wd_dir / "svssm_hmc_samples.npz"
    if nd_npz.exists() and wd_npz.exists():
        nd_sc, _ = load_samples(nd_npz)
        wd_sc, _ = load_samples(wd_npz)
        nd_summ = summarize_marginals(nd_sc)
        wd_summ = summarize_marginals(wd_sc)
        rows = shift_table(nd_summ, wd_summ)
        print_shift_table(rows,
                          f"NO-DATA vs WITH-DATA at T={wd_dir.name.split('_T')[-1]}")
    else:
        if not nd_npz.exists():
            print(f"\n[skip view 1] no-data samples missing: {nd_npz}")
        if not wd_npz.exists():
            print(f"\n[skip view 1] with-data samples missing: {wd_npz}")

    # ---- view (2): phi-grid ----
    if pg_root.exists():
        phi_grid_table(pg_root)
    else:
        print(f"\n[skip view 2] phi-grid root missing: {pg_root}")


if __name__ == "__main__":
    raise SystemExit(main())
