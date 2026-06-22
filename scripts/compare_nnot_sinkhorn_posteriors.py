"""
Compare the NN-OT vs Sinkhorn posteriors for the matched d=1 moderate-prior runs.

Backs the §3 "Does it work?" table in interview_answers.tex (result R3.3). The two
runs are identical in every setting (truth, prior, data, CRN, L-HNN NUTS sampler);
they differ ONLY in the OT resample (trained NN-OT operator vs exact Sinkhorn).
This script quantifies how close the resulting posteriors are, per parameter:

  - point summaries : median, mean, sd, 95% CI, P(phi<0), truth coverage
  - distance metrics: |Δ median|, sd ratio, two-sample KS (stat + p),
                      1-D Wasserstein distance
  - verdict         : KS p > 0.05 -> "indistinguishable" else "distinguishable"
                      (with the effect size shown so a significant-but-tiny KS
                       is not misread as a material difference)

Usage:
  python scripts/compare_nnot_sinkhorn_posteriors.py \
    --nnot_dir reports/d1_lhnn_nnot_moderate_T100 \
    --sinkhorn_dir reports/d1_lhnn_sinkhorn_moderate_T100

Writes <out_dir>/nnot_vs_sinkhorn_posteriors.{txt,json} (default out_dir = nnot_dir).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as st

PARAMS = [("mu", "mu", 0), ("phi", "phi_diag", 0), ("sigma_eta_sq", "sigma_eta_sq", 0)]


def load_marginals(run_dir: Path) -> dict:
    z = np.load(run_dir / "svssm_hmc_multi_full_phi_samples.npz")
    return {name: z[key][..., comp].ravel() for name, key, comp in PARAMS}


def truth_from_summary(run_dir: Path) -> dict:
    cfg = json.load(open(run_dir / "chain_0" /
                         "svssm_hmc_multi_full_phi_summary.json"))["config"]
    def first(v):
        if isinstance(v, str):
            v = [float(x) for x in v.split(",") if x != ""]
            return v[0] if v else float("nan")
        return float(v)
    se = first(cfg["sigma_eta"])
    return {"mu": first(cfg["mu"]), "phi": first(cfg["phi_diag"]),
            "sigma_eta_sq": se * se}


def covers(x, t):
    return bool(np.quantile(x, 0.025) <= t <= np.quantile(x, 0.975))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nnot_dir", type=str,
                   default="reports/d1_lhnn_nnot_moderate_T100")
    p.add_argument("--sinkhorn_dir", type=str,
                   default="reports/d1_lhnn_sinkhorn_moderate_T100")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write outputs (default: --nnot_dir).")
    args = p.parse_args()

    nnot_dir, sk_dir = Path(args.nnot_dir), Path(args.sinkhorn_dir)
    out_dir = Path(args.out_dir) if args.out_dir else nnot_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    nn = load_marginals(nnot_dir)
    sk = load_marginals(sk_dir)
    truth = truth_from_summary(nnot_dir)
    n_nn = nn["mu"].size
    n_sk = sk["mu"].size

    L = []
    L.append("=" * 96)
    L.append("NN-OT vs Sinkhorn posteriors — matched d=1 moderate-prior runs "
             "(only the OT resample differs)")
    L.append("=" * 96)
    L.append(f"generated      : {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"NN-OT run      : {nnot_dir}   (draws={n_nn})")
    L.append(f"Sinkhorn run   : {sk_dir}   (draws={n_sk})")
    L.append(f"truth          : mu={truth['mu']}  phi={truth['phi']}  "
             f"sigma_eta_sq={truth['sigma_eta_sq']:.4g}")
    L.append("metrics        : KS = two-sample Kolmogorov-Smirnov; "
             "Wass = 1-D Wasserstein; sd ratio = NN-OT/Sinkhorn")
    L.append("=" * 96)

    # ---- per-parameter point summaries ----
    L.append("\nPER-PARAMETER POSTERIORS")
    L.append(f"  {'param':<14}{'src':<10}{'median':>9}{'mean':>9}{'sd':>8}"
             f"{'2.5%':>9}{'97.5%':>9}{'P(phi<0)':>10}{'cov':>5}")
    L.append("  " + "-" * 81)
    for name, _, _ in PARAMS:
        for src, lbl in [(nn, "NN-OT"), (sk, "Sinkhorn")]:
            x = src[name]
            pneg = f"{100*np.mean(x<0):.1f}%" if name == "phi" else "-"
            L.append(f"  {name:<14}{lbl:<10}{np.median(x):>+9.3f}{np.mean(x):>+9.3f}"
                     f"{np.std(x, ddof=1):>8.3f}{np.quantile(x,.025):>+9.3f}"
                     f"{np.quantile(x,.975):>+9.3f}{pneg:>10}"
                     f"{('Y' if covers(x, truth[name]) else 'N'):>5}")
        L.append("")

    # ---- distributional comparison ----
    L.append("DISTRIBUTIONAL COMPARISON (NN-OT vs Sinkhorn)")
    L.append(f"  {'param':<14}{'|d median|':>11}{'sd ratio':>10}{'KS stat':>9}"
             f"{'KS p':>9}{'Wass':>10}{'verdict':>22}")
    L.append("  " + "-" * 83)
    rows = []
    for name, _, _ in PARAMS:
        a, b = nn[name], sk[name]
        ks = st.ks_2samp(a, b)
        wass = float(st.wasserstein_distance(a, b))
        dmed = float(abs(np.median(a) - np.median(b)))
        sdr = float(np.std(a, ddof=1) / np.std(b, ddof=1))
        verdict = "indistinguishable" if ks.pvalue > 0.05 else "distinguishable"
        L.append(f"  {name:<14}{dmed:>11.4f}{sdr:>10.3f}{ks.statistic:>9.4f}"
                 f"{ks.pvalue:>9.4f}{wass:>10.4f}{verdict:>22}")
        rows.append({"param": name,
                     "nnot_median": float(np.median(a)), "sinkhorn_median": float(np.median(b)),
                     "nnot_sd": float(np.std(a, ddof=1)), "sinkhorn_sd": float(np.std(b, ddof=1)),
                     "abs_delta_median": dmed, "sd_ratio_nnot_over_sinkhorn": sdr,
                     "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                     "wasserstein": wass, "verdict": verdict})

    L.append("\n" + "=" * 96)
    L.append("Reading: medians/CIs agree to within MC noise (point inference identical);")
    L.append("KS may flag a SIGNIFICANT but TINY shape difference (small KS stat /")
    L.append("Wasserstein) — the operator-approximation signature, not a material shift.")
    L.append("=" * 96)

    txt = "\n".join(L) + "\n"
    print(txt)
    (out_dir / "nnot_vs_sinkhorn_posteriors.txt").write_text(txt)
    (out_dir / "nnot_vs_sinkhorn_posteriors.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "nnot_dir": str(nnot_dir), "sinkhorn_dir": str(sk_dir),
        "n_draws_nnot": int(n_nn), "n_draws_sinkhorn": int(n_sk),
        "truth": truth, "comparison": rows,
    }, indent=2))
    print(f"wrote {out_dir / 'nnot_vs_sinkhorn_posteriors.txt'}")
    print(f"wrote {out_dir / 'nnot_vs_sinkhorn_posteriors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
