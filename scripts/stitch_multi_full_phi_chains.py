"""Stitch per-chain full-Phi samples npz files into one (chains, draws, .)
combined npz, so downstream diagnostics see a 4-chain dataset.

Usage:
    python scripts/stitch_multi_full_phi_chains.py \
        --chain_dirs chain_0 chain_1 chain_2 chain_3 \
        --out_dir <combined_dir>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chain_dirs", type=str, nargs="+", required=True,
                   help="Directories holding per-chain "
                        "svssm_hmc_multi_full_phi_samples.npz files.")
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npzs = []
    for d in args.chain_dirs:
        path = os.path.join(d, "svssm_hmc_multi_full_phi_samples.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        npzs.append(np.load(path))

    # Each per-chain npz is shape (1, draws, ...) -> concat on axis 0.
    def cat(key):
        return np.concatenate([n[key] for n in npzs], axis=0)

    out = {
        "samples_raw":  cat("samples_raw"),
        "mu":           cat("mu"),
        "phi_diag":     cat("phi_diag"),
        "phi_off":      cat("phi_off"),
        "sigma_eta_sq": cat("sigma_eta_sq"),
        "accept":       cat("accept"),
        "log_prob":     cat("log_prob"),
        "mu_truth":           npzs[0]["mu_truth"],
        "phi_diag_truth":     npzs[0]["phi_diag_truth"],
        "phi_off_truth":      npzs[0]["phi_off_truth"],
        "sigma_eta_sq_truth": npzs[0]["sigma_eta_sq_truth"],
        "Phi_truth":          npzs[0]["Phi_truth"],
    }
    out_path = os.path.join(args.out_dir,
                              "svssm_hmc_multi_full_phi_samples.npz")
    np.savez_compressed(out_path, **out)
    print(f"saved {out_path}  shapes: mu={out['mu'].shape}  "
          f"phi_off={out['phi_off'].shape}")

    # Also stitch summary.jsons into one combined summary.
    rows = []
    for i, d in enumerate(args.chain_dirs):
        sp = os.path.join(d, "svssm_hmc_multi_full_phi_summary.json")
        if os.path.exists(sp):
            rows.append({"chain_dir": d, "summary": json.load(open(sp))})
    if rows:
        with open(os.path.join(args.out_dir,
                                  "svssm_hmc_multi_full_phi_summary.json"),
                  "w") as f:
            json.dump({"num_chains": len(args.chain_dirs),
                        "stitched_from": args.chain_dirs,
                        "per_chain_summaries": rows,
                        # Carry the first chain's config so downstream
                        # diagnostics scripts that read summary.json["config"]
                        # still work (priors and truths match across chains).
                        "config": rows[0]["summary"]["config"]},
                       f, indent=2)
        print(f"saved combined summary.json")


if __name__ == "__main__":
    main()
