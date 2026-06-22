"""Convert a full-Phi driver samples npz into the aggregate-compatible
formats.

The full-Phi driver (exp_hmc_svssm_multivariate_full_phi) saves
constrained samples as named arrays (mu, phi_diag, phi_off,
sigma_eta_sq) because the upper-triangular phi_off block
(d(d-1)/2 entries) is ragged and does not fit the aggregators'
(3, d) param-type x component grid. This converter bridges back:

  d == 1            -> univariate format: samples_constrained
                       (chains, draws, 3) + truth (3,)
                       [phi_off is empty; consume with
                        src.experiments.exp_svssm_hmc_aggregate]
  d >= 2, phi_off=0 -> multivariate format: samples_constrained
                       (chains, draws, 3, d) + truth (3, d)
                       [consume with
                        src.experiments.exp_svssm_hmc_multivariate_aggregate;
                        ONLY valid when the run had zero off-diagonals,
                        otherwise phi_off would be silently dropped]
  d >= 2, phi_off!=0 -> refused (the aggregate format cannot represent
                        the off-diagonal block; extend the aggregator
                        instead of dropping parameters).

Writes <out>/svssm_hmc_samples.npz next to the input by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_npz", type=str, required=True,
                   help="Path to svssm_hmc_multi_full_phi_samples.npz")
    p.add_argument("--out_npz", type=str, default=None,
                   help="Output path (default: svssm_hmc_samples.npz "
                        "next to the input).")
    args = p.parse_args()

    src = Path(args.samples_npz).resolve()
    z = np.load(src)
    mu = z["mu"].astype(np.float32)                  # (c, s, d)
    phi_diag = z["phi_diag"].astype(np.float32)      # (c, s, d)
    phi_off = z["phi_off"].astype(np.float32)        # (c, s, d(d-1)/2)
    sig2 = z["sigma_eta_sq"].astype(np.float32)      # (c, s, d)

    chains, draws, d = mu.shape
    n_off = phi_off.shape[-1]

    mu_truth = z["mu_truth"].astype(np.float32)
    phi_diag_truth = z["phi_diag_truth"].astype(np.float32)
    sig2_truth = z["sigma_eta_sq_truth"].astype(np.float32)
    phi_off_truth = z["phi_off_truth"].astype(np.float32)

    accept = z["accept"] if "accept" in z.files else np.ones((chains, draws))
    log_prob = z["log_prob"] if "log_prob" in z.files else np.zeros((chains, draws))

    if n_off > 0 and (np.any(phi_off != 0.0) or np.any(phi_off_truth != 0.0)):
        raise SystemExit(
            f"Run has a non-trivial phi_off block ({n_off} entries; "
            f"truth={phi_off_truth.tolist()}). The aggregate npz format "
            f"cannot represent it -- converting would silently drop "
            f"parameters. Use the full-phi diagnostics scripts instead, "
            f"or extend the aggregator."
        )

    out = Path(args.out_npz) if args.out_npz else src.parent / "svssm_hmc_samples.npz"

    if d == 1:
        # Univariate format: samples_constrained (chains, draws, 3) ordered
        # [mu, phi, sigma_eta_sq], but truth = [mu, phi, sigma_eta] with the
        # third entry UNSQUARED -- exp_svssm_hmc_aggregate squares it itself
        # (see its "npz stores [mu, phi, sigma_eta]" comment).
        sc = np.concatenate([mu, phi_diag, sig2], axis=-1)   # (c, s, 3)
        truth = np.array([mu_truth[0], phi_diag_truth[0],
                           float(np.sqrt(sig2_truth[0]))],
                          dtype=np.float32)
        np.savez_compressed(
            out, samples_raw=z["samples_raw"], samples_constrained=sc,
            accept=accept, log_prob=log_prob,
            step_size=np.full((chains, draws), np.nan, dtype=np.float32),
            truth=truth,
        )
        print(f"wrote {out}  (univariate format: samples_constrained "
              f"{sc.shape}, truth {truth.shape})")
        print("consume with: python -m src.experiments.exp_svssm_hmc_aggregate "
              f"--samples_npz {out}")
    else:
        # Diagonal-multivariate format: (chains, draws, 3, d)
        sc = np.stack([mu, phi_diag, sig2], axis=2)          # (c, s, 3, d)
        truth = np.stack([mu_truth, phi_diag_truth, sig2_truth], axis=0)  # (3, d)
        np.savez_compressed(
            out, samples_raw=z["samples_raw"], samples_constrained=sc,
            accept=accept, log_prob=log_prob,
            step_size=np.full((chains, draws), np.nan, dtype=np.float32),
            truth=truth,
        )
        print(f"wrote {out}  (multivariate format: samples_constrained "
              f"{sc.shape}, truth {truth.shape})")
        print("consume with: python -m src.experiments."
              f"exp_svssm_hmc_multivariate_aggregate --samples_npz {out}")


if __name__ == "__main__":
    main()
