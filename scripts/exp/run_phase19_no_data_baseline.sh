#!/bin/bash
# Phase 19, experiment A: NO-DATA BASELINE.
#
# Run the v7 windowed-adaptive HMC kernel with the LIKELIHOOD term ZEROED.
# The chain samples from the prior alone. The resulting posterior is the
# empirical attractor location — wherever the prior + sampler geometry
# concentrate.
#
# Purpose: compare against the Phase 18 with-data posterior at the same
# truth + same prior + same kernel. If with-data and without-data
# posteriors overlap, the likelihood is NOT moving the chain — strong
# evidence the data is uninformative at this regime. If they differ
# (especially along mu and sigma_eta^2 — Phase 18.11 showed those CIs
# shrink with T), the likelihood IS informing.
#
# This is the experiment the user's critique gestured at: "Check the
# prior-only / no-data baseline directly. Run the sampler with the
# likelihood switched off. Where does it land? That's your attractor
# location empirically."
#
# Runtime: no filter is called, so each step is ~microseconds.
# 4 chains x 2200 steps should finish in <5 minutes on laptop.

set -uo pipefail
cd "/Users/amreshverma/Documents/Random Work/MLCOE_Q2_PF"
export PYTHONPATH=.

OUT_DIR="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/phase19_attractor/no_data_baseline"
mkdir -p "${OUT_DIR}"

LOG="${OUT_DIR}/run.log"

echo "=== Phase 19A no-data baseline starting at $(date) ==="
echo "Output: ${OUT_DIR}"

t0=$(date +%s)

# Same kernel + prior as Phase 16/18 wide-shifted sweep.
# T=100 to match the Phase 18.10/18.11 attractor experiments.
# --no_likelihood => chain samples the prior only (no filter call).
python -u -m scripts.exp.exp_hmc_svssm \
  --no_likelihood \
  --mu 0.0 --phi 0.95 --sigma_eta 0.3 \
  --T 100 --N 64 \
  --n_lambda 10 --L 5 --step_size 0.01 \
  --num_chains 4 \
  --num_burnin 200 \
  --num_results 2000 \
  --dispersion 0.2 \
  --data_seed 42 --base_seed 300 \
  --progress_every 200 \
  --prior_mu_loc 2.0 --prior_mu_scale 3.0 \
  --prior_phi_raw_loc 0.0 --prior_phi_raw_scale 2.0 \
  --prior_log_sigma_eta_sq_loc 1.5 --prior_log_sigma_eta_sq_scale 3.0 \
  --use_windowed_adaptive \
  --out_dir "${OUT_DIR}" \
  2>&1 | tee "${LOG}"

dt=$(( $(date +%s) - t0 ))
echo
echo "=== Phase 19A no-data baseline done in ${dt}s ==="
echo "Results: ${OUT_DIR}/svssm_hmc_results.txt"
echo "Compare to with-data posterior at:"
echo "  reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc_sweep_wide_T100/"
echo "(or the Phase 18 attractor run if different — see section3 §8 for paths)."
