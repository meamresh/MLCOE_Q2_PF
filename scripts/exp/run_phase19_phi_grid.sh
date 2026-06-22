#!/bin/bash
# Phase 19, experiment B: phi-GRID SIMULATION STUDY.
#
# Three datasets, one each at phi in {0.7, 0.85, 0.95}, sigma_eta=0.3,
# mu=0.0. Same wide-shifted prior. Same v7 windowed-adaptive kernel.
#
# Purpose: directly test whether the LEDH-PF-PF HMC pipeline can recover
# phi at DIFFERENT true values. The constant-volatility attractor lives
# at phi=1, sigma_eta^2=0. If our chain ALWAYS lands near phi ~ 0.99
# regardless of truth, the attractor is in control. If it tracks each
# true phi (~0.7, ~0.85, ~0.95), the likelihood is informing.
#
# This is THE single most diagnostic experiment for the attractor
# concern — much sharper than the Phase 18.11 T-sweep (which only
# established CI(phi) doesn't shrink with T; consistent with attractor
# OR with "need more T").
#
# Runtime: T=100 (default, matches Phase 18 directly), N=64, 2 chains x
# 800 steps/cell, 3 cells, sequential. Per-step ~4-5s on laptop, so
# ~75 min per chain x 2 chains x 3 cells ~= ~10h sequential (overnight).
# Override: T_FORCED=50 for ~5h proper; T_FORCED=20 for ~3h scout.

set -uo pipefail
cd "/Users/amreshverma/Documents/Random Work/MLCOE_Q2_PF"
export PYTHONPATH=.

# ---------------- knobs ----------------
T_FORCED=${T_FORCED:-100}     # T per cell (override: T_FORCED=50 or T_FORCED=20)
N_FORCED=${N_FORCED:-64}
NUM_CHAINS=${NUM_CHAINS:-2}
NUM_BURNIN=${NUM_BURNIN:-200}
NUM_RESULTS=${NUM_RESULTS:-600}
PHI_GRID=(0.70 0.85 0.95)

ROOT_OUT="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/phase19_attractor/phi_grid"
mkdir -p "${ROOT_OUT}"

echo "=== Phase 19B phi-grid starting at $(date) ==="
echo "T=${T_FORCED} N=${N_FORCED} chains=${NUM_CHAINS} burn=${NUM_BURNIN} samples=${NUM_RESULTS}"
echo "phi values: ${PHI_GRID[*]}"
echo "Output root: ${ROOT_OUT}"
echo

grand_t0=$(date +%s)

for PHI in "${PHI_GRID[@]}"; do
  CELL_OUT="${ROOT_OUT}/phi_${PHI}"
  mkdir -p "${CELL_OUT}"
  LOG="${CELL_OUT}/run.log"

  echo "--- phi=${PHI} starting at $(date +%H:%M:%S) ---"
  cell_t0=$(date +%s)

  # Same wide-shifted prior as Phase 16/17/18 sweep. Truth varies in phi;
  # mu and sigma_eta held at their Phase 18 values so each cell is a
  # pure A/B on phi-recovery.
  python -u -m scripts.exp.exp_hmc_svssm \
    --mu 0.0 --phi "${PHI}" --sigma_eta 0.3 \
    --T "${T_FORCED}" --N "${N_FORCED}" \
    --n_lambda 10 --K 10 --L 5 --step_size 0.01 \
    --num_chains "${NUM_CHAINS}" \
    --num_burnin "${NUM_BURNIN}" \
    --num_results "${NUM_RESULTS}" \
    --dispersion 0.1 \
    --data_seed 42 --base_seed 300 \
    --progress_every 100 \
    --prior_mu_loc 2.0 --prior_mu_scale 3.0 \
    --prior_phi_raw_loc 0.0 --prior_phi_raw_scale 2.0 \
    --prior_log_sigma_eta_sq_loc 1.5 --prior_log_sigma_eta_sq_scale 3.0 \
    --use_windowed_adaptive \
    --out_dir "${CELL_OUT}" \
    > "${LOG}" 2>&1

  cell_dt=$(( $(date +%s) - cell_t0 ))
  echo "--- phi=${PHI} done in ${cell_dt}s ---"
done

grand_dt=$(( $(date +%s) - grand_t0 ))
echo
echo "=== Phase 19B phi-grid done in ${grand_dt}s ($(awk "BEGIN{print ${grand_dt}/60}") min) ==="
echo

# ---------------- summary: posterior medians across the grid ----------------
echo "========= phi-GRID POSTERIOR MEDIANS (constrained scale) ========="
printf "%10s | %12s %12s %12s\n" "truth_phi" "post_mu" "post_phi" "post_sigma_eta"
echo "------------------------------------------------------"
for PHI in "${PHI_GRID[@]}"; do
  CELL_OUT="${ROOT_OUT}/phi_${PHI}"
  RESULTS="${CELL_OUT}/svssm_hmc_results.txt"
  if [[ -f "${RESULTS}" ]]; then
    # Pull medians from the standard svssm_hmc_results.txt format.
    # Expected layout: a "Posterior" or similar block with median per param.
    MU_MED=$(grep -E "^\s*mu\s" "${RESULTS}" | awk '{print $4}' | head -1)
    PHI_MED=$(grep -E "^\s*phi\s" "${RESULTS}" | awk '{print $4}' | head -1)
    SIG_MED=$(grep -E "^\s*sigma_eta\s" "${RESULTS}" | awk '{print $4}' | head -1)
    printf "%10s | %12s %12s %12s\n" "${PHI}" "${MU_MED:-?}" "${PHI_MED:-?}" "${SIG_MED:-?}"
  else
    printf "%10s | %12s %12s %12s\n" "${PHI}" "FAIL" "FAIL" "FAIL"
  fi
done
echo
echo "Interpretation:"
echo "  - If post_phi tracks truth_phi: filter informing on phi (good)."
echo "  - If post_phi piles up at ~0.99 across all 3: attractor dominates."
echo "  - If post_phi is in between but biased high: weak data + boundary pull."
echo
echo "Compare also against no-data baseline at:"
echo "  reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/phase19_attractor/no_data_baseline/"
