#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=SVSSM_MV_FP_WIDE
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --output=sv_logs/mv_full_phi_wide_%j.out
#SBATCH --error=sv_logs/mv_full_phi_wide_%j.err
#
# Upper-triangular Phi V1 multivariate SVSSM, T=200, WIDE + SHIFTED priors,
# 4 chains in PARALLEL via srun.
#
# Why this configuration:
#   * T=50 was a laptop expedient. The Phase 10 univariate HPC ran at T=100;
#     for the cross-asset-persistence question phi_off needs MORE data than
#     univariate identifiability, not less. T=200 gives the data a real
#     chance to dominate the prior.
#   * Priors are now SHIFTED away from truth so coverage is a real test:
#     if the data pulls the posterior back to truth, we know it's the data,
#     not the prior happening to be centered correctly.
#   * Chains run in parallel via 4 srun tasks (HPC CPUs are slow individually;
#     this matches the Phase 10/11/12 pipeline pattern).
#   * After all 4 chains complete, scripts/stitch_multi_full_phi_chains.py
#     combines them into one (4, draws, .) dataset for the standard
#     diagnostics.
#
# Truths (held fixed, same as every other run):
#   mu = (0.0, -0.3)   phi_diag = (0.95, 0.85)   phi_off = 0.05
#   sigma_eta = (0.3, 0.4)
#
# Wide + shifted priors (this run):
#   mu          ~ N( 1.0, 2.0^2 )    -- centered at +1, truth -0.3 / 0 in tail
#   phi_raw     ~ N( 0.0, 2.0^2 )    -- centered at phi=0, truth phi_raw~1.8
#                                        (truth phi_diag=0.95 well in the tail)
#   phi_off     ~ N( 0.0, 1.0^2 )    -- 5x wider than laptop's 0.2 cap;
#                                        the headline relaxation. Truth 0.05.
#   log_sig^2   ~ N( 0.0, 2.0^2 )    -- truth log_sig^2 ~ -2.4, -1.8 in tail
#
# Walltime budget:
#   Laptop XLA at T=50: 2.33s/step. HPC CPU node ~3x faster -> 0.78s/step.
#   T scaling: T=200 is 4x cost -> ~3.1s/step at T=200 HPC.
#   Per chain: 12000 steps * 3.1s = 37200s = 10.3h.
#   Chains run in PARALLEL -> total wall ~10.3h max + ~5 min overhead.
#   Walltime header 14h covers it with headroom.
#
# Resource footprint:
#   4 srun tasks * 8 cpus = 32 cores. Each TF process pinned to 8 threads
#   via SLURM_CPUS_PER_TASK. Matches Phase 10 pattern.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p sv_logs

echo "===================================================="
echo " SLURM JOB ID: ${SLURM_JOB_ID}"
echo " Node: ${SLURM_NODELIST}"
echo " Running in: ${PWD}"
echo " Date: $(date)"
echo "===================================================="

# ---------------- environment ----------------
module purge
module load python/3/3.10
module load gcc openmpi hdf5/serial hypre

source /home/amreshv/envs/mlenv/bin/activate

# ---------------- threading caps ----------------
# Each srun task gets SLURM_CPUS_PER_TASK threads. Let TF use them all.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTEROP_THREADS=1
# NOT setting XLA_FLAGS=--xla_dump_to (Phase 10 lesson: dumps HLO IR and
# inflated compile + 3 GB of dump).
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "CPUs on node: ${SLURM_CPUS_ON_NODE}"
echo "Tasks: ${SLURM_NTASKS}  CPUs/task: ${SLURM_CPUS_PER_TASK}"

# ---------------- output dirs ----------------
ROOT_OUT="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH"
OUT_BASE="${ROOT_OUT}/svssm_hmc_multivariate_full_phi_hpc_wide_T200"
COMBINED_DIR="${OUT_BASE}"
mkdir -p "${COMBINED_DIR}"

# ---------------- experiment knobs ----------------
T=200
N=64
NUM_CHAINS=4
NUM_BURNIN=2000
NUM_RESULTS=10000

# Truths (held fixed).
MU="0.0,-0.3"
PHI_DIAG="0.95,0.85"
PHI_OFF="0.05"
SIGMA_ETA="0.3,0.4"

# Wide + SHIFTED-from-truth priors.
PRIOR_MU_LOC=1.0          # truth (0, -0.3) sits 0.5 / 0.65 sigma off prior
PRIOR_MU_SCALE=2.0
PRIOR_PHI_RAW_LOC=0.0     # truth phi_raw ~ (1.83, 1.26); prior pulls to phi=0
PRIOR_PHI_RAW_SCALE=2.0
PRIOR_PHI_OFF_SCALE=1.0   # 5x wider than the 0.2 laptop cap
PRIOR_LOG_SIG_LOC=0.0     # truth log_sig^2 ~ (-2.4, -1.8) WAY off prior
PRIOR_LOG_SIG_SCALE=2.0

# Single base_seed shared by all chains so CRN (filter noise) is
# identical across chains -- standard HMC parameter-recovery pattern.
# Chains differ only in init_raw + HMC momentum seed, both of which
# are derived from base_seed + chain_id in the driver.
BASE_SEED=300

# Shared knobs (priors, truths, T, N, etc.). num_chains=4 tells the
# driver the total ensemble size; --chain_id N (added per srun task
# below) tells it to run ONLY that one chain in this task.
SHARED=(
  --d 2 --T "${T}" --N "${N}" --n_lambda 10 --K 10 --L 5
  --step_size 0.05
  --num_burnin "${NUM_BURNIN}"
  --num_results "${NUM_RESULTS}"
  --dispersion 0.15
  --data_seed 42
  --base_seed "${BASE_SEED}"
  --progress_every 500
  --mu "${MU}" --phi_diag "${PHI_DIAG}"
  --phi_off "${PHI_OFF}" --sigma_eta "${SIGMA_ETA}"
  --prior_mu_loc "${PRIOR_MU_LOC}" --prior_mu_scale "${PRIOR_MU_SCALE}"
  --prior_phi_raw_loc "${PRIOR_PHI_RAW_LOC}"
  --prior_phi_raw_scale "${PRIOR_PHI_RAW_SCALE}"
  --prior_phi_off_scale "${PRIOR_PHI_OFF_SCALE}"
  --prior_log_sigma_eta_sq_loc "${PRIOR_LOG_SIG_LOC}"
  --prior_log_sigma_eta_sq_scale "${PRIOR_LOG_SIG_SCALE}"
  --num_chains "${NUM_CHAINS}"
)

echo "T=${T}  N=${N}  num_chains=${NUM_CHAINS} (PARALLEL via srun)"
echo "Truths:  mu=${MU}  phi_diag=${PHI_DIAG}  phi_off=${PHI_OFF}  sigma_eta=${SIGMA_ETA}"
echo "Wide + SHIFTED priors:"
echo "  mu          ~ N(${PRIOR_MU_LOC}, ${PRIOR_MU_SCALE}^2)    (truth in tail)"
echo "  phi_raw     ~ N(${PRIOR_PHI_RAW_LOC}, ${PRIOR_PHI_RAW_SCALE}^2)  (pulls phi to 0)"
echo "  phi_off     ~ N(0, ${PRIOR_PHI_OFF_SCALE}^2)             (wide)"
echo "  log_sig^2   ~ N(${PRIOR_LOG_SIG_LOC}, ${PRIOR_LOG_SIG_SCALE}^2)  (truth in tail)"
echo "Output (combined): ${COMBINED_DIR}"

# ====================================================================
# Launch 4 chains in parallel via srun.
# ====================================================================
t0=$(date +%s)
pids=()
chain_dirs=()

for i in $(seq 0 $((NUM_CHAINS - 1))); do
  CHAIN_DIR="${OUT_BASE}/chain_${i}"
  chain_dirs+=("${CHAIN_DIR}")
  mkdir -p "${CHAIN_DIR}"
  echo "Launching chain ${i}  base_seed=${BASE_SEED}  chain_id=${i}  out=${CHAIN_DIR}"
  srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    python -u -m scripts.exp.exp_hmc_svssm_multivariate_full_phi \
      "${SHARED[@]}" \
      --chain_id "${i}" \
      --out_dir "${CHAIN_DIR}" \
      > "sv_logs/mv_full_phi_wide_chain_${i}_${SLURM_JOB_ID}.out" \
      2> "sv_logs/mv_full_phi_wide_chain_${i}_${SLURM_JOB_ID}.err" &
  pids+=($!)
done

echo
echo "  [wait] waiting for ${#pids[@]} chain processes..."
fail=0
for i in "${!pids[@]}"; do
  pid=${pids[i]}
  if wait "${pid}"; then
    echo "  [ok]   chain ${i} (pid ${pid}) exited 0"
  else
    rc=$?
    echo "  [FAIL] chain ${i} (pid ${pid}) exited ${rc}" >&2
    fail=$((fail + 1))
  fi
done
t_parallel=$(($(date +%s) - t0))
echo "  [wait] all chains done in ${t_parallel}s (parallel)"

if (( fail > 0 )); then
  echo "[error] ${fail} chain(s) failed; aborting before aggregation" >&2
  exit 2
fi

# ====================================================================
# Stitch per-chain npzs into one (4, draws, .) combined npz.
# ====================================================================
echo
echo "================================================"
echo "Stitching ${NUM_CHAINS} chains into ${COMBINED_DIR}"
echo "================================================"
python -u scripts/stitch_multi_full_phi_chains.py \
  --chain_dirs "${chain_dirs[@]}" \
  --out_dir "${COMBINED_DIR}"

# ====================================================================
# Prior-only HMC for Phase-19 confirmation (no filter call, very fast).
# Run 2 chains in parallel too.
# ====================================================================
echo
echo "================================================"
echo "Prior-only HMC (same priors, --no_likelihood)"
echo "================================================"
PRIOR_BASE="${OUT_BASE}_prior_only"
prior_chain_dirs=()
prior_pids=()
for i in 0 1; do
  PD="${PRIOR_BASE}/chain_${i}"
  prior_chain_dirs+=("${PD}")
  mkdir -p "${PD}"
  srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    python -u -m scripts.exp.exp_hmc_svssm_multivariate_full_phi \
      "${SHARED[@]}" \
      --chain_id "${i}" \
      --no_likelihood \
      --out_dir "${PD}" \
      > "sv_logs/mv_full_phi_wide_prior_chain_${i}_${SLURM_JOB_ID}.out" \
      2> "sv_logs/mv_full_phi_wide_prior_chain_${i}_${SLURM_JOB_ID}.err" &
  prior_pids+=($!)
done
for pid in "${prior_pids[@]}"; do wait "${pid}" || true; done

python -u scripts/stitch_multi_full_phi_chains.py \
  --chain_dirs "${prior_chain_dirs[@]}" \
  --out_dir "${PRIOR_BASE}"

# ====================================================================
# Diagnostics + comparison + plots on the combined directories.
# ====================================================================
echo
echo "================================================"
echo "Diagnostics + comparison"
echo "================================================"
python -u scripts/save_diagnostics_multi_full_phi.py \
  --out_dir "${COMBINED_DIR}" > /dev/null 2>&1 || true
python -u scripts/check_prior_dominance.py \
  --out_dir "${COMBINED_DIR}" > /dev/null 2>&1 || true
python -u scripts/compare_prior_vs_posterior.py \
  --prior_dir "${PRIOR_BASE}" \
  --posterior_dir "${COMBINED_DIR}" > /dev/null 2>&1 || true
python -u scripts/plot_prior_vs_posterior_overlay.py \
  --prior_dir "${PRIOR_BASE}" \
  --posterior_dir "${COMBINED_DIR}" > /dev/null 2>&1 || true
python -u scripts/plot_trace_multi_full_phi.py \
  --out_dir "${COMBINED_DIR}" > /dev/null 2>&1 || true
python -u scripts/plot_trace_stationary_cov.py \
  --out_dir "${COMBINED_DIR}" > /dev/null 2>&1 || true

# ====================================================================
# Print everything to the job log for at-a-glance review.
# ====================================================================
echo
echo "================================================"
echo "Diagnostics (combined run)"
echo "================================================"
test -f "${COMBINED_DIR}/svssm_hmc_multi_full_phi_diagnostics.txt" && \
  cat "${COMBINED_DIR}/svssm_hmc_multi_full_phi_diagnostics.txt" || \
  echo "(no diagnostics file)"

echo
echo "================================================"
echo "Prior dominance check"
echo "================================================"
test -f "${COMBINED_DIR}/svssm_hmc_multi_full_phi_prior_dominance.txt" && \
  cat "${COMBINED_DIR}/svssm_hmc_multi_full_phi_prior_dominance.txt" || \
  echo "(no prior dominance file)"

echo
echo "================================================"
echo "Prior-only HMC vs with-likelihood"
echo "================================================"
test -f "${COMBINED_DIR}/svssm_hmc_multi_full_phi_prior_vs_posterior.txt" && \
  cat "${COMBINED_DIR}/svssm_hmc_multi_full_phi_prior_vs_posterior.txt" || \
  echo "(no comparison file)"

t_total=$(($(date +%s) - t0))
echo
echo "================================================"
echo "TOTAL wall: ${t_total}s ($(python -c "print(f'{${t_total}/3600:.2f}')")h)"
echo "Combined output:  ${COMBINED_DIR}"
echo "Prior-only:       ${PRIOR_BASE}"
echo "Per-chain runs:   ${OUT_BASE}/chain_{0,1,2,3}"
echo "================================================"
