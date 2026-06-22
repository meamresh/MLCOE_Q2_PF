#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=SVSSM_Tsweep
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --output=sv_logs/master_%j.out
#SBATCH --error=sv_logs/master_%j.err
#
# T-sweep parameter-recovery experiment: T in {50, 100, 200}, 4 chains
# per T, all on the SAME wide-shifted prior. Canonical SV-model truth
# (mu=0, phi=0.95, sigma_eta=0.3). Prior is shifted 2-sigma off truth
# but with wide scale so truth is still in the prior bulk (matches the
# v4 laptop validation benchmark exactly --- this is the HPC scale-up
# of that test).
#
# Motivation: the v4 windowed kernel showed that posterior-prior agreement
# at T=50 was DUE TO THE OLD KERNEL not the data. With the new kernel,
# even T=50 produces data-informed posteriors (medians moved 60--97%
# of the way from prior to truth at the laptop scale). We don't know how
# much T is needed for "clean" recovery on each parameter. This sweep
# answers that, in one run.
#
# Diagnostic table (after the run):
#
#   all 3 T at truth (within +/- bias)  -> kernel + data sufficient at T=50
#   only T=100,200 at truth             -> T=50 still data-thin
#   only T=200 at truth                 -> HRS quasi-likelihood needs >100 obs
#   none at truth                       -> kernel issue we missed
#
# The single wide-shifted prior holds T as the only varying axis, which
# is what the sweep needs. TIGHT-shifted was dropped from this run (and
# can be re-added with a follow-up sweep at the winning T if needed).
#
# 12 srun tasks launched concurrently:
#   4 chains for T=50
#   4 chains for T=100
#   4 chains for T=200
# After all 12 finish, three aggregator + Vehtari diagnostic passes.
#
# Walltime budget with the v7 windowed-proper kernel (dense mass).
# Per-step costs MEASURED on HPC via slurm_diagnostic.slurm (job 3244):
#   per-step (HPC, T=50)    4.10 s   (130 steps in 532 s)
#   per-step (HPC, T=100)   7.25 s   (130 steps in 942 s)
# Linear fit: per_step(T) ~ 1.0 + 0.063 * T s/step
#   per-step (HPC, T=200) ~13.4 s   (extrapolated)
# (The earlier T=500 = ~200 s/step observation is OUTSIDE the linear
#  regime --- likely XLA compile/memory blowup at large unrolled graph
#  sizes. Not relevant for the T <= 200 sweep.)
#
# total per chain  200 burn + 2500 sample = 2700 steps:
#   T=50:    2700 * 4.10 s = 11100 s = 3.1 h per chain
#   T=100:   2700 * 7.25 s = 19600 s = 5.4 h per chain
#   T=200:   2700 *13.40 s = 36200 s =10.0 h per chain  <-- bottleneck
# Total wall (all 12 chains in parallel) = max = 10.0 h.
# Compile overhead: ~5 sample_chain calls per chain x ~10-30 s = baked
# into the linear fit's intercept (a = 1.0 s/step amortized over 130 steps).
# margin: --time=14:00:00 gives 4 h headroom for T=200 surprises.
#
# Sample-count rationale (Vehtari thresholds bulk-ESS / tail-ESS >= 400):
#   4 chains x 2500 samples = 10000 total.
#   At lag ~10 (pessimistic, no mass matrix help): 10000 / 10 = ESS ~1000
#   At lag ~5  (realistic with diagonal mass matrix):           ESS ~2000
#   At lag ~3  (optimistic):                                    ESS ~3300
#   Either way clears Vehtari thresholds with margin.
#
# Resource: 12 tasks * 3 cores/task = 36 cores; exactly fills a 36-core
# node. Each task is single-threaded (OMP=1) so 3 cores per task is
# generous for compile + XLA work.

set -uo pipefail

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

# ---------------- threading (avoid oversubscription) ----------------
# Each chain runs in its own srun step with 3 cores visible; cap TF/BLAS.
# 12 srun tasks * 3 cores = 36 cores exactly (full 36-core node).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# DO NOT set XLA_FLAGS=--xla_dump_to (Phase 10 lesson: that's a HLO
# dump flag, not a cache; eats 3+ GB and inflates compile time).

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "CPUs on node: ${SLURM_CPUS_ON_NODE}"
echo "Tasks: ${SLURM_NTASKS}  CPUs/task: ${SLURM_CPUS_PER_TASK}"

# ---------------- experiment knobs (edit here) ----------------
NUM_CHAINS=4
N=64
T_VALUES=(50 100 200)

ROOT_OUT="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH"
SWEEP_TAG="svssm_hmc_sweep_wide"   # one subdir per T under this tag

# Shared HMC + data knobs (identical across all T). Truth is the canonical
# SVSSM benchmark (financial-SV realistic):
#   mu=0      (zero-mean log-volatility)
#   phi=0.95  (high persistence, typical of daily SV)
#   sigma_eta=0.3 (typical volatility of log-volatility)
# NOTE: --T is intentionally NOT here; it varies per launch below.
SHARED=(
  --num_chains "${NUM_CHAINS}"
  --mu 0.0 --phi 0.95 --sigma_eta 0.3      # canonical truth (Phases 5-12 baseline)
  --N "${N}"
  --n_lambda 10
  --K 10
  --L 5
  --step_size 0.01                          # initial; v4 kernel adapts from here
  --num_burnin 200                          # gives windowed scheme room for
                                            # [20, 22, 45, 113]-step phases
  --num_results 2500                        # targets bulk-ESS >= 1000 even at
                                            # autocorrelation lag <= 10
                                            # (Vehtari threshold = 400)
  --dispersion 0.05                         # small: start near truth
  --data_seed 42
  --base_seed 300
  --progress_every 100                      # less log spam at long T
  # Windowed-proper kernel (v4): Stan-style expanding-window scheme,
  # 5 safeguards (NaN guard, variance clip [0.1,10], step clip [1e-4,0.5],
  # last window doubles as term, log_averaging_step smoothing).
  # Per-step cost ~= vanilla HMC at long T (compile overhead amortized).
  # See section3_kernel_upgrade.tex §4 for the full debugging story.
  --use_windowed_adaptive
)

# WIDE SHIFTED prior: same shift as TIGHT but wider scales so truth is
# still in the prior bulk despite the shift. This is the prior used in
# the v4 laptop validation, so the HPC sweep results are directly
# comparable to the laptop A/B.
# Prior modes (after shift):
#   mu_loc=2.0,       scale=3.0    -> mu mode at 2.0  (truth 0.0  is at -0.67 sigma)
#   phi_raw_loc=0.0,  scale=2.0    -> phi mode at 0.0 (truth phi_raw=1.83 is +0.92 sigma)
#   log_sig2_loc=1.5, scale=3.0    -> sig² mode at e^1.5=4.48
#                                     (truth log_sig²=-2.41 is at -1.3 sigma)
# All three: truth well within the prior bulk. "Uncertainly wrong" prior.
WIDE_PRIOR_ARGS=(
  --prior_mu_loc 2.0
  --prior_mu_scale 3.0
  --prior_phi_raw_loc 0.0
  --prior_phi_raw_scale 2.0
  --prior_log_sigma_eta_sq_loc 1.5
  --prior_log_sigma_eta_sq_scale 3.0
)

# Per-T output directories.
for T in "${T_VALUES[@]}"; do
  mkdir -p "${ROOT_OUT}/${SWEEP_TAG}_T${T}/chains"
done

# ---------------- 12 chains in parallel (4 chains x 3 T values) ----------------
t0=$(date +%s)
pids=()

for T in "${T_VALUES[@]}"; do
  OUT_DIR="${ROOT_OUT}/${SWEEP_TAG}_T${T}"
  for CHAIN_ID in $(seq 0 $((NUM_CHAINS - 1))); do
    echo "Launching T=${T} chain ${CHAIN_ID}"
    srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
      python -u -m scripts.exp.exp_hmc_svssm \
        "${SHARED[@]}" "${WIDE_PRIOR_ARGS[@]}" \
        --T "${T}" \
        --out_dir "${OUT_DIR}" \
        --chain_id "${CHAIN_ID}" \
        > "sv_logs/sweep_T${T}_chain_${CHAIN_ID}.out" \
        2> "sv_logs/sweep_T${T}_chain_${CHAIN_ID}.err" &
    pids+=($!)
  done
done

echo
echo "  [wait] waiting for ${#pids[@]} chain processes (4 chains x 3 T values)..."
fail=0
for i in "${!pids[@]}"; do
  pid=${pids[i]}
  if wait "${pid}"; then
    echo "  [ok]   task ${i} (pid ${pid}) exited 0"
  else
    rc=$?
    echo "  [FAIL] task ${i} (pid ${pid}) exited ${rc}" >&2
    fail=$((fail + 1))
  fi
done
t_parallel=$(($(date +%s) - t0))
echo "  [wait] all chains done in ${t_parallel}s (parallel)"

if (( fail > 0 )); then
  echo "[error] ${fail} chain(s) failed; running aggregators anyway on whatever completed" >&2
fi

# ---------------- Aggregators + Vehtari diagnostics (per T) ----------------
# Two aggregator passes per T:
#   1) the experiment driver's built-in --aggregate (basic split-Rhat table)
#   2) src.experiments.exp_svssm_hmc_aggregate for the full rank-Rhat +
#      bulk-ESS + tail-ESS table (Vehtari et al. 2021).
# Vehtari thresholds: rank-Rhat <= 1.01, bulk-ESS >= 400, tail-ESS >= 400.
for T in "${T_VALUES[@]}"; do
  OUT_DIR="${ROOT_OUT}/${SWEEP_TAG}_T${T}"
  echo
  echo "================================================"
  echo "Aggregating T=${T} -> ${OUT_DIR}"
  echo "================================================"
  python -u -m scripts.exp.exp_hmc_svssm \
    "${SHARED[@]}" "${WIDE_PRIOR_ARGS[@]}" \
    --T "${T}" --out_dir "${OUT_DIR}" --aggregate \
    > "sv_logs/sweep_T${T}_aggregate_${SLURM_JOB_ID}.out" \
    2> "sv_logs/sweep_T${T}_aggregate_${SLURM_JOB_ID}.err"

  echo
  echo "================================================"
  echo "Vehtari diagnostics: T=${T}"
  echo "================================================"
  python -u -m src.experiments.exp_svssm_hmc_aggregate \
    --samples_npz "${OUT_DIR}/svssm_hmc_samples.npz" \
    > "sv_logs/sweep_T${T}_vehtari_${SLURM_JOB_ID}.out" \
    2> "sv_logs/sweep_T${T}_vehtari_${SLURM_JOB_ID}.err"
done

echo
echo "================================================"
echo "Done. Outputs:"
for T in "${T_VALUES[@]}"; do
  echo "  T=${T}:  ${ROOT_OUT}/${SWEEP_TAG}_T${T}/svssm_hmc_results.txt"
done
echo "Total wall: $(( $(date +%s) - t0 ))s"
echo "================================================"
