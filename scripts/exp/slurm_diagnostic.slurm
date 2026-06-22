#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=SVSSM_diag
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=3
#SBATCH --time=05:30:00
#SBATCH --export=ALL
#SBATCH --output=sv_logs/diag_%j.out
#SBATCH --error=sv_logs/diag_%j.err
#
# DIAGNOSTIC: measure actual HPC per-step cost for the v7 windowed kernel
# (dense mass matrix + corrected direction) at two T values, so we can
# correctly budget the T-sweep production run.
#
# The previous T=500 run hit ~200 s/HMC step on HPC --- ~20x my laptop
# extrapolation. Two suspects:
#   (a) tf.linalg.expm inside the LEDH filter (used by integrator="exp")
#       may not fully fuse under XLA at large T, making compile + step
#       both grow super-linearly with T.
#   (b) HPC OMP_NUM_THREADS=1 vs laptop multi-threaded BLAS for the
#       matrix ops inside the filter.
#
# We can't distinguish those without measurements. So: run a short job
# at T=50 AND T=100 with a single chain each, measure total wall, and
# decompose compile vs per-step from the two T values.
#
# Per-chain budget here: 30 burnin + 100 sampling = 130 HMC steps.
# Expected wall (linear extrapolation from the T=500 -> 200 s/step
# observation):
#   T=50   ~20 s/step * 130 = 2600 s = 43 min  (compile-dominated for small steps)
#   T=100  ~40 s/step * 130 = 5200 s = 87 min
# If actual times are 5-10x smaller (i.e., T=500 was pathological,
# e.g. expm graph blowup at the largest T), we learn that too --- the
# point is to MEASURE, not extrapolate.
# --time=02:30:00 gives ~1 h safety margin on the worst-case T=100 estimate
# (the previous T=500 misfire taught us to over-provision).
#
# Both runs in parallel. 2 srun tasks * 3 cores = 6 cores; well within
# any partition limit.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p sv_logs

echo "===================================================="
echo " SLURM JOB ID: ${SLURM_JOB_ID}"
echo " Node: ${SLURM_NODELIST}"
echo " Date: $(date)"
echo " Purpose: measure HPC per-step cost at T=50 and T=100"
echo "===================================================="

# ---------------- environment ----------------
module purge
module load python/3/3.10
module load gcc openmpi hdf5/serial hypre

source /home/amreshv/envs/mlenv/bin/activate

# ---------------- threading (matches production SLURM) ----------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "CPUs on node: ${SLURM_CPUS_ON_NODE}"
echo "Tasks: ${SLURM_NTASKS}  CPUs/task: ${SLURM_CPUS_PER_TASK}"

# ---------------- experiment knobs ----------------
N=64

ROOT_OUT="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH"
DIAG_TAG="svssm_hmc_diag_perstep"

# Single-chain runs with the SAME kernel as production.
# --use_windowed_adaptive defaults to the v7 DENSE mass matrix; add
# --diagonal_mass to opt into the diagonal-only variant.
SHARED=(
  --num_chains 1
  --mu 0.0 --phi 0.95 --sigma_eta 0.3
  --N "${N}"
  --n_lambda 10
  --K 10
  --L 5
  --step_size 0.01
  --num_burnin 30                # tiny burnin: init=5, windows [4, 6, 15*]
                                 # (4 sample_chain compiles for warmup +
                                 # 1 for sampling = 5 total compiles)
  --num_results 100              # 100 sample steps after burnin
  --dispersion 0.05
  --data_seed 42
  --base_seed 300
  --progress_every 10            # frequent so we see if it's hanging
  --use_windowed_adaptive
)

WIDE_PRIOR_ARGS=(
  --prior_mu_loc 2.0
  --prior_mu_scale 3.0
  --prior_phi_raw_loc 0.0
  --prior_phi_raw_scale 2.0
  --prior_log_sigma_eta_sq_loc 1.5
  --prior_log_sigma_eta_sq_scale 3.0
)

# Two parallel tasks: one at T=50, one at T=100.
T_VALUES=(50 100)
for T in "${T_VALUES[@]}"; do
  mkdir -p "${ROOT_OUT}/${DIAG_TAG}_T${T}/chains"
done

t0=$(date +%s)
pids=()
for T in "${T_VALUES[@]}"; do
  OUT_DIR="${ROOT_OUT}/${DIAG_TAG}_T${T}"
  echo "Launching diagnostic T=${T} chain 0"
  srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    python -u -m scripts.exp.exp_hmc_svssm \
      "${SHARED[@]}" "${WIDE_PRIOR_ARGS[@]}" \
      --T "${T}" \
      --out_dir "${OUT_DIR}" \
      --chain_id 0 \
      > "sv_logs/diag_T${T}_${SLURM_JOB_ID}.out" \
      2> "sv_logs/diag_T${T}_${SLURM_JOB_ID}.err" &
  pids+=($!)
done

echo
echo "  [wait] waiting for ${#pids[@]} diagnostic chains..."
for pid in "${pids[@]}"; do wait "${pid}" || true; done
t_total=$(($(date +%s) - t0))
echo "  [wait] both diagnostics done in ${t_total}s"

echo
echo "================================================"
echo "Diagnostic results: extract per-step from the chain logs."
echo "================================================"
for T in "${T_VALUES[@]}"; do
  log="sv_logs/diag_T${T}_${SLURM_JOB_ID}.out"
  echo
  echo "--- T=${T} ---"
  echo "Last few lines (look for 'chain done in X.Xs'):"
  tail -25 "${log}" || echo "  (log not found)"
done

echo
echo "================================================"
echo "Total wall: ${t_total}s"
echo "Done."
echo "================================================"
