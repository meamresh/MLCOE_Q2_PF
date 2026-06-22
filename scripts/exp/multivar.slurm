#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=SVSSM_HMC
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --output=sv_logs/master_%j.out
#SBATCH --error=sv_logs/master_%j.err

cd "${SLURM_SUBMIT_DIR}"
mkdir -p sv_logs

echo "===================================================="
echo " SLURM JOB ID: ${SLURM_JOB_ID}"
echo " Node: ${SLURM_NODELIST}"
echo " Running in: ${PWD}"
echo "===================================================="

# ---------------- environment ----------------
module purge
module load python/3/3.10
module load gcc openmpi hdf5/serial hypre

source /home/amreshv/envs/mlenv/bin/activate

# ---------------- threading (avoid oversubscription) ----------------
# Each chain runs in its own srun step with up to 9 cores visible; cap TF/BLAS.
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

# ---------------- experiment knobs (edit here) ----------------
NUM_CHAINS=4
OUT_DIR="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc_T100_multivar"

COMMON=(
  --out_dir "${OUT_DIR}"
  --num_chains "${NUM_CHAINS}"
  --d 2
  --T 100
  --N 64
  --n_lambda 10
  --K 10
  --L 5
  --prior_mu_loc 2.0
  --prior_mu_scale 3.0
  --prior_phi_raw_loc 0.0
  --prior_phi_raw_scale 2.0
  --prior_log_sigma_eta_sq_loc 1.5
  --prior_log_sigma_eta_sq_scale 3.0
  --step_size 0.01
  --num_burnin 200
  --num_results 2500
  --dispersion 0.05
  --data_seed 42
  --base_seed 300
  --progress_every 50
)

mkdir -p "${OUT_DIR}/chains"

# ---------------- one Python process per chain (parallel) ----------------
for CHAIN_ID in $(seq 0 $((NUM_CHAINS - 1))); do
  echo "Launching SVSSM chain ${CHAIN_ID}"

  srun --exclusive \
       -N1 \
       -n1 \
       -c"${SLURM_CPUS_PER_TASK}" \
       --cpu-bind=cores \
    python -u -m scripts.exp.exp_hmc_svssm_multivariate \
      "${COMMON[@]}" \
      --chain_id "${CHAIN_ID}" \
      > "sv_logs/svssm_chain_${CHAIN_ID}.out" \
      2> "sv_logs/svssm_chain_${CHAIN_ID}.err" &

done
wait

echo "================================================"
echo "Aggregating chains -> ${OUT_DIR}"
echo "================================================"

python -u -m src.experiments.hmc_svssm \
  "${COMMON[@]}" \
  --aggregate \
  >> "sv_logs/svssm_aggregate_${SLURM_JOB_ID}.out" \
  2>> "sv_logs/svssm_aggregate_${SLURM_JOB_ID}.err"

python -u -m scripts.exp.exp_hmc_svssm_multivariate \
    --samples_npz "$NPZ" \
    --out_dir "$OUT_DIR" \
    >> sv_logs/aggregate.out \
    2>> sv_logs/aggregate.err


echo "================================================"
echo "Done. Outputs under ${OUT_DIR}/"
echo "  svssm_hmc_samples.npz  svssm_hmc_results.txt  plots"
echo "================================================"
