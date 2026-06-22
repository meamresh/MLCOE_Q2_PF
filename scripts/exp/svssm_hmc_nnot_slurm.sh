#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=SVSSM_NNOT_HMC
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --output=sv_logs/master_nnot_%j.out
#SBATCH --error=sv_logs/master_nnot_%j.err

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
# Each chain runs in its own srun step with up to 9 cores visible.
# NN-OT XLA leans on a small Keras forward + an XLA-fused outer loop;
# capping BLAS to 1 matches the Sinkhorn template. If you see CPU under-
# utilisation, try TF_NUM_INTRAOP_THREADS=4 (still safe at 9 cpus/task).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# Persistent XLA cache across the chain processes on the same node
# (each chain still pays the first compile; subsequent re-runs with the
# same (T, N, arch, graph_mode) re-use the cache). Comment out to disable.
export XLA_FLAGS="--xla_dump_to=${SLURM_SUBMIT_DIR}/sv_logs/xla_dump_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "CPUs on node: ${SLURM_CPUS_ON_NODE}"
echo "Tasks: ${SLURM_NTASKS}  CPUs/task: ${SLURM_CPUS_PER_TASK}"

# ---------------- experiment knobs (edit here) ----------------
NUM_CHAINS=4
OUT_DIR="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc_nnot_slurm"

# Architecture + checkpoint.
# Default: DeepONet trained at N=64 (Phase 3 checkpoint).
# For N=256 deployment use the Phase 9 retrained checkpoint:
#   ARCH=deeponet
#   CKPT="reports/.../section2_phase9/deeponet_N256.weights.h5"
ARCH="deeponet"
CKPT="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase3/deeponet.weights.h5"

COMMON=(
  --arch "${ARCH}"
  --graph_mode xla
  --checkpoint_path "${CKPT}"
  --n_ridges 64
  --grad_window 4
  --out_dir "${OUT_DIR}"
  --num_chains "${NUM_CHAINS}"
  --mu 0.0 --phi 0.95 --sigma_eta 0.3
  --T 100
  --N 64
  --n_lambda 10
  --L 5
  --prior_mu_loc 0.0
  --prior_mu_scale 1.0
  --prior_phi_raw_loc 2.0
  --prior_phi_raw_scale 0.5
  --prior_log_sigma_eta_sq_loc -2.0
  --prior_log_sigma_eta_sq_scale 1.0
  --step_size 0.05
  --num_burnin 2000
  --num_results 10000
  --dispersion 0.15
  --data_seed 42
  --base_seed 300
  --progress_every 50
)

mkdir -p "${OUT_DIR}/chains"

# Sanity-check the checkpoint exists before launching anything.
if [[ ! -f "${CKPT}" ]]; then
  echo "[error] checkpoint not found: ${CKPT}" >&2
  echo "        upload it from a local Phase-3/4/7/9 training run before submitting." >&2
  exit 1
fi
echo "Using arch=${ARCH}, checkpoint=${CKPT}"

# ---------------- one Python process per chain (parallel) ----------------
# Each chain compiles XLA independently (~130-200 s for the NN-OT filter
# at T=100). Compiles happen concurrently across the 4 srun steps, so the
# wall-time overhead is ~200 s once, not 4 x 200 s.
for CHAIN_ID in $(seq 0 $((NUM_CHAINS - 1))); do
  echo "Launching SVSSM NN-OT chain ${CHAIN_ID}"

  srun --exclusive \
       -N1 \
       -n1 \
       -c"${SLURM_CPUS_PER_TASK}" \
       --cpu-bind=cores \
    python -u scripts/exp/exp_hmc_svssm_neural_ot.py \
      "${COMMON[@]}" \
      --chain_id "${CHAIN_ID}" \
      > "sv_logs/svssm_nnot_chain_${CHAIN_ID}.out" \
      2> "sv_logs/svssm_nnot_chain_${CHAIN_ID}.err" &

done

wait

echo "================================================"
echo "All chains done. Aggregating -> ${OUT_DIR}"
echo "================================================"

# Aggregator (built into the driver via --aggregate); reads
# ${OUT_DIR}/chains/chain_*.npz and writes the combined report.
python -u scripts/exp/exp_hmc_svssm_neural_ot.py \
  "${COMMON[@]}" \
  --aggregate \
  > "sv_logs/svssm_nnot_aggregate_${SLURM_JOB_ID}.out" \
  2> "sv_logs/svssm_nnot_aggregate_${SLURM_JOB_ID}.err"

echo "================================================"
echo "Done. Outputs under ${OUT_DIR}/"
echo "  svssm_hmc_neural_ot_samples.npz"
echo "  svssm_hmc_neural_ot_results.txt"
echo "  svssm_hmc_neural_ot_summary.json"
echo "  svssm_hmc_neural_ot_traces.png"
echo "  chains/chain_*.npz   (per-chain raw)"
echo "================================================"
