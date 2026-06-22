#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=SVSSM_T50_CMP
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --output=sv_logs/master_T50_cmp_%j.out
#SBATCH --error=sv_logs/master_T50_cmp_%j.err
#
# Side-by-side T=50 comparison of Sinkhorn vs Neural-OT HMC pipelines.
#
# 8 srun tasks launched in parallel:
#   - 4 chains of the Sinkhorn pipeline (src.experiments.hmc_svssm)
#   - 4 chains of the NN-OT pipeline   (src.experiments.exp_hmc_svssm_neural_ot, DeepONet XLA)
#
# Everything else (priors, T, N, seeds, dispersion, step_size, num_burnin,
# num_results, integrator) is held identical so the posteriors are directly
# comparable. Two output directories side by side under ${ROOT_OUT}/.
#
# Walltime budget: per-chain at T=50 is roughly half the T=100 cost we
# measured on HPC (~21h/chain at T=100). So expect ~10-12 h per chain.
# Since all 8 chains run in parallel, max-chain wall (~10-12 h) is the
# total job wall, well inside the 24 h request.
#
# If your node is smaller than 8 x 4 = 32 cores: drop --cpus-per-task to 2
# or --ntasks to 4 and run the two pipelines sequentially (uncomment the
# "SEQUENTIAL" block below).

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

# ---------------- threading caps (avoid oversubscription) ----------------
# Match the existing Sinkhorn template; both pipelines respect the same caps
# so the comparison is fair. If you want to speed both up at the cost of
# fairness with prior runs, bump TF_NUM_INTRAOP_THREADS to 2-3.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# NOTE: deliberately NOT setting XLA_FLAGS=--xla_dump_to (Phase 10 lesson;
# that flag dumps HLO IR to disk and inflated compile time + ate 3.1 GB).
# If a persistent XLA cache is wanted, use the right env var for your TF.

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "CPUs on node: ${SLURM_CPUS_ON_NODE}"
echo "Tasks: ${SLURM_NTASKS}  CPUs/task: ${SLURM_CPUS_PER_TASK}"

# ---------------- experiment knobs (edit here) ----------------
NUM_CHAINS=4
T=50
N=64

ROOT_OUT="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH"
OUT_DIR_SK="${ROOT_OUT}/svssm_hmc_T50_sinkhorn"
OUT_DIR_NN="${ROOT_OUT}/svssm_hmc_T50_nnot"

# NN-OT architecture + checkpoint.
ARCH="deeponet"
CKPT="${ROOT_OUT}/section2_phase3/deeponet.weights.h5"

# Shared HMC knobs (held identical between pipelines).
SHARED=(
  --num_chains "${NUM_CHAINS}"
  --mu 0.0 --phi 0.95 --sigma_eta 0.3
  --T "${T}"
  --N "${N}"
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

# Pipeline-specific args.
SK_ARGS=(--K 10 --out_dir "${OUT_DIR_SK}")
NN_ARGS=(
  --arch "${ARCH}"
  --graph_mode xla
  --checkpoint_path "${CKPT}"
  --n_ridges 64
  --grad_window 4
  --out_dir "${OUT_DIR_NN}"
)

mkdir -p "${OUT_DIR_SK}/chains" "${OUT_DIR_NN}/chains"

# Sanity-check the NN-OT checkpoint exists.
if [[ ! -f "${CKPT}" ]]; then
  echo "[error] checkpoint not found: ${CKPT}" >&2
  exit 1
fi
echo "T=${T}  N=${N}  NUM_CHAINS=${NUM_CHAINS}"
echo "Sinkhorn out: ${OUT_DIR_SK}"
echo "NN-OT    out: ${OUT_DIR_NN}"
echo "NN-OT arch=${ARCH} ckpt=${CKPT}"

# ====================================================================
# Launch: 4 Sinkhorn chains + 4 NN-OT chains, all 8 in parallel.
# ====================================================================

t0=$(date +%s)
pids=()

# ----- Sinkhorn chains -----
for CHAIN_ID in $(seq 0 $((NUM_CHAINS - 1))); do
  echo "Launching Sinkhorn chain ${CHAIN_ID}"
  srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    python -u -m src.experiments.hmc_svssm \
      "${SHARED[@]}" "${SK_ARGS[@]}" \
      --chain_id "${CHAIN_ID}" \
      > "sv_logs/T50_sk_chain_${CHAIN_ID}.out" \
      2> "sv_logs/T50_sk_chain_${CHAIN_ID}.err" &
  pids+=($!)
done

# ----- NN-OT chains -----
for CHAIN_ID in $(seq 0 $((NUM_CHAINS - 1))); do
  echo "Launching NN-OT chain ${CHAIN_ID}"
  srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    python -u -m src.experiments.exp_hmc_svssm_neural_ot \
      "${SHARED[@]}" "${NN_ARGS[@]}" \
      --chain_id "${CHAIN_ID}" \
      > "sv_logs/T50_nn_chain_${CHAIN_ID}.out" \
      2> "sv_logs/T50_nn_chain_${CHAIN_ID}.err" &
  pids+=($!)
done

# ----- Wait + per-chain status -----
echo
echo "  [wait] waiting for ${#pids[@]} chain processes (4 SK + 4 NN-OT)..."
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
  echo "[error] ${fail} chain(s) failed; aborting before aggregation" >&2
  exit 2
fi

# ====================================================================
# Aggregate both pipelines (built-in --aggregate mode).
# ====================================================================

echo
echo "================================================"
echo "Aggregating Sinkhorn chains -> ${OUT_DIR_SK}"
echo "================================================"
python -u -m src.experiments.hmc_svssm \
  "${SHARED[@]}" "${SK_ARGS[@]}" --aggregate \
  > "sv_logs/T50_sk_aggregate_${SLURM_JOB_ID}.out" \
  2> "sv_logs/T50_sk_aggregate_${SLURM_JOB_ID}.err"

echo
echo "================================================"
echo "Aggregating NN-OT chains -> ${OUT_DIR_NN}"
echo "================================================"
python -u -m src.experiments.exp_hmc_svssm_neural_ot \
  "${SHARED[@]}" "${NN_ARGS[@]}" --aggregate \
  > "sv_logs/T50_nn_aggregate_${SLURM_JOB_ID}.out" \
  2> "sv_logs/T50_nn_aggregate_${SLURM_JOB_ID}.err"

# ====================================================================
# Side-by-side summary.
# ====================================================================

echo
echo "================================================"
echo "Side-by-side comparison (T=${T}, N=${N})"
echo "================================================"
echo
echo "--- Sinkhorn ---"
tail -n 14 "${OUT_DIR_SK}/svssm_hmc_results.txt" || echo "(no Sinkhorn results.txt)"
echo
echo "--- Neural-OT (${ARCH} XLA) ---"
tail -n 14 "${OUT_DIR_NN}/svssm_hmc_neural_ot_results.txt" || echo "(no NN-OT results.txt)"
echo

# Best-effort: per-chain wall comparison (extract elapsed_s from chain npz).
echo "Per-chain wall (seconds):"
python -u -c "
import numpy as np, glob
for label, d in [('Sinkhorn', '${OUT_DIR_SK}/chains'), ('Neural-OT', '${OUT_DIR_NN}/chains')]:
    files = sorted(glob.glob(d + '/chain_*.npz'))
    if not files:
        print(f'  {label}: no chain files'); continue
    walls = []
    for f in files:
        try:
            walls.append(float(np.load(f)['elapsed_s']))
        except Exception:
            walls.append(float('nan'))
    print(f'  {label:<10}  per-chain: {walls}  max: {max(walls):.1f}  mean: {sum(walls)/len(walls):.1f}')
" 2>/dev/null || echo "(per-chain wall extraction failed)"

echo
echo "================================================"
echo "Done. Outputs:"
echo "  ${OUT_DIR_SK}/svssm_hmc_results.txt          (Sinkhorn)"
echo "  ${OUT_DIR_NN}/svssm_hmc_neural_ot_results.txt (Neural-OT)"
echo "  ${OUT_DIR_SK}/chains/chain_*.npz             (Sinkhorn per-chain)"
echo "  ${OUT_DIR_NN}/chains/chain_*.npz             (Neural-OT per-chain)"
echo "Total wall: $(( $(date +%s) - t0 ))s"
echo "================================================"

# ====================================================================
# SEQUENTIAL FALLBACK (uncomment if your node cannot host 8 srun tasks)
# ====================================================================
#
# Re-run the script with these settings on a smaller node:
#   #SBATCH --ntasks=4
#   #SBATCH --cpus-per-task=9
#
# And replace the two "for CHAIN_ID in ..." blocks above with:
#
#   echo "Launching Sinkhorn first (4 chains in parallel)"
#   for CHAIN_ID in $(seq 0 $((NUM_CHAINS - 1))); do
#     srun ... python -m src.experiments.hmc_svssm ... &
#   done; wait
#
#   echo "Then NN-OT (4 chains in parallel)"
#   for CHAIN_ID in $(seq 0 $((NUM_CHAINS - 1))); do
#     srun ... python -m src.experiments.exp_hmc_svssm_neural_ot ... &
#   done; wait
#
# Total wall doubles (~20-24 h) but the resource ask halves.
