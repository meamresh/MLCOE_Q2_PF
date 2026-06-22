#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=SVSSM_NNOT_v7
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --output=sv_logs/master_nnot_%j.out
#SBATCH --error=sv_logs/master_nnot_%j.err
#
# NN-OT T-sweep production run, MIRRORING the Phase 16 Sinkhorn sweep
# (scripts/exp/slurm.slurm). Same kernel (v7 dense-mass windowed-adaptive
# PreconditionedHMC), same prior (wide-shifted), same T values, same
# chain/sample budget. Only the LIKELIHOOD differs: a trained DeepONet
# neural-operator monotone OT instead of the Sinkhorn iterative OT
# inside the LEDH filter.
#
# Compared to the existing scripts/exp/svssm_hmc_nnot_slurm.sh
# (single-T, vanilla HMC, canonical prior) this is the Phase 17
# upgrade: same kernel as the Sinkhorn Phase 16, so any difference
# in posteriors between the two pipelines is attributable to the OT
# resampler choice, not the kernel.
#
# Headline comparison after both runs complete:
#   Phase 16 (Sinkhorn): reports/.../svssm_hmc_sweep_wide_T{50,100,200}
#   Phase 17 (NN-OT):    reports/.../svssm_hmc_nnot_sweep_wide_T{50,100,200}
# Expected: posteriors statistically indistinguishable per-T (Phase 10
# pattern at canonical-truth scaled to wide-shifted prior).
#
# 12 srun tasks launched concurrently:
#   4 chains for T=50
#   4 chains for T=100
#   4 chains for T=200
# After all 12 finish, three aggregator + Vehtari diagnostic passes.
#
# Walltime budget:
#   The Phase 16 Sinkhorn sweep with the same kernel took 11h at T=200.
#   NN-OT per-step is typically 1.0-1.2x faster than Sinkhorn (Phase 11),
#   so NN-OT should land at 9-11h at T=200. Compile overhead is bigger
#   for NN-OT (XLA fusion of the trained Keras OT block) but amortizes
#   over 2700 steps per chain.
#   margin: --time=20:00:00 matches the Phase 16 sweep's allocation.

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

# ---------------- threading (matches Phase 16 Sinkhorn sweep) ----------------
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
SWEEP_TAG="svssm_hmc_nnot_sweep_wide"   # one subdir per T under this tag

# Architecture + checkpoint.
# DeepONet trained at N=64 (Phase 3) — matches our N=64 sweep.
# (Phase 8/9 showed DeepONet has shallower OOD-N cliff than mGradNet
#  and is the default architecture choice.)
ARCH="deeponet"
CKPT="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase3/deeponet.weights.h5"

# Shared HMC + data knobs (identical across all T AND identical to the
# Sinkhorn Phase 16 sweep except for the NN-OT-specific args).
# NOTE: --T is intentionally NOT here; it varies per launch below.
SHARED=(
  --arch "${ARCH}"
  --graph_mode xla
  --checkpoint_path "${CKPT}"
  --n_ridges 64
  --grad_window 4
  --num_chains "${NUM_CHAINS}"
  --mu 0.0 --phi 0.95 --sigma_eta 0.3      # canonical truth (Phases 5-12 baseline)
  --N "${N}"
  --n_lambda 10
  --L 5
  --step_size 0.01                          # initial; v7 kernel adapts from here
  --num_burnin 200                          # gives windowed scheme room for
                                            # [20, 22, 45, 113]-step phases
  --num_results 2500                        # targets bulk-ESS >= 1000 even at
                                            # autocorrelation lag <= 10
  --dispersion 0.05                         # small: start near truth
  --data_seed 42
  --base_seed 300
  --progress_every 100
  # v7 windowed-proper kernel (DENSE mass by default). Matches the
  # kernel used in scripts/exp/slurm.slurm so the resulting posteriors
  # are directly comparable to Phase 16 Sinkhorn.
  --use_windowed_adaptive
)

# WIDE SHIFTED prior (same as the Sinkhorn Phase 16 sweep):
#   mu_loc=2.0,       scale=3.0    -> mu mode at 2.0  (truth 0.0 is at -0.67 sigma)
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

# Sanity-check the checkpoint exists before launching anything.
if [[ ! -f "${CKPT}" ]]; then
  echo "[error] checkpoint not found: ${CKPT}" >&2
  echo "        upload it from a local Phase-3 training run before submitting." >&2
  exit 1
fi
echo "Using arch=${ARCH}, checkpoint=${CKPT}"

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
    echo "Launching NN-OT T=${T} chain ${CHAIN_ID}"
    srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
      python -u -m scripts.exp.exp_hmc_svssm_neural_ot \
        "${SHARED[@]}" "${WIDE_PRIOR_ARGS[@]}" \
        --T "${T}" \
        --out_dir "${OUT_DIR}" \
        --chain_id "${CHAIN_ID}" \
        > "sv_logs/nnot_sweep_T${T}_chain_${CHAIN_ID}.out" \
        2> "sv_logs/nnot_sweep_T${T}_chain_${CHAIN_ID}.err" &
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
#   1) the NN-OT driver's built-in --aggregate (basic split-Rhat table)
#   2) src.experiments.exp_svssm_hmc_aggregate for the full rank-Rhat +
#      bulk-ESS + tail-ESS table (Vehtari et al. 2021).
# Vehtari thresholds: rank-Rhat <= 1.01, bulk-ESS >= 400, tail-ESS >= 400.
for T in "${T_VALUES[@]}"; do
  OUT_DIR="${ROOT_OUT}/${SWEEP_TAG}_T${T}"
  echo
  echo "================================================"
  echo "Aggregating NN-OT T=${T} -> ${OUT_DIR}"
  echo "================================================"
  python -u -m scripts.exp.exp_hmc_svssm_neural_ot \
    "${SHARED[@]}" "${WIDE_PRIOR_ARGS[@]}" \
    --T "${T}" --out_dir "${OUT_DIR}" --aggregate \
    > "sv_logs/nnot_sweep_T${T}_aggregate_${SLURM_JOB_ID}.out" \
    2> "sv_logs/nnot_sweep_T${T}_aggregate_${SLURM_JOB_ID}.err"

  echo
  echo "================================================"
  echo "Vehtari diagnostics: NN-OT T=${T}"
  echo "================================================"
  python -u -m src.experiments.exp_svssm_hmc_aggregate \
    --samples_npz "${OUT_DIR}/svssm_hmc_samples.npz" \
    > "sv_logs/nnot_sweep_T${T}_vehtari_${SLURM_JOB_ID}.out" \
    2> "sv_logs/nnot_sweep_T${T}_vehtari_${SLURM_JOB_ID}.err"
done

echo
echo "================================================"
echo "Done. NN-OT outputs:"
for T in "${T_VALUES[@]}"; do
  echo "  T=${T}:  ${ROOT_OUT}/${SWEEP_TAG}_T${T}/svssm_hmc_results.txt"
done
echo "Total wall: $(( $(date +%s) - t0 ))s"
echo "================================================"
echo
echo "Compare against Phase 16 Sinkhorn at:"
for T in "${T_VALUES[@]}"; do
  echo "  T=${T}:  ${ROOT_OUT}/svssm_hmc_sweep_wide_T${T}/svssm_hmc_results.txt"
done
