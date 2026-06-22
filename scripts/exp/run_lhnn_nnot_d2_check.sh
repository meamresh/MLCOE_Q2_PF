#!/bin/bash
# =============================================================================
# End-to-end L-HNN NUTS + NN-OT pipeline at d=2 (reference / check run).
#
# This is the first END-TO-END check of the COMBINATION: surrogate-gradient
# NUTS (L-HNN) sampling the multivariate full-Phi SVSSM whose Sinkhorn
# resampler is replaced by a trained DeepONet operator (NN-OT). Each half is
# validated separately; this script wires them together at d=2.
#
# Two trained artifacts, in dependency order:
#   1. DeepONet operator   (deeponet_nd.weights.h5)  -- the NN-OT resampler.
#   2. L-HNN surrogate cache (lhnn_d2_nnot.weights.h5) -- trained AGAINST the
#      NN-OT target (pilot runs with --nnot_weights), so it learns the NN-OT
#      gradient field, NOT the Sinkhorn one. An L-HNN cache trained against
#      Sinkhorn is INVALID here.
#
# Retrain rules (full table in Reads/Run_Instruction_LHNN_NNOT.md):
#   - change d / N / n_lambda / n_basis / T  -> retrain BOTH (operator first).
#   - change truth / priors / data_seed      -> retrain L-HNN (operator only
#                                               if posterior leaves the grid).
#   - change pure NUTS knobs (draws, burnin, treedepth, accept, dispersion,
#     kinetic_mh, sampling step_size, chains) -> retrain NOTHING (cache reused).
#
# Usage:  bash scripts/exp/run_lhnn_nnot_d2_check.sh
# Re-runnable: skips operator training if its weights already exist; the
# launcher skips the L-HNN pilot if its cache already exists.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- config (edit here; these must be consistent across operator + run) -----
D=2
T=50
N=64
NLAMBDA=10            # MUST match operator training and the run
K=10                 # Sinkhorn iters (unused once NN-OT is in; kept for parity)
NBASIS=64            # operator n_basis; passed to BOTH trainer and run

# truth (d=2)
MU="0.0,-0.3"
PHI_DIAG="0.95,0.85"
PHI_OFF="0.05"            # length d(d-1)/2 = 1 at d=2
SIGMA_ETA="0.3,0.4"

# priors (informative recovery priors, same as the validated d=2 runs)
PRIORS="--prior_mu_loc -0.2 --prior_mu_scale 1.0 \
--prior_phi_raw_loc 1.5 --prior_phi_raw_scale 1.0 \
--prior_log_sigma_eta_sq_loc -2.0 --prior_log_sigma_eta_sq_scale 1.0 \
--prior_phi_off_scale 0.2"

DATA_SEED=42
BASE_SEED=300

# NUTS sampling (pure sampler knobs -> changing these does NOT need a retrain)
NUM_CHAINS=4
NUM_BURNIN=300
NUM_RESULTS=2500     # 2500 = the chain length where d=2 rank-Rhat cleared.
                     # For a faster smoke, drop to e.g. 500 (R-hat will be loose).
STEP_SIZE=0.01
MAX_TREEDEPTH=8
TARGET_ACCEPT=0.65
DISPERSION=0.05

# L-HNN surrogate architecture + pilot (validated at d=2)
HIDDEN_UNITS=128
NUM_HIDDEN=3
LHNN_EPOCHS=2500
PILOT_TRAJ=60
PILOT_STEPS=60

# ---- output locations -------------------------------------------------------
NNOT_DIR="reports/nnot_d${D}_T${T}_N${N}"
NNOT_WEIGHTS="${NNOT_DIR}/deeponet_nd.weights.h5"
RUN_DIR="reports/d${D}_lhnn_nnot_check"
LHNN_CACHE="${RUN_DIR}/lhnn_d${D}_nnot.weights.h5"   # note the _nnot suffix
mkdir -p "$NNOT_DIR" "$RUN_DIR"

echo "============================================================"
echo "L-HNN NUTS + NN-OT  d=${D} T=${T} N=${N}  (end-to-end check)"
echo "  operator weights : ${NNOT_WEIGHTS}"
echo "  L-HNN cache      : ${LHNN_CACHE}"
echo "  run output       : ${RUN_DIR}"
echo "============================================================"

# =============================================================================
# STEP 1 -- train the DeepONet (NN-OT) operator
# =============================================================================
if [ -f "$NNOT_WEIGHTS" ]; then
  echo "[step 1] operator already trained at ${NNOT_WEIGHTS} -> skip."
else
  echo "[step 1] training DeepONet operator (d=${D}, N=${N}, n_basis=${NBASIS}) ..."
  python -u -m scripts.exp.phase16_train_multi_nnot_nd \
    --d "$D" --T "$T" --N "$N" \
    --n_theta 60 --seeds_per_theta 2 \
    --max_epochs 60 --patience 10 --n_basis "$NBASIS" --seed 42 \
    --out_dir "$NNOT_DIR" 2>&1 | tee "${NNOT_DIR}/train_operator.log"
  if [ ! -f "$NNOT_WEIGHTS" ]; then
    echo "[step 1] ERROR: operator weights not produced at ${NNOT_WEIGHTS}" >&2
    exit 1
  fi
fi

# =============================================================================
# STEP 2 + 3 -- train L-HNN against the NN-OT target, then sample (parallel)
#   The launcher: no cache -> --train_only (pilot through the NN-OT filter) ->
#   N parallel --chain_id chains (each loads cache + operator) -> stitch ->
#   save_diagnostics. --nnot_weights is passed to BOTH stages so the surrogate
#   learns the NN-OT gradient field.
# =============================================================================
echo "[step 2+3] L-HNN train-against-NN-OT + ${NUM_CHAINS} parallel chains ..."
bash scripts/exp/launch_lhnn_nuts_parallel.sh \
  "$RUN_DIR" "$NUM_CHAINS" "$LHNN_CACHE" \
  --d "$D" --T "$T" --N "$N" --n_lambda "$NLAMBDA" --K "$K" \
  --mu "$MU" --phi_diag "$PHI_DIAG" --phi_off "$PHI_OFF" --sigma_eta "$SIGMA_ETA" \
  $PRIORS \
  --data_seed "$DATA_SEED" --base_seed "$BASE_SEED" \
  --nnot_weights "$NNOT_WEIGHTS" --nnot_n_basis "$NBASIS" \
  --num_burnin "$NUM_BURNIN" --num_results "$NUM_RESULTS" \
  --step_size "$STEP_SIZE" --max_treedepth "$MAX_TREEDEPTH" \
  --target_accept_prob "$TARGET_ACCEPT" --dispersion "$DISPERSION" \
  --hidden_units "$HIDDEN_UNITS" --num_hidden "$NUM_HIDDEN" --lhnn_epochs "$LHNN_EPOCHS" \
  --num_pilot_trajectories "$PILOT_TRAJ" --pilot_steps_per_trajectory "$PILOT_STEPS" \
  --kinetic_mh \
  2>&1 | tee "${RUN_DIR}/run.log"

# =============================================================================
# STEP 4 -- trace plots (both read the stitched npz from --out_dir, so they
#   must run AFTER the launcher has stitched the per-chain npz into RUN_DIR).
# =============================================================================
STITCHED="${RUN_DIR}/svssm_hmc_multi_full_phi_samples.npz"
if [ -f "$STITCHED" ]; then
  echo "[step 4] trace plots ..."
  python scripts/plot_trace_multi_full_phi.py --out_dir "$RUN_DIR"        # raw-param traces
  python scripts/plot_trace_stationary_cov.py --out_dir "$RUN_DIR"        # derived Sigma_h traces
else
  echo "[step 4] WARNING: stitched npz not found ($STITCHED); skipping plots." >&2
fi

echo "============================================================"
echo "DONE. Stitched samples + diagnostics + plots in: ${RUN_DIR}"
echo "  samples     : ${RUN_DIR}/svssm_hmc_multi_full_phi_samples.npz"
echo "  diagnostics : ${RUN_DIR}/svssm_hmc_multi_full_phi_diagnostics.txt"
echo "  trace plot  : ${RUN_DIR}/svssm_hmc_multi_full_phi_trace.png"
echo "  Sigma_h plot: ${RUN_DIR}/svssm_hmc_multi_full_phi_stationary_trace.png"
echo
echo "Sanity check (judge by COVERAGE + recompute rank-Rhat from the npz;"
echo "do NOT trust stored summary.json R-hat -- the formula changed over time):"
echo "  grep -E 'coverage|rank' ${RUN_DIR}/svssm_hmc_multi_full_phi_diagnostics.txt"
echo
echo "To compare against the Sinkhorn-target L-HNN posterior (the d=2 reference),"
echo "recompute rank_rhat on both npz with scripts/exp/compare_svssm_hmc_methods.py."
echo "============================================================"
