#!/bin/bash
# =============================================================================
# L-HNN NUTS + NN-OT at d=2, T=200, truth B, operator grid CENTERED on truth.
# Same config as the d=2 T=50 grid-fix run, only T=50 -> T=500, with a CLIPPED pilot (40x40).
#
# T changed => BOTH artifacts retrain (operator capture is at the new T; the
# L-HNN target log-density is at the new T). This script trains a fresh
# T=200 operator (B-centered grid) and a fresh T=200 L-HNN cache, then samples.
# NN-OT only (no Sinkhorn reference) -- per request.
#
# Truth B (matched SNR sigma_h^2 = 1 per dim):
#   mu=[1,-1]  phi_diag=[0.95,0.80]  sigma_eta=[0.3122,0.6]  phi_off=0.05
#
# Usage:  bash scripts/exp/run_lhnn_nnot_d2_T200.sh
# Re-runnable: skips operator/L-HNN whose artifacts exist.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- config (same as gridfix, T=200) ----------------------------------------
D=2; T=500; N=64; NLAMBDA=10; K=10; NBASIS=64

MU="1.0,-1.0"
PHI_DIAG="0.95,0.80"
SIGMA_ETA="0.3122,0.6"
PHI_OFF="0.05"

PRIORS="--prior_mu_loc 0.0 --prior_mu_scale 1.5 \
--prior_phi_raw_loc 1.5 --prior_phi_raw_scale 1.0 \
--prior_log_sigma_eta_sq_loc -1.7 --prior_log_sigma_eta_sq_scale 1.0 \
--prior_phi_off_scale 0.2"

DATA_SEED=42; BASE_SEED=300

NUM_CHAINS=4; NUM_BURNIN=300; NUM_RESULTS=1000
STEP_SIZE=0.01; MAX_TREEDEPTH=8; TARGET_ACCEPT=0.65; DISPERSION=0.05
HIDDEN_UNITS=128; NUM_HIDDEN=3; LHNN_EPOCHS=6000; PILOT_TRAJ=40; PILOT_STEPS=40   # CLIPPED pilot (1640 grads vs 3660) to cut the T=500 pilot ~2.2x

# ---- output locations -------------------------------------------------------
NNOT_DIR="reports/nnot_d${D}_T${T}_N${N}_matchedB"     # T=200 B-centered operator
NNOT_WEIGHTS="${NNOT_DIR}/deeponet_nd.weights.h5"
NN_DIR="reports/d${D}_lhnn_nnot_B_T${T}"
NN_CACHE="${NN_DIR}/lhnn_d${D}_nnot.weights.h5"
mkdir -p "$NNOT_DIR" "$NN_DIR"

echo "============================================================"
echo "L-HNN + NN-OT  d=${D} T=${T} N=${N}  truth B (grid-centered)"
echo "  operator: ${NNOT_WEIGHTS}"
echo "  run     : ${NN_DIR}"
echo "============================================================"

# ---- STEP 1: train the T=200 operator, grid centered on truth B -------------
if [ -f "$NNOT_WEIGHTS" ]; then
  echo "[1] T=${T} B-centered operator exists -> skip."
else
  echo "[1] training T=${T} operator, grid centered on truth B ..."
  python -u -m scripts.exp.phase16_train_multi_nnot_nd \
    --d "$D" --T "$T" --N "$N" --n_theta 60 --seeds_per_theta 2 \
    --max_epochs 60 --patience 10 --n_basis "$NBASIS" --seed 42 \
    --mu0 "$MU" --phi0 "$PHI_DIAG" --sig0 "$SIGMA_ETA" --phi_off0 0.05 \
    --out_dir "$NNOT_DIR" 2>&1 | tee "${NNOT_DIR}/train_operator.log"
  [ -f "$NNOT_WEIGHTS" ] || { echo "[1] ERROR: no operator weights" >&2; exit 1; }
fi

# ---- STEP 2+3: train L-HNN against T=200 NN-OT target, then sample ----------
echo "[2+3] L-HNN train-against-NN-OT (T=${T}) + ${NUM_CHAINS} parallel chains ..."
bash scripts/exp/launch_lhnn_nuts_parallel.sh "$NN_DIR" "$NUM_CHAINS" "$NN_CACHE" \
  --d "$D" --T "$T" --N "$N" --n_lambda "$NLAMBDA" --K "$K" \
  --mu "$MU" --phi_diag "$PHI_DIAG" --phi_off "$PHI_OFF" --sigma_eta "$SIGMA_ETA" \
  $PRIORS --data_seed "$DATA_SEED" --base_seed "$BASE_SEED" \
  --nnot_weights "$NNOT_WEIGHTS" --nnot_n_basis "$NBASIS" \
  --num_burnin "$NUM_BURNIN" --num_results "$NUM_RESULTS" \
  --step_size "$STEP_SIZE" --max_treedepth "$MAX_TREEDEPTH" \
  --target_accept_prob "$TARGET_ACCEPT" --dispersion "$DISPERSION" \
  --hidden_units "$HIDDEN_UNITS" --num_hidden "$NUM_HIDDEN" --lhnn_epochs "$LHNN_EPOCHS" \
  --num_pilot_trajectories "$PILOT_TRAJ" --pilot_steps_per_trajectory "$PILOT_STEPS" \
  --kinetic_mh --progress_every 250 2>&1 | tee "${NN_DIR}/run.log"

# ---- STEP 4: trace plots ----------------------------------------------------
if [ -f "${NN_DIR}/svssm_hmc_multi_full_phi_samples.npz" ]; then
  echo "[4] trace plots ..."
  python scripts/plot_trace_multi_full_phi.py --out_dir "$NN_DIR"
  python scripts/plot_trace_stationary_cov.py --out_dir "$NN_DIR"
fi

echo "============================================================"
echo "DONE. T=${T} NN-OT run in: ${NN_DIR}"
echo "  diagnostics: ${NN_DIR}/svssm_hmc_multi_full_phi_diagnostics.txt"
echo "  trace plots: svssm_hmc_multi_full_phi_{trace,stationary_trace}.png"
echo "Per-chain sampling wall (vs T=50 NN-OT ~1660s) shows the T-scaling:"
echo "  for c in 0 1 2 3; do python3 -c \"import json;print(json.load(open('${NN_DIR}/chain_\$c/svssm_hmc_multi_full_phi_summary.json'))['sampling_wall_s'])\"; done"
echo "============================================================"
