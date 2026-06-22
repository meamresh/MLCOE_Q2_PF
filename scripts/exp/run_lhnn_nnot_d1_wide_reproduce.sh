#!/bin/bash
# =============================================================================
# Cross-check: can the L-HNN NUTS + NN-OT (multivariate) pipeline, run at d=1,
# REPRODUCE the windowed-HMC wide-shifted-prior recovery used in the Q1 answer?
#
# Reference (windowed exact-gradient HMC, reports/.../new/svssm_hmc_sweep_wide_T*):
#   truth mu=0, phi=0.95, sigma_eta=0.3 ; WIDE-SHIFTED prior
#     mu ~ N(2,3^2), phi_raw ~ N(0,2^2), log sigma_eta^2 ~ N(1.5,3^2)
#   recovery (median): T=100 -> mu 0.357, phi 0.922, sigma_eta^2 0.286, P(phi<0) 8.0%
#                      T=200 -> mu 0.226, phi 0.954, sigma_eta^2 0.101, P(phi<0) 4.9%
#
# This runs the SAME truth + prior through L-HNN+NN-OT at d=1 and compares.
# Default T=100 (~55 min). For T=200 set T=200 below (and ideally PILOT 60x60);
# expect ~2 h. The default d=1 operator grid already centers on this truth, but
# we pass it explicitly for clarity.
#
# Usage:  bash scripts/exp/run_lhnn_nnot_d1_wide_reproduce.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

D=1; T=100; N=64; NLAMBDA=10; K=10; NBASIS=64      # <- set T=200 to match the headline (slower)
MU="0.0"; PHI_DIAG="0.95"; PHI_OFF=""; SIGMA_ETA="0.3"

# WIDE-SHIFTED prior (modes far from truth) -- identical to the windowed sweep.
PRIORS="--prior_mu_loc 2.0 --prior_mu_scale 3.0 \
--prior_phi_raw_loc 0.0 --prior_phi_raw_scale 2.0 \
--prior_log_sigma_eta_sq_loc 1.5 --prior_log_sigma_eta_sq_scale 3.0 \
--prior_phi_off_scale 0.5"

DATA_SEED=42; BASE_SEED=300
NUM_CHAINS=4; NUM_BURNIN=300; NUM_RESULTS=1000     # 1000 enough for the median-recovery check
STEP=0.01; TREEDEPTH=8; ACCEPT=0.65; DISP=0.05
HID=128; NHID=3; LHNN_EPOCHS=4000; PILOT_TRAJ=40; PILOT_STEPS=40   # clipped pilot

NNOT_DIR="reports/nnot_d${D}_T${T}_N${N}_wide"
NNOT_WEIGHTS="${NNOT_DIR}/deeponet_nd.weights.h5"
RUN_DIR="reports/d${D}_lhnn_nnot_wide_T${T}"
LHNN_CACHE="${RUN_DIR}/lhnn_d${D}_nnot.weights.h5"
mkdir -p "$NNOT_DIR" "$RUN_DIR"

echo "=== reproduce windowed wide-prior recovery via L-HNN+NN-OT  d=${D} T=${T} ==="

# ---- Step 1: train the d=1 operator (grid centered on the truth) ----
if [ -f "$NNOT_WEIGHTS" ]; then
  echo "[1] operator exists -> skip."
else
  echo "[1] training d=${D} operator (grid centered on truth) ..."
  python -u -m scripts.exp.phase16_train_multi_nnot_nd \
    --d "$D" --T "$T" --N "$N" --n_theta 60 --seeds_per_theta 2 \
    --max_epochs 60 --patience 10 --n_basis "$NBASIS" --seed 42 \
    --mu0 "$MU" --phi0 "$PHI_DIAG" --sig0 "$SIGMA_ETA" --phi_off0 0.05 \
    --out_dir "$NNOT_DIR" 2>&1 | tee "${NNOT_DIR}/train_operator.log"
  [ -f "$NNOT_WEIGHTS" ] || { echo "[1] ERROR: no operator weights" >&2; exit 1; }
fi

# ---- Step 2+3: L-HNN train-against-NN-OT + parallel chains ----
echo "[2+3] L-HNN(NN-OT) train + ${NUM_CHAINS} parallel chains ..."
bash scripts/exp/launch_lhnn_nuts_parallel.sh "$RUN_DIR" "$NUM_CHAINS" "$LHNN_CACHE" \
  --d "$D" --T "$T" --N "$N" --n_lambda "$NLAMBDA" --K "$K" \
  --mu "$MU" --phi_diag "$PHI_DIAG" --phi_off "$PHI_OFF" --sigma_eta "$SIGMA_ETA" \
  $PRIORS --data_seed "$DATA_SEED" --base_seed "$BASE_SEED" \
  --nnot_weights "$NNOT_WEIGHTS" --nnot_n_basis "$NBASIS" \
  --num_burnin "$NUM_BURNIN" --num_results "$NUM_RESULTS" \
  --step_size "$STEP" --max_treedepth "$TREEDEPTH" \
  --target_accept_prob "$ACCEPT" --dispersion "$DISP" \
  --hidden_units "$HID" --num_hidden "$NHID" --lhnn_epochs "$LHNN_EPOCHS" \
  --num_pilot_trajectories "$PILOT_TRAJ" --pilot_steps_per_trajectory "$PILOT_STEPS" \
  --kinetic_mh --progress_every 250 2>&1 | tee "${RUN_DIR}/run.log"

[ -f "${RUN_DIR}/svssm_hmc_multi_full_phi_samples.npz" ] && \
  python scripts/plot_trace_multi_full_phi.py --out_dir "$RUN_DIR"

# ---- Step 4: recovery comparison vs the windowed reference ----
echo "[4] L-HNN+NN-OT recovery vs windowed reference:"
python -u - "$RUN_DIR" "$T" <<'PYEOF'
import numpy as np, sys
RUN, T = sys.argv[1], int(sys.argv[2])
z = np.load(RUN + "/svssm_hmc_multi_full_phi_samples.npz")
mu  = z["mu"][...,0].ravel(); phi = z["phi_diag"][...,0].ravel(); s2 = z["sigma_eta_sq"][...,0].ravel()
ref = {100: dict(mu=0.357, phi=0.922, s2=0.286, pneg=8.0),
       200: dict(mu=0.226, phi=0.954, s2=0.101, pneg=4.9)}.get(T, {})
print(f"  {'quantity':<14}{'L-HNN+NN-OT':>14}{'windowed ref':>14}")
print(f"  {'median mu':<14}{np.median(mu):>14.3f}{ref.get('mu','?'):>14}")
print(f"  {'median phi':<14}{np.median(phi):>14.3f}{ref.get('phi','?'):>14}")
print(f"  {'median sig2':<14}{np.median(s2):>14.3f}{ref.get('s2','?'):>14}")
print(f"  {'P(phi<0) %':<14}{100*np.mean(phi<0):>14.1f}{ref.get('pneg','?'):>14}")
print("  (truth: mu=0, phi=0.95, sigma_eta^2=0.09)")
PYEOF
echo "=== DONE: ${RUN_DIR} ==="
