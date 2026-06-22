#!/bin/bash
# =============================================================================
# Companion to run_lhnn_nnot_d1_wide_reproduce.sh: the SAME truth at d=1, but
# with MODERATE near-truth priors (the L-HNN's comfort zone, matching the d=2
# sweep) instead of the wide-shifted prior. Expected: fast, err_trigs ~ 0,
# recovers the truth -- the in-distribution contrast to the wide-prior failure.
#
#   truth mu=0, phi=0.95, sigma_eta=0.3 (sigma^2=0.09)
#   MODERATE prior (= d=2 sweep): mu~N(0,1.5), phi_raw~N(1.5,1) [phi mode 0.905],
#                                 log sigma^2~N(-1.7,1) [sigma^2 mode 0.18]
#
# The operator depends only on the (truth-centred) theta-grid and T, NOT on the
# prior, so we REUSE the operator trained by the wide run. The L-HNN surrogate
# IS prior-dependent (prior is part of the target) -> retrained here.
#
# Usage:  bash scripts/exp/run_lhnn_nnot_d1_moderate.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

D=1; T=100; N=64; NLAMBDA=10; K=10; NBASIS=64
MU="0.0"; PHI_DIAG="0.95"; PHI_OFF=""; SIGMA_ETA="0.3"

# MODERATE near-truth prior (the surrogate's comfort zone)
PRIORS="--prior_mu_loc 0.0 --prior_mu_scale 1.5 \
--prior_phi_raw_loc 1.5 --prior_phi_raw_scale 1.0 \
--prior_log_sigma_eta_sq_loc -1.7 --prior_log_sigma_eta_sq_scale 1.0 \
--prior_phi_off_scale 0.2"

DATA_SEED=42; BASE_SEED=300
NUM_CHAINS=4; NUM_BURNIN=300; NUM_RESULTS=1000
STEP=0.01; TREEDEPTH=8; ACCEPT=0.65; DISP=0.05
HID=128; NHID=3; LHNN_EPOCHS=4000; PILOT_TRAJ=40; PILOT_STEPS=40

# REUSE the wide run's operator (same truth-grid + T; prior-independent).
NNOT_DIR="reports/nnot_d${D}_T${T}_N${N}_wide"
NNOT_WEIGHTS="${NNOT_DIR}/deeponet_nd.weights.h5"
RUN_DIR="reports/d${D}_lhnn_nnot_moderate_T${T}"
LHNN_CACHE="${RUN_DIR}/lhnn_d${D}_nnot.weights.h5"
mkdir -p "$NNOT_DIR" "$RUN_DIR"

echo "=== d=1 MODERATE-prior L-HNN+NN-OT recovery (comfort-zone contrast) ==="

# ---- Step 1: operator (reuse if present, else train at truth grid) ----
if [ -f "$NNOT_WEIGHTS" ]; then
  echo "[1] reusing operator at ${NNOT_WEIGHTS} (prior-independent)."
else
  echo "[1] training d=${D} operator (grid centered on truth) ..."
  python -u -m scripts.exp.phase16_train_multi_nnot_nd \
    --d "$D" --T "$T" --N "$N" --n_theta 60 --seeds_per_theta 2 \
    --max_epochs 60 --patience 10 --n_basis "$NBASIS" --seed 42 \
    --mu0 "$MU" --phi0 "$PHI_DIAG" --sig0 "$SIGMA_ETA" --phi_off0 0.05 \
    --out_dir "$NNOT_DIR" 2>&1 | tee "${NNOT_DIR}/train_operator.log"
  [ -f "$NNOT_WEIGHTS" ] || { echo "[1] ERROR: no operator weights" >&2; exit 1; }
fi

# ---- Step 2+3: L-HNN (retrain for moderate target) + parallel chains ----
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

# ---- Step 4: recovery + the comfort-zone contrast (err_trigs, speed) ----
echo "[4] recovery + err_trigs/timing (contrast vs the wide-prior run):"
for c in 0 1 2 3; do
  python3 -c "import json;s=json.load(open('${RUN_DIR}/chain_${c}/svssm_hmc_multi_full_phi_summary.json'));n=s['nuts_diagnostics'];print(f'  chain_${c}: err_trigs={n[\"error_triggers_per_chain\"][0]} real_grads={n[\"total_real_grad_evals\"]} sample={s[\"sampling_wall_s\"]:.0f}s acc={s[\"accept_rate_overall\"]:.3f}')" 2>/dev/null
done
python -u - "$RUN_DIR" <<'PYEOF'
import numpy as np, sys
z = np.load(sys.argv[1] + "/svssm_hmc_multi_full_phi_samples.npz")
mu=z["mu"][...,0].ravel(); phi=z["phi_diag"][...,0].ravel(); s2=z["sigma_eta_sq"][...,0].ravel()
def cov(x,t): l,h=np.quantile(x,[.025,.975]); return l<=t<=h
print(f"  median mu={np.median(mu):.3f} (truth 0, cov {cov(mu,0.0)})")
print(f"  median phi={np.median(phi):.3f} (truth 0.95, cov {cov(phi,0.95)})  P(phi<0)={100*np.mean(phi<0):.1f}%")
print(f"  median sig2={np.median(s2):.3f} (truth 0.09, cov {cov(s2,0.09)})")
PYEOF
echo "=== DONE: ${RUN_DIR}  (expect err_trigs ~0 and fast, vs wide-prior ~8.5 real-grads/iter) ==="
