#!/bin/bash
# =============================================================================
# Disambiguator: is the phi~=0.81 moderate-prior posterior the EXACT posterior,
# or an NN-OT operator-approximation shift? Run the SAME L-HNN NUTS sampler with
# EXACT Sinkhorn OT (no --nnot_weights) at the same d=1, T=100, moderate prior.
#
#   L-HNN+NN-OT moderate gave phi median 0.808 (truth 0.95).
#   - if L-HNN+Sinkhorn phi ~= 0.81 -> 0.81 IS the exact moderate-prior posterior
#                                       (NN-OT faithful; "phi biased" was wrong).
#   - if L-HNN+Sinkhorn phi ~= 0.92 -> the NN-OT operator shifts phi down
#                                       (operator approximation, not coverage).
#
# No operator stage (Sinkhorn). L-HNN pilot is trained against the Sinkhorn target.
# Usage:  bash scripts/exp/run_lhnn_sinkhorn_d1_moderate.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

D=1; T=100; N=64; NLAMBDA=10; K=10
MU="0.0"; PHI_DIAG="0.95"; PHI_OFF=""; SIGMA_ETA="0.3"
PRIORS="--prior_mu_loc 0.0 --prior_mu_scale 1.5 \
--prior_phi_raw_loc 1.5 --prior_phi_raw_scale 1.0 \
--prior_log_sigma_eta_sq_loc -1.7 --prior_log_sigma_eta_sq_scale 1.0 \
--prior_phi_off_scale 0.2"
DATA_SEED=42; BASE_SEED=300
NUM_CHAINS=4; NUM_BURNIN=300; NUM_RESULTS=1000
STEP=0.01; TREEDEPTH=8; ACCEPT=0.65; DISP=0.05
HID=128; NHID=3; LHNN_EPOCHS=4000; PILOT_TRAJ=40; PILOT_STEPS=40

RUN_DIR="reports/d1_lhnn_sinkhorn_moderate_T100"
LHNN_CACHE="${RUN_DIR}/lhnn_d1_sink.weights.h5"
mkdir -p "$RUN_DIR"

echo "=== d=1 MODERATE-prior L-HNN + EXACT Sinkhorn (no operator) ==="
bash scripts/exp/launch_lhnn_nuts_parallel.sh "$RUN_DIR" "$NUM_CHAINS" "$LHNN_CACHE" \
  --d "$D" --T "$T" --N "$N" --n_lambda "$NLAMBDA" --K "$K" \
  --mu "$MU" --phi_diag "$PHI_DIAG" --phi_off "$PHI_OFF" --sigma_eta "$SIGMA_ETA" \
  $PRIORS --data_seed "$DATA_SEED" --base_seed "$BASE_SEED" \
  --num_burnin "$NUM_BURNIN" --num_results "$NUM_RESULTS" \
  --step_size "$STEP" --max_treedepth "$TREEDEPTH" \
  --target_accept_prob "$ACCEPT" --dispersion "$DISP" \
  --hidden_units "$HID" --num_hidden "$NHID" --lhnn_epochs "$LHNN_EPOCHS" \
  --num_pilot_trajectories "$PILOT_TRAJ" --pilot_steps_per_trajectory "$PILOT_STEPS" \
  --kinetic_mh --progress_every 250 2>&1 | tee "${RUN_DIR}/run.log"

[ -f "${RUN_DIR}/svssm_hmc_multi_full_phi_samples.npz" ] && \
  python scripts/plot_trace_multi_full_phi.py --out_dir "$RUN_DIR"

echo "[done] L-HNN+Sinkhorn vs L-HNN+NN-OT (0.808) vs truth (0.95):"
for c in 0 1 2 3; do python3 -c "import json;s=json.load(open('${RUN_DIR}/chain_${c}/svssm_hmc_multi_full_phi_summary.json'));n=s['nuts_diagnostics'];print(f'  chain_${c}: err_trigs={n[\"error_triggers_per_chain\"][0]} sample={s[\"sampling_wall_s\"]:.0f}s acc={s[\"accept_rate_overall\"]:.3f}')" 2>/dev/null; done
python -u - "$RUN_DIR" <<'PYEOF'
import numpy as np, sys
z=np.load(sys.argv[1]+"/svssm_hmc_multi_full_phi_samples.npz")
mu=z["mu"][...,0].ravel(); phi=z["phi_diag"][...,0].ravel(); s2=z["sigma_eta_sq"][...,0].ravel()
def cov(x,t): l,h=np.quantile(x,[.025,.975]); return l<=t<=h
print(f"  phi  : median {np.median(phi):.3f}  CI [{np.quantile(phi,.025):.3f},{np.quantile(phi,.975):.3f}]  cov={cov(phi,0.95)}")
print(f"  mu   : median {np.median(mu):.3f}   cov={cov(mu,0.0)}")
print(f"  sig2 : median {np.median(s2):.3f}   cov={cov(s2,0.09)}")
print(f"  P(phi<0)={100*np.mean(phi<0):.1f}%")
print("  -> if phi~=0.81: NN-OT faithful (0.81 is exact); if phi~=0.92: NN-OT shifts phi down")
PYEOF
