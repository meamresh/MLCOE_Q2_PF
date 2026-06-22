#!/bin/bash
# =============================================================================
# Grid-centering fix validation: L-HNN NUTS + NN-OT at d=2, truth B, with the
# operator's theta-grid RE-CENTERED on the truth (using the new --mu0/--phi0/
# --sig0 trainer args).
#
# WHY two pipelines: the previous check found a per-parameter posterior shift
# (NN-OT vs Sinkhorn) that tracked the operator's theta-GRID edges, because the
# trainer's DEFAULT grid did not match the truth. To test whether centering the
# grid on the truth CLOSES that shift, we need a Sinkhorn reference AT THE SAME
# truth B (the old Sinkhorn reference is at the old truth). So this script runs
# BOTH L-HNN+Sinkhorn (reference) and L-HNN+NN-OT (grid-centered) at truth B and
# compares them.
#
# Truth B (matched SNR sigma_h^2 = sigma_eta^2/(1-phi^2) = 1 per dim, distinct
# individual values):
#   mu        = [ 1.0, -1.0 ]
#   phi_diag  = [ 0.95, 0.80 ]
#   sigma_eta = [ 0.3122, 0.6 ]   ->  sigma_h^2 = [1.0, 1.0]
#   phi_off   = 0.05
# The operator grid is centered on this truth (--mu0/--phi0/--sig0 = truth),
# which is the fix under test.
#
# Usage:  bash scripts/exp/run_lhnn_nnot_d2_gridfix.sh
# Re-runnable: skips operator training / L-HNN pilots whose artifacts exist.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- shared config ----------------------------------------------------------
D=2; T=50; N=64; NLAMBDA=10; K=10; NBASIS=64

# truth B (grid will be centered on this)
MU="1.0,-1.0"
PHI_DIAG="0.95,0.80"
SIGMA_ETA="0.3122,0.6"          # sqrt(1-0.95^2)=0.3122, sqrt(1-0.8^2)=0.6 -> sigma_h^2=1 each
PHI_OFF="0.05"

# priors -- chosen to cover truth B (same for BOTH pipelines, so the comparison
# is fair). mu is weakly identified at T=50, so its prior is wide and centered
# at 0; phi/sigma priors bracket the (well-identified) truth.
PRIORS="--prior_mu_loc 0.0 --prior_mu_scale 1.5 \
--prior_phi_raw_loc 1.5 --prior_phi_raw_scale 1.0 \
--prior_log_sigma_eta_sq_loc -1.7 --prior_log_sigma_eta_sq_scale 1.0 \
--prior_phi_off_scale 0.2"

DATA_SEED=42; BASE_SEED=300

# NUTS sampling (same for both pipelines)
NUM_CHAINS=4; NUM_BURNIN=300; NUM_RESULTS=1000
STEP_SIZE=0.01; MAX_TREEDEPTH=8; TARGET_ACCEPT=0.65; DISPERSION=0.05

# L-HNN architecture + pilot
HIDDEN_UNITS=128; NUM_HIDDEN=3; LHNN_EPOCHS=2500; PILOT_TRAJ=60; PILOT_STEPS=60

# ---- output locations -------------------------------------------------------
NNOT_DIR="reports/nnot_d${D}_T${T}_N${N}_matchedB"      # B-centered operator
NNOT_WEIGHTS="${NNOT_DIR}/deeponet_nd.weights.h5"
SK_DIR="reports/d${D}_lhnn_sinkhorn_B"                  # Sinkhorn reference @ truth B
NN_DIR="reports/d${D}_lhnn_nnot_B"                      # NN-OT @ truth B (grid-centered)
SK_CACHE="${SK_DIR}/lhnn_d${D}_sink.weights.h5"
NN_CACHE="${NN_DIR}/lhnn_d${D}_nnot.weights.h5"
mkdir -p "$NNOT_DIR" "$SK_DIR" "$NN_DIR"

# common model/prior/data/NUTS flags shared by both launcher calls
COMMON="--d $D --T $T --N $N --n_lambda $NLAMBDA --K $K \
--mu $MU --phi_diag $PHI_DIAG --phi_off $PHI_OFF --sigma_eta $SIGMA_ETA \
$PRIORS --data_seed $DATA_SEED --base_seed $BASE_SEED \
--num_burnin $NUM_BURNIN --num_results $NUM_RESULTS \
--step_size $STEP_SIZE --max_treedepth $MAX_TREEDEPTH \
--target_accept_prob $TARGET_ACCEPT --dispersion $DISPERSION \
--hidden_units $HIDDEN_UNITS --num_hidden $NUM_HIDDEN --lhnn_epochs $LHNN_EPOCHS \
--num_pilot_trajectories $PILOT_TRAJ --pilot_steps_per_trajectory $PILOT_STEPS \
--kinetic_mh"

echo "============================================================"
echo "GRID-FIX VALIDATION  d=${D} T=${T} N=${N}  truth B (sigma_h^2=[1,1])"
echo "  operator (B-centered): ${NNOT_WEIGHTS}"
echo "  Sinkhorn reference   : ${SK_DIR}"
echo "  NN-OT (grid-centered): ${NN_DIR}"
echo "============================================================"

# =============================================================================
# STEP 1 -- train the operator with the grid CENTERED ON TRUTH B (the fix)
# =============================================================================
if [ -f "$NNOT_WEIGHTS" ]; then
  echo "[1] B-centered operator exists -> skip."
else
  echo "[1] training operator, grid centered on truth B ..."
  python -u -m scripts.exp.phase16_train_multi_nnot_nd \
    --d "$D" --T "$T" --N "$N" --n_theta 60 --seeds_per_theta 2 \
    --max_epochs 60 --patience 10 --n_basis "$NBASIS" --seed 42 \
    --mu0 "$MU" --phi0 "$PHI_DIAG" --sig0 "$SIGMA_ETA" --phi_off0 0.05 \
    --out_dir "$NNOT_DIR" 2>&1 | tee "${NNOT_DIR}/train_operator.log"
  [ -f "$NNOT_WEIGHTS" ] || { echo "[1] ERROR: no operator weights" >&2; exit 1; }
fi

# =============================================================================
# STEP 2 -- L-HNN + Sinkhorn at truth B  (the reference; NO --nnot_weights)
# =============================================================================
echo "[2] L-HNN + Sinkhorn reference @ truth B ..."
bash scripts/exp/launch_lhnn_nuts_parallel.sh "$SK_DIR" "$NUM_CHAINS" "$SK_CACHE" \
  $COMMON --progress_every 250 2>&1 | tee "${SK_DIR}/run.log"

# =============================================================================
# STEP 3 -- L-HNN + NN-OT at truth B  (grid-centered operator)
# =============================================================================
echo "[3] L-HNN + NN-OT @ truth B (B-centered operator) ..."
bash scripts/exp/launch_lhnn_nuts_parallel.sh "$NN_DIR" "$NUM_CHAINS" "$NN_CACHE" \
  $COMMON --nnot_weights "$NNOT_WEIGHTS" --nnot_n_basis "$NBASIS" \
  --progress_every 250 2>&1 | tee "${NN_DIR}/run.log"

# =============================================================================
# STEP 4 -- trace plots (both runs) + NN-OT-vs-Sinkhorn comparison
# =============================================================================
for DIR in "$SK_DIR" "$NN_DIR"; do
  if [ -f "${DIR}/svssm_hmc_multi_full_phi_samples.npz" ]; then
    python scripts/plot_trace_multi_full_phi.py --out_dir "$DIR"
    python scripts/plot_trace_stationary_cov.py --out_dir "$DIR"
  fi
done

echo "[4] comparison (recompute rank_rhat from npz; never trust stored) ..."
python -u - "$SK_DIR" "$NN_DIR" <<'PYEOF'
import numpy as np, sys
sys.path.insert(0, 'scripts/exp')
from compare_svssm_hmc_methods import rank_rhat
from scipy.stats import ks_2samp
SK, NN = sys.argv[1], sys.argv[2]
def params(d):
    z = np.load(d + "/svssm_hmc_multi_full_phi_samples.npz"); o = {}
    for i in range(2): o[f"mu_{i}"] = z["mu"][..., i]
    for i in range(2): o[f"phi_{i}"] = z["phi_diag"][..., i]
    o["phi_off"] = z["phi_off"][..., 0]
    for i in range(2): o[f"sig2_{i}"] = z["sigma_eta_sq"][..., i]
    return o
psk, pnn = params(SK), params(NN)
# truth B: sigma_eta_sq = [0.3122^2, 0.6^2] = [0.0975, 0.36]
truth = {"mu_0": 1.0, "mu_1": -1.0, "phi_0": 0.95, "phi_1": 0.80,
         "phi_off": 0.05, "sig2_0": 0.0975, "sig2_1": 0.36}
def w95(x): l, h = np.quantile(x, [.025, .975]); return h - l
def cov(x, t): l, h = np.quantile(x, [.025, .975]); return l <= t <= h
print(f"\n{'param':<9}{'truth':>8} | {'NN med':>8}{'SK med':>8}{'KS p':>7} | "
      f"{'NN R̂':>7}{'SK R̂':>7} | {'NNcov':>5}{'w NN/SK':>8}")
print('-' * 72)
nagree = ncov = 0
for k in psk:
    nm, sm = np.median(pnn[k]), np.median(psk[k])
    ks = ks_2samp(pnn[k].ravel(), psk[k].ravel()).pvalue
    c = cov(pnn[k].ravel(), truth[k]); ncov += c
    nagree += (ks > 0.05)
    wr = w95(pnn[k].ravel()) / w95(psk[k].ravel())
    print(f"{k:<9}{truth[k]:>8.3f} | {nm:>8.3f}{sm:>8.3f}{ks:>7.3f} | "
          f"{rank_rhat(pnn[k]):>7.3f}{rank_rhat(psk[k]):>7.3f} | "
          f"{('Y' if c else 'N'):>5}{wr:>8.2f}")
print('-' * 72)
print(f"NN-OT: coverage {ncov}/7  |  KS-indistinguishable from Sinkhorn: {nagree}/7")
print(f"mean width ratio NN/SK: "
      f"{np.mean([w95(pnn[k].ravel())/w95(psk[k].ravel()) for k in psk]):.2f}")
print("(prev check, DEFAULT grid: 2/7 KS-agree, phi_1 the worst at 1.35x wider.")
print(" if grid-centering worked, KS-agree rises and phi_1 width ratio -> ~1.)")
PYEOF

echo "============================================================"
echo "DONE. Sinkhorn-B: ${SK_DIR}  |  NN-OT-B: ${NN_DIR}"
echo "============================================================"
