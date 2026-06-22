#!/bin/bash
# Auto-orchestration for the d=4 wave:
#   1. wait for the 4 Sinkhorn baseline chains to finish
#   2. stitch + diagnose the baseline
#   3. launch 4 NN-OT chains (v2 operator, n_basis=128), wait for them
#   4. stitch + diagnose NN-OT
#   5. SK-vs-NN comparison (KS / coverage / medians)
# Logs to /tmp/d4_orchestrate.log; writes DONE marker at the end.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
LOG=/tmp/d4_orchestrate.log
SKDIR=reports/d4_T200_sinkhorn
NNDIR=reports/d4_T200_nnot
W=reports/d4_T200_training_v2/deeponet_nd.weights.h5
echo "[orch $(date +%T)] start; waiting for baseline chains" | tee -a "$LOG"

# 1. wait for baseline (all 4 "chain done in")
until [ "$(grep -l 'chain done in' /tmp/d4_T200_sk_0.log /tmp/d4_T200_sk_1.log /tmp/d4_T200_sk_2.log /tmp/d4_T200_sk_3.log 2>/dev/null | wc -l | tr -d ' ')" = "4" ]; do
  sleep 60
done
echo "[orch $(date +%T)] baseline done; stitching" | tee -a "$LOG"

# 2. stitch + diagnose baseline
python -u scripts/stitch_multi_full_phi_chains.py \
  --chain_dirs $SKDIR/chain_0 $SKDIR/chain_1 $SKDIR/chain_2 $SKDIR/chain_3 \
  --out_dir $SKDIR >> "$LOG" 2>&1
python -u scripts/save_diagnostics_multi_full_phi.py --out_dir $SKDIR >> "$LOG" 2>&1 || true
echo "[orch $(date +%T)] baseline diagnostics saved" | tee -a "$LOG"

# 3. launch 4 NN-OT chains
if [ ! -f "$W" ]; then echo "[orch] MISSING weights $W; abort" | tee -a "$LOG"; exit 2; fi
mkdir -p $NNDIR/chain_0 $NNDIR/chain_1 $NNDIR/chain_2 $NNDIR/chain_3
pids=()
for i in 0 1 2 3; do
  python -u -m scripts.exp.exp_hmc_svssm_multivariate_full_phi \
    --d 4 --T 200 --N 64 --n_lambda 10 --K 10 --L 5 --step_size 0.02 \
    --num_chains 4 --num_burnin 400 --num_results 750 --dispersion 0.10 \
    --data_seed 42 --base_seed 300 --progress_every 150 \
    --mu "0.0,0.0,0.0,0.0" --phi_diag "0.95,0.9,0.85,0.8" \
    --phi_off "0.05,0.05,0.05,0.05,0.05,0.05" --sigma_eta "0.3,0.35,0.4,0.45" \
    --prior_mu_loc 0.0 --prior_mu_scale 1.0 \
    --prior_phi_raw_loc 2.0 --prior_phi_raw_scale 0.5 \
    --prior_phi_off_scale 0.2 \
    --prior_log_sigma_eta_sq_loc -2.0 --prior_log_sigma_eta_sq_scale 1.0 \
    --nnot_weights "$W" --nnot_n_basis 128 --chain_id $i \
    --out_dir $NNDIR/chain_$i > /tmp/d4_T200_nn_$i.log 2>&1 &
  pids+=($!)
done
echo "[orch $(date +%T)] NN-OT chains launched (pids ${pids[*]})" | tee -a "$LOG"
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[orch $(date +%T)] NN-OT chains done; stitching" | tee -a "$LOG"

# 4. stitch + diagnose NN-OT
python -u scripts/stitch_multi_full_phi_chains.py \
  --chain_dirs $NNDIR/chain_0 $NNDIR/chain_1 $NNDIR/chain_2 $NNDIR/chain_3 \
  --out_dir $NNDIR >> "$LOG" 2>&1
python -u scripts/save_diagnostics_multi_full_phi.py --out_dir $NNDIR >> "$LOG" 2>&1 || true

# 5. SK-vs-NN comparison
python -u - "$SKDIR" "$NNDIR" >> "$LOG" 2>&1 <<'PYEOF'
import sys, numpy as np, glob
from scipy import stats
sys.path.insert(0, 'scripts/exp')
from compare_svssm_hmc_methods import rank_rhat
skd, nnd = sys.argv[1], sys.argv[2]
def load(d):
    z = np.load(d + "/svssm_hmc_multi_full_phi_samples.npz")
    return z
sk, nn = load(skd), load(nnd)
dd = sk["mu"].shape[2]
off = [(i,j) for i in range(dd) for j in range(i+1,dd)]
print("\n===== d=4 Sinkhorn vs NN-OT comparison =====")
print(f"{'param':<16}{'truth':>8}{'SK med':>9}{'NN med':>9}{'|Δ|':>7}{'KS p':>8}{'covSK':>6}{'covNN':>6}")
def rows():
    for i in range(dd): yield f"mu_{i}", sk["mu"][...,i], nn["mu"][...,i], float(sk["mu_truth"][i])
    for i in range(dd): yield f"phi_diag_{i}", sk["phi_diag"][...,i], nn["phi_diag"][...,i], float(sk["phi_diag_truth"][i])
    for k,(i,j) in enumerate(off): yield f"phi_off_{i}{j}", sk["phi_off"][...,k], nn["phi_off"][...,k], float(sk["phi_off_truth"][k])
    for i in range(dd): yield f"sigma_eta_sq_{i}", sk["sigma_eta_sq"][...,i], nn["sigma_eta_sq"][...,i], float(sk["sigma_eta_sq_truth"][i])
ks_pass=0; tot=0
for name, s, n, t in rows():
    sm, nm = float(np.median(s)), float(np.median(n))
    p = stats.ks_2samp(s.ravel(), n.ravel()).pvalue
    cs = "Y" if np.percentile(s,2.5)<=t<=np.percentile(s,97.5) else "N"
    cn = "Y" if np.percentile(n,2.5)<=t<=np.percentile(n,97.5) else "N"
    ks_pass += p>0.05; tot+=1
    print(f"{name:<16}{t:>+8.3f}{sm:>+9.3f}{nm:>+9.3f}{abs(sm-nm):>7.3f}{p:>8.3f}{cs:>6}{cn:>6}")
print(f"KS pass (p>0.05): {ks_pass}/{tot}")
PYEOF

echo "[orch $(date +%T)] DONE — comparison appended above" | tee -a "$LOG"
touch reports/d4_T200_nnot/ORCHESTRATION_DONE
