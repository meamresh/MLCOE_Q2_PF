#!/bin/bash
# =============================================================================
# Re-run the L-HNN+NN-OT d=2 T-sweep (T=50,200,500) at 2500 draws for converged
# R-hat/ESS, reusing the EXISTING operators + L-HNN caches (sampling-only --
# no pilots, no operator training). Then the cross-T stationary-vs-individual
# comparison on the converged chains.
#
# All three share truth B (matched SNR) and identical priors; only T differs.
# Each cache was trained against its own T's NN-OT target, so we point each
# re-run at the matching cache/operator. New _2500 out_dirs preserve the
# 1000-draw runs.
#
# Usage:  bash scripts/exp/rerun_lhnn_nnot_tsweep_2500.sh
# Re-runnable: launcher skips train (cache present) -> chains only.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# shared config (truth B + priors; identical to the 1000-draw runs except draws)
MU="1.0,-1.0"; PHI_DIAG="0.95,0.80"; SIGMA_ETA="0.3122,0.6"; PHI_OFF="0.05"
PRIORS="--prior_mu_loc 0.0 --prior_mu_scale 1.5 \
--prior_phi_raw_loc 1.5 --prior_phi_raw_scale 1.0 \
--prior_log_sigma_eta_sq_loc -1.7 --prior_log_sigma_eta_sq_scale 1.0 \
--prior_phi_off_scale 0.2"
N=64; NLAMBDA=10; K=10; NBASIS=64
NUM_CHAINS=4; NUM_BURNIN=500; NUM_RESULTS=3000
STEP=0.01; TREEDEPTH=8; ACCEPT=0.65; DISP=0.05
HID=128; NHID=3   # MUST match the cached L-HNN architecture

for T in 50 200 500; do
  if [ "$T" = "50" ]; then SRC="reports/d2_lhnn_nnot_B"; else SRC="reports/d2_lhnn_nnot_B_T${T}"; fi
  OP="reports/nnot_d2_T${T}_N64_matchedB/deeponet_nd.weights.h5"
  CACHE="${SRC}/lhnn_d2_nnot.weights.h5"
  OUT="reports/d2_lhnn_nnot_B_T${T}_2500"
  echo "############################################################"
  echo "# T=${T}  ->  ${OUT}  (cache ${CACHE})"
  echo "############################################################"
  if [ ! -f "$CACHE" ] || [ ! -f "$OP" ]; then
    echo "ERROR: missing cache or operator for T=${T}" >&2; exit 1
  fi
  bash scripts/exp/launch_lhnn_nuts_parallel.sh "$OUT" "$NUM_CHAINS" "$CACHE" \
    --d 2 --T "$T" --N "$N" --n_lambda "$NLAMBDA" --K "$K" \
    --mu "$MU" --phi_diag "$PHI_DIAG" --phi_off "$PHI_OFF" --sigma_eta "$SIGMA_ETA" \
    $PRIORS --data_seed 42 --base_seed 300 \
    --nnot_weights "$OP" --nnot_n_basis "$NBASIS" \
    --num_burnin "$NUM_BURNIN" --num_results "$NUM_RESULTS" \
    --step_size "$STEP" --max_treedepth "$TREEDEPTH" \
    --target_accept_prob "$ACCEPT" --dispersion "$DISP" \
    --hidden_units "$HID" --num_hidden "$NHID" \
    --kinetic_mh --progress_every 500 2>&1 | tee "${OUT}/run.log"
  if [ -f "${OUT}/svssm_hmc_multi_full_phi_samples.npz" ]; then
    python scripts/plot_trace_multi_full_phi.py --out_dir "$OUT"
    python scripts/plot_trace_stationary_cov.py --out_dir "$OUT"
  fi
done

# ---- cross-T stationary-vs-individual comparison (converged chains) ----------
echo "############################################################"
echo "# cross-T: derived stationary Sigma_h vs individual params"
echo "############################################################"
python -u - <<'PYEOF'
import numpy as np, sys
sys.path.insert(0,'scripts/exp'); sys.path.insert(0,'scripts')
from compare_svssm_hmc_methods import rank_rhat, bulk_ess
from plot_trace_stationary_cov import smith_doubling
Phi_t=np.array([[0.95,0.05],[0.0,0.80]]); Se_t=np.diag([0.0975,0.36])
Sh_t=smith_doubling(Phi_t,Se_t); rho_t=Sh_t[0,1]/np.sqrt(Sh_t[0,0]*Sh_t[1,1])
print(f"TRUTH: sigma_h^2=[{Sh_t[0,0]:.3f},{Sh_t[1,1]:.3f}] rho_h={rho_t:.3f} | phi_1=0.80 sig2_1=0.360")
def derived(d):
    z=np.load(d+"/svssm_hmc_multi_full_phi_samples.npz")
    pd=z["phi_diag"].reshape(-1,2);po=z["phi_off"].reshape(-1,1);s2=z["sigma_eta_sq"].reshape(-1,2)
    n=pd.shape[0];Phi=np.zeros((n,2,2));Phi[:,0,0]=pd[:,0];Phi[:,1,1]=pd[:,1];Phi[:,0,1]=po[:,0]
    Se=np.zeros((n,2,2));Se[:,0,0]=s2[:,0];Se[:,1,1]=s2[:,1];Sh=smith_doubling(Phi,Se)
    return Sh[:,0,0],Sh[:,1,1],Sh[:,0,1]/np.sqrt(Sh[:,0,0]*Sh[:,1,1]),pd[:,1],s2[:,1],z
def L(nm,x,t):
    lo,hi=np.quantile(x,[.025,.975]);md=np.median(x)
    return f"{nm:<12} med={md:7.3f} CIw={hi-lo:6.3f} err%={100*(md-t)/t:+6.1f} cov={'Y' if lo<=t<=hi else 'N'}"
for T in [50,200,500]:
    d=f"reports/d2_lhnn_nnot_B_T{T}_2500"
    sh0,sh1,rho,phi1,s21,z=derived(d)
    mx=max(rank_rhat(z[k][...,i]) for k in ["mu","phi_diag","sigma_eta_sq"] for i in range(2))
    print(f"\nT={T}  (max raw rank-Rhat {mx:.3f})")
    print("  "+L("sigma_h^2_0",sh0,Sh_t[0,0])); print("  "+L("sigma_h^2_1",sh1,Sh_t[1,1]))
    print("  "+L("rho_h",rho,rho_t))
    print("  "+L("phi_1",phi1,0.80)+"   "+L("sig2_1",s21,0.36))
PYEOF
echo "DONE."
