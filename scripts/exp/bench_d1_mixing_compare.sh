#!/bin/bash
# d=1 chain-mixing benchmark: full-Phi driver (heavy vs light numerical
# guards) vs the diagonal driver, at IDENTICAL HMC config.
#
# Motivation: check whether the full-Phi `call_mat_phi` path (1e-3 ridge +
# tight ±50/±1e3 clips + matrix solves) mixes worse than the diagonal path
# (clean scalar arithmetic), and whether reducing the ridge/clips
# ("light") helps.
#
# Outputs per-parameter rank-Rhat + bulk-ESS for all three.
#
# CAVEAT (important): at d=1 the full-Phi driver and the diagonal driver
# generate DIFFERENT y realizations from the same --data_seed (different
# RNG consumption in gen_svssm_multi_phi_mat vs gen_svssm_multi), so this
# is NOT a perfectly controlled comparison — the diagonal run fits a
# different dataset. For a strict apples-to-apples test, generate one y
# series and feed both drivers the same observations. Also: at the small
# trimmed budget (2 chains x 150) rank-Rhat/ESS are noisy.
#
# Usage:  bash scripts/exp/bench_d1_mixing_compare.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- shared config (edit here) ----
T=100; N=64; NCHAINS=2; BURN=100; RESULTS=150
STEP=0.05; DISP=0.10; DSEED=42; BSEED=300
# truth (d=1): mu, phi, sigma_eta
MU=0.0; PHI=0.95; SIG=0.3
# informative recovery priors
PR="--prior_mu_loc 0.0 --prior_mu_scale 1.0 \
--prior_phi_raw_loc 2.0 --prior_phi_raw_scale 0.5 \
--prior_log_sigma_eta_sq_loc -2.0 --prior_log_sigma_eta_sq_scale 1.0"

HEAVY=reports/bench_d1_matphi_heavy
LIGHT=reports/bench_d1_matphi_light
DIAG=reports/bench_d1_diag
rm -rf $HEAVY $LIGHT $DIAG; mkdir -p $HEAVY $LIGHT $DIAG

# ---- (A) full-Phi driver, HEAVY guards (defaults: ridge 1e-3, clip 50/1e3) ----
python -u -m scripts.exp.exp_hmc_svssm_multivariate_full_phi \
  --d 1 --T $T --N $N --n_lambda 10 --K 10 --L 5 --step_size $STEP \
  --num_chains $NCHAINS --num_burnin $BURN --num_results $RESULTS --dispersion $DISP \
  --data_seed $DSEED --base_seed $BSEED --progress_every 75 \
  --mu $MU --phi_diag $PHI --phi_off "" --sigma_eta $SIG $PR --prior_phi_off_scale 0.2 \
  --out_dir $HEAVY > /tmp/bench_heavy.log 2>&1 &

# ---- (B) full-Phi driver, LIGHT guards (ridge 1e-6, loose clips) ----
python -u -m scripts.exp.exp_hmc_svssm_multivariate_full_phi \
  --d 1 --T $T --N $N --n_lambda 10 --K 10 --L 5 --step_size $STEP \
  --num_chains $NCHAINS --num_burnin $BURN --num_results $RESULTS --dispersion $DISP \
  --data_seed $DSEED --base_seed $BSEED --progress_every 75 \
  --mu $MU --phi_diag $PHI --phi_off "" --sigma_eta $SIG $PR --prior_phi_off_scale 0.2 \
  --mat_phi_ridge 1e-6 --mat_phi_clip_particle 500 --mat_phi_clip_P 1e5 \
  --out_dir $LIGHT > /tmp/bench_light.log 2>&1 &

# ---- (C) diagonal driver (clean scalar path) ----
# CRITICAL: the diagonal driver defaults --use_windowed_adaptive to FALSE
# (the full-Phi driver defaults it to TRUE). Pass it explicitly so both
# pipelines use the SAME windowed-adaptive + dense-mass kernel — otherwise
# the diagonal run uses vanilla HMC and the comparison is unfair.
python -u -m scripts.exp.exp_hmc_svssm_multivariate \
  --d 1 --T $T --N $N --n_lambda 10 --L 5 --step_size $STEP \
  --num_chains $NCHAINS --num_burnin $BURN --num_results $RESULTS --dispersion $DISP \
  --data_seed $DSEED --base_seed $BSEED \
  --mu $MU --phi $PHI --sigma_eta $SIG $PR \
  --use_windowed_adaptive \
  --out_dir $DIAG > /tmp/bench_diag.log 2>&1 &

echo "launched 3 runs; waiting..."
wait
echo "all done."

# ---- per-parameter rank-Rhat + bulk-ESS comparison ----
python -u - "$HEAVY" "$LIGHT" "$DIAG" <<'PYEOF'
import numpy as np, sys
sys.path.insert(0, 'scripts/exp')
from compare_svssm_hmc_methods import rank_rhat, bulk_ess

heavy, light, diag = sys.argv[1], sys.argv[2], sys.argv[3]

def rr(x):  # x: (chains, draws)
    return rank_rhat(x), bulk_ess(x)

def load_matphi(base):
    z = np.load(base + "/svssm_hmc_multi_full_phi_samples.npz")
    return {"mu": z["mu"][..., 0], "phi": z["phi_diag"][..., 0],
            "sigma_eta_sq": z["sigma_eta_sq"][..., 0]}

def load_diag(base):
    z = np.load(base + "/svssm_hmc_multi_samples.npz")
    sc = z["samples_constrained"]          # (chains, draws, 3, 1) at d=1
    if sc.ndim == 4:
        sc = sc[..., 0]                    # -> (chains, draws, 3)
    return {"mu": sc[..., 0], "phi": sc[..., 1], "sigma_eta_sq": sc[..., 2]}

runs = [("heavy mat-phi", load_matphi(heavy)),
        ("light mat-phi", load_matphi(light)),
        ("diagonal     ", load_diag(diag))]

print(f"\n{'run':<16}{'param':<14}{'rank-Rhat':>11}{'bulk-ESS':>10}")
print('-' * 51)
for name, d in runs:
    for p in ["mu", "phi", "sigma_eta_sq"]:
        r, b = rr(d[p])
        print(f"{name:<16}{p:<14}{r:>11.3f}{b:>10.0f}")
    print('-' * 51)

print("\nNOTE: full-Phi and diagonal drivers generate different y at the same")
print("--data_seed (different RNG in their gen functions). For a strict test,")
print("feed both the SAME observations. rank-Rhat/ESS noisy at 2x150.")
PYEOF
