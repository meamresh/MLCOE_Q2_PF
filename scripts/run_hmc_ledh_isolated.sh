#!/usr/bin/env bash
# Run HMC-LEDH chains one at a time in fresh Python processes.
# Each chain saves to <out_dir>/chains/chain_{id}.npz; final invocation
# aggregates and writes the diagnostic report.
#
# Usage:
#   scripts/run_hmc_ledh_isolated.sh [NUM_CHAINS] [SAMPLES_PER_CHAIN] [BURN_PER_CHAIN] [SAMPLER]
#   scripts/run_hmc_ledh_isolated.sh 4 1500 750 custom
#   scripts/run_hmc_ledh_isolated.sh 4 200 100 tfp

set -euo pipefail

NUM_CHAINS=${1:-4}
SAMPLES=${2:-1500}
BURN=${3:-750}
SAMPLER=${4:-custom}

echo "================================================================"
echo "  HMC-LEDH isolated-chain driver"
echo "  num_chains=${NUM_CHAINS}  samples/chain=${SAMPLES}  burn/chain=${BURN}  sampler=${SAMPLER}"
echo "================================================================"

for c in $(seq 0 $((NUM_CHAINS - 1))); do
    echo ""
    echo "================================================================"
    echo "  Launching HMC-LEDH chain ${c} (fresh Python process)"
    echo "================================================================"
    python -m src.experiments.exp_part3_bonus1b_hmc_ledh_only_old \
        --chain_id "${c}" \
        --num_chains "${NUM_CHAINS}" \
        --samples_per_chain "${SAMPLES}" \
        --burn_per_chain "${BURN}" \
        --sampler "${SAMPLER}"
done

echo ""
echo "================================================================"
echo "  Aggregating saved chains and running diagnostics"
echo "================================================================"
python -m src.experiments.exp_part3_bonus1b_hmc_ledh_only_old \
    --aggregate \
    --num_chains "${NUM_CHAINS}" \
    --samples_per_chain "${SAMPLES}" \
    --burn_per_chain "${BURN}" \
    --sampler "${SAMPLER}"
