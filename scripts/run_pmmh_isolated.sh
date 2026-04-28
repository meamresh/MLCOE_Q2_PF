#!/usr/bin/env bash
# Run PMMH chains one at a time in fresh Python processes to avoid
# TFP eager memory leaks accumulating across chains. Each chain saves
# its samples to <out_dir>/chains/chain_{id}.npz; final invocation
# aggregates all chains and writes the diagnostic report.
#
# Usage:
#   scripts/run_pmmh_isolated.sh [NUM_CHAINS] [SAMPLES_PER_CHAIN] [BURN_PER_CHAIN]
#   scripts/run_pmmh_isolated.sh 4 20000 3000

set -euo pipefail

NUM_CHAINS=${1:-4}
SAMPLES=${2:-18000}
BURN=${3:-3000}

echo "================================================================"
echo "  PMMH isolated-chain driver"
echo "  num_chains=${NUM_CHAINS}  samples/chain=${SAMPLES}  burn/chain=${BURN}"
echo "================================================================"

for c in $(seq 0 $((NUM_CHAINS - 1))); do
    echo ""
    echo "================================================================"
    echo "  Launching chain ${c} (fresh Python process)"
    echo "================================================================"
    python -m src.experiments.exp_part3_bonus1b_pmmh_only \
        --chain_id "${c}" \
        --num_chains "${NUM_CHAINS}" \
        --samples_per_chain "${SAMPLES}" \
        --burn_per_chain "${BURN}"
done

echo ""
echo "================================================================"
echo "  Aggregating saved chains and running diagnostics"
echo "================================================================"
python -m src.experiments.exp_part3_bonus1b_pmmh_only \
    --aggregate \
    --num_chains "${NUM_CHAINS}" \
    --samples_per_chain "${SAMPLES}" \
    --burn_per_chain "${BURN}"
