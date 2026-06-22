#!/usr/bin/env bash
# Run L-HNN HMC training (dispersed pilots, once) and then 4 isolated
# sampling chains in fresh Python processes.
#
# Steps:
#   1. --train  : train L-HNN once with dispersed-pilot trajectories
#                 (fixes the wrong-mode issue from a single starting
#                 point); save weights + training meta to disk.
#   2. --chain_id 0..N-1 : load trained L-HNN, run one chain, save .npz.
#   3. --aggregate : load chains + meta, write diagnostic report.
#
# Usage:
#   scripts/run_lhnn_isolated.sh [NUM_CHAINS] [SAMPLES_PER_CHAIN] [BURN_PER_CHAIN]
#   scripts/run_lhnn_isolated.sh 4 5000 1000

set -euo pipefail

NUM_CHAINS=${1:-4}
SAMPLES=${2:-3000}
BURN=${3:-1000}

echo "================================================================"
echo "  L-HNN HMC isolated-chain driver  (dispersed-pilot training)"
echo "  num_chains=${NUM_CHAINS}  samples/chain=${SAMPLES}  burn/chain=${BURN}"
echo "================================================================"

echo ""
echo "================================================================"
echo "  STEP 1: Train L-HNN with dispersed pilots"
echo "================================================================"
python -m src.experiments.exp_part3_bonus1b_lhnn_only \
    --train \
    --num_chains "${NUM_CHAINS}"

for c in $(seq 0 $((NUM_CHAINS - 1))); do
    echo ""
    echo "================================================================"
    echo "  STEP 2.${c}: Launching L-HNN sampling chain ${c} (fresh process)"
    echo "================================================================"
    python -m src.experiments.exp_part3_bonus1b_lhnn_only \
        --chain_id "${c}" \
        --num_chains "${NUM_CHAINS}" \
        --samples_per_chain "${SAMPLES}" \
        --burn_per_chain "${BURN}"
done

echo ""
echo "================================================================"
echo "  STEP 3: Aggregating saved chains and running diagnostics"
echo "================================================================"
python -m src.experiments.exp_part3_bonus1b_lhnn_only \
    --aggregate \
    --num_chains "${NUM_CHAINS}" \
    --samples_per_chain "${SAMPLES}" \
    --burn_per_chain "${BURN}"
