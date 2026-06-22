#!/usr/bin/env bash
# Run SVSSM HMC chains in PARALLEL (one process per chain), then aggregate.
#
# Each chain runs in its own Python process and writes to
#   <OUT_DIR>/chains/chain_<id>.npz
# After all N processes finish, a final --aggregate run loads them all and
# writes the combined results.txt / summary.json / .npz / plots in <OUT_DIR>.
#
# Per-process TF threading is capped so 4 simultaneous processes don't
# fight for the same CPU cores. Adjust CPU_PER_CHAIN to match your machine
# (e.g. 8-core box, 4 chains -> 2 cores/chain).
#
# Usage:
#   scripts/run_svssm_hmc_parallel.sh
#   scripts/run_svssm_hmc_parallel.sh --T 100 --num_chains 4 --num_burnin 300 --num_results 300
#   scripts/run_svssm_hmc_parallel.sh --out_dir reports/.../svssm_hmc_parallel_long
#
# Pass any additional --flag value pairs as extra args; they're forwarded to
# the Python entry point. See `python -m scripts.exp.exp_hmc_svssm --help`.

set -euo pipefail

# Defaults (overridable by passing --num_chains / --T / etc.):
NUM_CHAINS=4
OUT_DIR="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc_parallel"
CPU_PER_CHAIN=2

# Pull --num_chains and --out_dir out of the args list so we can target the
# right number of subprocesses and the right output directory. All other
# args are forwarded verbatim to each Python invocation.
PYARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_chains)
            NUM_CHAINS="$2"; PYARGS+=("--num_chains" "$2"); shift 2;;
        --out_dir)
            OUT_DIR="$2"; PYARGS+=("--out_dir" "$2"); shift 2;;
        --cpu_per_chain)
            CPU_PER_CHAIN="$2"; shift 2;;
        *)
            PYARGS+=("$1"); shift;;
    esac
done

mkdir -p "${OUT_DIR}/chains"

echo "============================================================"
echo "  SVSSM HMC parallel driver"
echo "  num_chains   = ${NUM_CHAINS}"
echo "  out_dir      = ${OUT_DIR}"
echo "  cpu/chain    = ${CPU_PER_CHAIN}"
echo "  extra args   = ${PYARGS[*]:-(none)}"
echo "============================================================"

PIDS=()
for c in $(seq 0 $((NUM_CHAINS - 1))); do
    LOG="${OUT_DIR}/chains/chain_${c}.log"
    echo "  launching chain ${c} -> ${LOG}"
    TF_NUM_INTEROP_THREADS=${CPU_PER_CHAIN} \
    TF_NUM_INTRAOP_THREADS=${CPU_PER_CHAIN} \
    OMP_NUM_THREADS=${CPU_PER_CHAIN} \
    python -m scripts.exp.exp_hmc_svssm \
        "${PYARGS[@]}" \
        --chain_id "${c}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "  waiting for ${NUM_CHAINS} chains (PIDs: ${PIDS[*]})..."
EXIT=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "  chain (PID ${pid}) FAILED"
        EXIT=1
    fi
done

if [[ ${EXIT} -ne 0 ]]; then
    echo ""
    echo "  one or more chains failed; tails of chain logs:"
    for c in $(seq 0 $((NUM_CHAINS - 1))); do
        echo "  --- chain ${c} ---"
        tail -8 "${OUT_DIR}/chains/chain_${c}.log" || true
    done
    exit ${EXIT}
fi

echo ""
echo "  all ${NUM_CHAINS} chains finished. aggregating..."
echo ""
python -m scripts.exp.exp_hmc_svssm \
    "${PYARGS[@]}" \
    --aggregate

echo ""
echo "  done. results in ${OUT_DIR}/"
