#!/usr/bin/env bash
# Launch N chains of the NN-OT HMC driver in parallel (one process per chain),
# then run an aggregation pass that combines them into the standard
# svssm_hmc_neural_ot_{samples.npz, results.txt, summary.json}.
#
# Usage:
#     bash scripts/exp/run_nnot_parallel.sh
#     # or with overrides:
#     NUM_CHAINS=4 GRAPH_MODE=xla ARCH=deeponet \
#         bash scripts/exp/run_nnot_parallel.sh
#
# All env vars are optional. Anything else passed positionally is forwarded
# to the Python driver (after the per-process --chain_id), so you can do:
#     bash scripts/exp/run_nnot_parallel.sh --T 50 --num_burnin 200 --num_results 400
#
# Output:
#     ${OUT_DIR}/chains/chain_<i>.npz   (one per parallel process)
#     ${OUT_DIR}/svssm_hmc_neural_ot_samples.npz   (combined)
#     ${OUT_DIR}/svssm_hmc_neural_ot_results.txt   (combined)
#     ${OUT_DIR}/svssm_hmc_neural_ot_summary.json  (combined)
#
# CPU note: each parallel process pays its own ~133 s XLA compile cost
# when graph_mode=xla. At NUM_CHAINS=2 the parallel saving is marginal;
# at NUM_CHAINS=4+ it pays off cleanly.

set -euo pipefail

# ---------- Config (env-overrideable) ----------
NUM_CHAINS="${NUM_CHAINS:-2}"
ARCH="${ARCH:-deeponet}"
GRAPH_MODE="${GRAPH_MODE:-xla}"
T="${T:-20}"
N="${N:-64}"
NUM_BURNIN="${NUM_BURNIN:-100}"
NUM_RESULTS="${NUM_RESULTS:-200}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
OUT_DIR="${OUT_DIR:-reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/svssm_hmc_nnot_parallel}"
PY="${PY:-python}"

# Extra flags forwarded to every invocation (positional).
EXTRA_ARGS=("$@")

# ---------- Sanity ----------
if ! [[ "${NUM_CHAINS}" =~ ^[0-9]+$ ]] || (( NUM_CHAINS < 1 )); then
    echo "[error] NUM_CHAINS must be a positive integer (got '${NUM_CHAINS}')" >&2
    exit 1
fi

if [[ ! -f "scripts/exp/exp_hmc_svssm_neural_ot.py" ]]; then
    echo "[error] run this from the repo root (scripts/exp/exp_hmc_svssm_neural_ot.py not found)" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}/chains" "${OUT_DIR}/logs"

echo "==================================================================="
echo "Parallel NN-OT HMC launcher"
echo "==================================================================="
echo "  arch         = ${ARCH}"
echo "  graph_mode   = ${GRAPH_MODE}"
echo "  num_chains   = ${NUM_CHAINS} (parallel processes)"
echo "  T / N        = ${T} / ${N}"
echo "  burnin / res = ${NUM_BURNIN} / ${NUM_RESULTS}"
echo "  out_dir      = ${OUT_DIR}"
echo "  extra args   = ${EXTRA_ARGS[*]:-(none)}"
echo "==================================================================="

# Common args shared by all per-chain invocations.
COMMON_ARGS=(
    --arch "${ARCH}"
    --graph_mode "${GRAPH_MODE}"
    --num_chains "${NUM_CHAINS}"
    --num_burnin "${NUM_BURNIN}"
    --num_results "${NUM_RESULTS}"
    --T "${T}"
    --N "${N}"
    --progress_every "${PROGRESS_EVERY}"
    --out_dir "${OUT_DIR}"
)

# ---------- Launch chains in parallel ----------
pids=()
t0=$(date +%s)
echo
for ((c = 0; c < NUM_CHAINS; c++)); do
    log="${OUT_DIR}/logs/chain_${c}.log"
    echo "  [launch] chain ${c} -> ${log}"
    PYTHONPATH=. "${PY}" scripts/exp/exp_hmc_svssm_neural_ot.py \
        "${COMMON_ARGS[@]}" \
        --chain_id "${c}" \
        "${EXTRA_ARGS[@]}" \
        >"${log}" 2>&1 &
    pids+=($!)
done

# ---------- Wait + report exit codes ----------
echo
echo "  [wait] waiting for ${#pids[@]} chain process(es)..."
fail=0
for i in "${!pids[@]}"; do
    pid=${pids[i]}
    if wait "${pid}"; then
        echo "  [ok]   chain ${i} (pid ${pid}) exited 0"
    else
        rc=$?
        echo "  [FAIL] chain ${i} (pid ${pid}) exited ${rc}; see ${OUT_DIR}/logs/chain_${i}.log" >&2
        fail=$((fail + 1))
    fi
done
t_parallel=$(($(date +%s) - t0))
echo "  [wait] all chains done in ${t_parallel}s wall (parallel)"

if (( fail > 0 )); then
    echo "[error] ${fail} chain(s) failed; aborting before aggregation" >&2
    exit 2
fi

# ---------- Aggregate ----------
echo
echo "  [aggregate] combining ${NUM_CHAINS} chains into final report"
PYTHONPATH=. "${PY}" scripts/exp/exp_hmc_svssm_neural_ot.py \
    "${COMMON_ARGS[@]}" \
    --aggregate \
    "${EXTRA_ARGS[@]}"

echo
echo "==================================================================="
echo "Parallel run complete."
echo "  per-chain logs: ${OUT_DIR}/logs/chain_*.log"
echo "  per-chain npz : ${OUT_DIR}/chains/chain_*.npz"
echo "  combined      : ${OUT_DIR}/svssm_hmc_neural_ot_{samples.npz,results.txt,summary.json}"
echo "==================================================================="
