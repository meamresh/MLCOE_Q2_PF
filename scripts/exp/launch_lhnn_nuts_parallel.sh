#!/bin/bash
# Parallel launcher for the multivariate full-Phi L-HNN NUTS driver.
#
# The chains in an L-HNN NUTS run are independent, but the engine's
# multi-chain helper runs them in a sequential Python loop (4 chains = 4x
# the sampling wall). The pilot+train, by contrast, happens ONCE. So the
# right pattern is:
#   1. train_only  -> pilot + train + cache the L-HNN (single process)
#   2. N chains in PARALLEL, each loading the cached L-HNN (--chain_id c)
#   3. stitch the per-chain npzs into one (chains, draws, .) dataset
#   4. run save_diagnostics on the stitched dir
#
# Each parallel chain is bit-identical to the sequential multi-chain run:
# same init_raw (rng(base_seed) drawn for all chains, indexed by chain_id),
# same chain_seed (base_seed + 1009*(c+1)), same shared CRN (base_seed).
#
# Usage:
#   bash scripts/exp/launch_lhnn_nuts_parallel.sh \
#        <out_dir> <num_chains> <weights_cache> [driver args ...]
#
# Do NOT put --out_dir, --weights_cache, --chain_id, --train_only, or
# --num_chains in the passthrough args; the launcher supplies them. Pass
# everything else (-d, --T, --N, truth, priors, NUTS/L-HNN settings).
#
# Example (d=2 T=50):
#   bash scripts/exp/launch_lhnn_nuts_parallel.sh \
#        reports/d2_T50_lhnn_nuts 4 reports/d2_T50_lhnn_nuts/lhnn_d2.weights.h5 \
#        --d 2 --T 50 --N 64 --n_lambda 10 --K 10 \
#        --mu 0.0,-0.3 --phi_diag 0.95,0.85 --phi_off 0.05 --sigma_eta 0.3,0.4 \
#        --prior_phi_off_scale 0.2 --num_burnin 200 --num_results 1000 \
#        --step_size 0.01 --max_treedepth 8 --progress_every 50
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <out_dir> <num_chains> <weights_cache> [driver args ...]" >&2
  exit 2
fi
OUT_DIR="$1"; NUM_CHAINS="$2"; CACHE="$3"; shift 3
DRIVER="scripts.exp.exp_hmc_svssm_multivariate_full_phi_lhnn_nuts"
mkdir -p "$OUT_DIR"

# ---- 1. pre-warm the cache (pilot + train) if missing ----
if [ -f "$CACHE" ]; then
  echo "[launch] cache present at $CACHE -> skipping pilot+train."
else
  echo "[launch] no cache at $CACHE -> running --train_only (pilot+train) ..."
  python -u -m "$DRIVER" "$@" \
    --num_chains "$NUM_CHAINS" --weights_cache "$CACHE" \
    --out_dir "$OUT_DIR" --train_only 2>&1 | tee "$OUT_DIR/train.log"
  if [ ! -f "$CACHE" ]; then
    echo "[launch] ERROR: train_only did not produce $CACHE" >&2; exit 1
  fi
fi

# ---- 2. launch N chains in parallel ----
echo "[launch] starting $NUM_CHAINS chains in parallel ..."
PIDS=()
for c in $(seq 0 $((NUM_CHAINS - 1))); do
  CDIR="$OUT_DIR/chain_$c"
  mkdir -p "$CDIR"
  python -u -m "$DRIVER" "$@" \
    --num_chains "$NUM_CHAINS" --weights_cache "$CACHE" \
    --chain_id "$c" --out_dir "$CDIR" > "$CDIR/chain.log" 2>&1 &
  pid=$!                       # macOS bash 3.2 has no ${arr[-1]}; capture $! directly
  PIDS+=("$pid")
  echo "  chain $c -> pid $pid  (log: $CDIR/chain.log)"
done

# ---- wait, tracking failures ----
FAIL=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[launch] chain $i (pid ${PIDS[$i]}) FAILED -- see $OUT_DIR/chain_$i/chain.log" >&2
    FAIL=1
  fi
done
if [ "$FAIL" -ne 0 ]; then
  echo "[launch] one or more chains failed; not stitching." >&2; exit 1
fi
echo "[launch] all $NUM_CHAINS chains done."

# ---- 3. stitch ----
CHAIN_DIRS=()
for c in $(seq 0 $((NUM_CHAINS - 1))); do CHAIN_DIRS+=("$OUT_DIR/chain_$c"); done
python scripts/stitch_multi_full_phi_chains.py \
  --chain_dirs "${CHAIN_DIRS[@]}" --out_dir "$OUT_DIR"

# ---- 4. diagnostics on the stitched dir ----
python scripts/save_diagnostics_multi_full_phi.py --out_dir "$OUT_DIR"
echo "[launch] done. Stitched samples + diagnostics in $OUT_DIR"
