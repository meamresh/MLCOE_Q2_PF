#!/bin/bash
# =============================================================================
# Parallel h_0 ablation: run the N chains of ablate_init_h0.py CONCURRENTLY
# (one process per chain, each covering all three init types), then aggregate.
#
# Each chain process writes raw samples to {OUT}/chain_C/{init_type}.npy;
# the final --aggregate pass stacks them across chains, computes split-Rhat +
# summaries, prints the comparison table, and writes h0_ablation_results.json.
#
# Usage:
#   bash scripts/exp/launch_ablate_h0_parallel.sh <out_dir> <num_chains> [extra args...]
#
# Example (windowed dense mass, 4 chains x (300 burn + 1000 sample), T=50):
#   bash scripts/exp/launch_ablate_h0_parallel.sh \
#     reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/h0_ablation_windowed 4 \
#     --use_windowed_adaptive --T 50 --num_burnin 300 --num_results 1000
#
# bash 3.2 compatible (no ${arr[-1]}); captures each PID with $!.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT="${1:?usage: launch_ablate_h0_parallel.sh <out_dir> <num_chains> [extra args...]}"
NCH="${2:?need num_chains}"
shift 2
EXTRA="$@"   # passthrough: --use_windowed_adaptive --T ... --num_burnin ... etc.

mkdir -p "$OUT"
echo "=== parallel h_0 ablation: $NCH chains -> $OUT ==="
echo "    extra args: $EXTRA"

PIDS=""
for c in $(seq 0 $((NCH - 1))); do
  python3 -u -m scripts.exp.ablate_init_h0 \
    --chain_id "$c" --num_chains "$NCH" --out_dir "$OUT" $EXTRA \
    > "$OUT/chain_${c}.log" 2>&1 &
  pid=$!
  PIDS="$PIDS $pid"
  echo "  launched chain $c  (pid $pid)  -> $OUT/chain_${c}.log"
done

echo "  waiting for $NCH chains ..."
FAIL=0
for p in $PIDS; do
  if ! wait "$p"; then
    echo "  [WARN] pid $p exited non-zero" >&2
    FAIL=1
  fi
done
[ "$FAIL" -eq 0 ] && echo "  all chains finished OK" || echo "  [WARN] >=1 chain failed (see logs)"

echo "=== aggregating ==="
python3 -u -m scripts.exp.ablate_init_h0 \
  --aggregate --num_chains "$NCH" --out_dir "$OUT" $EXTRA \
  2>&1 | tee "$OUT/aggregate.log"

echo "=== done: $OUT/h0_ablation_results.json ==="
