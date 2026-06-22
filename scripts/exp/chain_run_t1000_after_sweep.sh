#!/bin/bash
# Wait for the running T-sweep (rerun_lhnn_nnot_tsweep_2500.sh) to finish,
# then auto-launch the T=1000 run. Lets the T=1000 run start hands-off after
# the sweep without oversubscribing cores (they run sequentially).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[chain] $(date)  waiting for T-sweep wrapper to exit ..."
sleep 60
while pgrep -f "rerun_lhnn_nnot_tsweep_2500" >/dev/null 2>&1; do
  sleep 120
done
echo "[chain] $(date)  sweep finished -> launching T=1000 ..."
bash scripts/exp/run_lhnn_nnot_d2_T1000.sh
echo "[chain] $(date)  T=1000 done."
