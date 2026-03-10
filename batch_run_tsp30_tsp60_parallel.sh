#!/bin/bash
# 并行运行 Tsp30 和 Tsp60 两批实验（同一终端后台 + wait）
# 日志分别落在 logs/batch_run_tsp30_* 和 logs/batch_run_tsp60_*

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Tsp30 and Tsp60 in parallel..."
./batch_run_tsp30.sh &  PID1=$!
./batch_run_tsp60.sh &  PID2=$!
wait $PID1; R1=$?
wait $PID2; R2=$?
echo "Tsp30 exit: $R1, Tsp60 exit: $R2"
[ $R1 -eq 0 ] && [ $R2 -eq 0 ] && exit 0 || exit 1
