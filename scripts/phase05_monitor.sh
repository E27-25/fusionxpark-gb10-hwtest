#!/usr/bin/env bash
# Phase 0.5 — Background GPU Monitor
# Logs nvidia-smi stats every second alongside any experiment
# Usage: ./phase05_monitor.sh [tag] &   (run in background)
#        kill %1  (to stop)
#
# Output: logs/monitor_<tag>_<timestamp>.csv

BASE=/home/student/Desktop/Test
LOGS=$BASE/logs
mkdir -p "$LOGS"

TAG="${1:-monitor}"
TS=$(date '+%Y%m%d_%H%M%S')
OUT="$LOGS/${TAG}_${TS}.csv"

echo "timestamp,power_w,temp_c,mem_used_mb,mem_total_mb,gpu_util_pct,mem_util_pct" > "$OUT"

echo "  [monitor] Logging to: $OUT (Ctrl+C or kill $$ to stop)"

trap "echo '  [monitor] Stopped. Saved: $OUT'" EXIT INT TERM

while true; do
    ts=$(date '+%Y-%m-%dT%H:%M:%S')
    stats=$(nvidia-smi \
        --query-gpu=power.draw,temperature.gpu,memory.used,memory.total,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits 2>/dev/null || echo "0,0,0,0,0,0")
    echo "${ts},${stats}" | tr -d ' ' >> "$OUT"
    sleep 1
done
