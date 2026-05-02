#!/usr/bin/env bash
# Run Qwen2-Audio ASR bench — meant to run after T6 finishes
# Can run alongside T11 (combined ~60 GB, within 119 GB limit)
set -euo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/asr_bench_qwen2audio.log
cd "$BASE"
echo "[$(date '+%H:%M:%S')] Starting Qwen2-Audio ASR bench" | tee "$LOG"
python3 phase1_asr_bench.py --models qwen2-audio-7b --samples 50 2>&1 | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] Done." | tee -a "$LOG"
