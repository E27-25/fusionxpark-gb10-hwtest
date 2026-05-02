#!/usr/bin/env bash
# Run inference sweep for non-gated models safe alongside T21 (~23 GB used)
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/sweep_now.log
cd "$BASE"
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()  { echo "  ✓ $*" | tee -a "$LOG"; }
fail(){ echo "  ✗ $*" | tee -a "$LOG"; }
mkdir -p "$BASE/logs"

log "=== Sweep: non-gated models (safe while T21 runs) ==="

# XL: DeepSeek-R1-Distill models (~16 GB and ~64 GB)
log "--- XL: DeepSeek-R1-Distill-Llama-8B (BF16, ~16 GB) ---"
python3 phase1_inference_bench.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --dtype bfloat16 \
    --batch-sizes 1 4 8 --prompts short medium \
    2>&1 | tee -a "$LOG" && ok "DeepSeek-R1-8B done" || fail "DeepSeek-R1-8B failed"
sleep 5

# VLM: Qwen2.5-VL-7B (~16 GB)
log "--- VLM: Qwen2.5-VL-7B-Instruct (BF16, ~16 GB) ---"
python3 phase1_inference_bench.py \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" --dtype bfloat16 \
    --batch-sizes 1 4 --prompts short \
    2>&1 | tee -a "$LOG" && ok "Qwen2.5-VL-7B done" || fail "Qwen2.5-VL-7B failed"
sleep 5

log "=== Sweep done (safe models). Large/XL/MoE deferred until T21 finishes ==="
