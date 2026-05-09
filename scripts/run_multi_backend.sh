#!/usr/bin/env bash
# Multi-backend inference comparison: HF vs vLLM vs SGLang vs TRT-LLM
# Usage:
#   ./run_multi_backend.sh              — all available backends, Qwen3-8B
#   ./run_multi_backend.sh vllm sglang  — specific backends only
#
# Prerequisites:
#   HF     : always available
#   vLLM   : sudo apt install python3-dev && pip3 install vllm --break-system-packages
#   SGLang : pip3 install "sglang[all]" --break-system-packages
#   TRT-LLM: pip3 install tensorrt-llm --break-system-packages
set -uo pipefail

BASE=/home/student/Desktop/Test
LOG=$BASE/logs/multi_backend.log
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$BASE/logs"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

MODEL=${MODEL:-"Qwen/Qwen3-8B"}
DTYPE=${DTYPE:-"bfloat16"}
BATCH_SIZES=${BATCH_SIZES:-"1 4 8"}

# Default: run all backends (each will print "not_installed" if missing)
if [ $# -eq 0 ]; then
    BACKENDS="hf vllm sglang trtllm"
else
    BACKENDS="$*"
fi

log "=== MULTI-BACKEND INFERENCE BENCHMARK ==="
log "Model: $MODEL   dtype: $DTYPE   batch: $BATCH_SIZES"
log "Backends: $BACKENDS"

python3 "$BASE/phase1_inference_bench.py" \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --batch-sizes $BATCH_SIZES \
    --prompts short medium \
    --backend $BACKENDS \
    2>&1 | tee -a "$LOG"

log "=== DONE ==="
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
