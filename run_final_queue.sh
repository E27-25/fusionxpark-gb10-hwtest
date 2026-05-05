#!/usr/bin/env bash
# Final queue: T8/T12/T14/T15/T23 (retries) + Phase 3 eval
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/final_queue.log
cd "$BASE"
export PATH="$HOME/.local/bin:$PATH"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }
mkdir -p "$BASE/logs" "$BASE/results/training"

run_exp() {
    local EXP="$1"; shift
    local CMD="$*"
    log "─── $EXP ───"
    if eval "$CMD" 2>&1 | tee -a "$LOG"; then
        ok "$EXP done"
    else
        warn "$EXP had errors"
    fi
    python3 "$BASE/tracker.py" 2>/dev/null | tail -5 | tee -a "$LOG"
    sleep 5
}

log "=== FINAL QUEUE START ==="

# T15: Whisper LoRA (fixed: processing_class instead of tokenizer)
run_exp T15 python3 "$BASE/phase2_asr_train.py" --experiment T15 \
    --batch-size 4 --grad-accum 4 --max-steps 500 --max-samples 10000

# T14: Whisper Full FT (fixed: processing_class instead of tokenizer)
run_exp T14 python3 "$BASE/phase2_asr_train.py" --experiment T14 \
    --batch-size 2 --grad-accum 8 --max-steps 500 --max-samples 10000

# T12: Mixtral-8x7B QLoRA (fixed: device_map={"":0} to prevent CPU offload)
run_exp T12 python3 "$BASE/phase2_train.py" --experiment T12 \
    --batch-size 1 --grad-accum 16 --max-seq-len 512 --max-steps 150

# T8: Qwen2.5-72B QLoRA (fixed: device_map={"":0}, reduced seq_len)
run_exp T8 python3 "$BASE/phase2_train.py" --experiment T8 \
    --batch-size 1 --grad-accum 8 --max-seq-len 512 --max-steps 300

# T23: CPT Qwen3-8B OpenWebText (never ran — was after Phase3 in old queue)
run_exp T23 python3 "$BASE/phase2_cpt.py" \
    --max-steps 1000 --max-seq-len 2048 --batch-size 2 --grad-accum 16

log "=== TRAINING DONE — STARTING PHASE 3 EVAL ==="

# Phase 3 eval — all trained checkpoints
run_exp "lm_eval" bash "$BASE/phase3_eval.sh"

log "=== FINAL QUEUE COMPLETE ==="
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
