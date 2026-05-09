#!/usr/bin/env bash
# Run T7 (MoE LoRA) and retry T8 (72B QLoRA with fixed device_map=auto)
# Launch after main phase2 queue finishes
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/phase2_train.log
export PATH="$HOME/.local/bin:$PATH"
log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }

log "=== RETRY QUEUE: T7 + T8 ==="

# T7: Qwen3-30B-A3B MoE LoRA r=16
log "─── Experiment T7 ───"
if python3 "$BASE/phase2_train.py" \
    --experiment T7 \
    --batch-size 2 --grad-accum 8 \
    --max-seq-len 2048 --max-steps 500 \
    2>&1 | tee -a "$LOG"; then
    ok "T7 done"
else
    warn "T7 had errors (continuing)"
fi
python3 "$BASE/tracker.py" 2>/dev/null | tail -10 | tee -a "$LOG"
sleep 10

# T8: Qwen2.5-72B QLoRA NF4 — reduced settings for memory safety
log "─── Experiment T8 (retry, device_map=auto) ───"
if python3 "$BASE/phase2_train.py" \
    --experiment T8 \
    --batch-size 1 --grad-accum 16 \
    --max-seq-len 1024 --max-steps 300 \
    2>&1 | tee -a "$LOG"; then
    ok "T8 done"
else
    warn "T8 had errors"
fi

log "=== RETRY DONE ==="
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
