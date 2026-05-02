#!/usr/bin/env bash
# Phase 2 Queue 2 — T2, T5, T7, T8 (run after main queue T6/T11/T12/T20/T21 completes)
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/phase2_q2.log
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$BASE/logs"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }

SESSION="gb10q2"

if [ -z "${TMUX:-}" ]; then
    log "Launching in tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" "bash $BASE/run_queue2.sh 2>&1 | tee -a $LOG"
    echo "  Started tmux '$SESSION' — attach: tmux attach -t $SESSION"
    exit 0
fi

log "=== PHASE 2 QUEUE 2: T2 T5 T7 T8 ==="
log "Start: $(date)"

run_exp() {
    local EXP="$1"; shift
    local CMD="$*"
    log "─── Experiment $EXP ───"
    if eval "$CMD" 2>&1 | tee -a "$LOG"; then
        ok "$EXP done"
    else
        warn "$EXP had errors (continuing)"
    fi
    python3 "$BASE/tracker.py" 2>/dev/null | tail -10 | tee -a "$LOG"
    sleep 10
}

# T2: Mistral-7B Full FT (~50-60 GB, 500 steps @ ~30s/step = ~4h)
run_exp T2 python3 "$BASE/phase2_train.py" --experiment T2 \
    --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 500

# T5: Qwen3-8B LoRA r=64 (~28-35 GB, 500 steps @ ~22s/step = ~3h)
run_exp T5 python3 "$BASE/phase2_train.py" --experiment T5 \
    --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 500

# T7: Qwen3-30B-A3B MoE LoRA r=16 (~65-75 GB, 150 steps @ ~60s/step = ~2.5h)
run_exp T7 python3 "$BASE/phase2_train.py" --experiment T7 \
    --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 150

# T8: Qwen2.5-72B QLoRA NF4 (~48-56 GB, 300 steps @ ~30s/step = ~2.5h)
run_exp T8 python3 "$BASE/phase2_train.py" --experiment T8 \
    --batch-size 1 --grad-accum 16 --max-seq-len 1024 --max-steps 300

log "=== QUEUE 2 DONE ==="
log "End: $(date)"
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
