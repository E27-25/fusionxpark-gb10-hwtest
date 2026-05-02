#!/usr/bin/env bash
# Run after main Phase 2 queue (T4→T6→T11→T12→T20→T21) finishes
# Sequence: T7+T8 retry → ASR bench re-run → TTS bench → stress test → Phase 3 eval
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/post_training.log
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$BASE/logs"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }

SESSION="gb10post"

if [ -z "${TMUX:-}" ]; then
    log "Launching in tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" "bash $BASE/run_post_training.sh 2>&1 | tee -a $LOG"
    echo "  Started: tmux attach -t $SESSION"
    echo "  Log:     tail -f $LOG"
    exit 0
fi

log "=== POST-TRAINING PIPELINE ==="
log "Start: $(date)"

# ── Phase 2 remaining: T7 + T8 ───────────────────────────────
bash "$BASE/retry_t8.sh" 2>&1 | tee -a "$LOG"

# ── Phase 1 re-runs ─────────────────────────────────────────
log "=== RE-RUNNING PHASE 1 BENCHMARKS ==="

log "--- ASR bench (re-run with numpy fix) ---"
if python3 "$BASE/phase1_asr_bench.py" 2>&1 | tee -a "$LOG"; then
    ok "ASR bench done"
else
    warn "ASR bench had errors"
fi
sleep 15

log "--- TTS bench (Bark + Qwen3-TTS attempt) ---"
if python3 "$BASE/phase1_tts_bench.py" 2>&1 | tee -a "$LOG"; then
    ok "TTS bench done"
else
    warn "TTS bench had errors"
fi
sleep 15

log "--- Stress test (HF backend) ---"
if python3 "$BASE/phase1_stress_test.py" \
    --backend hf \
    --duration 60 \
    --concurrent 1 \
    2>&1 | tee -a "$LOG"; then
    ok "Stress test done"
else
    warn "Stress test had errors"
fi
sleep 15

# ── Phase 3 Evaluation ───────────────────────────────────────
log "=== PHASE 3: EVALUATION ==="

log "--- Base model eval (Qwen3-8B) ---"
bash "$BASE/phase3_eval.sh" "Qwen/Qwen3-8B" "qwen3-8b-base" 2>&1 | tee -a "$LOG" &
EVAL_PID=$!

# Run ASR eval in parallel (different GPU workload pattern)
log "--- ASR eval (Whisper base WER) ---"
python3 "$BASE/phase3_asr_eval.py" --n-samples 200 2>&1 | tee -a "$LOG" || warn "ASR eval had errors"

wait $EVAL_PID || warn "Base model eval had errors"

log "=== POST-TRAINING PIPELINE DONE ==="
log "End: $(date)"
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
