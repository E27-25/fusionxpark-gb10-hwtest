#!/usr/bin/env bash
# Phase 2 — Training experiments (T3 → T4 → T24 → T20 → T21 → T1 → T8 → T6 → ...)
# Runs inside tmux session "gb10bench" so SSH disconnects are safe.
# Usage:
#   ./run_phase2.sh              — run default order starting from T3
#   ./run_phase2.sh T3 T4 T24   — run specific experiments
#   ./run_phase2.sh smoke        — smoke-test all (20 steps each)
set -uo pipefail

BASE=/home/student/Desktop/Test
LOG=$BASE/logs/phase2_train.log
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$BASE/logs" "$BASE/results/training"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }

SESSION="gb10bench"

# ── Default experiment order (cheapest/most important first) ──
DEFAULT_ORDER="T3 T4 T24 T20 T21 T1 T8 T6 T7 T11 T12 T2 T5"

# ── Smoke test: 20 steps only, fast validation ─────────────────
if [ "${1:-}" = "smoke" ]; then
    log "=== PHASE 2 SMOKE TEST (20 steps each) ==="
    for EXP in T3 T4 T24 T1 T8; do
        log "--- Smoke: $EXP ---"
        python3 "$BASE/phase2_train.py" \
            --experiment "$EXP" \
            --smoke-test \
            --batch-size 2 \
            --grad-accum 4 \
            2>&1 | tee -a "$LOG"
    done
    log "=== SMOKE TEST DONE ==="
    python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
    exit 0
fi

# ── Select which experiments to run ───────────────────────────
if [ $# -eq 0 ]; then
    EXPERIMENTS=$DEFAULT_ORDER
else
    EXPERIMENTS="$*"
fi

# ── Launch in tmux if not already in a tmux session ───────────
if [ -z "${TMUX:-}" ]; then
    log "Launching in tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "bash $BASE/run_phase2.sh $EXPERIMENTS 2>&1 | tee -a $LOG"
    echo "  Started tmux session '$SESSION'"
    echo "  Attach with: tmux attach -t $SESSION"
    echo "  Tail log:    tail -f $LOG"
    exit 0
fi

log "=== PHASE 2 TRAINING RUN ==="
log "Experiments: $EXPERIMENTS"
log "Start: $(date)"

run_experiment() {
    local EXP="$1"
    log "─── Experiment $EXP ───"
    # Route to correct script; use fewer steps for large models (86s/step vs 19s/step)
    case "$EXP" in
        T20|T22)
            CMD="python3 $BASE/phase2_dpo.py --experiment $EXP --max-steps 300" ;;
        T21)
            CMD="python3 $BASE/phase2_grpo.py --max-steps 200" ;;
        # Large models (32B+): 150 steps ~3.5h each
        T6|T7|T12)
            CMD="python3 $BASE/phase2_train.py --experiment $EXP --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 150" ;;
        *)
            CMD="python3 $BASE/phase2_train.py --experiment $EXP --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 500" ;;
    esac
    if eval "$CMD" 2>&1 | tee -a "$LOG"; then
        ok "$EXP done"
    else
        warn "$EXP had errors (continuing)"
    fi
    python3 "$BASE/tracker.py" 2>/dev/null | tail -10 | tee -a "$LOG"
    # Cool-down between experiments
    sleep 10
}

for EXP in $EXPERIMENTS; do
    run_experiment "$EXP"
done

log "=== PHASE 2 DONE ==="
log "End: $(date)"
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
