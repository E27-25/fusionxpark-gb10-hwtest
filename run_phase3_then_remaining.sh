#!/usr/bin/env bash
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/phase3_remaining.log
cd "$BASE"
export PATH="$HOME/.local/bin:$PATH"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }
mkdir -p "$BASE/logs"

run_exp() {
    local EXP="$1"; shift
    log "─── $EXP ───"
    if eval "$@" 2>&1 | tee -a "$LOG"; then ok "$EXP done"
    else warn "$EXP had errors"; fi
    sleep 5
}

# 1. Phase 3 eval — all existing checkpoints
log "=== PHASE 3 EVAL ==="
run_exp lm_eval bash "$BASE/phase3_eval.sh"

# 2. T23: CPT (never ran)
log "=== T23: CPT ==="
run_exp T23 python3 "$BASE/phase2_cpt.py" \
    --max-steps 1000 --max-seq-len 2048 --batch-size 2 --grad-accum 16

# 3. T8: 72B QLoRA (retry with seq_len=256 to reduce peak mem)
log "=== T8: 72B QLoRA ==="
run_exp T8 python3 "$BASE/phase2_train.py" --experiment T8 \
    --batch-size 1 --grad-accum 8 --max-seq-len 256 --max-steps 300

# 4. Phase 3 eval again for T23/T8 checkpoints only (skips .done ones)
log "=== PHASE 3 EVAL (T23/T8 top-up) ==="
run_exp lm_eval_topup bash "$BASE/phase3_eval.sh"

log "=== ALL DONE ==="
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
