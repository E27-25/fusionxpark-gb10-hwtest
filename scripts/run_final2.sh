#!/usr/bin/env bash
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/final2.log
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

log "=== FINAL2 START ==="

# Phase 3 eval (all existing checkpoints)
run_exp lm_eval bash "$BASE/phase3_eval.sh"

# T23 CPT (never ran)
run_exp T23 python3 "$BASE/phase2_cpt.py" \
    --max-steps 1000 --max-seq-len 2048 --batch-size 2 --grad-accum 16

# T8 72B QLoRA (seq_len=256 to reduce peak memory)
run_exp T8 python3 "$BASE/phase2_train.py" --experiment T8 \
    --batch-size 1 --grad-accum 8 --max-seq-len 256 --max-steps 300

# Phase 3 eval top-up (picks up T23/T8 checkpoints, skips .done ones)
run_exp lm_eval_topup bash "$BASE/phase3_eval.sh"

log "=== ALL DONE ==="
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
