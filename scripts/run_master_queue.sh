#!/usr/bin/env bash
# Master queue: SGLang precision sweep → Phase 3 eval → remaining training
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/master_queue.log
mkdir -p "$BASE/logs"
export PATH="$HOME/.local/bin:$PATH"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }

log "=== MASTER QUEUE v2 START ==="

# 1. HF Precision sweep: BF16/INT8/NF4/AWQ (SGLang broken - sgl_kernel ABI mismatch)
log "--- HF Precision Sweep (8B: BF16/INT8/NF4/AWQ, 32B: BF16/INT8/NF4) ---"
if python3 "$BASE/phase1_precision_sweep_hf.py" --model 8b 2>&1 | tee -a "$LOG"; then
    ok "Precision sweep 8B done"
else
    warn "Precision sweep 8B had errors"
fi

if python3 "$BASE/phase1_precision_sweep_hf.py" --model 32b 2>&1 | tee -a "$LOG"; then
    ok "Precision sweep 32B done"
else
    warn "Precision sweep 32B had errors"
fi

# 2. Phase 3 evaluation (limited benchmarks, ~70 min/model)
log "--- Phase 3: Standard Evaluation ---"
if bash "$BASE/phase3_eval.sh" 2>&1 | tee -a "$LOG"; then
    ok "Phase 3 eval done"
else
    warn "Phase 3 eval had errors"
fi

# 3. Remaining training experiments
log "--- Remaining Training: T23 CPT ---"
if python3 "$BASE/phase2_cpt.py" \
    --max-steps 500 --max-seq-len 1024 --batch-size 2 --grad-accum 16 \
    2>&1 | tee -a "$LOG"; then
    ok "T23 CPT done"
else
    warn "T23 CPT had errors"
fi

log "--- Remaining Training: T8 QLoRA 72B ---"
if python3 "$BASE/phase2_train.py" --experiment T8 \
    --batch-size 1 --grad-accum 8 --max-seq-len 256 --max-steps 150 \
    2>&1 | tee -a "$LOG"; then
    ok "T8 done"
else
    warn "T8 had errors"
fi

# Final tracker summary
log "=== MASTER QUEUE COMPLETE ==="
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
