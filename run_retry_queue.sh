#!/usr/bin/env bash
# Retry queue for experiments that failed in the first pass:
#   T11 — DeepSeek-V2-Lite LoRA (patched is_torch_fx_available)
#   T12 — Mixtral-8x7B QLoRA NF4 (changed from LoRA to QLoRA to avoid OOM)
#   T20 — Qwen3-8B DPO (fixed max_prompt_length kwarg)
# Then runs Queue 2: T2 T5 T7 T8
set -uo pipefail
BASE=/home/student/Desktop/Test
LOG=$BASE/logs/retry_queue.log
cd "$BASE"
export PATH="$HOME/.local/bin:$PATH"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }
mkdir -p "$BASE/logs" "$BASE/results/training"

# ── VLM Inference Sweep (missed in watcher chain) ────────────
log "=== VLM INFERENCE SWEEP ==="
bash "$BASE/run_inference_sweep2.sh" vlm 2>&1 | tee -a "$LOG" && ok "VLM sweep done" || warn "VLM sweep had errors"
sleep 10

# ── Retry failed experiments ──────────────────────────────────
log "=== RETRY QUEUE START ==="

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
    sleep 10
}

# T11: DeepSeek-V2-Lite LoRA r=32 (patched)
run_exp T11 python3 "$BASE/phase2_train.py" --experiment T11 \
    --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 500

# T12: Mixtral-8x7B QLoRA NF4 r=16 (was BF16 LoRA → OOM; now QLoRA)
run_exp T12 python3 "$BASE/phase2_train.py" --experiment T12 \
    --batch-size 2 --grad-accum 8 --max-seq-len 1024 --max-steps 150

# T20: Qwen3-8B DPO — start from T3 SFT checkpoint
run_exp T20 python3 "$BASE/phase2_dpo.py" --experiment T20 \
    --sft-checkpoint "$BASE/models/T3_Qwen3-8B_lora_r16" --max-steps 300

# T22: Qwen3-32B DPO (after T20 — uses T6 SFT checkpoint)
run_exp T22 python3 "$BASE/phase2_dpo.py" --experiment T22 \
    --sft-checkpoint "$BASE/models/T6_Qwen3-32B_lora_r32" --max-steps 150

# ── Queue 2: Longer / larger experiments ─────────────────────
log "=== QUEUE 2 START ==="

# T2: Mistral-7B Full FT
run_exp T2 python3 "$BASE/phase2_train.py" --experiment T2 \
    --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 500

# T5: Qwen3-8B LoRA r=64 (larger LoRA rank)
run_exp T5 python3 "$BASE/phase2_train.py" --experiment T5 \
    --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 500

# T7: Qwen3-30B-A3B MoE LoRA r=16
run_exp T7 python3 "$BASE/phase2_train.py" --experiment T7 \
    --batch-size 2 --grad-accum 8 --max-seq-len 2048 --max-steps 150

# T8: Qwen2.5-72B QLoRA NF4
run_exp T8 python3 "$BASE/phase2_train.py" --experiment T8 \
    --batch-size 2 --grad-accum 4 --max-seq-len 1024 --max-steps 300

# ── ASR Training (T14, T15) ──────────────────────────────────
log "=== QUEUE 2b: ASR TRAINING ==="

# T14: Whisper-large-v3 Full FT (CommonVoice en, 10K samples)
run_exp T14 python3 "$BASE/phase2_asr_train.py" --experiment T14 \
    --max-samples 10000 --max-steps 500

# T15: Whisper-large-v3 LoRA r=16
run_exp T15 python3 "$BASE/phase2_asr_train.py" --experiment T15 \
    --max-samples 10000 --max-steps 500

# ── Model Merges (CPU-heavy, run after training frees GPU) ───
log "=== QUEUE 3: MODEL MERGES ==="

# M1: SLERP(Qwen3-8B-base, T3-SFT) — needs T3 checkpoint (already done)
run_exp M1 python3 "$BASE/phase2_merge.py" --experiment M1

# M2: TIES(T3-SFT, T20-DPO) — needs T20 checkpoint
run_exp M2 python3 "$BASE/phase2_merge.py" --experiment M2

# M3: DARE+TIES(Qwen3-8B, Qwen2.5-Coder-7B)
run_exp M3 python3 "$BASE/phase2_merge.py" --experiment M3

# ── Stress Test (HF concurrent simulation, runs between merges and eval) ──
log "=== QUEUE 3b: STRESS TEST ==="
run_exp stress_test python3 "$BASE/phase1_stress_test.py" \
    --model "Qwen/Qwen3-8B" --backend hf \
    --concurrent 1 10 50 --duration 60 --max-tokens 128

# ── Phase 3: lm-eval on base + fine-tuned models ─────────────
log "=== QUEUE 4: PHASE 3 EVALUATION ==="
run_exp "lm_eval_base" bash "$BASE/phase3_eval.sh" 2>&1

# ── T23: CPT (long — run after Phase 3 eval to avoid blocking eval) ──
log "=== QUEUE 5: CPT ==="
run_exp T23 python3 "$BASE/phase2_cpt.py" \
    --max-steps 1000 --max-seq-len 2048 --batch-size 2 --grad-accum 16

log "=== ALL QUEUES DONE ==="
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
