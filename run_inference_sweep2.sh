#!/usr/bin/env bash
# Inference sweep — models not yet benchmarked in Phase 1
# Run after T6 finishes (T11 will be running, ~38-45 GB — leaves ~74-80 GB free)
# Usage: ./run_inference_sweep2.sh [small|mid|large|xl|moe|vlm|all]
set -uo pipefail

BASE=/home/student/Desktop/Test
LOG=$BASE/logs/inference_sweep2.log
cd "$BASE"
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()  { echo "  ✓ $*" | tee -a "$LOG"; }
fail(){ echo "  ✗ $*" | tee -a "$LOG"; }

mkdir -p "$BASE/logs"

GROUP="${1:-all}"
log "=== Inference Sweep 2 — group=$GROUP ==="

# ── Small BF16 (safe alongside any training) ──────────────────────
run_small() {
    log "--- Small BF16 models ---"
    for MODEL in \
        "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
        "Qwen/Qwen3-14B" \
        "Qwen/Qwen2.5-7B-Instruct" \
        "Qwen/Qwen2.5-Coder-7B-Instruct" \
    ; do
        log "Benchmarking $MODEL"
        python3 phase1_inference_bench.py \
            --model "$MODEL" --dtype bfloat16 \
            --batch-sizes 1 4 8 --prompts short medium \
            2>&1 | tee -a "$LOG" \
            && ok "$MODEL done" || fail "$MODEL failed"
        sleep 5
    done
}

# ── Mid BF16 (safe alongside T11 ~40 GB) ──────────────────────────
run_mid() {
    log "--- Mid BF16 models ---"
    for MODEL in \
        "meta-llama/Llama-3.1-8B-Instruct" \
        "google/gemma-3-4b-it" \
        "google/gemma-3-12b-it" \
        "microsoft/phi-4" \
    ; do
        log "Benchmarking $MODEL"
        python3 phase1_inference_bench.py \
            --model "$MODEL" --dtype bfloat16 \
            --batch-sizes 1 4 8 --prompts short medium \
            2>&1 | tee -a "$LOG" \
            && ok "$MODEL done" || fail "$MODEL failed"
        sleep 5
    done
}

# ── Large BF16 (need ~64 GB free — safe after T11 starts) ─────────
run_large() {
    log "--- Large BF16 models (~64 GB each) ---"
    for MODEL in \
        "Qwen/Qwen2.5-32B-Instruct" \
        "Qwen/Qwen2.5-Coder-32B-Instruct" \
        "google/gemma-3-27b-it" \
    ; do
        log "Benchmarking $MODEL"
        python3 phase1_inference_bench.py \
            --model "$MODEL" --dtype bfloat16 \
            --batch-sizes 1 4 --prompts short medium \
            2>&1 | tee -a "$LOG" \
            && ok "$MODEL done" || fail "$MODEL failed"
        sleep 10
    done
}

# ── XL NF4 (35-40 GB — can run alongside small training jobs) ─────
run_xl() {
    log "--- XL NF4 models ---"
    # DeepSeek-R1-Distill-8B fits in BF16; others need NF4
    log "Benchmarking DeepSeek-R1-Distill-Llama-8B (BF16)"
    python3 phase1_inference_bench.py \
        --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --dtype bfloat16 \
        --batch-sizes 1 4 8 --prompts short medium \
        2>&1 | tee -a "$LOG" \
        && ok "DeepSeek-R1-Distill-8B done" || fail "DeepSeek-R1-Distill-8B failed"
    sleep 5

    log "Benchmarking DeepSeek-R1-Distill-Qwen-32B (BF16)"
    python3 phase1_inference_bench.py \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --dtype bfloat16 \
        --batch-sizes 1 4 --prompts short \
        2>&1 | tee -a "$LOG" \
        && ok "DeepSeek-R1-Distill-32B done" || fail "DeepSeek-R1-Distill-32B failed"
    sleep 10

    for MODEL in \
        "meta-llama/Llama-3.3-70B-Instruct" \
        "Qwen/Qwen2.5-72B-Instruct" \
    ; do
        log "Benchmarking $MODEL (NF4)"
        python3 phase1_inference_bench.py \
            --model "$MODEL" --nf4 \
            --batch-sizes 1 4 --prompts short \
            2>&1 | tee -a "$LOG" \
            && ok "$MODEL done" || fail "$MODEL failed"
        sleep 10
    done
}

# ── MoE (need most free memory for 235B) ──────────────────────────
run_moe() {
    log "--- MoE models ---"
    # Qwen3-30B-A3B: ~60 GB BF16 — needs T6 done, safe alongside T11
    log "Benchmarking Qwen3-30B-A3B"
    python3 phase1_inference_bench.py \
        --model "Qwen/Qwen3-30B-A3B" --dtype bfloat16 \
        --batch-sizes 1 4 --prompts short medium \
        2>&1 | tee -a "$LOG" \
        && ok "Qwen3-30B-A3B done" || fail "Qwen3-30B-A3B failed"
    sleep 10

    # Qwen3-235B-A22B: ~118 GB NF4 — only attempt if already cached (470 GB BF16 download = impractical)
    Q235B_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct"
    if [ -d "$Q235B_CACHE" ] && [ "$(find "$Q235B_CACHE" -name '*.safetensors' 2>/dev/null | wc -l)" -gt 10 ]; then
        log "Benchmarking Qwen3-235B-A22B (NF4, cached — attempting)"
        python3 phase1_inference_bench.py \
            --model "Qwen/Qwen3-235B-A22B-Instruct" --nf4 \
            --batch-sizes 1 --prompts short \
            2>&1 | tee -a "$LOG" \
            && ok "Qwen3-235B-A22B done" || fail "Qwen3-235B-A22B failed (OOM likely)"
        sleep 15
    else
        log "Skipping Qwen3-235B-A22B: not cached locally (470 GB download impractical)"
        fail "Qwen3-235B-A22B skipped (not cached)"
    fi
}

# ── VLMs (image+text) ─────────────────────────────────────────────
run_vlm() {
    log "--- VLM models ---"
    for MODEL in \
        "Qwen/Qwen2.5-VL-7B-Instruct" \
        "meta-llama/Llama-3.2-11B-Vision-Instruct" \
    ; do
        log "Benchmarking $MODEL"
        python3 phase1_inference_bench.py \
            --model "$MODEL" --dtype bfloat16 \
            --batch-sizes 1 4 --prompts short \
            2>&1 | tee -a "$LOG" \
            && ok "$MODEL done" || fail "$MODEL failed"
        sleep 5
    done
}

case "$GROUP" in
    small) run_small ;;
    mid)   run_mid ;;
    large) run_large ;;
    xl)    run_xl ;;
    moe)   run_moe ;;
    vlm)   run_vlm ;;
    all)
        run_small
        run_mid
        run_large
        run_xl
        run_moe
        run_vlm
        ;;
    *)
        echo "Usage: $0 [small|mid|large|xl|moe|vlm|all]"
        exit 1
        ;;
esac

log "=== Inference sweep 2 complete ==="
python3 "$BASE/tracker.py" 2>/dev/null | tail -5 | tee -a "$LOG"
