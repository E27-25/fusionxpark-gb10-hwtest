#!/usr/bin/env bash
# Phase 1 — Run remaining experiments (quant_sweep, ASR, embed, speculative, longctx)
# Already done: inference_bench (Qwen3-8B baseline)
set -uo pipefail

BASE=/home/student/Desktop/Test
LOG=$BASE/logs/bench_run.log
export PATH="$HOME/.local/bin:$PATH"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$LOG"; }

run_py() {
    local script=$1; shift
    log "Running: python3 $script $*"
    if python3 "$BASE/$script" "$@" 2>&1 | tee -a "$LOG"; then
        ok "$script done"
    else
        warn "$script had errors (continuing)"
    fi
}

log "=== PHASE 1 REMAINING EXPERIMENTS ==="
log "Start: $(date)"

# 1.2 Quantization sweep (BF16, INT8, NF4 will work; AWQ/GGUF will error-record)
log "─── 1.2 Quantization Sweep ───"
run_py phase1_quant_sweep.py

# 1.3 ASR Benchmark (Whisper large-v3 + turbo)
log "─── 1.3 ASR Benchmark ───"
run_py phase1_asr_bench.py \
    --models openai/whisper-large-v3 openai/whisper-large-v3-turbo \
    --samples 50

# 1.5 Embedding Benchmark (no reranker — avoids mteb pytrec-eval dep)
log "─── 1.5 Embedding Benchmark ───"
run_py phase1_embed_rerank_bench.py --mode embed --batch-sizes 1 8 32

# 1.6 Speculative Decoding (Qwen3-0.6B → Qwen3-8B)
log "─── 1.6 Speculative Decoding ───"
run_py phase1_speculative_bench.py --pairs 0.6B-8B --backend hf --baseline

# 1.7 Long-Context (shorter list to avoid very long runs)
log "─── 1.7 Long-Context ───"
run_py phase1_longctx_bench.py \
    --models qwen3-8b \
    --ctx-lengths 4096 8192 16384 \
    --kv-scaling

# 1.1 More models: smaller Qwen3 variants
log "─── 1.x Small Models ───"
for MODEL in "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B"; do
    run_py phase1_inference_bench.py \
        --model "$MODEL" \
        --batch-sizes 1 4 8 \
        --prompts short
done

# 1.1 Larger models (32B)
log "─── 1.x Large Model: Qwen3-32B ───"
run_py phase1_inference_bench.py \
    --model Qwen/Qwen3-32B \
    --batch-sizes 1 4 \
    --prompts short || true

log "=== PHASE 1 COMPLETE ==="
log "End: $(date)"
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
