#!/usr/bin/env bash
# Master Runner — GB10 Grace Blackwell Experiment Suite
# Runs all phases in sequence with logging and error recovery.
# Usage:
#   ./run_all.sh                  # Full run (all phases)
#   ./run_all.sh --phase 0        # Only install
#   ./run_all.sh --phase 0.5      # Only HW bench
#   ./run_all.sh --phase 1        # Only inference benchmarks
#   ./run_all.sh --phase 2        # Only training
#   ./run_all.sh --phase 3        # Only evaluation
#   ./run_all.sh --smoke-test     # Quick 20-step smoke test of everything
set -uo pipefail

BASE=/home/student/Desktop/Test
LOGS=$BASE/logs
mkdir -p "$LOGS"

TS=$(date '+%Y%m%d_%H%M%S')
MASTER_LOG="$LOGS/run_all_${TS}.log"
PHASE="all"
SMOKE=""

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --smoke-test) SMOKE="--smoke-test"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
ok()   { echo "  ✓ $*" | tee -a "$MASTER_LOG"; }
warn() { echo "  ⚠ $*" | tee -a "$MASTER_LOG"; }
fail() { echo "  ✗ $*" | tee -a "$MASTER_LOG"; }
sep()  { echo "$(printf '─%.0s' {1..70})" | tee -a "$MASTER_LOG"; }

run_py() {
    local script=$1; shift
    local args="$*"
    log "Running: python3 $script $args"
    if python3 "$BASE/$script" $args 2>&1 | tee -a "$MASTER_LOG"; then
        ok "$script done"
        return 0
    else
        warn "$script had errors (continuing)"
        return 1
    fi
}

run_sh() {
    local script=$1; shift
    local args="$*"
    log "Running: bash $script $args"
    if bash "$BASE/$script" $args 2>&1 | tee -a "$MASTER_LOG"; then
        ok "$script done"
        return 0
    else
        warn "$script had errors (continuing)"
        return 1
    fi
}

# Start background monitor
start_monitor() {
    local tag=$1
    bash "$BASE/phase05_monitor.sh" "$tag" &
    MONITOR_PID=$!
    log "GPU monitor started (PID $MONITOR_PID)"
}

stop_monitor() {
    if [ -n "${MONITOR_PID:-}" ]; then
        kill $MONITOR_PID 2>/dev/null || true
        log "GPU monitor stopped"
    fi
}

trap stop_monitor EXIT

# ─────────────────────────────────────────────────────────────
phase_0() {
    sep
    log "=== PHASE 0: Environment Setup ==="
    run_sh phase0_install.sh || true
    run_py phase0_verify_env.py
}

phase_05() {
    sep
    log "=== PHASE 0.5: Hardware Characterization ==="
    start_monitor "hw_bench"
    run_py phase05_hw_bench.py
    stop_monitor
}

phase_1() {
    sep
    log "=== PHASE 1: Inference Benchmarks ==="
    start_monitor "inference"

    log "1.1 LLM Inference Benchmark"
    run_py phase1_inference_bench.py --model Qwen/Qwen3-8B --batch-sizes 1 4 8 --prompts short medium

    log "1.2 Quantization Sweep"
    run_py phase1_quant_sweep.py

    log "1.3 ASR Benchmark"
    run_py phase1_asr_bench.py --models whisper-large-v3 whisper-large-v3-turbo --samples 100

    log "1.4 TTS Benchmark"
    run_py phase1_tts_bench.py --models qwen3-tts-base qwen3-tts-voice

    log "1.5 Embedding & Reranker"
    run_py phase1_embed_rerank_bench.py --mode all --batch-sizes 1 8 32

    log "1.6 Speculative Decoding"
    run_py phase1_speculative_bench.py --pairs 0.6B-8B --backend hf --baseline

    log "1.7 Long-Context Benchmark"
    run_py phase1_longctx_bench.py --models qwen3-8b --ctx-lengths 4096 8192 16384 32768 --kv-scaling

    log "1.8 Stress Test"
    run_py phase1_stress_test.py --concurrent 1 10 50 --duration 30 --backend vllm

    # Bonus: larger models
    for model in "Qwen/Qwen3-32B" "Qwen/Qwen3-30B-A3B"; do
        tag=$(echo $model | tr '/' '_')
        log "1.x Benchmark $model"
        run_py phase1_inference_bench.py --model "$model" --batch-sizes 1 4 --prompts short || true
    done

    stop_monitor
}

phase_2() {
    sep
    log "=== PHASE 2: Training Experiments ==="
    start_monitor "training"

    # Run cheapest experiments first
    log "2.1 LoRA experiments (T3, T4, T24 — cheapest)"
    for exp in T3 T4 T24; do
        run_py phase2_train.py --experiment $exp $SMOKE --max-seq-len 2048 --batch-size 4 --grad-accum 4
    done

    log "2.2 ASR fine-tuning (T15 LoRA — cheap)"
    run_py phase2_asr_train.py --experiment T15 $SMOKE

    log "2.3 DPO (T20)"
    run_py phase2_dpo.py --experiment T20 $SMOKE

    log "2.4 GRPO (T21)"
    run_py phase2_grpo.py $SMOKE

    log "2.5 Full FT (T1 — larger)"
    run_py phase2_train.py --experiment T1 $SMOKE --batch-size 2 --grad-accum 8

    log "2.6 QLoRA 72B (T8)"
    run_py phase2_train.py --experiment T8 $SMOKE --batch-size 1 --grad-accum 16

    log "2.7 32B LoRA (T6)"
    run_py phase2_train.py --experiment T6 $SMOKE --batch-size 1 --grad-accum 16

    log "2.8 CPT (T23)"
    run_py phase2_cpt.py --max-steps 5000 $SMOKE || true

    log "2.9 Model Merging (M1, M2, M3)"
    bash "$BASE/phase2_merge.sh" || true

    stop_monitor
}

phase_3() {
    sep
    log "=== PHASE 3: Evaluation ==="
    start_monitor "evaluation"

    log "3.1 Standard benchmarks (lm-eval)"
    bash "$BASE/phase3_eval.sh" Qwen/Qwen3-8B qwen3-8b-base || true

    log "3.2 ASR evaluation"
    run_py phase3_asr_eval.py --n-samples 200

    log "3.3 RAG pipeline"
    run_py phase3_rag_pipeline.py --configs rag-base rag-qwen3 no-rag --corpus-size 500

    stop_monitor
}

# ─────────────────────────────────────────────────────────────
main() {
    log "=== GB10 Grace Blackwell — Full Experiment Suite ==="
    log "Phase: $PHASE   Smoke: ${SMOKE:-no}"
    log "Log: $MASTER_LOG"
    log "Start: $(date)"
    sep

    # Print live tracker in background
    python3 "$BASE/tracker.py" 2>/dev/null | head -20 || true

    case "$PHASE" in
        "0")       phase_0 ;;
        "0.5")     phase_05 ;;
        "1")       phase_1 ;;
        "2")       phase_2 ;;
        "3")       phase_3 ;;
        "all")
            [[ "$SMOKE" == "" ]] && phase_0 || true
            phase_05
            phase_1
            phase_2
            phase_3
            ;;
        *)
            fail "Unknown phase: $PHASE (use 0, 0.5, 1, 2, 3, or all)"
            exit 1 ;;
    esac

    sep
    log "=== ALL PHASES COMPLETE ==="
    log "End: $(date)"
    log ""
    log "Final tracker report:"
    python3 "$BASE/tracker.py" 2>/dev/null || true
    log ""
    log "HTML dashboard: $BASE/results/dashboard.html"
    python3 "$BASE/tracker.py" --html 2>/dev/null || true
}

main
