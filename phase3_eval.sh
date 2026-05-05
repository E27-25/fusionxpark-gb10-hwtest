#!/usr/bin/env bash
# Phase 3 — lm-evaluation-harness standard benchmarks
# Runs MMLU, HellaSwag, ARC, GSM8K, HumanEval, TruthfulQA, Winogrande
# Also runs Qwen3 thinking vs no-think comparison
set -uo pipefail

BASE=/home/student/Desktop/Test
RESULTS=$BASE/results/evaluation
mkdir -p "$RESULTS"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS/eval.log"; }
ok()  { echo "  ✓ $*" | tee -a "$RESULTS/eval.log"; }
fail(){ echo "  ✗ $*" | tee -a "$RESULTS/eval.log"; }

TASKS="mmlu,hellaswag,arc_challenge,arc_easy,gsm8k,truthfulqa_mc1,winogrande"
BATCH=auto
MAX_BATCH=4
DTYPE=bfloat16

run_lmeval() {
    local model_path=$1
    local tag=$2
    local extra_args="${3:-}"
    local out_dir="$RESULTS/$tag"
    mkdir -p "$out_dir"

    log "Evaluating: $tag ($model_path)"
    lm-eval run \
        --model hf \
        --model_args "pretrained=$model_path,dtype=$DTYPE,trust_remote_code=True" \
        --tasks "$TASKS" \
        --num_fewshot 5 \
        --batch_size "$BATCH" \
        --max_batch_size "$MAX_BATCH" \
        --output_path "$out_dir" \
        $extra_args \
        2>&1 | tee "$out_dir/lmeval.log"

    if [ $? -eq 0 ]; then
        ok "$tag eval complete"
        python3 - "$out_dir" "$tag" << 'EOF'
import sys, json, pathlib, glob

out_dir, tag = sys.argv[1], sys.argv[2]
files = glob.glob(f"{out_dir}/**/*.json", recursive=True)
for f in sorted(files):
    try:
        data = json.loads(pathlib.Path(f).read_text())
        results = data.get("results", {})
        if not results: continue
        print(f"\n  {tag} Results:")
        for task, scores in results.items():
            acc = scores.get("acc,none") or scores.get("acc_norm,none") or scores.get("exact_match,none")
            if acc is not None:
                print(f"    {task:<30} {acc*100:.1f}%")
        break
    except:
        pass
EOF
    else
        fail "$tag eval failed"
    fi
}

run_thinking_comparison() {
    local model_path=$1
    local tag=$2
    log "Qwen3 thinking comparison: $tag"

    # No-think mode (standard — Qwen3 has thinking disabled by default via system prompt)
    mkdir -p "$RESULTS/${tag}_nothink"
    lm-eval run \
        --model hf \
        --model_args "pretrained=$model_path,dtype=$DTYPE,trust_remote_code=True" \
        --tasks "mmlu,gsm8k" \
        --num_fewshot 5 \
        --batch_size 4 \
        --output_path "$RESULTS/${tag}_nothink" \
        2>&1 | tee "$RESULTS/${tag}_nothink/lmeval.log" || fail "no-think eval failed"

    # Think mode (enable_thinking=True activates chain-of-thought reasoning)
    mkdir -p "$RESULTS/${tag}_think"
    lm-eval run \
        --model hf \
        --model_args "pretrained=$model_path,dtype=$DTYPE,trust_remote_code=True,enable_thinking=True" \
        --tasks "mmlu,gsm8k" \
        --num_fewshot 5 \
        --batch_size 1 \
        --output_path "$RESULTS/${tag}_think" \
        2>&1 | tee "$RESULTS/${tag}_think/lmeval.log" || fail "think eval failed"

    ok "Thinking comparison complete for $tag"
}

# ── Main ─────────────────────────────────────────────────────
eval_checkpoint() {
    local checkpoint=$1
    local exp_name
    exp_name=$(basename "$checkpoint")

    if [ ! -d "$checkpoint" ]; then return; fi
    if [ -f "$RESULTS/ft_${exp_name}/.done" ]; then
        log "  Skipping $exp_name (already done)"
        return
    fi

    mkdir -p "$RESULTS/ft_${exp_name}"

    if [ -f "$checkpoint/adapter_config.json" ]; then
        base=$(python3 -c "import json; print(json.load(open('$checkpoint/adapter_config.json'))['base_model_name_or_path'])")
        log "Evaluating PEFT: $exp_name (base=$base)"
        # Large models (70B+) need 4-bit loading to fit in 119 GB unified memory
        local extra_model_args=""
        if echo "$base" | grep -qiE "72B|70B|Mixtral|235B"; then
            extra_model_args=",load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16"
            log "  (using NF4 loading for large base model)"
        fi
        lm-eval run \
            --model hf \
            --model_args "pretrained=$base,peft=$checkpoint,dtype=$DTYPE,trust_remote_code=True${extra_model_args}" \
            --tasks "$TASKS" \
            --num_fewshot 5 \
            --batch_size "$BATCH" \
            --max_batch_size "$MAX_BATCH" \
            --output_path "$RESULTS/ft_${exp_name}" \
            2>&1 | tee "$RESULTS/ft_${exp_name}/lmeval.log" \
            && { ok "$exp_name done"; touch "$RESULTS/ft_${exp_name}/.done"; } \
            || fail "FT eval failed for $exp_name"
    else
        log "Evaluating full model: $exp_name"
        run_lmeval "$checkpoint" "ft_${exp_name}" \
            && touch "$RESULTS/ft_${exp_name}/.done" \
            || fail "FT eval failed for $exp_name"
    fi
}

main() {
    log "=== Phase 3: Standard Evaluation ==="
    python3 -c "import lm_eval; print('lm_eval', lm_eval.__version__)" 2>/dev/null || {
        fail "lm-eval not installed. Run: pip install lm-eval"
        exit 1
    }

    # 1. Base model baseline
    log "--- Base model: Qwen3-8B ---"
    run_lmeval "Qwen/Qwen3-8B" "qwen3-8b-base"
    run_thinking_comparison "Qwen/Qwen3-8B" "qwen3-8b-base"

    # 2. All trained checkpoints in models/ dir
    TRAIN_DIR="$BASE/models"
    log "--- Fine-tuned checkpoints ---"
    for checkpoint in "$TRAIN_DIR"/T1_* "$TRAIN_DIR"/T2_* "$TRAIN_DIR"/T3_* \
                      "$TRAIN_DIR"/T4_* "$TRAIN_DIR"/T5_* "$TRAIN_DIR"/T6_* \
                      "$TRAIN_DIR"/T7_* "$TRAIN_DIR"/T8_* "$TRAIN_DIR"/T11_* \
                      "$TRAIN_DIR"/T12_* "$TRAIN_DIR"/T20_* "$TRAIN_DIR"/T21_* \
                      "$TRAIN_DIR"/T22_* "$TRAIN_DIR"/T23_* "$TRAIN_DIR"/T24_* \
                      "$TRAIN_DIR"/M1_* "$TRAIN_DIR"/M2_* "$TRAIN_DIR"/M3_*; do
        eval_checkpoint "$checkpoint"
    done

    # 3. Summary table
    log "--- Results Summary ---"
    python3 - "$RESULTS" << 'EOF'
import sys, json, pathlib, glob

results_dir = pathlib.Path(sys.argv[1])
print(f"\n{'Model':<40} {'MMLU':>6} {'GSM8K':>6} {'ARC-C':>6} {'WG':>5} {'HS':>5}")
print("-" * 70)
for out_dir in sorted(results_dir.iterdir()):
    if not out_dir.is_dir(): continue
    files = sorted(out_dir.glob("**/*.json"))
    for f in files:
        try:
            data = json.loads(f.read_text())
            r = data.get("results", {})
            if not r: continue
            mmlu  = r.get("mmlu",{}).get("acc,none", 0) * 100
            gsm8k = r.get("gsm8k",{}).get("exact_match,none", 0) * 100
            arc_c = r.get("arc_challenge",{}).get("acc_norm,none", 0) * 100
            wg    = r.get("winogrande",{}).get("acc,none", 0) * 100
            hs    = r.get("hellaswag",{}).get("acc_norm,none", 0) * 100
            print(f"{out_dir.name:<40} {mmlu:>5.1f}% {gsm8k:>5.1f}% {arc_c:>5.1f}% {wg:>4.1f}% {hs:>4.1f}%")
            break
        except: pass
EOF

    log "=== Evaluation complete. Results in: $RESULTS ==="
}

main "$@"
