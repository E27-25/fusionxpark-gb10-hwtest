#!/usr/bin/env bash
# Phase 3 — lm-evaluation-harness standard benchmarks
# v3: per-task limits to keep each model under ~90 min total
#   Rate measured: ~0.83 req/sec at batch=8 on GB10 → budget ~4500 req/hr
#   LL tasks (--limit 100 each): ~1800 req → ~36 min
#   MMLU (--limit 5 per subtask): ~1140 req → ~23 min
#   GSM8K (--limit 100, batch=1): ~100 req → ~10 min
#   Total per model: ~70 min
set -uo pipefail

BASE=/home/student/Desktop/Test
RESULTS=$BASE/results/evaluation
mkdir -p "$RESULTS"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS/eval.log"; }
ok()  { echo "  ✓ $*" | tee -a "$RESULTS/eval.log"; }
fail(){ echo "  ✗ $*" | tee -a "$RESULTS/eval.log"; }

# Loglikelihood tasks — NO mmlu here (too many subtasks × limit)
LL_TASKS="hellaswag,arc_challenge,arc_easy,truthfulqa_mc1,winogrande"
LL_LIMIT=100   # per task → 100×(4+4+4+4+2)=1800 req → ~36 min/model

# MMLU — separate call so its limit doesn't blow up
MMLU_TASKS="mmlu"
MMLU_LIMIT=5   # per subtask → 57×5×4=1140 req → ~23 min/model

# Generate task — separate call, batch=1
GEN_TASKS="gsm8k"
GEN_LIMIT=100

DTYPE=bfloat16

# ── run one lm-eval call, return 0/1 ──────────────────────────
_lmeval() {
    local out_dir=$1; local model_args=$2; local tasks=$3
    local fewshot=${4:-5}; local batch=${5:-8}; local extra=${6:-}
    mkdir -p "$out_dir"
    lm-eval run \
        --model hf \
        --model_args "$model_args" \
        --tasks "$tasks" \
        --num_fewshot "$fewshot" \
        --batch_size "$batch" \
        --output_path "$out_dir" \
        $extra \
        2>&1 | tee -a "$out_dir/lmeval.log"
}

# ── evaluate one model (path, tag, extra_model_args) ─────────
eval_model() {
    local model_path=$1
    local tag=$2
    local extra_model_args="${3:-}"
    local out_dir="$RESULTS/$tag"

    if [ -f "$out_dir/.done" ]; then
        log "  Skipping $tag (already done)"
        return 0
    fi
    mkdir -p "$out_dir"
    log "Evaluating: $tag"

    local margs="pretrained=$model_path,dtype=$DTYPE,trust_remote_code=True${extra_model_args}"

    # 1. Non-MMLU loglikelihood tasks (limit 100/task → ~36 min)
    if [ ! -f "$out_dir/.ll_done" ]; then
        log "  $tag: LL tasks limit=$LL_LIMIT (batch=8, ~36 min)"
        if _lmeval "$out_dir" "$margs" "$LL_TASKS" 5 8 "--limit $LL_LIMIT"; then
            touch "$out_dir/.ll_done"
            ok "$tag LL tasks done"
        else
            fail "$tag LL tasks failed"
        fi
    else
        log "  $tag: LL tasks already done, skipping"
    fi

    # 2. MMLU separately (limit 5/subtask → 57×5×4=1140 req → ~23 min)
    if [ ! -f "$out_dir/.mmlu_done" ]; then
        log "  $tag: MMLU limit=$MMLU_LIMIT/subtask (batch=8, ~23 min)"
        if _lmeval "$out_dir/mmlu" "$margs" "$MMLU_TASKS" 5 8 "--limit $MMLU_LIMIT"; then
            touch "$out_dir/.mmlu_done"
            ok "$tag MMLU done"
        else
            fail "$tag MMLU failed (non-fatal)"
        fi
    else
        log "  $tag: MMLU already done, skipping"
    fi

    # 3. GSM8K (generate_until, batch=1, limit 100 samples → ~10 min)
    if [ ! -f "$out_dir/.gsm_done" ]; then
        log "  $tag: gsm8k (batch=1, limit=$GEN_LIMIT, ~10 min)"
        if _lmeval "$out_dir/gsm8k" "$margs" "$GEN_TASKS" 5 1 "--limit $GEN_LIMIT"; then
            touch "$out_dir/.gsm_done"
            ok "$tag GSM8K done"
        else
            fail "$tag GSM8K failed (non-fatal)"
        fi
    fi

    touch "$out_dir/.done"
    ok "$tag all done"

    # Print quick summary
    python3 - "$out_dir" "$tag" << 'PYEOF'
import sys, json, pathlib, glob
out_dir, tag = sys.argv[1], sys.argv[2]
r_all = {}
for f in sorted(pathlib.Path(out_dir).glob("**/*.json")):
    try:
        data = json.loads(f.read_text())
        r = data.get("results", {})
        if r: r_all.update(r)
    except: pass
if not r_all:
    print(f"  {tag}: no results yet")
else:
    mmlu  = r_all.get("mmlu",{}).get("acc,none")
    gsm8k = r_all.get("gsm8k",{}).get("exact_match,none")
    arc_c = r_all.get("arc_challenge",{}).get("acc_norm,none")
    wg    = r_all.get("winogrande",{}).get("acc,none")
    hs    = r_all.get("hellaswag",{}).get("acc_norm,none")
    def fmt(v): return f"{v*100:.1f}%" if v is not None else "  —  "
    print(f"  {tag}: MMLU={fmt(mmlu)} GSM8K={fmt(gsm8k)} ARC-C={fmt(arc_c)} WG={fmt(wg)} HS={fmt(hs)}")
PYEOF
}

# ── evaluate a fine-tuned checkpoint ─────────────────────────
eval_checkpoint() {
    local checkpoint=$1
    if [ ! -d "$checkpoint" ]; then return; fi
    local exp_name
    exp_name=$(basename "$checkpoint")
    local out_dir="$RESULTS/ft_${exp_name}"

    if [ -f "$out_dir/.done" ]; then
        log "  Skipping $exp_name (already done)"
        return
    fi

    if [ -f "$checkpoint/adapter_config.json" ]; then
        local base
        base=$(python3 -c "import json; print(json.load(open('$checkpoint/adapter_config.json'))['base_model_name_or_path'])")
        log "Evaluating PEFT: $exp_name (base=$base)"
        local extra=""
        if echo "$base" | grep -qiE "72B|70B|Mixtral|235B"; then
            extra=",load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16"
            log "  (NF4 loading for large base)"
        fi
        eval_model "$base" "ft_${exp_name}" ",peft=$checkpoint${extra}"
    else
        log "Evaluating full model: $exp_name"
        eval_model "$checkpoint" "ft_${exp_name}"
    fi
}

# ── Summary table ─────────────────────────────────────────────
print_summary() {
    log "--- Results Summary ---"
    python3 - "$RESULTS" << 'PYEOF'
import sys, json, pathlib

results_dir = pathlib.Path(sys.argv[1])
print(f"\n{'Model':<42} {'MMLU':>6} {'GSM8K':>6} {'ARC-C':>6} {'WG':>6} {'HS':>6}")
print("-" * 74)
rows = []
for d in sorted(results_dir.iterdir()):
    if not d.is_dir() or d.name.startswith("."): continue
    r_all = {}
    for f in sorted(d.glob("**/*.json")):
        try:
            data = json.loads(f.read_text())
            r = data.get("results", {})
            if r: r_all.update(r)
        except: pass
    if not r_all: continue
    def g(task, key):
        v = r_all.get(task, {}).get(key)
        return f"{v*100:5.1f}%" if v is not None else "   — "
    mmlu  = g("mmlu",         "acc,none")
    gsm8k = g("gsm8k",        "exact_match,none")
    arc_c = g("arc_challenge", "acc_norm,none")
    wg    = g("winogrande",    "acc,none")
    hs    = g("hellaswag",     "acc_norm,none")
    print(f"{d.name:<42} {mmlu} {gsm8k} {arc_c} {wg} {hs}")
PYEOF
}

# ── Main ─────────────────────────────────────────────────────
main() {
    log "=== Phase 3: Standard Evaluation (v2 — per-task saves) ==="
    python3 -c "import lm_eval; print('lm_eval', lm_eval.__version__)" 2>/dev/null || {
        fail "lm-eval not installed"; exit 1
    }

    # 1. Base model baseline
    log "--- Base model: Qwen3-8B ---"
    eval_model "Qwen/Qwen3-8B" "qwen3-8b-base"

    # 2. All fine-tuned checkpoints
    TRAIN_DIR="$BASE/models"
    log "--- Fine-tuned checkpoints ---"
    for checkpoint in \
        "$TRAIN_DIR"/T1_*  "$TRAIN_DIR"/T2_*  "$TRAIN_DIR"/T3_*  \
        "$TRAIN_DIR"/T4_*  "$TRAIN_DIR"/T5_*  "$TRAIN_DIR"/T6_*  \
        "$TRAIN_DIR"/T7_*  "$TRAIN_DIR"/T8_*  "$TRAIN_DIR"/T11_* \
        "$TRAIN_DIR"/T12_* "$TRAIN_DIR"/T20_* "$TRAIN_DIR"/T21_* \
        "$TRAIN_DIR"/T22_* "$TRAIN_DIR"/T23_* "$TRAIN_DIR"/T24_* \
        "$TRAIN_DIR"/M1_*  "$TRAIN_DIR"/M2_*  "$TRAIN_DIR"/M3_*; do
        eval_checkpoint "$checkpoint"
    done

    # 3. Summary
    print_summary
    log "=== Evaluation complete. Results in: $RESULTS ==="
}

main "$@"
