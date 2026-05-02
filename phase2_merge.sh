#!/usr/bin/env bash
# Phase 2 — Model Merging via mergekit (M1, M2, M3)
# No GPU training required — CPU merge then load to GPU for validation
set -euo pipefail

BASE=/home/student/Desktop/Test
MODELS=$BASE/models
RESULTS=$BASE/results/training
mkdir -p "$MODELS" "$RESULTS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
ok()  { echo "  ✓ $*"; }
fail(){ echo "  ✗ $*"; }

# ── Helpers ──────────────────────────────────────────────────
check_mergekit() {
    python3 -c "import mergekit" 2>/dev/null || {
        fail "mergekit not installed. Run: pip install mergekit"
        exit 1
    }
    ok "mergekit available"
}

run_merge() {
    local exp_id=$1
    local config_path=$2
    local out_dir=$3

    log "Running $exp_id: $out_dir"
    mkdir -p "$out_dir"

    python3 -m mergekit.merge "$config_path" "$out_dir" \
        --allow-crimes \
        --copy-tokenizer \
        --trust-remote-code \
        2>&1 | tee "$RESULTS/${exp_id}_merge.log"

    if [ $? -eq 0 ]; then
        ok "$exp_id merge complete: $out_dir"
    else
        fail "$exp_id merge failed — check $RESULTS/${exp_id}_merge.log"
        return 1
    fi
}

validate_merge() {
    local exp_id=$1
    local model_path=$2
    log "Validating $exp_id at $model_path..."
    python3 - <<EOF
import json, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

model_path = "$model_path"
results_path = "$RESULTS/${exp_id}_validation.json"

try:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    m.eval()
    mem = torch.cuda.memory_allocated() / 1e9

    prompt = "Explain the concept of gradient descent in neural networks."
    inp = tok(prompt, return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=100, do_sample=False)
    elapsed = time.perf_counter() - t0
    generated = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    result = {
        "experiment": "$exp_id",
        "model_path": model_path,
        "status": "ok",
        "memory_gb": round(mem, 2),
        "generation_sec": round(elapsed, 2),
        "generated": generated[:200],
    }
    print(f"  ✓ {result['experiment']}: {result['generated'][:80]}...")
    del m; import gc; gc.collect(); torch.cuda.empty_cache()

except Exception as e:
    result = {"experiment": "$exp_id", "status": "error", "error": str(e)}
    print(f"  ✗ Error: {e}")

Path(results_path).write_text(__import__("json").dumps(result, indent=2))
print(f"  Saved: {results_path}")
EOF
}

# ── M1: SLERP — Qwen3-8B base + SFT adapter merged ──────────
write_m1_config() {
    local sft_path="${1:-Qwen/Qwen3-8B}"  # Use T3 output if available
    cat > /tmp/m1_slerp.yaml << EOF
models:
  - model: Qwen/Qwen3-8B
  - model: $sft_path
merge_method: slerp
base_model: Qwen/Qwen3-8B
parameters:
  t: 0.5
dtype: bfloat16
EOF
    echo /tmp/m1_slerp.yaml
}

# ── M2: TIES — SFT + DPO ─────────────────────────────────────
write_m2_config() {
    local sft_path="${1:-Qwen/Qwen3-8B}"
    local dpo_path="${2:-Qwen/Qwen3-8B}"
    cat > /tmp/m2_ties.yaml << EOF
models:
  - model: $sft_path
    parameters:
      weight: 0.6
  - model: $dpo_path
    parameters:
      weight: 0.4
merge_method: ties
base_model: Qwen/Qwen3-8B
parameters:
  density: 0.5
  normalize: true
dtype: bfloat16
EOF
    echo /tmp/m2_ties.yaml
}

# ── M3: DARE+TIES — General + Code specialist ─────────────────
write_m3_config() {
    cat > /tmp/m3_dare_ties.yaml << EOF
models:
  - model: Qwen/Qwen3-8B
    parameters:
      density: 0.7
      weight: 0.5
  - model: Qwen/Qwen2.5-Coder-7B-Instruct
    parameters:
      density: 0.7
      weight: 0.5
merge_method: dare_ties
base_model: Qwen/Qwen3-8B
parameters:
  density: 0.7
  normalize: true
dtype: bfloat16
EOF
    echo /tmp/m3_dare_ties.yaml
}

# ── Main ─────────────────────────────────────────────────────
main() {
    log "=== Phase 2: Model Merging ==="
    check_mergekit

    # Auto-detect and merge LoRA adapters into full models if needed
    T3_ADAPTER="$MODELS/T3_Qwen3-8B_lora_r16"
    T20_ADAPTER="$MODELS/T20_Qwen3-8B_dpo"

    merge_lora_if_needed() {
        local adapter_dir="$1"
        local out_dir="${adapter_dir}_merged"
        if [ -f "$adapter_dir/adapter_config.json" ] && [ ! -d "$out_dir" ]; then
            log "Merging LoRA adapter: $adapter_dir → $out_dir"
            python3 "$BASE/merge_lora_to_full.py" \
                --adapter "$adapter_dir" \
                --output  "$out_dir" 2>&1 | tee -a "$RESULTS/merge_prep.log"
        fi
        echo "$out_dir"
    }

    if [ -f "$T3_ADAPTER/adapter_config.json" ]; then
        T3_PATH=$(merge_lora_if_needed "$T3_ADAPTER")
    else
        T3_PATH=${T3_PATH:-"Qwen/Qwen3-8B"}
        log "T3 adapter not found, using base: $T3_PATH"
    fi

    if [ -f "$T20_ADAPTER/adapter_config.json" ]; then
        T20_PATH=$(merge_lora_if_needed "$T20_ADAPTER")
    else
        T20_PATH=${T20_PATH:-"$T3_PATH"}
        log "T20 adapter not found, using: $T20_PATH"
    fi

    # M1: SLERP
    log "--- M1: SLERP merge ---"
    M1_CFG=$(write_m1_config "$T3_PATH")
    M1_OUT="$MODELS/M1_qwen3-8b_slerp"
    if run_merge "M1" "$M1_CFG" "$M1_OUT"; then
        validate_merge "M1" "$M1_OUT"
    fi

    # M2: TIES
    log "--- M2: TIES merge (SFT + DPO) ---"
    M2_CFG=$(write_m2_config "$T3_PATH" "$T20_PATH")
    M2_OUT="$MODELS/M2_qwen3-8b_ties"
    if run_merge "M2" "$M2_CFG" "$M2_OUT"; then
        validate_merge "M2" "$M2_OUT"
    fi

    # M3: DARE+TIES
    log "--- M3: DARE+TIES (General + Code) ---"
    M3_CFG=$(write_m3_config)
    M3_OUT="$MODELS/M3_qwen3-8b_dare_ties"
    if run_merge "M3" "$M3_CFG" "$M3_OUT"; then
        validate_merge "M3" "$M3_OUT"
    fi

    log "=== Model merging complete ==="
    log "Merged models saved to: $MODELS"
    log "Results saved to: $RESULTS"
}

main "$@"
