#!/usr/bin/env bash
# Re-run only the parts that failed in Phase 1
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

log "=== PHASE 1 FIX RUN ==="

# Fix 1: INT8 quant sweep (fixed BitsAndBytesConfig API)
log "─── Re-run: INT8 quantization only ───"
python3 - <<'EOF' 2>&1 | tee -a "$LOG"
import json, sys, time, torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
MODEL   = "Qwen/Qwen3-8B"
PROMPT  = "Explain the difference between supervised and unsupervised learning."
MAX_NEW = 200
ITERS   = 5

def bench_hf(model, tok):
    inp = tok(PROMPT, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / ITERS
    n_new = out.shape[1] - inp["input_ids"].shape[1]
    tps = n_new / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return {"ttft_ms": round(ttft,1), "tokens_per_sec": round(tps,1),
            "tpot_ms": round(elapsed/n_new*1000,2), "peak_memory_gb": round(mem,2)}

print("  [INT8] Loading with BitsAndBytesConfig...")
try:
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                              device_map="cuda", trust_remote_code=True)
    m.eval()
    r = bench_hf(m, tok)
    print(f"  INT8 ✓  TPS={r['tokens_per_sec']}  TTFT={r['ttft_ms']}ms  mem={r['peak_memory_gb']}GB")
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    r = {"error": str(e)}
    print(f"  INT8 ❌ {e}")

# Patch the existing quant_sweep file
import glob
files = sorted(glob.glob(str(RESULTS / "quant_sweep_*.json")))
if files:
    d = json.loads(Path(files[-1]).read_text())
    d["int8"] = r
    Path(files[-1]).write_text(json.dumps(d, indent=2))
    print(f"  Updated: {files[-1]}")
EOF

# Fix 2: ASR Benchmark (correct model name format)
log "─── Re-run: ASR Benchmark ───"
run_py phase1_asr_bench.py \
    --models whisper-large-v3 whisper-large-v3-turbo \
    --samples 50

# Fix 3: Embedding bench with available models
log "─── Re-run: Embedding Benchmark (BAAI/bge-m3) ───"
run_py phase1_embed_rerank_bench.py --mode embed --batch-sizes 1 8 32

log "=== PHASE 1 FIX RUN DONE ==="
python3 "$BASE/tracker.py" 2>/dev/null | tee -a "$LOG"
