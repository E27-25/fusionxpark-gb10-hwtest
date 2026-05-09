#!/usr/bin/env bash
# Phase 0 — Quick Essential Install (no source builds)
# GB10 Grace Blackwell | aarch64 | CUDA 13.0 | Ubuntu 24.04
# Run via: ./run_bench.sh install
set -uo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1

BASE=/home/student/Desktop/Test
LOG=$BASE/logs/install.log
mkdir -p "$BASE/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()  { echo "  ✓ $*"  | tee -a "$LOG"; }
warn(){ echo "  ⚠ $*"  | tee -a "$LOG"; }
pip_install() { pip3 install --break-system-packages -q "$@" 2>&1 | tail -3 | tee -a "$LOG"; }

log "=== Phase 0: Quick Install ==="
log "Platform: $(uname -m)  Python: $(python3 --version 2>&1)"

# 1. PyTorch — already installed
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ok "PyTorch $(python3 -c 'import torch; print(torch.__version__)')"
else
    log "Installing PyTorch nightly (cu128, aarch64)..."
    pip3 install --break-system-packages -q --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128
    ok "PyTorch installed"
fi

# 2. HuggingFace core stack
log "--- 2. HuggingFace Core Stack ---"
pip_install \
    "transformers>=4.45" \
    "accelerate>=0.34" \
    "datasets>=2.20" \
    "peft>=0.12" \
    "trl>=0.10" \
    "tokenizers>=0.19" \
    sentencepiece \
    einops \
    huggingface_hub \
    && ok "HuggingFace core" || warn "HF core partial fail"

# 3. bitsandbytes (QLoRA)
log "--- 3. bitsandbytes ---"
pip_install bitsandbytes && ok "bitsandbytes" || warn "bitsandbytes failed — QLoRA blocked"

# 4. vLLM (inference server)
log "--- 4. vLLM ---"
pip_install vllm && ok "vLLM" || warn "vLLM failed"

# 5. Audio / ASR / TTS
log "--- 5. Audio stack ---"
pip_install librosa soundfile && ok "librosa + soundfile" || warn "audio partial"
pip_install evaluate jiwer && ok "evaluate + jiwer" || warn "evaluate partial"

# 6. Evaluation frameworks
log "--- 6. Evaluation ---"
pip_install lm_eval && ok "lm-eval" || warn "lm-eval failed"
pip_install mteb && ok "mteb" || warn "mteb failed"
pip_install evalplus && ok "evalplus" || warn "evalplus failed"

# 7. RAG / Retrieval
log "--- 7. RAG stack ---"
pip_install faiss-gpu && ok "faiss-gpu" || {
    pip_install faiss-cpu && ok "faiss-cpu (fallback)" || warn "faiss failed"
}

# 8. Quantization
log "--- 8. Quantization libs ---"
pip_install autoawq && ok "autoawq" || warn "autoawq failed"
pip_install auto-gptq optimum && ok "auto-gptq + optimum" || warn "auto-gptq failed"

# 9. Model merging
log "--- 9. mergekit ---"
pip_install mergekit && ok "mergekit" || warn "mergekit failed"

# 10. Utilities
log "--- 10. Utilities ---"
pip_install psutil pandas matplotlib seaborn tqdm pyyaml rich && ok "utilities"

# 11. Optional
log "--- 11. Optional (may fail) ---"
pip_install utmos resemblyzer && ok "utmos + resemblyzer" || warn "utmos/resemblyzer failed"
pip_install beir && ok "beir" || warn "beir failed"
pip_install "sglang[all]" && ok "SGLang" || warn "SGLang failed"

# Summary
log "=== Install complete ==="
log "Checking installs..."
python3 - <<'EOF' 2>&1 | tee -a "$LOG"
import importlib
pkgs = {
    'torch':       'CRITICAL',
    'transformers':'CRITICAL',
    'accelerate':  'CRITICAL',
    'datasets':    'CRITICAL',
    'peft':        'CRITICAL',
    'trl':         'CRITICAL',
    'bitsandbytes':'HIGH',
    'vllm':        'HIGH',
    'librosa':     'MEDIUM',
    'evaluate':    'MEDIUM',
    'lm_eval':     'MEDIUM',
    'faiss':       'MEDIUM',
    'autoawq':     'LOW',
    'mergekit':    'LOW',
}
missing = []
for p, priority in pkgs.items():
    try:
        m = importlib.import_module(p)
        ver = getattr(m, '__version__', '?')
        print(f"  ✓ {p:<20} {ver}")
    except ImportError:
        print(f"  ✗ {p:<20} MISSING [{priority}]")
        if priority == 'CRITICAL':
            missing.append(p)
if missing:
    print(f"\n  ⚠ CRITICAL packages missing: {missing}")
    print("  Phase 1+ experiments will fail without these.")
else:
    print("\n  All CRITICAL packages installed ✓")
EOF

log ""
log "Next: python3 $BASE/phase0_verify_env.py"
log "Then: ./run_bench.sh phase1"
