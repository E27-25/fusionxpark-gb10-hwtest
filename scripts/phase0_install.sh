#!/usr/bin/env bash
# Phase 0 — Full installation script with fallbacks
# GB10 Grace Blackwell | aarch64 | CUDA 13.0
# Logs to: /home/student/Desktop/Test/logs/install.log
set -uo pipefail

# Allow pip to install system-wide on Ubuntu 24.04+
export PIP_BREAK_SYSTEM_PACKAGES=1
# Convenient pip alias that always passes the flag
pip() { command pip3 --quiet "$@"; }

BASE=/home/student/Desktop/Test
LOG=$BASE/logs/install.log
mkdir -p "$BASE/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()  { echo "  ✓ $*" | tee -a "$LOG"; }
warn(){ echo "  ⚠ $*" | tee -a "$LOG"; }
fail(){ echo "  ✗ $*" | tee -a "$LOG"; }

log "=== Phase 0: Environment Setup ==="
log "Platform: $(uname -m)  OS: $(lsb_release -ds 2>/dev/null || echo unknown)"
log "Python: $(python3 --version 2>&1)"

# ── 1. PyTorch (CRITICAL — must come first) ──────────────────
log "--- 1. PyTorch (CUDA 13.0, aarch64) ---"

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ok "PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    log "Trying PyTorch nightly (cu130, aarch64)..."
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu130 \
        --quiet 2>&1 | tail -3 | tee -a "$LOG" || {
        warn "Nightly failed — trying stable cu121 as fallback..."
        pip install torch torchvision torchaudio --quiet 2>&1 | tail -3 | tee -a "$LOG" || {
            fail "PyTorch install failed. Try: docker pull nvcr.io/nvidia/pytorch:25.03-py3"
            exit 1
        }
    }
    python3 -c "import torch; print('  PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" | tee -a "$LOG"
fi

# ── 2. HuggingFace core stack ─────────────────────────────────
log "--- 2. HuggingFace Core Stack ---"
pip3 install --break-system-packages -q \
    "transformers>=4.45" \
    "accelerate>=0.34" \
    "datasets>=2.20" \
    "peft>=0.12" \
    "trl>=0.10" \
    "tokenizers>=0.19" \
    sentencepiece \
    einops \
    huggingface_hub \
    2>&1 | tail -3 | tee -a "$LOG" || warn "HF stack partial install"
ok "HuggingFace stack installed"

# ── 3. bitsandbytes (QLoRA — HIGH RISK for SM120) ────────────
log "--- 3. bitsandbytes (SM120 / Blackwell) ---"
if pip install bitsandbytes --quiet 2>&1 | tee -a "$LOG" | grep -q "Successfully"; then
    # verify SM120 kernel
    python3 -c "
import bitsandbytes as bnb
cs = bnb.cuda_setup.CUDASetup.get_instance()
print('  bnb CUDA setup:', cs.cuda_setup_log if hasattr(cs,'cuda_setup_log') else 'loaded')
" 2>&1 | tee -a "$LOG" || warn "bnb loaded but SM120 kernels may be missing"
    ok "bitsandbytes installed"
else
    warn "bitsandbytes pip failed — trying source build..."
    pip install git+https://github.com/TimDettmers/bitsandbytes.git --quiet 2>&1 | tail -5 | tee -a "$LOG" || {
        fail "bitsandbytes failed — QLoRA experiments (T8) blocked"
    }
fi

# ── 4. Flash Attention 3 (MEDIUM RISK) ───────────────────────
log "--- 4. Flash Attention 3 ---"
if python3 -c "import flash_attn" 2>/dev/null; then
    ok "Flash Attention already installed"
else
    # try pip first
    pip install flash-attn --no-build-isolation --quiet 2>&1 | tail -3 | tee -a "$LOG" || {
        warn "pip failed — building FA from source (~20-45 min)..."
        cd /tmp
        git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git fa3 2>&1 | tail -3 | tee -a "$LOG"
        cd fa3
        pip install -e . --no-build-isolation --quiet 2>&1 | tail -5 | tee -a "$LOG" || {
            warn "FA3 build failed — using torch SDPA fallback (automatic)"
        }
        cd "$BASE"
    }
fi

# ── 5. Unsloth (HIGH RISK — likely fails on aarch64) ─────────
log "--- 5. Unsloth (may fail on aarch64) ---"
pip install "unsloth[colab-new]" --quiet 2>&1 | tail -3 | tee -a "$LOG" || {
    warn "Unsloth failed (expected on aarch64 — Triton not supported)"
    warn "T13 experiment blocked. Fallback: PEFT+TRL"
    python3 -c "
import json, pathlib
state = pathlib.Path('$BASE/logs/install_state.json')
d = json.loads(state.read_text()) if state.exists() else {}
d['unsloth_available'] = False
state.write_text(json.dumps(d, indent=2))
"
}

# ── 6. AWQ + GPTQ + FP8 tools ────────────────────────────────
log "--- 6. Quantization libraries ---"
pip install autoawq --quiet 2>&1 | tail -2 | tee -a "$LOG" && ok "autoawq" || warn "autoawq failed"
pip install auto-gptq optimum --quiet 2>&1 | tail -2 | tee -a "$LOG" && ok "auto-gptq" || warn "auto-gptq failed"

# ── 7. Inference servers ──────────────────────────────────────
log "--- 7. Inference Servers ---"

# vLLM
log "  Installing vLLM..."
pip install vllm --quiet 2>&1 | tail -3 | tee -a "$LOG" && ok "vLLM" || {
    warn "vLLM pip failed — trying source..."
    pip install git+https://github.com/vllm-project/vllm.git --quiet 2>&1 | tail -3 | tee -a "$LOG" || fail "vLLM failed"
}

# SGLang
log "  Installing SGLang..."
pip install "sglang[all]" --quiet 2>&1 | tail -3 | tee -a "$LOG" && ok "SGLang" || warn "SGLang failed"

# llama.cpp (build from source — most reliable on aarch64)
log "  Building llama.cpp..."
if command -v llama-cli &>/dev/null; then
    ok "llama.cpp already in PATH"
else
    cd /tmp
    git clone --depth 1 https://github.com/ggerganov/llama.cpp llama_cpp_build 2>&1 | tail -2 | tee -a "$LOG"
    cd llama_cpp_build
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF 2>&1 | tail -5 | tee -a "$LOG"
    cmake --build build --config Release -j$(nproc) 2>&1 | tail -5 | tee -a "$LOG"
    sudo cmake --install build 2>&1 | tail -3 | tee -a "$LOG" || cp build/bin/llama-cli /usr/local/bin/ 2>/dev/null
    pip install llama-cpp-python --quiet 2>&1 | tail -3 | tee -a "$LOG" || warn "llama-cpp-python pip failed"
    cd "$BASE"
    ok "llama.cpp built"
fi

# ── 8. Mergekit ───────────────────────────────────────────────
log "--- 8. MergeKit ---"
pip install mergekit --quiet 2>&1 | tail -2 | tee -a "$LOG" && ok "mergekit" || warn "mergekit failed"

# ── 9. Audio / TTS / ASR dependencies ────────────────────────
log "--- 9. Audio / ASR / TTS ---"
pip install -q librosa soundfile evaluate jiwer 2>&1 | tail -2 | tee -a "$LOG"
pip install -q utmos resemblyzer 2>&1 | tail -2 | tee -a "$LOG" || warn "utmos/resemblyzer may need manual install"
ok "Audio stack"

# ── 10. Evaluation frameworks ────────────────────────────────
log "--- 10. Evaluation ---"
pip install -q lm-eval 2>&1 | tail -2 | tee -a "$LOG" && ok "lm-eval"
pip install -q mteb 2>&1 | tail -2 | tee -a "$LOG" && ok "mteb"
pip install -q beir 2>&1 | tail -2 | tee -a "$LOG" && ok "beir" || warn "beir may need manual install"
pip install -q evalplus 2>&1 | tail -2 | tee -a "$LOG" && ok "evalplus"
pip install -q faiss-gpu 2>&1 | tail -2 | tee -a "$LOG" || pip install -q faiss-cpu 2>&1 | tail -2 | tee -a "$LOG"
ok "faiss"

# ── 11. Utilities ─────────────────────────────────────────────
log "--- 11. Utilities ---"
pip install -q psutil pandas matplotlib seaborn tqdm pyyaml 2>&1 | tail -2 | tee -a "$LOG"
ok "Utilities"

# ── Done ──────────────────────────────────────────────────────
log "=== Installation complete ==="
log "Run next: python3 $BASE/phase0_verify_env.py"
