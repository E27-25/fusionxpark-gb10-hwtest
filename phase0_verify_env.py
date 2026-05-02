#!/usr/bin/env python3
"""
Phase 0 — Environment Verification
Checks all installs, CUDA capability, SM120 kernels, writes install_state.json
"""
import sys, json, subprocess
from pathlib import Path
from datetime import datetime

BASE = Path("/home/student/Desktop/Test")
STATE = BASE / "logs/install_state.json"
STATE.parent.mkdir(parents=True, exist_ok=True)

state = {"timestamp": datetime.now().isoformat(), "checks": {}, "blockers": []}

def check(name, fn):
    try:
        result = fn()
        state["checks"][name] = {"status": "ok", "detail": str(result)}
        print(f"  ✓ {name:<35} {result}")
        return True
    except Exception as e:
        state["checks"][name] = {"status": "fail", "detail": str(e)}
        print(f"  ✗ {name:<35} {e}")
        return False

print(f"\n{'='*60}")
print(f"  Phase 0 — Environment Verification")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

# ── PyTorch + CUDA ────────────────────────────────────────────
print("[ PyTorch & CUDA ]")
import torch
ok = check("PyTorch version", lambda: torch.__version__)
ok &= check("CUDA available", lambda: f"{'YES' if torch.cuda.is_available() else 'NO'}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    ok &= check("GPU name", lambda: torch.cuda.get_device_name(0))
    ok &= check("Compute capability (need 12.x)", lambda: f"{cap[0]}.{cap[1]}")
    ok &= check("Total memory (GB)", lambda: f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}")
    # matmul sanity
    ok &= check("BF16 matmul on CUDA", lambda: (
        torch.mm(
            torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16),
            torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
        ).shape
    ))
    if cap[0] < 12:
        state["blockers"].append("Compute capability < 12 — not Blackwell GB10")
else:
    state["blockers"].append("CUDA not available — PyTorch install failed")

# ── HuggingFace ───────────────────────────────────────────────
print("\n[ HuggingFace Stack ]")
check("transformers", lambda: __import__("transformers").__version__)
check("accelerate",   lambda: __import__("accelerate").__version__)
check("datasets",     lambda: __import__("datasets").__version__)
check("peft",         lambda: __import__("peft").__version__)
check("trl",          lambda: __import__("trl").__version__)

# ── Quantization ──────────────────────────────────────────────
print("\n[ Quantization Libraries ]")
bnb_ok = check("bitsandbytes", lambda: __import__("bitsandbytes").__version__)
if not bnb_ok:
    state["blockers"].append("bitsandbytes missing — T8 QLoRA blocked")
check("autoawq",   lambda: __import__("awq").__version__ if hasattr(__import__("awq"), "__version__") else "installed")
check("auto-gptq", lambda: __import__("auto_gptq").__version__ if hasattr(__import__("auto_gptq"), "__version__") else "installed")

# ── Flash Attention ───────────────────────────────────────────
print("\n[ Attention Kernels ]")
fa3_ok = check("flash-attn (FA3)", lambda: __import__("flash_attn").__version__)
state["checks"]["flash_attn_available"] = fa3_ok
if not fa3_ok:
    print("     → fallback: torch SDPA (automatic, no action needed)")

# ── Unsloth ───────────────────────────────────────────────────
print("\n[ Unsloth ]")
unsloth_ok = check("unsloth", lambda: __import__("unsloth").__version__ if hasattr(__import__("unsloth"), "__version__") else "installed")
state["checks"]["unsloth_available"] = unsloth_ok
if not unsloth_ok:
    print("     → expected on aarch64 — T13 skipped, PEFT+TRL fallback active")

# ── Inference Servers ────────────────────────────────────────
print("\n[ Inference Servers ]")
vllm_ok = check("vLLM", lambda: __import__("vllm").__version__)
if not vllm_ok:
    state["blockers"].append("vLLM missing — use SGLang or llama.cpp instead")
check("SGLang", lambda: __import__("sglang").__version__)

r = subprocess.run(["which", "llama-cli"], capture_output=True, text=True)
llamacpp_ok = r.returncode == 0
state["checks"]["llama_cpp"] = {"status": "ok" if llamacpp_ok else "fail"}
print(f"  {'✓' if llamacpp_ok else '✗'} llama.cpp (llama-cli)       {'found' if llamacpp_ok else 'not in PATH'}")

# ── Audio / ASR ───────────────────────────────────────────────
print("\n[ Audio / ASR / TTS ]")
check("librosa",    lambda: __import__("librosa").__version__)
check("soundfile",  lambda: __import__("soundfile").__version__)
check("evaluate",   lambda: __import__("evaluate").__version__)
check("jiwer (WER)", lambda: __import__("jiwer").__version__)

# ── Evaluation ────────────────────────────────────────────────
print("\n[ Evaluation Frameworks ]")
check("lm-eval",  lambda: __import__("lm_eval").__version__)
check("mteb",     lambda: __import__("mteb").__version__)
check("evalplus", lambda: __import__("evalplus").__version__ if hasattr(__import__("evalplus"), "__version__") else "installed")
check("faiss",    lambda: "gpu" if __import__("faiss").get_num_gpus() > 0 else "cpu")

# ── Utilities ────────────────────────────────────────────────
print("\n[ Utilities ]")
check("mergekit", lambda: "installed" if __import__("mergekit") else "")
check("psutil",   lambda: __import__("psutil").__version__)
check("pandas",   lambda: __import__("pandas").__version__)
check("matplotlib",lambda: __import__("matplotlib").__version__)
check("pyyaml",   lambda: __import__("yaml").__version__)

# ── Write state ──────────────────────────────────────────────
STATE.write_text(json.dumps(state, indent=2))

n_ok   = sum(1 for v in state["checks"].values() if isinstance(v, dict) and v.get("status") == "ok")
n_fail = sum(1 for v in state["checks"].values() if isinstance(v, dict) and v.get("status") == "fail")
n_bool = sum(1 for v in state["checks"].values() if isinstance(v, bool))

print(f"\n{'='*60}")
print(f"  Results: {n_ok} ✓   {n_fail} ✗")
if state["blockers"]:
    print(f"\n  BLOCKERS:")
    for b in state["blockers"]:
        print(f"    ✗ {b}")
else:
    print(f"\n  No blockers — ready for Phase 0.5 hardware benchmark")
print(f"\n  State saved: {STATE}")
print(f"  Next: python3 {BASE}/phase05_hw_bench.py")
print(f"{'='*60}")
