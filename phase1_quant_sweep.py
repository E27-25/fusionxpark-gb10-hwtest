#!/usr/bin/env python3
"""
Phase 1 — Quantization Format Sweep
BF16 → FP8 → INT8 → AWQ → GPTQ → NF4 → GGUF Q4/Q5/Q8 on Qwen3-8B
Measures: tokens/sec, TTFT, memory, BW util, perplexity proxy
"""
import json, time, subprocess, argparse, statistics
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL   = "Qwen/Qwen3-8B"
PROMPT  = "Explain the difference between supervised and unsupervised learning."
MAX_NEW = 200
N_RUNS  = 5


def percentile(data, p):
    data = sorted(data)
    return data[min(int(p / 100 * len(data)), len(data) - 1)]


def get_power():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        p, t = r.stdout.strip().split(",")
        return float(p.strip()), int(t.strip())
    except:
        return 0.0, 0


def bench_hf(model, tok, max_new=MAX_NEW):
    import torch

    # Count parameters for MFU/BW calculations
    n_params = sum(p.numel() for p in model.parameters())
    # model_load_gb: estimate from parameter count at ~2 bytes/param (bf16 reference)
    model_load_gb = n_params * 2 / 1e9
    batch_size = 1

    inp = tok(PROMPT, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()

    # warmup
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()

    # Collect N_RUNS TTFT measurements
    ttft_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        ttft_times.append((time.perf_counter() - t0) * 1000)

    # Collect N_RUNS generation time measurements
    power_start, _ = get_power()
    gen_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        gen_times.append(time.perf_counter() - t0)
    power_end, temp = get_power()

    n_new = out.shape[1] - inp["input_ids"].shape[1]

    mean_elapsed = statistics.mean(gen_times)
    tps_list = [n_new / t for t in gen_times]
    tps_mean = statistics.mean(tps_list)
    tps_stdev = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0

    ttft_mean = statistics.mean(ttft_times)
    ttft_p95  = percentile(ttft_times, 95)

    mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    power = (power_start + power_end) / 2
    tok_per_watt = tps_mean / power if power > 0 else None

    # MFU: 2 * n_params * tps / 1e12 / 67.0  (2N for inference)
    mfu = 2 * n_params * tps_mean / 1e12 / 67.0

    # BW: model_load_gb * (tps / batch_size) / 273.0 * 100
    bw_util_pct = model_load_gb * (tps_mean / batch_size) / 273.0 * 100

    return {
        "ttft_ms":        round(ttft_mean, 1),
        "ttft_p95_ms":    round(ttft_p95, 1),
        "tokens_per_sec": round(tps_mean, 1),
        "tps_stdev":      round(tps_stdev, 2),
        "tpot_ms":        round(mean_elapsed / n_new * 1000, 2),
        "peak_memory_gb": round(mem, 2),
        "power_w":        round(power, 1),
        "temp_c":         temp,
        "tokens_per_watt": round(tok_per_watt, 2) if tok_per_watt else None,
        "mfu":            round(mfu, 4),
        "bw_util_pct":    round(bw_util_pct, 1),
    }

# ─────────────────────────────────────────────────────────────
def run_bf16():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("  [BF16] Loading...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                              device_map="cuda", trust_remote_code=True)
    m.eval()
    try:
        r = bench_hf(m, tok)
    except torch.cuda.OutOfMemoryError as e:
        r = {"error": f"OOM: {e}"}
    except Exception as e:
        r = {"error": str(e)}
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
    return r

def run_int8():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print("  [INT8] Loading...")
    try:
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                   device_map="cuda", trust_remote_code=True)
        m.eval()
        try:
            r = bench_hf(m, tok)
        except torch.cuda.OutOfMemoryError as e:
            r = {"error": f"OOM: {e}"}
        except Exception as e:
            r = {"error": str(e)}
        del m; import gc; gc.collect(); torch.cuda.empty_cache()
        return r
    except Exception as e:
        return {"error": str(e)}

def run_nf4():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print("  [NF4] Loading...")
    try:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                   device_map="cuda", trust_remote_code=True)
        m.eval()
        try:
            r = bench_hf(m, tok)
        except torch.cuda.OutOfMemoryError as e:
            r = {"error": f"OOM: {e}"}
        except Exception as e:
            r = {"error": str(e)}
        del m; import gc; gc.collect(); torch.cuda.empty_cache()
        return r
    except Exception as e:
        return {"error": str(e)}

def run_awq():
    print("  [AWQ] Loading...")
    try:
        import torch
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        # First quantize if not already done
        awq_path = BASE / "models/qwen3-8b-awq"
        if not awq_path.exists():
            print("    Quantizing to AWQ (this takes ~5 min)...")
            m = AutoAWQForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
            m.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
            m.save_quantized(str(awq_path))
            tok.save_pretrained(str(awq_path))

        m = AutoAWQForCausalLM.from_quantized(str(awq_path), trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(str(awq_path), trust_remote_code=True)
        try:
            r = bench_hf(m, tok)
        except torch.cuda.OutOfMemoryError as e:
            r = {"error": f"OOM: {e}"}
        except Exception as e:
            r = {"error": str(e)}
        del m; import gc; gc.collect()
        return r
    except Exception as e:
        return {"error": str(e)}

def run_gguf(quant_type):
    print(f"  [GGUF {quant_type}] Loading via llama-cpp-python...")
    try:
        gguf_path = BASE / f"models/qwen3-8b-{quant_type.lower()}.gguf"
        if not gguf_path.exists():
            print(f"    GGUF file not found at {gguf_path}")
            print(f"    Convert with: python3 llama.cpp/convert_hf_to_gguf.py {MODEL}")
            return {"error": "gguf_not_found"}

        from llama_cpp import Llama

        m = Llama(model_path=str(gguf_path), n_gpu_layers=-1, n_ctx=2048, verbose=False)

        # Collect N_RUNS TTFT measurements
        ttft_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            m(PROMPT, max_tokens=1, echo=False)
            ttft_times.append((time.perf_counter() - t0) * 1000)

        # Collect N_RUNS generation time measurements
        gen_times = []
        n_new_last = 0
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            out = m(PROMPT, max_tokens=MAX_NEW, echo=False)
            gen_times.append(time.perf_counter() - t0)
            n_new_last = out["usage"]["completion_tokens"]

        tps_list  = [n_new_last / t for t in gen_times]
        tps_mean  = statistics.mean(tps_list)
        tps_stdev = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
        ttft_mean = statistics.mean(ttft_times)
        ttft_p95  = percentile(ttft_times, 95)
        mean_elapsed = statistics.mean(gen_times)

        del m
        return {
            "ttft_ms":        round(ttft_mean, 1),
            "ttft_p95_ms":    round(ttft_p95, 1),
            "tokens_per_sec": round(tps_mean, 1),
            "tps_stdev":      round(tps_stdev, 2),
            "tpot_ms":        round(mean_elapsed / n_new_last * 1000, 2),
            "peak_memory_gb": None,
            "power_w":        None,
            "temp_c":         None,
            "tokens_per_watt": None,
        }
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────
def main():
    results = {
        "model": MODEL, "timestamp": datetime.now().isoformat(),
        "reference_bw_gbs": 273, "reference_tflops_bf16": 67,
    }

    formats = [
        ("bf16",     run_bf16),
        ("int8",     run_int8),
        ("nf4",      run_nf4),
        ("awq",      run_awq),
        ("gguf_q4",  lambda: run_gguf("Q4_K_M")),
        ("gguf_q5",  lambda: run_gguf("Q5_K_M")),
        ("gguf_q8",  lambda: run_gguf("Q8_0")),
    ]

    for name, fn in formats:
        print(f"\n── {name.upper()} ──")
        try:
            r = fn()
            results[name] = r
            if "error" not in r:
                print(f"    TPS={r.get('tokens_per_sec')} ±{r.get('tps_stdev')}  "
                      f"TTFT={r.get('ttft_ms')}ms (p95={r.get('ttft_p95_ms')}ms)  "
                      f"MEM={r.get('peak_memory_gb')}GB  {r.get('power_w')}W  "
                      f"{r.get('tokens_per_watt')} tok/W")
            else:
                print(f"    Error: {r['error']}")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"    Exception: {e}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  {'Format':<10} {'TPS':>6} {'±stdev':>7} {'TTFT':>7} {'p95':>7} "
          f"{'Mem':>6} {'BW%':>5} {'W':>5} {'tok/W':>6}")
    print(f"  {'-'*70}")
    bf16_tps = results.get("bf16", {}).get("tokens_per_sec", 1)
    for name in ["bf16", "int8", "nf4", "awq", "gguf_q4", "gguf_q5", "gguf_q8"]:
        r = results.get(name, {})
        if "error" not in r and r:
            tps = r.get("tokens_per_sec", 0)
            ratio = f"{tps/bf16_tps:.1f}x" if bf16_tps else "?"
            print(f"  {name:<10} {tps:>6.0f} {str(r.get('tps_stdev','?')):>7}  "
                  f"{r.get('ttft_ms',0):>6.0f}ms "
                  f"{r.get('ttft_p95_ms',0):>6.0f}ms "
                  f"{r.get('peak_memory_gb',0) or 0:>5.1f}G "
                  f"{str(r.get('bw_util_pct','?')):>5}% "
                  f"{r.get('power_w',0) or 0:>5.0f} "
                  f"{str(r.get('tokens_per_watt','?')):>6}  ({ratio})")

    out = RESULTS / f"quant_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out}")

if __name__ == "__main__":
    main()
