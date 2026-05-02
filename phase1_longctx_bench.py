#!/usr/bin/env python3
"""
Phase 1 — Long-Context Benchmark
Tests: needle-in-haystack, multi-hop retrieval, LongBench tasks
Context lengths: 4K, 8K, 16K, 32K, 64K, 128K tokens
"""
import json, time, subprocess, argparse, random, re, statistics
from pathlib import Path
from datetime import datetime

N_RUNS = 3

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen3-8b":  "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
}

CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

NEEDLE = "The secret number is 42391."
NEEDLE_QUESTION = "What is the secret number mentioned in the document?"
NEEDLE_ANSWER   = "42391"

HAYSTACK_SENTENCE = (
    "The advancement of artificial intelligence requires careful consideration "
    "of computational resources, training methodologies, and evaluation frameworks. "
)


def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        p, t, m = r.stdout.strip().split(",")
        return {"power_w": float(p.strip()), "temp_c": int(t.strip()),
                "mem_mb": int(m.strip())}
    except:
        return {}


def build_needle_haystack(target_tokens, tokenizer, needle_depth_pct=50):
    hay = HAYSTACK_SENTENCE
    long_hay = (hay * (target_tokens // len(hay.split()) + 100))
    tokens = tokenizer.encode(long_hay)
    half = len(tokens) // 2
    needle_pos = int(len(tokens) * needle_depth_pct / 100)

    needle_tokens = tokenizer.encode(NEEDLE, add_special_tokens=False)
    full_tokens = tokens[:needle_pos] + needle_tokens + tokens[needle_pos:]
    full_tokens = full_tokens[:target_tokens]
    text = tokenizer.decode(full_tokens)
    return text


def extract_number(text):
    numbers = re.findall(r'\b\d{5}\b', text)
    return numbers[0] if numbers else None


# ─────────────────────────────────────────────────────────────
def bench_needle_hf(model_id, model_key, ctx_len, depth_pct=50):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        m   = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )
        m.eval()
        mem_load = torch.cuda.memory_allocated() / 1e9

        context = build_needle_haystack(ctx_len - 100, tok, depth_pct)
        prompt = (f"{context}\n\nBased on the document above, "
                  f"{NEEDLE_QUESTION} Answer with just the number.")

        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=ctx_len).to("cuda")
        actual_ctx = inputs["input_ids"].shape[1]

        # TTFT — run N_RUNS times, collect list
        ttft_list = []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                m.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            ttft_list.append((time.perf_counter() - t0) * 1000)

        ttft_ms     = statistics.mean(ttft_list)
        ttft_p95_ms = sorted(ttft_list)[int(0.95 * len(ttft_list))]

        # Full generation (for correctness + tps_decode)
        gpu_before = get_gpu_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=20, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        gpu_after = get_gpu_stats()

        tps_decode = 20 / elapsed if elapsed > 0 else None

        generated = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
        found = extract_number(generated)
        correct = found == NEEDLE_ANSWER

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        result = {
            "model": model_id, "model_key": model_key,
            "target_ctx_len": ctx_len,
            "actual_ctx_len": actual_ctx,
            "needle_depth_pct": depth_pct,
            "ttft_ms":     round(ttft_ms, 1),
            "ttft_p95_ms": round(ttft_p95_ms, 1),
            "tps_decode":  round(tps_decode, 2) if tps_decode is not None else None,
            "generation_sec": round(elapsed, 3),
            "correct": correct,
            "generated": generated[:100],
            "found_number": found,
            "peak_memory_gb": round(peak_mem, 2),
            "model_load_gb":  round(mem_load, 2),
            "power_w":  gpu_after.get("power_w"),
            "temp_c":   gpu_after.get("temp_c"),
        }
        status = "✓" if correct else "✗"
        print(f"    ctx={actual_ctx:>7}  {status}  TTFT={ttft_ms:.0f}ms(p95={ttft_p95_ms:.0f}ms)  "
              f"TPS={tps_decode:.1f}  Mem={peak_mem:.1f}GB  [{generated[:30]}...]")

        del m
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError:
        print(f"    ctx={ctx_len}: OOM")
        torch.cuda.empty_cache()
        return {"model": model_id, "model_key": model_key,
                "target_ctx_len": ctx_len, "error": "OOM"}
    except Exception as e:
        print(f"    ctx={ctx_len}: Error: {e}")
        return {"model": model_id, "model_key": model_key,
                "target_ctx_len": ctx_len, "error": str(e)}


def bench_kv_cache_scaling(model_id, model_key, ctx_lengths=CONTEXT_LENGTHS):
    """Measure memory and TTFT scaling with context length."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n  KV Cache Scaling for {model_id}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        m   = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )
        m.eval()
        mem_base = torch.cuda.memory_allocated() / 1e9
        rows = []

        for ctx_len in ctx_lengths:
            try:
                # Generate a context of roughly ctx_len tokens
                filler = HAYSTACK_SENTENCE * (ctx_len // 10)
                inputs = tok(filler, return_tensors="pt", truncation=True,
                             max_length=ctx_len).to("cuda")
                actual = inputs["input_ids"].shape[1]

                torch.cuda.reset_peak_memory_stats()
                ttft_list = []
                for _ in range(N_RUNS):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        m.generate(**inputs, max_new_tokens=1, do_sample=False)
                    torch.cuda.synchronize()
                    ttft_list.append((time.perf_counter() - t0) * 1000)

                ttft = statistics.mean(ttft_list)
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                kv_mem = peak_mem - mem_base

                # tps_decode: generate 20 new tokens, measure tokens/sec
                t0 = time.perf_counter()
                with torch.no_grad():
                    m.generate(**inputs, max_new_tokens=20, do_sample=False)
                torch.cuda.synchronize()
                decode_elapsed = time.perf_counter() - t0
                tps_decode = 20 / decode_elapsed if decode_elapsed > 0 else None

                row = {
                    "target_ctx": ctx_len, "actual_ctx": actual,
                    "ttft_ms":    round(ttft, 1),
                    "tps_decode": round(tps_decode, 2) if tps_decode is not None else None,
                    "total_mem_gb": round(peak_mem, 2),
                    "kv_cache_gb":  round(kv_mem, 2),
                }
                rows.append(row)
                print(f"    ctx={actual:>7}  TTFT={ttft:.0f}ms  TPS={tps_decode:.1f}  "
                      f"TotalMem={peak_mem:.1f}GB  KV={kv_mem:.1f}GB")

            except torch.cuda.OutOfMemoryError:
                print(f"    ctx={ctx_len}: OOM")
                rows.append({"target_ctx": ctx_len, "error": "OOM"})
                torch.cuda.empty_cache()
                break

        del m
        torch.cuda.empty_cache()
        return {"model": model_id, "model_key": model_key,
                "kv_scaling": rows, "base_model_gb": round(mem_base, 2)}

    except Exception as e:
        return {"model": model_id, "model_key": model_key, "error": str(e)}


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=list(MODELS.keys()) + ["all"], default=["all"])
    parser.add_argument("--ctx-lengths", nargs="+", type=int,
                        default=[4096, 8192, 16384, 32768, 65536, 131072])
    parser.add_argument("--depths", nargs="+", type=int, default=[10, 50, 90],
                        help="Needle depth percentages")
    parser.add_argument("--kv-scaling", action="store_true",
                        help="Run KV cache memory scaling test")
    args = parser.parse_args()

    model_keys = list(MODELS.keys()) if "all" in args.models else args.models
    all_results = {"needle": [], "kv_scaling": []}

    for model_key in model_keys:
        model_id = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"  Model: {model_id}")

        # Needle-in-haystack at multiple context lengths and depths
        print(f"\n  Needle-in-Haystack Test:")
        for ctx_len in args.ctx_lengths:
            for depth in args.depths:
                result = bench_needle_hf(model_id, model_key, ctx_len, depth)
                result["needle_depth_pct"] = depth
                all_results["needle"].append(result)
                if "OOM" in result.get("error", ""):
                    print(f"  OOM at ctx={ctx_len} — skipping larger contexts")
                    break

        # KV cache scaling
        if args.kv_scaling:
            scaling = bench_kv_cache_scaling(model_id, model_key, args.ctx_lengths)
            all_results["kv_scaling"].append(scaling)

    # Summary: needle accuracy by context length
    print(f"\n{'='*70}")
    print("  NEEDLE ACCURACY BY CONTEXT LENGTH")
    print(f"  {'Model':<15} {'Ctx':>8} {'Depth':>6} {'Correct':>8} {'TTFT':>8} {'Mem':>6}")
    print(f"  {'-'*70}")
    for r in all_results["needle"]:
        if "error" not in r:
            status = "YES" if r.get("correct") else "NO"
            print(f"  {r.get('model_key',''):<15} "
                  f"{r.get('actual_ctx_len',0):>8} "
                  f"{r.get('needle_depth_pct',0):>5}% "
                  f"{status:>8} "
                  f"{r.get('ttft_ms',0):>7.0f}ms "
                  f"{r.get('peak_memory_gb',0):>5.1f}G")

    out = RESULTS / f"longctx_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
