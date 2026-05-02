#!/usr/bin/env python3
"""
Phase 1 — Speculative Decoding Benchmark
Draft (small) + Target (large) model pairs via vLLM or HF-native speculative
Measures: accepted token rate, effective TPS vs baseline, memory overhead
"""
import json, time, subprocess, argparse, statistics
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
RESULTS.mkdir(parents=True, exist_ok=True)

SPECULATIVE_PAIRS = [
    {"draft": "Qwen/Qwen3-0.6B", "target": "Qwen/Qwen3-8B",  "label": "0.6B→8B"},
    {"draft": "Qwen/Qwen3-1.7B", "target": "Qwen/Qwen3-32B", "label": "1.7B→32B"},
    {"draft": "Qwen/Qwen3-4B",   "target": "Qwen/Qwen3-32B", "label": "4B→32B"},
]

PROMPTS = [
    "Explain the difference between supervised and unsupervised machine learning.",
    "What are the key advantages of transformer architecture over RNNs?",
    "Describe the process of gradient descent in neural network training.",
]

MAX_NEW = 200
N_RUNS  = 5


def percentile(data, p):
    data = sorted(data)
    return data[min(int(p / 100 * len(data)), len(data) - 1)]


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


# ─────────────────────────────────────────────────────────────
def bench_baseline_hf(model_id, prompt):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    m   = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    m.eval()

    inp = tok(prompt, return_tensors="pt").to("cuda")

    # warmup
    with torch.no_grad():
        m.generate(**inp, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()

    # Collect N_RUNS TTFT measurements
    ttft_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            m.generate(**inp, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        ttft_times.append((time.perf_counter() - t0) * 1000)

    # Collect N_RUNS generation time measurements
    gpu_stats_start = get_gpu_stats()
    gen_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
        gen_times.append(time.perf_counter() - t0)
    gpu_stats_end = get_gpu_stats()

    n_new = out.shape[1] - inp["input_ids"].shape[1]

    tps_list  = [n_new / t for t in gen_times]
    tps_mean  = statistics.mean(tps_list)
    tps_stdev = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
    ttft_mean = statistics.mean(ttft_times)
    ttft_p95  = percentile(ttft_times, 95)

    mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    power = ((gpu_stats_start.get("power_w", 0) + gpu_stats_end.get("power_w", 0)) / 2)

    del m
    torch.cuda.empty_cache()
    return {
        "tokens_per_sec":  round(tps_mean, 1),
        "tps_stdev":       round(tps_stdev, 2),
        "ttft_ms":         round(ttft_mean, 1),
        "ttft_p95_ms":     round(ttft_p95, 1),
        "peak_memory_gb":  round(mem, 2),
        "power_w":         round(power, 1),
        "temp_c":          gpu_stats_end.get("temp_c"),
        "tokens_per_watt": round(tps_mean / power, 2) if power > 0 else None,
    }


def bench_speculative_hf(draft_id, target_id, prompt, n_speculative=5):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"    Loading target {target_id}...")
    tok    = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_id, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    target.eval()

    print(f"    Loading draft {draft_id}...")
    draft = AutoModelForCausalLM.from_pretrained(
        draft_id, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    draft.eval()

    inp = tok(prompt, return_tensors="pt").to("cuda")
    mem_both = torch.cuda.memory_allocated() / 1e9

    # HF native assisted generation (speculative) — warmup
    with torch.no_grad():
        target.generate(**inp, max_new_tokens=32, do_sample=False,
                        assistant_model=draft)
    torch.cuda.synchronize()

    # Collect N_RUNS TTFT measurements
    ttft_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            target.generate(**inp, max_new_tokens=1, do_sample=False,
                            assistant_model=draft)
        torch.cuda.synchronize()
        ttft_times.append((time.perf_counter() - t0) * 1000)

    # Collect N_RUNS generation time measurements
    gpu_stats_start = get_gpu_stats()
    gen_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                                  assistant_model=draft)
        torch.cuda.synchronize()
        gen_times.append(time.perf_counter() - t0)
    gpu_stats_end = get_gpu_stats()

    n_new = out.shape[1] - inp["input_ids"].shape[1]

    tps_list  = [n_new / t for t in gen_times]
    tps_mean  = statistics.mean(tps_list)
    tps_stdev = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
    ttft_mean = statistics.mean(ttft_times)
    ttft_p95  = percentile(ttft_times, 95)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    power = ((gpu_stats_start.get("power_w", 0) + gpu_stats_end.get("power_w", 0)) / 2)

    del target, draft
    torch.cuda.empty_cache()
    return {
        "tokens_per_sec":  round(tps_mean, 1),
        "tps_stdev":       round(tps_stdev, 2),
        "ttft_ms":         round(ttft_mean, 1),
        "ttft_p95_ms":     round(ttft_p95, 1),
        "peak_memory_gb":  round(peak_mem, 2),
        "power_w":         round(power, 1),
        "temp_c":          gpu_stats_end.get("temp_c"),
        "tokens_per_watt": round(tps_mean / power, 2) if power > 0 else None,
        "both_models_gb":  round(mem_both, 2),
    }


def bench_speculative_vllm(draft_id, target_id, prompt, label):
    print(f"    Trying vLLM speculative for {label}...")
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=target_id,
            speculative_model=draft_id,
            num_speculative_tokens=5,
            use_v2_block_manager=True,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            dtype="bfloat16",
        )

        params = SamplingParams(max_tokens=MAX_NEW, temperature=0.0)

        # warmup
        llm.generate([prompt], params)

        # Collect N_RUNS generation time measurements
        gen_times = []
        n_new_last = 0
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], params)
            gen_times.append(time.perf_counter() - t0)
            n_new_last = len(outputs[0].outputs[0].token_ids)

        tps_list  = [n_new_last / t for t in gen_times]
        tps_mean  = statistics.mean(tps_list)
        tps_stdev = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0

        # accepted token rate from vLLM metrics if available
        accepted_rate = None
        try:
            metrics = llm.get_tokenizer()  # placeholder
        except:
            pass

        del llm
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()

        return {
            "tokens_per_sec":       round(tps_mean, 1),
            "tps_stdev":            round(tps_stdev, 2),
            "accepted_token_rate":  accepted_rate,
            "backend":              "vllm",
        }
    except Exception as e:
        return {"error": str(e), "backend": "vllm"}


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+",
                        choices=["0.6B-8B", "1.7B-32B", "4B-32B", "all"],
                        default=["all"])
    parser.add_argument("--backend", choices=["hf", "vllm", "both"], default="hf")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run baseline (target model alone) for speedup ratio")
    args = parser.parse_args()

    pair_labels = {
        "0.6B-8B":  SPECULATIVE_PAIRS[0],
        "1.7B-32B": SPECULATIVE_PAIRS[1],
        "4B-32B":   SPECULATIVE_PAIRS[2],
    }
    selected = (SPECULATIVE_PAIRS if "all" in args.pairs
                else [pair_labels[p] for p in args.pairs])

    all_results = []
    prompt = PROMPTS[0]

    for pair in selected:
        label      = pair["label"]
        draft_id   = pair["draft"]
        target_id  = pair["target"]
        print(f"\n── {label} ──")
        result = {"label": label, "draft": draft_id, "target": target_id}

        # Baseline (target only)
        if args.baseline:
            print(f"  Baseline (target only)...")
            try:
                base = bench_baseline_hf(target_id, prompt)
                result["baseline_tps"]      = base["tokens_per_sec"]
                result["baseline_tps_stdev"] = base["tps_stdev"]
                result["baseline_ttft_ms"]  = base["ttft_ms"]
                result["baseline_ttft_p95_ms"] = base["ttft_p95_ms"]
                result["baseline_mem_gb"]   = base["peak_memory_gb"]
                result["baseline_power_w"]  = base["power_w"]
                print(f"    Baseline TPS: {base['tokens_per_sec']} ±{base['tps_stdev']}  "
                      f"TTFT={base['ttft_ms']}ms (p95={base['ttft_p95_ms']}ms)")
            except torch.cuda.OutOfMemoryError as e:
                result["baseline_error"] = f"OOM: {e}"
            except Exception as e:
                result["baseline_error"] = str(e)

        # Speculative — HF
        if args.backend in ("hf", "both"):
            print(f"  HF Speculative ({label})...")
            try:
                spec = bench_speculative_hf(draft_id, target_id, prompt)
                result["speculative_hf"] = spec
                baseline_tps       = result.get("baseline_tps", None)
                baseline_tps_stdev = result.get("baseline_tps_stdev", 0.0)
                spec_tps    = spec["tokens_per_sec"]
                spec_stdev  = spec["tps_stdev"]

                if baseline_tps and baseline_tps > 0:
                    speedup_ratio = spec_tps / baseline_tps
                    # propagate relative uncertainty: sqrt((sa/a)^2 + (sb/b)^2) * ratio
                    rel_unc = (
                        (spec_stdev / spec_tps) ** 2 +
                        (baseline_tps_stdev / baseline_tps) ** 2
                    ) ** 0.5 if spec_tps > 0 and baseline_tps > 0 else 0.0
                    speedup_stdev = rel_unc * speedup_ratio
                else:
                    speedup_ratio = None
                    speedup_stdev = None

                result["speedup_ratio"]  = round(speedup_ratio, 3) if speedup_ratio is not None else None
                result["speedup_stdev"]  = round(speedup_stdev, 4) if speedup_stdev is not None else None
                result["speedup_hf"]     = result["speedup_ratio"]  # backward compat alias
                print(f"    Spec TPS: {spec['tokens_per_sec']} ±{spec['tps_stdev']}  "
                      f"TTFT={spec['ttft_ms']}ms (p95={spec['ttft_p95_ms']}ms)  "
                      f"Mem: {spec['peak_memory_gb']:.1f}GB  "
                      f"Speedup: {speedup_ratio:.3f}x ±{speedup_stdev:.4f}" if speedup_ratio else
                      f"    Spec TPS: {spec['tokens_per_sec']} ±{spec['tps_stdev']}  "
                      f"TTFT={spec['ttft_ms']}ms  Mem: {spec['peak_memory_gb']:.1f}GB")
            except Exception as e:
                result["speculative_hf_error"] = str(e)
                print(f"    Error: {e}")

        # Speculative — vLLM
        if args.backend in ("vllm", "both"):
            print(f"  vLLM Speculative ({label})...")
            vllm_result = bench_speculative_vllm(draft_id, target_id, prompt, label)
            result["speculative_vllm"] = vllm_result
            if "error" not in vllm_result:
                baseline_tps = result.get("baseline_tps", None)
                spec_tps     = vllm_result["tokens_per_sec"]
                if baseline_tps and baseline_tps > 0:
                    speedup = spec_tps / baseline_tps
                else:
                    speedup = None
                result["speedup_vllm"] = round(speedup, 3) if speedup is not None else None
                print(f"    Spec TPS: {vllm_result['tokens_per_sec']} ±{vllm_result['tps_stdev']}  "
                      f"Speedup: {speedup:.3f}x" if speedup else
                      f"    Spec TPS: {vllm_result['tokens_per_sec']}")

        all_results.append(result)

    print(f"\n{'='*70}")
    print(f"  {'Pair':<15} {'Base TPS':>9} {'±':>5} {'Spec TPS':>9} {'±':>5} "
          f"{'Speedup':>8} {'±':>6} {'Mem':>6}")
    print(f"  {'-'*70}")
    for r in all_results:
        base        = r.get("baseline_tps", "?")
        base_std    = r.get("baseline_tps_stdev", "?")
        spec        = r.get("speculative_hf", {}).get("tokens_per_sec", "?")
        spec_std    = r.get("speculative_hf", {}).get("tps_stdev", "?")
        speedup     = r.get("speedup_ratio", "?")
        speedup_std = r.get("speedup_stdev", "?")
        mem         = r.get("speculative_hf", {}).get("peak_memory_gb", "?")
        print(f"  {r['label']:<15} {str(base):>9} {str(base_std):>5} "
              f"{str(spec):>9} {str(spec_std):>5} "
              f"{str(speedup):>8} {str(speedup_std):>6} {str(mem):>6}")

    out = RESULTS / f"speculative_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
