#!/usr/bin/env python3
"""
Phase 1 — Continuous Batching Stress Test
Simulates concurrent multi-user load via vLLM
Metrics: p50/p95/p99 latency, throughput, GPU memory under sustained load
"""
import json, time, subprocess, argparse, threading, queue, statistics
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
RESULTS.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "Qwen/Qwen3-8B"

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe the architecture of a transformer model.",
    "How does gradient descent work in neural networks?",
    "What is the role of attention mechanisms in language models?",
    "Explain the concept of transfer learning.",
    "What are the advantages of using CUDA for deep learning?",
    "Describe the process of tokenization in NLP.",
    "How does beam search differ from greedy decoding?",
    "What is the purpose of layer normalization in transformers?",
]

PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096]


def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,"
             "memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        parts = [x.strip() for x in r.stdout.strip().split(",")]
        return {
            "power_w": float(parts[0]),
            "temp_c": int(parts[1]),
            "mem_used_mb": int(parts[2]),
            "util_pct": int(parts[3]),
        }
    except:
        return {}


# ─────────────────────────────────────────────────────────────
class RequestResult:
    def __init__(self, prompt_len, output_len, ttft_ms, total_ms, error=None):
        self.prompt_len  = prompt_len
        self.output_len  = output_len
        self.ttft_ms     = ttft_ms
        self.total_ms    = total_ms
        self.error       = error


def stress_test_vllm(model_id, concurrent_users, duration_sec=60,
                     max_output_tokens=128):
    print(f"  vLLM stress: {concurrent_users} concurrent users, {duration_sec}s")
    try:
        from vllm import LLM, SamplingParams
        import random

        llm = LLM(
            model=model_id,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_prefix_caching=True,
        )
        params = SamplingParams(max_tokens=max_output_tokens, temperature=0.7)

        results = []
        stop_event = threading.Event()
        result_queue = queue.Queue()

        def worker():
            while not stop_event.is_set():
                prompt = random.choice(PROMPTS)
                t0 = time.perf_counter()
                try:
                    outputs = llm.generate([prompt], params)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    n_out = len(outputs[0].outputs[0].token_ids)
                    result_queue.put(RequestResult(
                        prompt_len=len(prompt.split()),
                        output_len=n_out,
                        ttft_ms=None,
                        total_ms=round(elapsed_ms, 1),
                    ))
                except Exception as e:
                    result_queue.put(RequestResult(0, 0, None, 0, error=str(e)))

        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(concurrent_users)]

        gpu_snapshots = []
        t_start = time.perf_counter()
        for t in threads:
            t.start()

        while (time.perf_counter() - t_start) < duration_sec:
            gpu_snapshots.append(get_gpu_stats())
            time.sleep(2)

        stop_event.set()
        for t in threads:
            t.join(timeout=10)

        while not result_queue.empty():
            results.append(result_queue.get_nowait())

        total_elapsed = time.perf_counter() - t_start
        good = [r for r in results if r.error is None]
        errors = [r for r in results if r.error is not None]
        latencies = [r.total_ms for r in good]
        output_tokens = [r.output_len for r in good]

        stats = {
            "concurrent_users": concurrent_users,
            "duration_sec": round(total_elapsed, 1),
            "total_requests": len(results),
            "successful": len(good),
            "errors": len(errors),
            "throughput_req_per_sec": round(len(good) / total_elapsed, 2),
            "throughput_tok_per_sec": round(sum(output_tokens) / total_elapsed, 1),
        }
        if latencies:
            latencies.sort()
            stats.update({
                "latency_p50_ms":  round(statistics.median(latencies), 1),
                "latency_p95_ms":  round(latencies[int(0.95 * len(latencies))], 1),
                "latency_p99_ms":  round(latencies[int(0.99 * len(latencies))], 1),
                "latency_mean_ms": round(statistics.mean(latencies), 1),
            })
        if gpu_snapshots:
            stats.update({
                "mean_power_w":    round(sum(s.get("power_w", 0) for s in gpu_snapshots) / len(gpu_snapshots), 1),
                "mean_gpu_util":   round(sum(s.get("util_pct", 0) for s in gpu_snapshots) / len(gpu_snapshots), 1),
                "peak_mem_mb":     max(s.get("mem_used_mb", 0) for s in gpu_snapshots),
                "mean_temp_c":     round(sum(s.get("temp_c", 0) for s in gpu_snapshots) / len(gpu_snapshots), 1),
            })

        del llm
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()
        return stats

    except ImportError:
        return {"error": "vllm not installed", "concurrent_users": concurrent_users}
    except Exception as e:
        return {"error": str(e), "concurrent_users": concurrent_users}


def stress_test_hf(model_id, concurrent_users, duration_sec=30,
                   max_output_tokens=128):
    """Simple HF generate stress test (no PagedAttention — for comparison)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  HF stress: {concurrent_users} concurrent users, {duration_sec}s")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        m   = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )
        m.eval()
        import random

        results = []
        t_start = time.perf_counter()
        gpu_snapshots = []

        # Sequential (HF doesn't do true concurrent batching in this simple mode)
        while (time.perf_counter() - t_start) < duration_sec:
            prompt = random.choice(PROMPTS)
            inp = tok(prompt, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    out = m.generate(**inp, max_new_tokens=max_output_tokens,
                                     do_sample=False)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                n_out = out.shape[1] - inp["input_ids"].shape[1]
                results.append(RequestResult(len(prompt.split()), n_out,
                                             None, round(elapsed_ms, 1)))
            except Exception as e:
                results.append(RequestResult(0, 0, None, 0, error=str(e)))
            gpu_snapshots.append(get_gpu_stats())

        total_elapsed = time.perf_counter() - t_start
        good = [r for r in results if r.error is None]
        latencies = sorted([r.total_ms for r in good])
        output_tokens = [r.output_len for r in good]

        stats = {
            "backend": "hf_generate",
            "concurrent_users": 1,  # HF is sequential here
            "duration_sec": round(total_elapsed, 1),
            "total_requests": len(results),
            "throughput_tok_per_sec": round(sum(output_tokens) / total_elapsed, 1),
        }
        if latencies:
            stats.update({
                "latency_p50_ms":  round(statistics.median(latencies), 1),
                "latency_p95_ms":  round(latencies[int(0.95 * len(latencies))], 1),
                "latency_mean_ms": round(statistics.mean(latencies), 1),
            })

        del m
        torch.cuda.empty_cache()
        return stats

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrent", nargs="+", type=int,
                        default=[1, 10, 50, 100])
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration per concurrency level (seconds)")
    parser.add_argument("--backend", choices=["vllm", "hf", "both"], default="hf")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    all_results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend,
        "vllm_results": [],
        "hf_results": [],
    }

    if args.backend in ("vllm", "both"):
        print(f"\n{'='*60}")
        print(f"  vLLM Stress Test: {args.model}")
        for n in args.concurrent:
            print(f"\n  ── Concurrent users: {n} ──")
            result = stress_test_vllm(args.model, n, args.duration, args.max_tokens)
            result["backend"] = "vllm"
            all_results["vllm_results"].append(result)
            if "error" not in result:
                print(f"    Throughput: {result.get('throughput_tok_per_sec',0):.0f} tok/s  "
                      f"P50={result.get('latency_p50_ms','?')}ms  "
                      f"P95={result.get('latency_p95_ms','?')}ms  "
                      f"GPU={result.get('mean_gpu_util','?')}%")

    if args.backend in ("hf", "both"):
        print(f"\n{'='*60}")
        print(f"  HF Stress Test (sequential baseline): {args.model}")
        result = stress_test_hf(args.model, 1, args.duration, args.max_tokens)
        all_results["hf_results"].append(result)
        print(f"    TPS: {result.get('throughput_tok_per_sec','?')}  "
              f"P50={result.get('latency_p50_ms','?')}ms")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  STRESS TEST SUMMARY")
    print(f"  {'Users':>6} {'TPS':>8} {'P50ms':>8} {'P95ms':>8} {'GPU%':>6} {'Mem':>6}")
    print(f"  {'-'*70}")
    for r in all_results["vllm_results"]:
        if "error" not in r:
            print(f"  {r.get('concurrent_users',0):>6} "
                  f"{r.get('throughput_tok_per_sec',0):>8.0f} "
                  f"{r.get('latency_p50_ms','?'):>8} "
                  f"{r.get('latency_p95_ms','?'):>8} "
                  f"{r.get('mean_gpu_util','?'):>6} "
                  f"{r.get('peak_mem_mb',0)//1024:>5}G")

    out = RESULTS / f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
