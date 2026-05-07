#!/usr/bin/env python3
"""
SGLang Precision Sweep — GB10 Grace Blackwell
Tests: BF16, FP8, W8A8-INT8, AWQ-INT4 on Qwen3-8B and Qwen3-32B
Measures: tok/s, TTFT, memory, bandwidth utilization

Usage:
    python3 phase1_sglang_precision_sweep.py [--model 8b|32b|all] [--quick]
"""
import argparse, json, time, subprocess, sys, os, signal
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results" / "inference"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Benchmark prompts ──────────────────────────────────────────
PROMPT_128  = "Explain quantum entanglement in simple terms. " * 5
PROMPT_512  = ("Quantum computing is a fundamentally different paradigm from classical computing. "
               "Explain the key differences, the role of qubits, superposition, entanglement, "
               "and give three practical applications being developed today. ") * 3
PROMPT_2048 = PROMPT_512 * 4

OUTPUT_LEN = 256  # tokens to generate per request

# ── SGLang server manager ──────────────────────────────────────
def start_server(model_id: str, port: int, dtype: str = "bfloat16",
                 quantization: str = None, kv_cache_dtype: str = None,
                 mem_fraction: float = 0.88, extra_args: list = None) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_id,
        "--port", str(port),
        "--dtype", dtype,
        "--mem-fraction-static", str(mem_fraction),
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if kv_cache_dtype:
        cmd += ["--kv-cache-dtype", kv_cache_dtype]
    if extra_args:
        cmd += extra_args
    print(f"  Starting SGLang: {' '.join(cmd[-6:])}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    return proc


def wait_ready(port: int, timeout: int = 180) -> bool:
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(3)
    return False


def stop_server(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(3)


# ── Single benchmark call ──────────────────────────────────────
def bench_one(port: int, prompt: str, n_requests: int = 10) -> dict:
    import urllib.request, urllib.error
    import json as _json

    url = f"http://localhost:{port}/generate"
    payload = json.dumps({
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": OUTPUT_LEN,
            "temperature": 0.0,
        }
    }).encode()
    headers = {"Content-Type": "application/json"}

    # Warmup
    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        urllib.request.urlopen(req, timeout=120)
    except Exception as e:
        return {"error": str(e)}

    # Timed runs
    latencies = []
    ttfts = []
    for _ in range(n_requests):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120)
            data = _json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        # SGLang response may include meta_info with TTFT
        meta = data.get("meta_info", {})
        if "ttft_s" in meta:
            ttfts.append(meta["ttft_s"])
        elif "time_to_first_token" in meta:
            ttfts.append(meta["time_to_first_token"])

    avg_latency = sum(latencies) / len(latencies)
    tok_per_sec = OUTPUT_LEN / avg_latency
    result = {
        "avg_latency_s": round(avg_latency, 3),
        "tok_per_sec": round(tok_per_sec, 2),
        "n_requests": n_requests,
    }
    if ttfts:
        result["ttft_ms"] = round(sum(ttfts) / len(ttfts) * 1000, 1)
    return result


def get_mem_gb() -> float:
    """Read GPU memory in GB from /proc or free."""
    try:
        out = subprocess.check_output(
            ["python3", "-c",
             "import torch; print(torch.cuda.memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9)"],
            timeout=10, text=True
        ).strip()
        parts = out.split()
        return round(float(parts[1]), 1)  # reserved (includes model weights)
    except Exception:
        return -1.0


# ── Configs per model ──────────────────────────────────────────
CONFIGS = {
    "8b": {
        "model_id": "Qwen/Qwen3-8B",
        "param_b": 8.0,
        "port": 30000,
        "mem_fraction": 0.70,
        "precision_configs": [
            {
                "name": "BF16",
                "dtype": "bfloat16",
                "quantization": None,
                "kv_cache_dtype": None,
                "bytes_per_param": 2.0,
            },
            {
                "name": "FP8 (W8A8)",
                "dtype": "bfloat16",
                "quantization": "w8a8_fp8",
                "kv_cache_dtype": None,
                "bytes_per_param": 1.0,
            },
            {
                "name": "INT8 (W8A8)",
                "dtype": "bfloat16",
                "quantization": "w8a8_int8",
                "kv_cache_dtype": None,
                "bytes_per_param": 1.0,
            },
            {
                "name": "FP8 + FP8-KV",
                "dtype": "bfloat16",
                "quantization": "w8a8_fp8",
                "kv_cache_dtype": "fp8_e5m2",
                "bytes_per_param": 1.0,
            },
            {
                "name": "AWQ-INT4",
                "dtype": "float16",
                "quantization": "awq",
                "kv_cache_dtype": None,
                "bytes_per_param": 0.5,
                "model_id_override": str(BASE / "models" / "qwen3-8b-awq"),
            },
        ],
    },
    "32b": {
        "model_id": "Qwen/Qwen3-32B",
        "param_b": 32.0,
        "port": 30001,
        "mem_fraction": 0.85,
        "precision_configs": [
            {
                "name": "BF16",
                "dtype": "bfloat16",
                "quantization": None,
                "kv_cache_dtype": None,
                "bytes_per_param": 2.0,
            },
            {
                "name": "FP8 (W8A8)",
                "dtype": "bfloat16",
                "quantization": "w8a8_fp8",
                "kv_cache_dtype": None,
                "bytes_per_param": 1.0,
            },
            {
                "name": "INT8 (W8A8)",
                "dtype": "bfloat16",
                "quantization": "w8a8_int8",
                "kv_cache_dtype": None,
                "bytes_per_param": 1.0,
            },
        ],
    },
}


# ── Prompt configs ─────────────────────────────────────────────
PROMPT_CONFIGS = [
    ("128t_in",  PROMPT_128,  10),
    ("512t_in",  PROMPT_512,  8),
    ("2048t_in", PROMPT_2048, 5),
]


# ── Main sweep ────────────────────────────────────────────────
def run_model_sweep(model_key: str, args):
    cfg = CONFIGS[model_key]
    model_id = cfg["model_id"]
    param_b  = cfg["param_b"]
    port     = cfg["port"]
    results  = []

    print(f"\n{'='*60}")
    print(f"Model: {model_id} ({param_b}B params)")
    print(f"{'='*60}")

    for pc in cfg["precision_configs"]:
        name = pc["name"]
        mid  = pc.get("model_id_override", model_id)
        bytes_pp = pc["bytes_per_param"]
        weight_gb = param_b * 1e9 * bytes_pp / 1e9

        print(f"\n  [{name}] model={mid.split('/')[-1]} est_weights={weight_gb:.1f}GB")

        proc = start_server(
            model_id=mid, port=port,
            dtype=pc["dtype"],
            quantization=pc.get("quantization"),
            kv_cache_dtype=pc.get("kv_cache_dtype"),
            mem_fraction=cfg["mem_fraction"],
        )

        if not wait_ready(port, timeout=300):
            print(f"    FAILED to start server for {name}")
            stop_server(proc)
            # Log stderr for debugging
            if proc.stdout:
                lines = []
                proc.stdout.flush()
            results.append({
                "model": model_id, "precision": name,
                "status": "server_failed",
            })
            continue

        print(f"    Server ready.")
        row = {
            "model": model_id,
            "precision": name,
            "quantization": pc.get("quantization"),
            "kv_cache_dtype": pc.get("kv_cache_dtype"),
            "est_weight_gb": round(weight_gb, 1),
            "status": "ok",
        }

        for pname, prompt, n_req in PROMPT_CONFIGS:
            if args.quick:
                n_req = min(n_req, 3)
            r = bench_one(port, prompt, n_req)
            if "error" in r:
                print(f"    {pname}: ERROR — {r['error']}")
                row[pname] = r
            else:
                bw_util = (param_b * 1e9 * bytes_pp) / 273e9 * r["tok_per_sec"] * 100
                r["bw_util_pct"] = round(bw_util, 1)
                ttft_str = f" TTFT={r['ttft_ms']}ms" if "ttft_ms" in r else ""
                print(f"    {pname}: {r['tok_per_sec']:.1f} tok/s"
                      f"{ttft_str} BW={r['bw_util_pct']}% lat={r['avg_latency_s']:.2f}s")
                row[pname] = r

        results.append(row)
        stop_server(proc)
        time.sleep(5)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["8b", "32b", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Fewer requests per config (faster)")
    args = parser.parse_args()

    models = ["8b", "32b"] if args.model == "all" else [args.model]

    all_results = {}
    for mkey in models:
        rows = run_model_sweep(mkey, args)
        all_results[mkey] = rows

    # Save results
    out_path = RESULTS / f"sglang_precision_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'Model':<20} {'Precision':<18} {'Est GB':>7} {'128t tok/s':>11} {'512t tok/s':>11} {'2048t tok/s':>12} {'BW%':>5}")
    print("-" * 90)
    for mkey, rows in all_results.items():
        for row in rows:
            if row.get("status") != "ok":
                print(f"  {row['model'].split('/')[-1]:<18} {row['precision']:<18} FAILED")
                continue
            def gs(pname):
                d = row.get(pname, {})
                return f"{d.get('tok_per_sec','—'):>6.1f}" if isinstance(d, dict) and 'tok_per_sec' in d else "    —"
            def bw(pname):
                d = row.get(pname, {})
                return d.get('bw_util_pct', 0) if isinstance(d, dict) else 0
            best_bw = max(bw(p[0]) for p in PROMPT_CONFIGS)
            print(f"  {row['model'].split('/')[-1]:<18} {row['precision']:<18} {row['est_weight_gb']:>6.1f}GB"
                  f" {gs('128t_in')} {gs('512t_in')} {gs('2048t_in')} {best_bw:>4.0f}%")


if __name__ == "__main__":
    main()
