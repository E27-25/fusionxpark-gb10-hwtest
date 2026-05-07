#!/usr/bin/env python3
"""
Precision Sweep using HuggingFace generate() + bitsandbytes quantization
GB10: BF16, INT8 (W8A8), NF4 (QLoRA-style 4-bit), SGLang (if available)
All numbers for Qwen3-8B and Qwen3-32B at batch=1 and batch=8

Usage:
    python3 phase1_precision_sweep_hf.py [--model 8b|32b|all] [--quick]
"""
import argparse, json, time, gc
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results" / "inference"
RESULTS.mkdir(parents=True, exist_ok=True)

PROMPT_SHORT  = "Explain quantum entanglement in simple terms."
PROMPT_MEDIUM = ("Quantum computing is a fundamentally different paradigm from classical computing. "
                 "Explain the key differences, the role of qubits, superposition, and entanglement. "
                 "Give three practical applications being developed today.")
PROMPT_LONG   = PROMPT_MEDIUM * 4

OUTPUT_TOKENS = 200

# ── Benchmark one configuration ───────────────────────────────
def bench_config(model, tokenizer, prompt: str, batch: int, n_runs: int = 5) -> dict:
    inputs = tokenizer([prompt] * batch, return_tensors="pt", padding=True).to("cuda")
    in_len = inputs.input_ids.shape[1]

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32, do_sample=False)
    torch.cuda.reset_peak_memory_stats()

    # Timed
    times = []
    out_tokens = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=OUTPUT_TOKENS, do_sample=False)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        new_tok = (out.shape[1] - in_len) * batch
        out_tokens.append(new_tok)

    peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    avg_time = sum(times) / len(times)
    avg_tokens = sum(out_tokens) / len(out_tokens)
    tok_per_sec = avg_tokens / avg_time
    ttft_ms = times[0] * 1000  # TTFT estimate from first run

    return {
        "batch": batch,
        "in_tokens": in_len,
        "out_tokens": int(avg_tokens / batch),
        "tok_per_sec": round(tok_per_sec, 2),
        "ttft_ms": round(ttft_ms, 1),
        "avg_latency_s": round(avg_time, 3),
        "peak_mem_gb": round(peak, 2),
    }


# ── Load model in different precision modes ────────────────────
def load_bf16(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    mem = torch.cuda.memory_allocated() / 1e9
    return model, round(mem, 2)


def load_int8(model_id):
    cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg,
        device_map={"": 0}, trust_remote_code=True,
    )
    mem = torch.cuda.memory_allocated() / 1e9
    return model, round(mem, 2)


def load_nf4(model_id):
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg,
        device_map={"": 0}, trust_remote_code=True,
    )
    mem = torch.cuda.memory_allocated() / 1e9
    return model, round(mem, 2)


def load_awq(model_id_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True,
    )
    mem = torch.cuda.memory_allocated() / 1e9
    return model, round(mem, 2)


# ── BW utilization calc ────────────────────────────────────────
def bw_util(param_b, bytes_pp, tok_per_sec):
    bw_used = param_b * 1e9 * bytes_pp * tok_per_sec / 1e9
    return round(bw_used / 134 * 100, 1)  # 134 GB/s measured


# ── Main sweep ─────────────────────────────────────────────────
CONFIGS = {
    "8b": {
        "model_id": "Qwen/Qwen3-8B",
        "param_b": 8.0,
        "precisions": [
            ("BF16",     load_bf16,  2.0, "Qwen/Qwen3-8B"),
            ("INT8",     load_int8,  1.0, "Qwen/Qwen3-8B"),
            ("NF4",      load_nf4,   0.5, "Qwen/Qwen3-8B"),
            ("AWQ-INT4", load_awq,   0.5, str(BASE / "models" / "qwen3-8b-awq")),
        ],
    },
    "32b": {
        "model_id": "Qwen/Qwen3-32B",
        "param_b": 32.0,
        "precisions": [
            ("BF16",   load_bf16,  2.0, "Qwen/Qwen3-32B"),
            ("INT8",   load_int8,  1.0, "Qwen/Qwen3-32B"),
            ("NF4",    load_nf4,   0.5, "Qwen/Qwen3-32B"),
        ],
    },
}


def run_sweep(model_key, args):
    cfg = CONFIGS[model_key]
    param_b = cfg["param_b"]
    all_results = []

    print(f"\n{'='*60}")
    print(f"Model: {cfg['model_id']} ({param_b}B params)")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batches = [1] if args.quick else [1, 8]
    prompts = [("short", PROMPT_SHORT)] if args.quick else [
        ("short", PROMPT_SHORT),
        ("medium", PROMPT_MEDIUM),
    ]
    n_runs = 3 if args.quick else 5

    for prec_name, load_fn, bytes_pp, model_path in cfg["precisions"]:
        print(f"\n  [{prec_name}]")
        model, load_mem = None, 0.0

        try:
            model, load_mem = load_fn(model_path)
            est_weight = param_b * bytes_pp
            print(f"    Loaded. mem={load_mem:.1f}GB (est weights={est_weight:.0f}GB)")
        except Exception as e:
            print(f"    FAILED to load: {e}")
            all_results.append({
                "model": cfg["model_id"], "precision": prec_name,
                "status": "load_failed", "error": str(e)[:200],
            })
            continue

        row = {
            "model": cfg["model_id"],
            "precision": prec_name,
            "bytes_per_param": bytes_pp,
            "est_weight_gb": round(param_b * bytes_pp, 1),
            "load_mem_gb": load_mem,
            "status": "ok",
        }

        for pname, prompt in prompts:
            for batch in batches:
                try:
                    r = bench_config(model, tokenizer, prompt, batch, n_runs)
                    bw = bw_util(param_b, bytes_pp, r["tok_per_sec"])
                    r["bw_util_pct"] = bw
                    key = f"b{batch}_{pname}"
                    row[key] = r
                    ttft = f" TTFT={r['ttft_ms']:.0f}ms" if batch == 1 else ""
                    print(f"    {pname} b={batch}: {r['tok_per_sec']:.1f} tok/s"
                          f"{ttft} BW={bw}% mem={r['peak_mem_gb']:.1f}GB")
                except Exception as e:
                    print(f"    {pname} b={batch}: ERROR — {e}")
                    row[f"b{batch}_{pname}"] = {"error": str(e)[:100]}

        all_results.append(row)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["8b", "32b", "all"], default="all")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    models = ["8b", "32b"] if args.model == "all" else [args.model]
    all_results = {}
    for mk in models:
        all_results[mk] = run_sweep(mk, args)

    out = RESULTS / f"precision_sweep_hf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out}")

    # Summary
    print(f"\n{'Model':<18} {'Precision':<12} {'Est GB':>7} {'B=1 tok/s':>10} {'B=8 tok/s':>10} {'BW%':>5}")
    print("-" * 70)
    for mk, rows in all_results.items():
        for row in rows:
            if row.get("status") != "ok":
                print(f"  {row['model'].split('/')[-1]:<16} {row['precision']:<12} FAILED")
                continue
            def gs(k):
                d = row.get(k, {})
                return f"{d.get('tok_per_sec', '—'):>6.1f}" if isinstance(d, dict) and 'tok_per_sec' in d else "     —"
            def bw(k):
                d = row.get(k, {})
                return d.get('bw_util_pct', 0) if isinstance(d, dict) else 0
            best_bw = max(bw("b1_short"), bw("b8_short"), bw("b1_medium"), bw("b8_medium"))
            print(f"  {row['model'].split('/')[-1]:<16} {row['precision']:<12} {row['est_weight_gb']:>6.1f}GB"
                  f" {gs('b1_short')} {gs('b8_short')} {best_bw:>4.0f}%")


if __name__ == "__main__":
    main()
