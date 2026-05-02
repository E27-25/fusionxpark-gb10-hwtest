#!/usr/bin/env python3
"""
Phase 1 — Inference Benchmark (LLM / VLM / MoE)
Backends: HF generate(), vLLM, SGLang, TensorRT-LLM
Metrics: TTFT (mean/p95), TPS (mean/stdev), TPOT, memory, BW%, MFU, tokens/watt
"""
import os, sys, json, time, argparse, subprocess, statistics, gc
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
RESULTS.mkdir(parents=True, exist_ok=True)

PEAK_BW_GBS   = 273.0   # GB10 LPDDR5X theoretical
PEAK_TFLOPS   = 67.0    # BF16 TFLOPS
THROTTLE_C    = 53
N_RUNS        = 5       # timing iterations for mean/stdev

PROMPT_SHORT  = "Explain quantum entanglement in simple terms."
PROMPT_MEDIUM = " ".join(["The history of artificial intelligence begins"] * 64)
PROMPT_LONG   = " ".join(["Context: " + "word " * 100] * 20)

PROMPTS = {
    "short":  (PROMPT_SHORT,  128),
    "medium": (PROMPT_MEDIUM, 256),
    "long":   (PROMPT_LONG,   256),
}

# ─────────────────────────────────────────────────────────────
def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        p, t, u = r.stdout.strip().split(",")
        return {"power_w": float(p.strip()), "temp_c": int(t.strip()), "util_pct": int(u.strip())}
    except:
        return {}

def compute_bw_util(model_bytes_gb, tokens_per_sec, batch_size):
    """
    Each decode step reads model weights once and emits batch_size tokens.
    BW = model_gb × (tps / batch_size)
    """
    try:
        decode_steps = tokens_per_sec / max(batch_size, 1)
        achieved = model_bytes_gb * decode_steps
        return round(achieved, 1), round(achieved / PEAK_BW_GBS * 100, 1)
    except:
        return None, None

def compute_mfu(n_params, tokens_per_sec):
    """
    MFU for inference: 2N FLOPs/token (forward pass only — not the 6N training rule).
    """
    try:
        achieved_tflops = 2 * n_params * tokens_per_sec / 1e12
        return round(achieved_tflops, 3), round(achieved_tflops / PEAK_TFLOPS * 100, 2)
    except:
        return None, None

def percentile(data, p):
    data = sorted(data)
    idx = int(p / 100 * len(data))
    return data[min(idx, len(data) - 1)]

def make_row(backend, model_id, dtype_str, bs, prompt_key,
             ttft_times, gen_times, n_tokens_per_run,
             peak_mem_gb, model_load_gb=None, n_params=None,
             gpu_stats=None):
    """Build a standardised result dict from raw timing lists."""
    ttft_ms = statistics.mean(ttft_times)
    ttft_p95 = percentile(ttft_times, 95)
    elapsed_mean = statistics.mean(gen_times)
    elapsed_stdev = statistics.stdev(gen_times) if len(gen_times) > 1 else 0.0
    n_tok = statistics.mean(n_tokens_per_run)  # tokens in last batch
    tps = n_tok / elapsed_mean if elapsed_mean > 0 else 0
    tps_stdev = tps * (elapsed_stdev / elapsed_mean) if elapsed_mean > 0 else 0

    gpu = gpu_stats or {}
    power = gpu.get("power_w", 0)

    bw_gbs, bw_util = (compute_bw_util(model_load_gb, tps, bs)
                       if model_load_gb else (None, None))
    tflops, mfu = (compute_mfu(n_params, tps) if n_params else (None, None))

    return {
        "backend": backend,
        "model": model_id,
        "dtype": dtype_str,
        "batch": bs,
        "prompt": prompt_key,
        "ttft_ms": round(ttft_ms, 1),
        "ttft_p95_ms": round(ttft_p95, 1),
        "tokens_per_sec": round(tps, 1),
        "tps_stdev": round(tps_stdev, 1),
        "tpot_ms": round(elapsed_mean / (n_tok / bs) * 1000, 2) if n_tok > 0 else None,
        "peak_memory_gb": round(peak_mem_gb, 2),
        "model_load_gb": round(model_load_gb, 2) if model_load_gb else None,
        "n_params": n_params,
        "bw_achieved_gbs": bw_gbs,
        "bw_util_pct": bw_util,
        "mfu_tflops": tflops,
        "mfu_pct": mfu,
        "power_w": power,
        "tokens_per_watt": round(tps / power, 2) if power > 0 else None,
        "temp_c": gpu.get("temp_c"),
        "throttled": (gpu.get("temp_c", 0) >= THROTTLE_C),
    }


# ═══════════════════════════════════════════════════════════════
# BACKEND: HuggingFace generate()
# ═══════════════════════════════════════════════════════════════
def benchmark_hf(model_id, dtype_str="bfloat16", batch_sizes=(1, 4, 8),
                 prompt_key="short", thinking=False, load_in_4bit=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    dtype = getattr(torch, dtype_str)
    prompt, out_len = PROMPTS[prompt_key]
    if thinking:
        prompt = "/think\n" + prompt

    label = "nf4" if load_in_4bit else dtype_str
    print(f"\n  [HF] Loading {model_id} ({label})...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg,
            device_map="cuda", trust_remote_code=True
        )
        dtype_str = "nf4"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="cuda", trust_remote_code=True
        )
    model.eval()
    torch.cuda.synchronize()
    mem_after_load = torch.cuda.memory_allocated() / 1e9
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Loaded: {n_params/1e9:.1f}B params, {mem_after_load:.1f} GB GPU")

    rows = []
    for bs in batch_sizes:
        try:
            inputs = tok([prompt] * bs, return_tensors="pt",
                         truncation=True, max_length=2048).to("cuda")
            torch.cuda.synchronize()

            # warmup
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=32, do_sample=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # TTFT
            ttft_times = []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=1, do_sample=False)
                torch.cuda.synchronize()
                ttft_times.append((time.perf_counter() - t0) * 1000)

            # Throughput
            gen_times, n_tokens_per_run = [], []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=out_len, do_sample=False)
                torch.cuda.synchronize()
                gen_times.append(time.perf_counter() - t0)
                n_new = (out.shape[1] - inputs["input_ids"].shape[1]) * bs
                n_tokens_per_run.append(n_new)

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            gpu = get_gpu_stats()

            row = make_row("hf", model_id, dtype_str, bs, prompt_key,
                           ttft_times, gen_times, n_tokens_per_run,
                           peak_mem, mem_after_load, n_params, gpu)
            row["thinking"] = thinking
            rows.append(row)

            tps = row["tokens_per_sec"]
            print(f"    bs={bs:2d}  TTFT={row['ttft_ms']:.0f}ms(p95={row['ttft_p95_ms']:.0f})  "
                  f"TPS={tps:.0f}±{row['tps_stdev']:.0f}  "
                  f"mem={peak_mem:.1f}GB  BW={row['bw_util_pct']}%  "
                  f"MFU={row['mfu_pct']}%  {row['power_w']:.0f}W")

        except Exception as e:
            msg = "OOM" if "out of memory" in str(e).lower() else str(e)
            print(f"    bs={bs}: {msg}")
            rows.append({"backend": "hf", "model": model_id, "batch": bs, "error": msg})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════
# BACKEND: vLLM
# ═══════════════════════════════════════════════════════════════
def benchmark_vllm(model_id, dtype_str="bfloat16", batch_sizes=(1, 4, 8),
                   prompt_key="short"):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  [vLLM] Not installed. To install:")
        print("    sudo apt install python3-dev")
        print("    pip3 install vllm --break-system-packages")
        return [{"backend": "vllm", "model": model_id, "error": "not_installed"}]

    import torch
    prompt, out_len = PROMPTS[prompt_key]
    print(f"\n  [vLLM] Loading {model_id} ({dtype_str})...")

    try:
        llm = LLM(model=model_id, dtype=dtype_str,
                  gpu_memory_utilization=0.85,
                  trust_remote_code=True,
                  max_model_len=4096)
    except Exception as e:
        print(f"  [vLLM] Load failed: {e}")
        return [{"backend": "vllm", "model": model_id, "error": str(e)}]

    mem_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"    Loaded. GPU memory: {mem_after_load:.1f} GB")

    rows = []
    for bs in batch_sizes:
        try:
            prompts_list = [prompt] * bs

            # warmup
            llm.generate(prompts_list, SamplingParams(temperature=0, max_tokens=32))
            torch.cuda.reset_peak_memory_stats()

            # TTFT (single output token measures prefill latency)
            ttft_times = []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                llm.generate(prompts_list, SamplingParams(temperature=0, max_tokens=1))
                ttft_times.append((time.perf_counter() - t0) * 1000)

            # Throughput
            gen_times, n_tokens_per_run = [], []
            gen_params = SamplingParams(temperature=0, max_tokens=out_len)
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                outputs = llm.generate(prompts_list, gen_params)
                gen_times.append(time.perf_counter() - t0)
                # count actual tokens generated
                n_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
                n_tokens_per_run.append(n_tok)

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            gpu = get_gpu_stats()

            row = make_row("vllm", model_id, dtype_str, bs, prompt_key,
                           ttft_times, gen_times, n_tokens_per_run,
                           peak_mem, mem_after_load, None, gpu)
            rows.append(row)
            print(f"    bs={bs:2d}  TTFT={row['ttft_ms']:.0f}ms(p95={row['ttft_p95_ms']:.0f})  "
                  f"TPS={row['tokens_per_sec']:.0f}±{row['tps_stdev']:.0f}  "
                  f"mem={peak_mem:.1f}GB  {row['power_w']:.0f}W")
        except Exception as e:
            print(f"    bs={bs}: {e}")
            rows.append({"backend": "vllm", "model": model_id, "batch": bs, "error": str(e)})

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════
# BACKEND: SGLang
# ═══════════════════════════════════════════════════════════════
def benchmark_sglang(model_id, dtype_str="bfloat16", batch_sizes=(1, 4, 8),
                     prompt_key="short"):
    try:
        import sglang as sgl
    except ImportError:
        print("  [SGLang] Not installed. To install:")
        print('    pip3 install "sglang[all]" --break-system-packages')
        return [{"backend": "sglang", "model": model_id, "error": "not_installed"}]

    import torch
    prompt, out_len = PROMPTS[prompt_key]
    print(f"\n  [SGLang] Loading {model_id} ({dtype_str})...")

    engine = None
    try:
        # SGLang >= 0.3: Engine API
        engine = sgl.Engine(model_path=model_id, dtype=dtype_str,
                            trust_remote_code=True,
                            mem_fraction_static=0.85,
                            log_level="error")
    except Exception as e:
        print(f"  [SGLang] Engine init failed: {e}")
        return [{"backend": "sglang", "model": model_id, "error": str(e)}]

    mem_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"    Loaded. GPU memory: {mem_after_load:.1f} GB")

    sampling_ttft = {"temperature": 0, "max_new_tokens": 1}
    sampling_gen  = {"temperature": 0, "max_new_tokens": out_len}

    rows = []
    for bs in batch_sizes:
        try:
            prompts_list = [prompt] * bs

            # warmup
            engine.generate(prompts_list, sampling_params={"temperature": 0, "max_new_tokens": 32})
            torch.cuda.reset_peak_memory_stats()

            # TTFT
            ttft_times = []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                engine.generate(prompts_list, sampling_params=sampling_ttft)
                ttft_times.append((time.perf_counter() - t0) * 1000)

            # Throughput
            gen_times, n_tokens_per_run = [], []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                outputs = engine.generate(prompts_list, sampling_params=sampling_gen)
                gen_times.append(time.perf_counter() - t0)
                # SGLang returns list of dicts; extract token count from meta_info or text
                n_tok = 0
                for o in (outputs if isinstance(outputs, list) else [outputs]):
                    meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
                    if "completion_tokens" in meta:
                        n_tok += meta["completion_tokens"]
                    else:
                        # fallback: estimate from output text word count
                        text = o.get("text", "") if isinstance(o, dict) else str(o)
                        n_tok += len(text.split())
                n_tokens_per_run.append(max(n_tok, 1))

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            gpu = get_gpu_stats()

            row = make_row("sglang", model_id, dtype_str, bs, prompt_key,
                           ttft_times, gen_times, n_tokens_per_run,
                           peak_mem, mem_after_load, None, gpu)
            rows.append(row)
            print(f"    bs={bs:2d}  TTFT={row['ttft_ms']:.0f}ms(p95={row['ttft_p95_ms']:.0f})  "
                  f"TPS={row['tokens_per_sec']:.0f}±{row['tps_stdev']:.0f}  "
                  f"mem={peak_mem:.1f}GB  {row['power_w']:.0f}W")
        except Exception as e:
            print(f"    bs={bs}: {e}")
            rows.append({"backend": "sglang", "model": model_id, "batch": bs, "error": str(e)})

    try:
        engine.shutdown()
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════
# BACKEND: TensorRT-LLM
# ═══════════════════════════════════════════════════════════════
def benchmark_trtllm(model_id, dtype_str="bfloat16", batch_sizes=(1, 4, 8),
                     prompt_key="short"):
    """
    TRT-LLM high-level API (v0.12+): mirrors vLLM interface.
    Engines are built on first use and cached under models/trtllm-engines/.
    Falls back to engine-build instructions if API unavailable.
    """
    try:
        from tensorrt_llm import LLM as TRTLLM
        from tensorrt_llm import SamplingParams as TRTSamplingParams
        trt_highlevel = True
    except ImportError:
        trt_highlevel = False
        try:
            import tensorrt_llm  # noqa — check import only
        except ImportError:
            print("  [TRT-LLM] Not installed. To install:")
            print("    pip3 install tensorrt-llm --break-system-packages")
            print("    (May require building from source for SM120/Blackwell)")
            return [{"backend": "trtllm", "model": model_id, "error": "not_installed"}]

    import torch
    prompt, out_len = PROMPTS[prompt_key]
    model_name = model_id.replace("/", "--")
    engine_dir = BASE / f"models/trtllm-engines/{model_name}-{dtype_str}"
    print(f"\n  [TRT-LLM] Using {model_id} ({dtype_str})...")

    if trt_highlevel:
        # High-level API — builds engine automatically
        try:
            print(f"    Building/loading TRT engine (first run may take 10–30 min)...")
            llm = TRTLLM(model=model_id, dtype=dtype_str,
                         tensor_parallel_size=1,
                         trust_remote_code=True)
        except Exception as e:
            print(f"  [TRT-LLM] Load failed: {e}")
            return [{"backend": "trtllm", "model": model_id, "error": str(e)}]

        mem_after_load = torch.cuda.memory_allocated() / 1e9
        print(f"    Loaded. GPU memory: {mem_after_load:.1f} GB")

        rows = []
        for bs in batch_sizes:
            try:
                prompts_list = [prompt] * bs

                # warmup
                llm.generate(prompts_list, TRTSamplingParams(max_tokens=32, temperature=1e-5))
                torch.cuda.reset_peak_memory_stats()

                # TTFT
                ttft_times = []
                for _ in range(N_RUNS):
                    t0 = time.perf_counter()
                    llm.generate(prompts_list, TRTSamplingParams(max_tokens=1, temperature=1e-5))
                    ttft_times.append((time.perf_counter() - t0) * 1000)

                # Throughput
                gen_params = TRTSamplingParams(max_tokens=out_len, temperature=1e-5)
                gen_times, n_tokens_per_run = [], []
                for _ in range(N_RUNS):
                    t0 = time.perf_counter()
                    outputs = llm.generate(prompts_list, gen_params)
                    gen_times.append(time.perf_counter() - t0)
                    n_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
                    n_tokens_per_run.append(n_tok)

                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                gpu = get_gpu_stats()

                row = make_row("trtllm", model_id, dtype_str, bs, prompt_key,
                               ttft_times, gen_times, n_tokens_per_run,
                               peak_mem, mem_after_load, None, gpu)
                rows.append(row)
                print(f"    bs={bs:2d}  TTFT={row['ttft_ms']:.0f}ms(p95={row['ttft_p95_ms']:.0f})  "
                      f"TPS={row['tokens_per_sec']:.0f}±{row['tps_stdev']:.0f}  "
                      f"mem={peak_mem:.1f}GB  {row['power_w']:.0f}W")
            except Exception as e:
                print(f"    bs={bs}: {e}")
                rows.append({"backend": "trtllm", "model": model_id, "batch": bs, "error": str(e)})

        try:
            del llm
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        return rows

    else:
        # Low-level ModelRunner API (TRT-LLM < 0.12)
        if not engine_dir.exists():
            print(f"  [TRT-LLM] Pre-built engine not found: {engine_dir}")
            print(f"  Build engine first:")
            print(f"    trtllm-build --model_dir {model_id} \\")
            print(f"      --output_dir {engine_dir} \\")
            print(f"      --dtype {dtype_str} --tp_size 1")
            return [{"backend": "trtllm", "model": model_id,
                     "error": f"engine_not_built:{engine_dir}"}]

        try:
            from tensorrt_llm.runtime import ModelRunner
        except Exception as e:
            return [{"backend": "trtllm", "model": model_id, "error": str(e)}]

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        try:
            runner = ModelRunner.from_dir(engine_dir=str(engine_dir), rank=0)
        except Exception as e:
            print(f"  [TRT-LLM] Runner init failed: {e}")
            return [{"backend": "trtllm", "model": model_id, "error": str(e)}]

        mem_after_load = torch.cuda.memory_allocated() / 1e9
        rows = []
        for bs in batch_sizes:
            try:
                input_ids = tok([prompt] * bs, return_tensors="pt",
                                truncation=True, max_length=2048)["input_ids"].cuda()
                in_len = torch.tensor([input_ids.shape[1]] * bs, dtype=torch.int32).cuda()

                runner.generate(input_ids, in_len,
                                sampling_config={"max_new_tokens": 32, "temperature": 1e-5})
                torch.cuda.reset_peak_memory_stats()

                ttft_times = []
                for _ in range(N_RUNS):
                    t0 = time.perf_counter()
                    runner.generate(input_ids, in_len,
                                    sampling_config={"max_new_tokens": 1, "temperature": 1e-5})
                    torch.cuda.synchronize()
                    ttft_times.append((time.perf_counter() - t0) * 1000)

                gen_times, n_tokens_per_run = [], []
                for _ in range(N_RUNS):
                    t0 = time.perf_counter()
                    out = runner.generate(input_ids, in_len,
                                         sampling_config={"max_new_tokens": out_len, "temperature": 1e-5})
                    torch.cuda.synchronize()
                    gen_times.append(time.perf_counter() - t0)
                    n_tok = out_len * bs  # runner doesn't always expose exact count
                    n_tokens_per_run.append(n_tok)

                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                gpu = get_gpu_stats()

                row = make_row("trtllm", model_id, dtype_str, bs, prompt_key,
                               ttft_times, gen_times, n_tokens_per_run,
                               peak_mem, mem_after_load, None, gpu)
                rows.append(row)
                print(f"    bs={bs:2d}  TTFT={row['ttft_ms']:.0f}ms  "
                      f"TPS={row['tokens_per_sec']:.0f}  mem={peak_mem:.1f}GB")
            except Exception as e:
                print(f"    bs={bs}: {e}")
                rows.append({"backend": "trtllm", "model": model_id, "batch": bs, "error": str(e)})

        try:
            del runner
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        return rows


# ═══════════════════════════════════════════════════════════════
# Thinking mode comparison (HF only — needs prompt injection)
# ═══════════════════════════════════════════════════════════════
def run_thinking_comparison(model_id):
    print(f"\n  Qwen3 Thinking Mode Comparison: {model_id}")
    rows = []
    for thinking, label in [(False, "no_think"), (True, "think")]:
        for r in benchmark_hf(model_id, batch_sizes=(1,), prompt_key="short", thinking=thinking):
            r["mode"] = label
            rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════════
def save_results(tag, rows):
    out = RESULTS / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"  Saved: {out}")
    return out


def print_summary(all_rows):
    print(f"\n{'='*96}")
    print(f"  {'Backend':<8} {'Model':<26} {'bs':>3} {'TPS':>6} {'±stdev':>7} "
          f"{'TTFT':>7} {'p95':>6} {'Mem':>6} {'BW%':>5} {'MFU%':>5} {'tok/W':>6}")
    print(f"  {'-'*96}")
    for r in all_rows:
        if "error" not in r:
            print(f"  {str(r.get('backend','?')):<8} "
                  f"{str(r.get('model',''))[-26:]:<26} "
                  f"{r.get('batch',0):>3} "
                  f"{r.get('tokens_per_sec',0):>6.0f} "
                  f"{r.get('tps_stdev',0):>7.0f} "
                  f"{r.get('ttft_ms',0):>6.0f}ms "
                  f"{r.get('ttft_p95_ms',0):>5.0f}ms "
                  f"{r.get('peak_memory_gb',0):>5.1f}G "
                  f"{str(r.get('bw_util_pct','?')):>5}% "
                  f"{str(r.get('mfu_pct','?')):>5}% "
                  f"{str(r.get('tokens_per_watt','?')):>6}")
        else:
            print(f"  {str(r.get('backend','?')):<8} "
                  f"{str(r.get('model',''))[-26:]:<26} "
                  f"bs={r.get('batch','?')}  ERROR: {r.get('error','?')}")
    print()


# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype",       default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--prompts",     nargs="+", default=["short", "medium"],
                        choices=list(PROMPTS.keys()))
    parser.add_argument("--backend",     nargs="+", default=["hf"],
                        choices=["hf", "vllm", "sglang", "trtllm", "all"])
    parser.add_argument("--thinking",    action="store_true",
                        help="Run Qwen3 thinking vs no-think comparison (HF only)")
    parser.add_argument("--nf4",         action="store_true",
                        help="Load model in NF4 (4-bit) via bitsandbytes (for 70B+ models)")
    parser.add_argument("--all-models",  action="store_true",
                        help="Run full model list from config/models.yaml")
    args = parser.parse_args()

    backends = (["hf", "vllm", "sglang", "trtllm"]
                if "all" in args.backend else list(args.backend))

    def run_backend(be, mid, dtyp, bsizes, pk):
        if be == "hf":
            return benchmark_hf(mid, dtyp, bsizes, pk, load_in_4bit=args.nf4)
        elif be == "vllm":
            return benchmark_vllm(mid, dtyp, bsizes, pk)
        elif be == "sglang":
            return benchmark_sglang(mid, dtyp, bsizes, pk)
        elif be == "trtllm":
            return benchmark_trtllm(mid, dtyp, bsizes, pk)
        return []

    all_rows = []

    if args.all_models:
        try:
            import yaml
            cfg = yaml.safe_load((BASE / "config/models.yaml").read_text())
        except Exception as e:
            print(f"Could not load config/models.yaml: {e}")
            return

        models_to_run = []
        for cat in ["small", "mid", "large"]:
            for m in cfg["llms"].get(cat, []):
                models_to_run.append((m["id"], "bfloat16"))
        for m in cfg["llms"].get("xl", []):
            models_to_run.append((m["id"], m.get("precision", "bfloat16")))

        for model_id, dtype in models_to_run:
            for be in backends:
                rows = run_backend(be, model_id, dtype, args.batch_sizes, "short")
                all_rows.extend(rows)
                save_results(f"{be}_{model_id.split('/')[-1]}", rows)
    else:
        for prompt_key in args.prompts:
            for be in backends:
                rows = run_backend(be, args.model, args.dtype, args.batch_sizes, prompt_key)
                all_rows.extend(rows)

        if args.thinking and "Qwen3" in args.model:
            rows = run_thinking_comparison(args.model)
            all_rows.extend(rows)

    save_results("inference_bench", all_rows)
    print_summary(all_rows)


if __name__ == "__main__":
    main()
