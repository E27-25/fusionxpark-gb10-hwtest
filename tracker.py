#!/usr/bin/env python3
"""
Experiment Progress Tracker — GB10 Grace Blackwell
Scans results/ directory and prints a live dashboard of all completed experiments.
Usage: python3 tracker.py             # one-shot report
       python3 tracker.py --watch     # refresh every 30s
       python3 tracker.py --html      # write results/dashboard.html
"""
import json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results"

EXPERIMENT_PLAN = {
    # Phase 0.5 hardware
    "hw_bench":       {"phase": "0.5", "desc": "Hardware characterization (BW, TFLOPS, NVLink, thermal)"},
    # Phase 1 inference
    "inference_bench":     {"phase": "1", "desc": "LLM/VLM/MoE inference (TPS, TTFT, BW, MFU)"},
    "quant_sweep":         {"phase": "1", "desc": "Quantization format sweep (BF16→GGUF)"},
    "asr_bench":           {"phase": "1", "desc": "ASR benchmark (WER, RTF)"},
    "tts_bench":           {"phase": "1", "desc": "TTS benchmark (UTMOS, RTF)"},
    "embed_rerank_bench":  {"phase": "1", "desc": "Embedding & Reranker throughput"},
    "speculative_bench":   {"phase": "1", "desc": "Speculative decoding speedup"},
    "longctx_bench":       {"phase": "1", "desc": "Long-context needle-in-haystack"},
    "stress_test":         {"phase": "1", "desc": "Concurrent batching stress test"},
    # Phase 2 training
    "T1":  {"phase": "2", "desc": "Qwen3-8B Full FT"},
    "T2":  {"phase": "2", "desc": "Mistral-7B Full FT"},
    "T3":  {"phase": "2", "desc": "Qwen3-8B LoRA r=16"},
    "T4":  {"phase": "2", "desc": "Mistral-7B LoRA r=16"},
    "T5":  {"phase": "2", "desc": "Qwen3-8B LoRA r=64"},
    "T6":  {"phase": "2", "desc": "Qwen3-32B LoRA r=32"},
    "T7":  {"phase": "2", "desc": "Qwen3-30B-A3B LoRA r=16 (MoE)"},
    "T8":  {"phase": "2", "desc": "Qwen2.5-72B QLoRA NF4"},
    "T9":  {"phase": "2", "desc": "Qwen2.5-VL-7B LoRA (VLM)"},
    "T10": {"phase": "2", "desc": "Qwen2.5-VL-7B LoRA (VLM alt)"},
    "T11": {"phase": "2", "desc": "DeepSeek-V2-Lite LoRA r=32"},
    "T12": {"phase": "2", "desc": "Mixtral-8x7B LoRA r=16"},
    "T14": {"phase": "2", "desc": "Whisper Full FT (ASR)"},
    "T15": {"phase": "2", "desc": "Whisper LoRA r=16 (ASR)"},
    "T20": {"phase": "2", "desc": "Qwen3-8B DPO"},
    "T21": {"phase": "2", "desc": "Qwen3-8B GRPO (GSM8K math)"},
    "T22": {"phase": "2", "desc": "Qwen3-32B DPO"},
    "T23": {"phase": "2", "desc": "Qwen3-8B CPT (OpenWebText)"},
    "T24": {"phase": "2", "desc": "DeepSeek-R1-Distill-8B LoRA"},
    "M1":  {"phase": "2", "desc": "SLERP merge (base + SFT)"},
    "M2":  {"phase": "2", "desc": "TIES merge (SFT + DPO)"},
    "M3":  {"phase": "2", "desc": "DARE+TIES merge (General + Code)"},
    # Phase 3 eval
    "asr_eval":  {"phase": "3", "desc": "WER evaluation (base vs fine-tuned)"},
    "rag_pipeline": {"phase": "3B", "desc": "End-to-end RAG pipeline"},
}


def find_result_files():
    """Scan results/ for all JSON result files."""
    found = {}
    for json_file in RESULTS.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            name = json_file.stem
            # Match key by requiring it as an exact prefix before underscore.
            # e.g. "T3_..." matches "T3" but NOT "T2"; "hw_bench_..." matches
            # "hw_bench" but not "hw". Avoids T2 falsely matching T24 files.
            for key in EXPERIMENT_PLAN:
                if name.startswith(key + "_") or name == key or name.endswith(f"_{key}"):
                    found.setdefault(key, []).append({
                        "file": json_file,
                        "mtime": json_file.stat().st_mtime,
                        "data": data,
                    })
        except:
            pass
    return found


def get_gpu_live():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,"
             "memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        parts = [x.strip() for x in r.stdout.strip().split(",")]
        return {
            "power_w":     float(parts[0]),
            "temp_c":      int(parts[1]),
            "mem_used_gb": round(int(parts[2]) / 1024, 1),
            "mem_total_gb":round(int(parts[3]) / 1024, 1),
            "util_pct":    int(parts[4]),
        }
    except:
        return {}


def extract_key_metric(key, data):
    """Extract the most important metric for each experiment type."""
    # ── Phase 2 training ──────────────────────────────────────
    if key.startswith("T") or key.startswith("M"):
        loss = data.get("final_loss")
        mem  = data.get("peak_memory_gb")
        t    = data.get("elapsed_min")
        tps  = data.get("train_tokens_per_sec")
        mfu  = data.get("train_mfu_pct")
        parts = []
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if tps is not None:
            parts.append(f"tok/s={tps:.0f}")
        if mfu is not None:
            parts.append(f"MFU={mfu:.1f}%")
        if mem is not None:
            parts.append(f"mem={mem}GB")
        if t is not None:
            parts.append(f"time={t}min")
        return "  ".join(parts) if parts else "done"

    # ── Phase 1 inference ─────────────────────────────────────
    if "inference" in key:
        rows = data if isinstance(data, list) else []
        good = [r for r in rows if "error" not in r]
        if good:
            r = good[0]
            tps = r.get("tokens_per_sec", 0)
            std = r.get("tps_stdev")
            std_str = f"±{std:.0f}" if std else ""
            return (f"TPS={tps}{std_str}  TTFT={r.get('ttft_ms')}ms  "
                    f"BW={r.get('bw_util_pct')}%  MFU={r.get('mfu_pct')}%")

    if "quant" in key:
        bf16 = data.get("bf16", {})
        n_formats = len([k for k in data
                         if k not in ("model", "timestamp",
                                      "reference_bw_gbs", "reference_tflops_bf16")])
        return (f"BF16 TPS={bf16.get('tokens_per_sec')}  "
                f"Formats tested: {n_formats}")

    if "asr" in key.lower() and "eval" not in key.lower():
        if isinstance(data, list) and data:
            good = [r for r in data if "error" not in r]
            if good:
                r = good[0]
                wer = r.get("wer_pct")
                rtf = r.get("mean_rtf")
                wer_str = f"{wer:.1f}%" if wer is not None else "?"
                rtf_str = f"{rtf:.3f}" if rtf is not None else "?"
                return f"WER={wer_str}  RTF={rtf_str}  model={r.get('model_key','?')}"

    if "tts" in key.lower():
        if isinstance(data, list) and data:
            r = data[0]
            return f"UTMOS={r.get('mean_utmos')}  RTF={r.get('mean_rtf')}"

    if "embed" in key.lower():
        rows = data.get("embedding", []) if isinstance(data, dict) else []
        good = [r for r in rows if "error" not in r]
        if good:
            r = good[0]
            return (f"sent/s={r.get('sentences_per_sec')}  "
                    f"dim={r.get('embed_dim')}  mem={r.get('peak_memory_gb')}GB")

    if "speculative" in key.lower():
        if isinstance(data, list) and data:
            r = data[0]
            speedup = r.get("speedup_hf") or r.get("speedup_ratio")
            base    = r.get("baseline_tps")
            spec    = r.get("speculative_hf", {}).get("tokens_per_sec")
            if speedup:
                return f"speedup={speedup}×  base={base} TPS  spec={spec} TPS"

    if "longctx" in key.lower():
        needles = data.get("needle", []) if isinstance(data, dict) else []
        correct = sum(1 for r in needles if r.get("correct"))
        total   = len([r for r in needles if "error" not in r])
        max_ctx = max((r.get("actual_ctx_len", 0) for r in needles if "error" not in r),
                      default=0)
        return (f"needle={correct}/{total} correct  "
                f"max_ctx={max_ctx//1000}K tokens") if total > 0 else ""

    if "hw_bench" in key:
        bw  = data.get("hw1_memory_bw", {})
        vals = [v.get("bw_gbs", 0) for v in bw.values() if isinstance(v, dict)]
        peak = max(vals) if vals else None
        hw2  = data.get("hw2_compute", {})
        bf16_tflops = hw2.get("bf16", {}).get("peak_tflops")
        fp8_tflops  = hw2.get("fp8",  {}).get("peak_tflops")
        dev  = data.get("device_info", {})
        if peak:
            return (f"BW peak={peak:.0f} GB/s  BF16={bf16_tflops} TFLOPS  "
                    f"FP8={fp8_tflops} TFLOPS  GPU={dev.get('name','?')}  "
                    f"Mem={dev.get('total_memory_gb','?')} GB")
        return "No data"

    if "rag" in key:
        cfgs = list(data.keys())
        return f"Configs: {', '.join(cfgs)}"

    return ""


def print_dashboard(found):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu = get_gpu_live()

    print(f"\n{'='*80}")
    print(f"  GB10 Grace Blackwell — Experiment Tracker   {now}")
    if gpu:
        print(f"  GPU: {gpu.get('util_pct')}% util  "
              f"{gpu.get('power_w')}W  "
              f"{gpu.get('mem_used_gb')}/{gpu.get('mem_total_gb')} GB  "
              f"{gpu.get('temp_c')}°C")
    print(f"{'='*80}")

    current_phase = None
    done_count = 0
    total_count = len(EXPERIMENT_PLAN)

    for key, info in EXPERIMENT_PLAN.items():
        phase = info["phase"]
        if phase != current_phase:
            current_phase = phase
            print(f"\n  ── Phase {phase} ─────────────────────────────────")

        if key in found:
            results = sorted(found[key], key=lambda x: x["mtime"], reverse=True)
            latest  = results[0]
            mtime   = datetime.fromtimestamp(latest["mtime"]).strftime("%m-%d %H:%M")
            metric  = extract_key_metric(key, latest["data"])
            n_runs  = len(results)
            suffix  = f"  (×{n_runs})" if n_runs > 1 else ""
            print(f"  ✓ {key:<20} [{mtime}]  {metric}{suffix}")
            done_count += 1
        else:
            print(f"  · {key:<20} {info['desc']}")

    print(f"\n{'='*80}")
    print(f"  Progress: {done_count}/{total_count} experiments complete "
          f"({done_count/total_count*100:.0f}%)")
    print(f"  Results dir: {RESULTS}")

    # List recent result files
    all_jsons = sorted(RESULTS.rglob("*.json"), key=lambda f: f.stat().st_mtime,
                       reverse=True)[:5]
    if all_jsons:
        print(f"\n  Recent files:")
        for f in all_jsons:
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")
            size  = f.stat().st_size // 1024
            print(f"    [{mtime}] {f.relative_to(BASE)}  ({size} KB)")
    print()


def write_html_report(found):
    out_path = RESULTS / "dashboard.html"
    rows_html = []
    for key, info in EXPERIMENT_PLAN.items():
        if key in found:
            latest = sorted(found[key], key=lambda x: x["mtime"], reverse=True)[0]
            mtime  = datetime.fromtimestamp(latest["mtime"]).strftime("%Y-%m-%d %H:%M")
            metric = extract_key_metric(key, latest["data"])
            status = "<td style='color:green'>✓ Done</td>"
        else:
            mtime  = ""
            metric = info["desc"]
            status = "<td style='color:#aaa'>· Pending</td>"
        rows_html.append(f"""
        <tr>
            {status}
            <td>Phase {info['phase']}</td>
            <td><b>{key}</b></td>
            <td>{metric}</td>
            <td>{mtime}</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html><head>
<title>GB10 Experiment Tracker</title>
<meta http-equiv="refresh" content="60">
<style>
  body {{ font-family: monospace; background: #111; color: #eee; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 6px 12px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ background: #222; color: #888; }}
  h1 {{ color: #88aaff; }}
</style>
</head><body>
<h1>GB10 Grace Blackwell — Experiment Dashboard</h1>
<p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (auto-refresh 60s)</p>
<table>
<tr><th>Status</th><th>Phase</th><th>Experiment</th><th>Metrics</th><th>Completed</th></tr>
{''.join(rows_html)}
</table>
</body></html>"""
    out_path.write_text(html)
    print(f"  HTML dashboard: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every 30 seconds")
    parser.add_argument("--html", action="store_true",
                        help="Write HTML dashboard to results/dashboard.html")
    parser.add_argument("--interval", type=int, default=30,
                        help="Refresh interval in seconds (with --watch)")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                import os
                os.system("clear")
                found = find_result_files()
                print_dashboard(found)
                if args.html:
                    write_html_report(found)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  Tracker stopped.")
    else:
        found = find_result_files()
        print_dashboard(found)
        if args.html:
            write_html_report(found)


if __name__ == "__main__":
    main()
