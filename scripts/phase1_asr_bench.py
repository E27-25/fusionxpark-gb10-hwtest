#!/usr/bin/env python3
"""
Phase 1 — ASR Benchmark
Models: Whisper-large-v3, Whisper-large-v3-turbo, Qwen2-Audio-7B
Metrics: WER, CER, RTF, TTFT, tokens/sec, memory
"""
import json, time, subprocess, argparse, tempfile, os, statistics, io
from pathlib import Path
from datetime import datetime
import numpy as np

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = {
    "whisper-large-v3":       "openai/whisper-large-v3",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "qwen2-audio-7b":         "Qwen/Qwen2-Audio-7B-Instruct",
}

LIBRISPEECH_SAMPLES = 50  # number of samples to evaluate


def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        p, t, m = r.stdout.strip().split(",")
        return {"power_w": float(p.strip()), "temp_c": int(t.strip()),
                "mem_used_mb": int(m.strip())}
    except:
        return {}


def compute_wer(references, hypotheses):
    try:
        import jiwer, re
        # Whisper adds punctuation; LibriSpeech refs have none — strip for fair WER
        def normalize(t):
            t = t.lower()
            t = re.sub(r"[^\w\s']", " ", t)  # remove punctuation, keep apostrophes
            return re.sub(r"\s+", " ", t).strip()
        refs_n = [normalize(r) for r in references]
        hyps_n = [normalize(h) for h in hypotheses]
        wer = jiwer.wer(refs_n, hyps_n)
        cer = jiwer.cer(refs_n, hyps_n)
        return round(wer * 100, 2), round(cer * 100, 2)
    except Exception as e:
        return None, None


def make_synthetic_samples(n):
    import numpy as np
    samples = []
    for _ in range(n):
        array = np.sin(2 * np.pi * 440 * np.arange(16000 * 5) / 16000).astype(np.float32)
        samples.append({
            "audio": {"array": array, "sampling_rate": 16000},
            "text": "four four three two one",
        })
    return samples


def load_librispeech(n=LIBRISPEECH_SAMPLES):
    try:
        from datasets import load_dataset, Audio
        import soundfile as sf, io
        print(f"  Loading LibriSpeech test-clean ({n} samples)...")
        # decode=False to avoid torchcodec; we decode with soundfile instead
        ds = load_dataset("librispeech_asr", "clean", split=f"test[:{n}]")
        ds = ds.cast_column("audio", Audio(decode=False))

        # Convert to list with manual soundfile decoding so pipeline receives np arrays
        samples = []
        for row in ds:
            raw = row["audio"]
            path = raw.get("path") or raw.get("bytes")
            try:
                if raw.get("bytes"):
                    arr, sr = sf.read(io.BytesIO(raw["bytes"]))
                else:
                    arr, sr = sf.read(path)
                arr = arr.astype(np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                samples.append({
                    "audio": {"array": arr, "sampling_rate": sr},
                    "text": row["text"],
                })
            except Exception as decode_err:
                print(f"  Warning: decode failed for sample: {decode_err}")
        print(f"  Loaded {len(samples)} samples via soundfile.")
        return samples
    except Exception as e:
        print(f"  LibriSpeech load failed: {e}")
        print("  WARNING: Falling back to synthetic dataset.")
        return make_synthetic_samples(10)


# ─────────────────────────────────────────────────────────────
def bench_whisper(model_id, dataset, model_key):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    print(f"\n  Loading {model_id}...")
    try:
        dtype = torch.bfloat16
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=dtype, device_map="cuda",
            low_cpu_mem_usage=True, use_safetensors=True
        )
        model.eval()
        mem_load = torch.cuda.memory_allocated() / 1e9

        refs, hyps = [], []
        rtf_list, ttft_list = [], []
        gpu_before = get_gpu_stats()

        for i, sample in enumerate(dataset):
            audio = sample["audio"]
            ref   = sample["text"].lower().strip()
            arr   = audio["array"]
            sr    = audio["sampling_rate"]
            audio_sec = len(arr) / sr

            # Use processor directly — avoids torchcodec path in pipeline
            inputs = processor(arr, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to("cuda", dtype=dtype)

            t0 = time.perf_counter()
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                )
            elapsed = time.perf_counter() - t0

            hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()
            rtf = elapsed / audio_sec
            refs.append(ref)
            hyps.append(hyp)
            rtf_list.append(rtf)
            if i < 10:
                ttft_list.append(elapsed * 1000)

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(dataset)} done, RTF={rtf:.3f}")

        gpu_after = get_gpu_stats()
        wer, cer = compute_wer(refs, hyps)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        rtf_stdev = statistics.stdev(rtf_list) if len(rtf_list) > 1 else 0.0
        rtf_p95   = sorted(rtf_list)[int(0.95 * len(rtf_list))]

        result_row = {
            "model": model_id,
            "model_key": model_key,
            "n_samples": len(dataset),
            "wer_pct": wer,
            "cer_pct": cer,
            "mean_rtf":  round(sum(rtf_list) / len(rtf_list), 4),
            "rtf_stdev": round(rtf_stdev, 4),
            "rtf_p95":   round(rtf_p95, 4),
            "min_rtf":   round(min(rtf_list), 4),
            "max_rtf":   round(max(rtf_list), 4),
            "mean_ttft_ms": round(sum(ttft_list) / len(ttft_list), 1) if ttft_list else None,
            "peak_memory_gb": round(peak_mem, 2),
            "model_load_gb":  round(mem_load, 2),
            "power_w":        gpu_after.get("power_w"),
            "temp_c":         gpu_after.get("temp_c"),
            "tokens_per_watt": None,
        }
        print(f"    WER={wer}%  CER={cer}%  RTF={result_row['mean_rtf']:.3f}  "
              f"RTFstd={rtf_stdev:.3f}  Mem={peak_mem:.1f}GB")

        del model
        torch.cuda.empty_cache()
        return result_row

    except Exception as e:
        print(f"    Error: {e}")
        return {"model": model_id, "model_key": model_key, "error": str(e)}


def bench_qwen_audio(model_id, dataset, model_key):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    print(f"\n  Loading {model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )
        model.eval()
        mem_load = torch.cuda.memory_allocated() / 1e9

        refs, hyps = [], []
        rtf_list = []

        for i, sample in enumerate(dataset):
            audio = sample["audio"]
            ref   = sample["text"].lower().strip()
            arr   = audio["array"]
            # Resample to 16kHz if needed (Qwen2-Audio expects 16kHz)
            if audio["sampling_rate"] != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=audio["sampling_rate"], target_sr=16000)
            audio_sec = len(arr) / 16000

            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "file://dummy"},
                    {"type": "text", "text": "Please transcribe this audio exactly as spoken. Output only the transcription, no other text."}
                ]}
            ]
            text_prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            # Pass audio array directly via feature processor
            inputs = processor(
                text=text_prompt,
                audio=arr,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to("cuda")

            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            elapsed = time.perf_counter() - t0

            decoded = processor.batch_decode(
                output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].lower().strip()
            refs.append(ref)
            hyps.append(decoded)
            rtf_list.append(elapsed / audio_sec)

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(dataset)} done, RTF={elapsed/audio_sec:.3f}")

        wer, cer = compute_wer(refs, hyps)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        rtf_stdev = statistics.stdev(rtf_list) if len(rtf_list) > 1 else 0.0
        rtf_p95   = sorted(rtf_list)[int(0.95 * len(rtf_list))]
        gpu_after = get_gpu_stats()

        result_row = {
            "model": model_id,
            "model_key": model_key,
            "n_samples": len(dataset),
            "wer_pct": wer,
            "cer_pct": cer,
            "mean_rtf":  round(sum(rtf_list) / len(rtf_list), 4),
            "rtf_stdev": round(rtf_stdev, 4),
            "rtf_p95":   round(rtf_p95, 4),
            "peak_memory_gb": round(peak_mem, 2),
            "model_load_gb":  round(mem_load, 2),
            "power_w":        gpu_after.get("power_w"),
            "temp_c":         gpu_after.get("temp_c"),
            "tokens_per_watt": None,
        }
        print(f"    WER={wer}%  CER={cer}%  RTF={result_row['mean_rtf']:.3f}  "
              f"RTFstd={rtf_stdev:.3f}  Mem={peak_mem:.1f}GB")

        del model
        torch.cuda.empty_cache()
        return result_row

    except Exception as e:
        print(f"    Error: {e}")
        return {"model": model_id, "model_key": model_key, "error": str(e)}


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=list(MODELS.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--samples", type=int, default=LIBRISPEECH_SAMPLES)
    args = parser.parse_args()

    keys = list(MODELS.keys()) if "all" in args.models else args.models
    dataset = load_librispeech(args.samples)

    all_rows = []
    for key in keys:
        model_id = MODELS[key]
        print(f"\n── {key.upper()} ──")
        if "qwen" in key.lower():
            row = bench_qwen_audio(model_id, dataset, key)
        else:
            row = bench_whisper(model_id, dataset, key)
        all_rows.append(row)

    print(f"\n{'='*80}")
    print(f"  {'Model':<30} {'WER%':>6} {'CER%':>6} {'RTF':>6} {'RTFstd':>8} {'Mem':>6}")
    print(f"  {'-'*80}")
    for r in all_rows:
        if "error" not in r:
            print(f"  {r.get('model_key',''):<30} "
                  f"{str(r.get('wer_pct','?')):>6} "
                  f"{str(r.get('cer_pct','?')):>6} "
                  f"{r.get('mean_rtf',0):>6.3f} "
                  f"{r.get('rtf_stdev',0):>8.3f} "
                  f"{r.get('peak_memory_gb',0):>5.1f}G")
        else:
            print(f"  {r.get('model_key',''):<30} ERROR: {r.get('error','')[:40]}")

    out = RESULTS / f"asr_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_rows, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
