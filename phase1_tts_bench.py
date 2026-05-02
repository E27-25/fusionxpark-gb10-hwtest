#!/usr/bin/env python3
"""
Phase 1 — TTS Benchmark
Models: Qwen3-TTS-0.6B-Base, Qwen3-TTS-1.7B-VoiceDesign, Qwen3-TTS-1.7B-CustomVoice
Metrics: RTF, UTMOS (MOS proxy), speaker similarity (CustomVoice), memory
"""
import json, time, subprocess, argparse, tempfile, os, wave, struct, statistics
from pathlib import Path
from datetime import datetime
import numpy as np

N_RUNS = 3  # RTF runs per text sample (warm cache after first)


def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
RESULTS.mkdir(parents=True, exist_ok=True)

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming every industry.",
    "The NVIDIA GB10 Grace Blackwell chip features 125 gigabytes of unified memory.",
    "Natural language processing enables machines to understand human speech.",
    "Deep learning models require significant computational resources for training.",
]

SAMPLE_RATE = 24000  # typical for TTS models


def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        p, t = r.stdout.strip().split(",")
        return {"power_w": float(p.strip()), "temp_c": int(t.strip())}
    except:
        return {}


def compute_utmos(audio_array, sample_rate=SAMPLE_RATE):
    # UTMOS requires fairseq which doesn't build on aarch64 — skip gracefully
    return None


def compute_speaker_similarity(ref_audio, gen_audio, sample_rate=SAMPLE_RATE):
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        import numpy as np
        encoder = VoiceEncoder()
        ref_np = np.array(ref_audio, dtype=np.float32)
        gen_np = np.array(gen_audio, dtype=np.float32)
        ref_embed = encoder.embed_utterance(preprocess_wav(ref_np, source_sr=sample_rate))
        gen_embed = encoder.embed_utterance(preprocess_wav(gen_np, source_sr=sample_rate))
        sim = np.dot(ref_embed, gen_embed) / (
            np.linalg.norm(ref_embed) * np.linalg.norm(gen_embed))
        return round(float(sim), 4)
    except Exception as e:
        return None


def save_wav(audio_array, path, sample_rate=SAMPLE_RATE):
    audio_int = (np.array(audio_array) * 32767).astype(np.int16)
    with wave.open(str(path), 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_int.tobytes())


# ─────────────────────────────────────────────────────────────
def bench_qwen_tts(model_id, model_key, custom_voice_ref=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n  Loading {model_id}...")
    try:
        from transformers import pipeline
        # Qwen3-TTS requires qwen3_tts architecture support in transformers.
        # If not available, raise early so the caller catches it gracefully.
        try:
            from transformers import Qwen3TTSForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                f"Qwen3-TTS architecture not supported by transformers "
                f"{__import__('transformers').__version__}. "
                f"Model '{model_id}' skipped."
            )
        pipe = pipeline("text-to-speech", model=model_id,
                        dtype=torch.bfloat16, device_map="cuda",
                        trust_remote_code=True)
        import gc; gc.collect()
        mem_load = torch.cuda.memory_allocated() / 1e9

        rows = []
        audio_dir = RESULTS / "audio_samples"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(TEST_TEXTS):
            kwargs = {}
            if custom_voice_ref is not None:
                kwargs["voice"] = custom_voice_ref

            rtf_list = []
            last_audio = None
            last_sr    = SAMPLE_RATE
            for run in range(N_RUNS):
                t0 = time.perf_counter()
                output = pipe(text, **kwargs)
                elapsed = time.perf_counter() - t0
                audio = output["audio"].squeeze()
                sr    = output.get("sampling_rate", SAMPLE_RATE)
                audio_sec = len(audio) / sr
                rtf_list.append(elapsed / audio_sec)
                last_audio, last_sr = audio, sr

            gpu_after = get_gpu_stats()
            utmos_score = compute_utmos(last_audio, last_sr)

            # Save audio sample from last run
            wav_path = audio_dir / f"{model_key}_sample{i}.wav"
            save_wav(last_audio, wav_path, last_sr)

            rtf_mean = statistics.mean(rtf_list)
            rtf_std  = statistics.stdev(rtf_list) if len(rtf_list) > 1 else 0.0
            rtf_p95  = percentile(rtf_list, 95)

            row = {
                "model": model_id,
                "model_key": model_key,
                "text_idx": i,
                "text_len_chars": len(text),
                "audio_sec": round(len(last_audio) / last_sr, 3),
                "rtf_mean": round(rtf_mean, 4),
                "rtf_stdev": round(rtf_std, 4),
                "rtf_p95": round(rtf_p95, 4),
                "utmos": utmos_score,
                "power_w": gpu_after.get("power_w"),
                "temp_c": gpu_after.get("temp_c"),
                "wav_path": str(wav_path),
            }
            rows.append(row)
            print(f"    [{i}] RTF={rtf_mean:.3f}±{rtf_std:.3f}  "
                  f"audio={row['audio_sec']:.1f}s  UTMOS={utmos_score}")

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        rtf_all = [r["rtf_mean"] for r in rows]
        utmos_vals = [r["utmos"] for r in rows if r.get("utmos") is not None]
        summary = {
            "model": model_id,
            "model_key": model_key,
            "mean_rtf": round(statistics.mean(rtf_all), 4),
            "rtf_stdev": round(statistics.stdev(rtf_all) if len(rtf_all) > 1 else 0.0, 4),
            "rtf_p95": round(percentile(rtf_all, 95), 4),
            "mean_utmos": round(statistics.mean(utmos_vals), 3) if utmos_vals else None,
            "peak_memory_gb": round(peak_mem, 2),
            "model_load_gb":  round(mem_load, 2),
            "samples": rows,
        }

        del pipe
        import gc; gc.collect()
        torch.cuda.empty_cache()
        return summary

    except Exception as e:
        print(f"    Error: {e}")
        return {"model": model_id, "model_key": model_key, "error": str(e)}


def bench_bark(model_key="bark"):
    import torch
    print(f"\n  Loading Bark...")
    try:
        from transformers import pipeline
        pipe = pipeline("text-to-speech", model="suno/bark",
                        dtype=torch.bfloat16, device_map="cuda")
        mem_load = torch.cuda.memory_allocated() / 1e9

        rows = []
        for i, text in enumerate(TEST_TEXTS[:3]):  # Bark is slow, fewer samples
            rtf_list = []
            for run in range(N_RUNS):
                t0 = time.perf_counter()
                output = pipe(text)
                elapsed = time.perf_counter() - t0
                audio = output["audio"].squeeze()
                sr    = output.get("sampling_rate", 24000)
                audio_sec = len(audio) / sr
                rtf_list.append(elapsed / audio_sec)
            rtf_mean = statistics.mean(rtf_list)
            rtf_std  = statistics.stdev(rtf_list) if len(rtf_list) > 1 else 0.0
            rows.append({"text_idx": i, "rtf_mean": round(rtf_mean, 4),
                          "rtf_stdev": round(rtf_std, 4),
                          "audio_sec": round(audio_sec, 3)})
            print(f"    [{i}] RTF={rtf_mean:.3f}±{rtf_std:.3f}")

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        rtf_all = [r["rtf_mean"] for r in rows]
        summary = {
            "model": "suno/bark", "model_key": model_key,
            "mean_rtf": round(statistics.mean(rtf_all), 4),
            "rtf_stdev": round(statistics.stdev(rtf_all) if len(rtf_all) > 1 else 0.0, 4),
            "rtf_p95": round(percentile(rtf_all, 95), 4),
            "peak_memory_gb": round(peak_mem, 2),
            "model_load_gb":  round(mem_load, 2),
            "samples": rows,
        }
        del pipe
        torch.cuda.empty_cache()
        return summary
    except Exception as e:
        return {"model": "suno/bark", "model_key": model_key, "error": str(e)}


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["all"],
                        choices=["qwen3-tts-base", "qwen3-tts-voice",
                                 "qwen3-tts-custom", "bark", "all"])
    parser.add_argument("--voice-ref", default=None,
                        help="Path to reference audio for CustomVoice cloning")
    args = parser.parse_args()

    model_configs = {
        "qwen3-tts-base":   "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "qwen3-tts-voice":  "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "qwen3-tts-custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "bark":             None,
    }

    keys = list(model_configs.keys()) if "all" in args.models else args.models
    all_results = []

    for key in keys:
        print(f"\n── {key.upper()} ──")
        if key == "bark":
            result = bench_bark(key)
        else:
            model_id = model_configs[key]
            ref = args.voice_ref if key == "qwen3-tts-custom" else None
            result = bench_qwen_tts(model_id, key, custom_voice_ref=ref)
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"  {'Model':<25} {'RTF':>6} {'UTMOS':>6} {'Mem':>6}")
    print(f"  {'-'*60}")
    for r in all_results:
        if "error" not in r:
            print(f"  {r.get('model_key',''):<25} "
                  f"{r.get('mean_rtf',0):>6.3f} "
                  f"{str(r.get('mean_utmos','?')):>6} "
                  f"{r.get('peak_memory_gb',0):>5.1f}G")
        else:
            print(f"  {r.get('model_key',''):<25} ERROR: {r.get('error','')[:35]}")

    out = RESULTS / f"tts_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
