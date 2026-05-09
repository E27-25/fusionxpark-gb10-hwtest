#!/usr/bin/env python3
"""
Phase 3 — ASR Evaluation: WER before/after fine-tuning
Compares base Whisper vs T14 (full FT) vs T15 (LoRA) on LibriSpeech test-clean
"""
import json, time, argparse
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/evaluation"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE / "models"

EVAL_CONFIGS = {
    "whisper-base":  "openai/whisper-large-v3",
    "whisper-T14":   None,  # set to T14 output dir
    "whisper-T15":   None,  # set to T15 output dir
}


def compute_wer(references, hypotheses):
    try:
        import jiwer
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        return round(wer * 100, 2), round(cer * 100, 2)
    except Exception as e:
        return None, None


def evaluate_whisper(model_path, label, n_samples=500):
    import torch
    from transformers import pipeline
    from datasets import load_dataset

    print(f"\n  Evaluating: {label} ({model_path})")
    try:
        pipe = pipeline(
            "automatic-speech-recognition", model=model_path,
            dtype=torch.bfloat16, device_map="cuda",
            chunk_length_s=30,
        )
        ds = load_dataset("librispeech_asr", "clean",
                          split=f"test[:{n_samples}]", trust_remote_code=True)

        refs, hyps, rtf_list = [], [], []
        for i, sample in enumerate(ds):
            audio    = sample["audio"]
            ref      = sample["text"].lower().strip()
            audio_s  = len(audio["array"]) / audio["sampling_rate"]

            audio_input = {"raw": audio["array"], "sampling_rate": audio["sampling_rate"]}
            t0 = time.perf_counter()
            result = pipe(audio_input, return_timestamps=False)
            elapsed = time.perf_counter() - t0

            hyp = result["text"].lower().strip()
            refs.append(ref)
            hyps.append(hyp)
            rtf_list.append(elapsed / audio_s)

            if (i + 1) % 100 == 0:
                partial_wer, _ = compute_wer(refs, hyps)
                print(f"    {i+1}/{n_samples}  WER={partial_wer}%  "
                      f"RTF={sum(rtf_list)/len(rtf_list):.3f}")

        wer, cer = compute_wer(refs, hyps)
        mean_rtf = sum(rtf_list) / len(rtf_list)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        result = {
            "label": label,
            "model": str(model_path),
            "n_samples": n_samples,
            "wer_pct": wer,
            "cer_pct": cer,
            "mean_rtf": round(mean_rtf, 4),
            "peak_memory_gb": round(peak_mem, 2),
            "timestamp": datetime.now().isoformat(),
        }
        print(f"  {label}: WER={wer}%  CER={cer}%  RTF={mean_rtf:.3f}  "
              f"Mem={peak_mem:.1f}GB")

        del pipe
        torch.cuda.empty_cache()
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"label": label, "model": str(model_path), "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["base", "T14", "T15", "all"], default=["all"])
    parser.add_argument("--n-samples", type=int, default=500, dest="n_samples")
    args = parser.parse_args()

    # Auto-detect fine-tuned checkpoints
    for exp_id, dir_prefix in [("T14", "T14_whisper"), ("T15", "T15_whisper")]:
        for d in MODELS_DIR.glob(f"{dir_prefix}*"):
            if d.is_dir():
                key = f"whisper-{exp_id}"
                EVAL_CONFIGS[key] = str(d)
                print(f"  Found {exp_id} checkpoint: {d}")
                break

    configs_to_run = []
    if "all" in args.models:
        configs_to_run = list(EVAL_CONFIGS.items())
    else:
        key_map = {"base": "whisper-base", "T14": "whisper-T14", "T15": "whisper-T15"}
        configs_to_run = [(key_map[m], EVAL_CONFIGS[key_map[m]]) for m in args.models]

    all_results = []
    for label, model_path in configs_to_run:
        if model_path is None:
            print(f"  Skipping {label}: no checkpoint found")
            continue
        result = evaluate_whisper(model_path, label, args.n_samples)
        all_results.append(result)

    print(f"\n{'='*60}")
    print("  ASR EVALUATION RESULTS (LibriSpeech test-clean)")
    print(f"  {'Model':<20} {'WER%':>7} {'CER%':>7} {'RTF':>7}")
    print(f"  {'-'*60}")
    for r in all_results:
        if "error" not in r:
            print(f"  {r['label']:<20} "
                  f"{str(r.get('wer_pct','?')):>7} "
                  f"{str(r.get('cer_pct','?')):>7} "
                  f"{r.get('mean_rtf',0):>7.3f}")

    out = RESULTS / f"asr_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
