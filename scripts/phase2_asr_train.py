#!/usr/bin/env python3
"""
Phase 2 — ASR Fine-tuning (T14: Whisper Full FT, T15: Whisper LoRA, T16: Qwen2-Audio LoRA)
Dataset: mozilla-foundation/common_voice_17_0 (en subset)
"""
import json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/training"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

EXPERIMENTS = {
    "T14": {"model": "openai/whisper-large-v3", "method": "full_ft"},
    "T15": {"model": "openai/whisper-large-v3", "method": "lora"},
    "T16": {"model": "Qwen/Qwen2-Audio-7B-Instruct", "method": "lora"},
}


def load_asr_dataset(max_samples=None):
    """Load LibriSpeech train.clean.100 with soundfile decoding (avoids torchcodec)."""
    from datasets import load_dataset, Audio
    import soundfile as sf, io, numpy as np
    print("  Loading LibriSpeech train.clean.100...")
    ds = load_dataset("librispeech_asr", "clean", split="train.100")
    ds = ds.cast_column("audio", Audio(decode=False))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Manually decode audio bytes using soundfile (no torchcodec needed)
    decoded_samples = []
    for row in ds:
        raw = row["audio"]
        try:
            if raw.get("bytes"):
                arr, sr = sf.read(io.BytesIO(raw["bytes"]))
            else:
                arr, sr = sf.read(raw["path"])
            arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)  # stereo → mono
            decoded_samples.append({
                "audio": {"array": arr, "sampling_rate": sr},
                "text": row["text"],
            })
        except Exception as e:
            pass  # skip undecodable samples

    from datasets import Dataset
    return Dataset.from_list(decoded_samples)


def prepare_whisper_dataset(ds, processor, max_input_length=480000):
    def preprocess(examples):
        audio = [a["array"] for a in examples["audio"]]
        inputs = processor(audio, sampling_rate=16000, return_tensors="np")
        examples["input_features"] = inputs.input_features
        labels = processor.tokenizer(
            examples["text"], truncation=True, max_length=128
        )
        examples["labels"] = labels["input_ids"]
        return examples

    ds = ds.map(preprocess, batched=True, batch_size=8,
                remove_columns=ds.column_names)
    return ds


# ─────────────────────────────────────────────────────────────
def run_whisper_full_ft(args):
    import torch
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                               Seq2SeqTrainingArguments, Seq2SeqTrainer,
                               WhisperTokenizer)
    import evaluate
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    model_id = EXPERIMENTS["T14"]["model"]
    out_dir  = MODELS_DIR / f"T14_{model_id.split('/')[-1]}_full_ft"

    print(f"\n  T14: Whisper Full FT ({model_id})")
    processor = WhisperProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    mem_load = torch.cuda.memory_allocated() / 1e9

    ds = load_asr_dataset(max_samples=args.max_samples)
    ds = prepare_whisper_dataset(ds, processor)

    wer_metric = evaluate.load("wer")

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str  = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1 if args.smoke_test else 3,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-5,
        warmup_steps=100,
        bf16=True,
        predict_with_generate=False,
        logging_steps=25,
        eval_strategy="no",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    trainer.save_model(str(out_dir))

    logs = trainer.state.log_history
    final_loss = next((x.get("loss") for x in reversed(logs) if "loss" in x), None)
    final_wer  = next((x.get("eval_wer") for x in reversed(logs) if "eval_wer" in x), None)

    result = {
        "experiment": "T14",
        "model": model_id, "method": "full_ft",
        "elapsed_min": round(elapsed / 60, 1),
        "steps": trainer.state.global_step,
        "final_loss": final_loss, "final_wer": final_wer,
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_gb": round(mem_load, 2),
        "output_dir": str(out_dir),
        "timestamp": datetime.now().isoformat(),
    }
    out = RESULTS / f"T14_whisper_full_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  T14: loss={final_loss}  WER={final_wer}  "
          f"time={result['elapsed_min']}min  mem={peak_mem:.1f}GB  Saved: {out}")

    import gc; gc.collect(); torch.cuda.empty_cache()
    return result


def run_whisper_lora(args):
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    model_id = EXPERIMENTS["T15"]["model"]
    out_dir  = MODELS_DIR / f"T15_{model_id.split('/')[-1]}_lora"
    print(f"\n  T15: Whisper LoRA ({model_id})")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj",
                        "fc1", "fc2"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    mem_load = torch.cuda.memory_allocated() / 1e9

    ds = load_asr_dataset(max_samples=args.max_samples)
    ds = prepare_whisper_dataset(ds, processor)

    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorSpeech:
        processor: Any
        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1 if args.smoke_test else 3,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-4,
        warmup_steps=100,
        bf16=True,
        predict_with_generate=False,
        logging_steps=25,
        eval_strategy="no",
        save_steps=500, save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorSpeech(processor=processor),
        processing_class=processor.feature_extractor,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    model.save_pretrained(str(out_dir))

    logs = trainer.state.log_history
    final_loss = next((x.get("loss") for x in reversed(logs) if "loss" in x), None)

    result = {
        "experiment": "T15",
        "model": model_id, "method": "lora_r16",
        "elapsed_min": round(elapsed / 60, 1),
        "steps": trainer.state.global_step,
        "final_loss": final_loss,
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_gb": round(mem_load, 2),
        "output_dir": str(out_dir),
        "timestamp": datetime.now().isoformat(),
    }
    out = RESULTS / f"T15_whisper_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  T15: loss={final_loss}  time={result['elapsed_min']}min  "
          f"mem={peak_mem:.1f}GB  Saved: {out}")

    import gc; gc.collect(); torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="T15",
                        choices=["T14", "T15", "T16", "all"])
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    parser.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    parser.add_argument("--max-steps", type=int, default=-1, dest="max_steps")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_steps = 20
        args.max_samples = 200

    exps = {"T14": run_whisper_full_ft, "T15": run_whisper_lora}

    if args.experiment == "all":
        for fn in [run_whisper_lora, run_whisper_full_ft]:
            fn(args)
    elif args.experiment in exps:
        exps[args.experiment](args)
    else:
        print(f"Experiment {args.experiment} not yet implemented")


if __name__ == "__main__":
    main()
