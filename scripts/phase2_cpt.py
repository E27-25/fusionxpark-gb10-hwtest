#!/usr/bin/env python3
"""
Phase 2 — Continued Pre-Training (T23: Qwen3-8B on OpenWebText)
Large-batch, long-sequence pre-training on raw text
"""
import json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/training"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run_cpt(args):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               TrainingArguments, DataCollatorForLanguageModeling,
                               Trainer)
    from datasets import load_dataset
    import bitsandbytes as bnb

    model_id = "Qwen/Qwen3-8B"
    out_dir  = MODELS_DIR / f"T23_{model_id.split('/')[-1]}_cpt"

    print(f"\n  CPT T23: {model_id}")
    print(f"  Dataset: openwebtext, LR=1e-4, seq_len={args.max_seq_len}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    m.gradient_checkpointing_enable()
    mem_load = torch.cuda.memory_allocated() / 1e9

    print("  Loading OpenWebText (streaming)...")
    ds = load_dataset("Skylion007/openwebtext", split="train",
                      streaming=not args.download)

    def tokenize_fn(examples):
        return tok(examples["text"], truncation=True,
                   max_length=args.max_seq_len, padding=False)

    if args.download:
        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
    else:
        # Streaming: take a subset
        n = args.max_samples or 50000
        ds = ds.take(n)
        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        from datasets import Dataset
        ds = Dataset.from_generator(lambda: ds)

    optimizer = bnb.optim.AdamW8bit(m.parameters(), lr=args.lr)
    collator  = DataCollatorForLanguageModeling(tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.max_steps if args.max_steps > 0 else 5000,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=m, args=training_args,
        train_dataset=ds,
        data_collator=collator,
        optimizers=(optimizer, None),
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    logs = trainer.state.log_history
    losses = [x.get("loss") for x in logs if "loss" in x]

    result = {
        "experiment": "T23",
        "model": model_id,
        "method": "cpt",
        "dataset": "openwebtext",
        "elapsed_min": round(elapsed / 60, 1),
        "steps": trainer.state.global_step,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_curve": losses[::max(1, len(losses)//20)],  # 20 checkpoints
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_gb": round(mem_load, 2),
        "output_dir": str(out_dir),
        "timestamp": datetime.now().isoformat(),
    }

    out = RESULTS / f"T23_cpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    print(f"  T23: init_loss={result['initial_loss']:.4f}  "
          f"final_loss={result['final_loss']:.4f}  "
          f"time={result['elapsed_min']}min  mem={peak_mem:.1f}GB")

    import gc; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5000, dest="max_steps")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--batch-size", type=int, default=2, dest="batch_size")
    parser.add_argument("--grad-accum", type=int, default=16, dest="grad_accum")
    parser.add_argument("--max-seq-len", type=int, default=4096, dest="max_seq_len")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--download", action="store_true",
                        help="Download full dataset (slow) instead of streaming")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_steps = 20
        args.max_samples = 500
        args.max_seq_len = 1024

    run_cpt(args)


if __name__ == "__main__":
    main()
