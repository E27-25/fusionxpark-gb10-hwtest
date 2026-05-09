#!/usr/bin/env python3
"""
Phase 2 — DPO Training (T20: Qwen3-8B, T22: Qwen3-32B)
Dataset: HuggingFaceH4/ultrafeedback_binarized
Metrics: loss, chosen/rejected reward margin, memory, time
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
    "T20": {
        "model": "Qwen/Qwen3-8B",
        "sft_checkpoint": None,  # set to T3 output dir if available
        "lr": 5e-7, "beta": 0.1,
        "batch_size": 2, "grad_accum": 8,
    },
    "T22": {
        "model": "Qwen/Qwen3-32B",
        "sft_checkpoint": None,  # set to T6 output dir if available
        "lr": 5e-7, "beta": 0.1,
        "batch_size": 1, "grad_accum": 16,
    },
}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        parts = [x.strip() for x in r.stdout.strip().split(",")]
        def _f(s, d=0.0):
            try: return float(s)
            except: return d
        def _i(s, d=0):
            try: return int(float(s))
            except: return d
        return (_f(parts[0]) if parts else 0.0,
                _i(parts[1]) if len(parts) > 1 else 0,
                _i(parts[2]) if len(parts) > 2 else 0)
    except:
        return 0.0, 0, 0


# ─────────────────────────────────────────────────────────────
def run_dpo(exp_id, args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset

    exp_cfg  = EXPERIMENTS[exp_id]
    sft_ckpt = exp_cfg.get("sft_checkpoint")
    base_id  = exp_cfg["model"]
    out_dir  = MODELS_DIR / f"{exp_id}_{base_id.split('/')[-1]}_dpo"

    # Detect if sft_checkpoint is a PEFT adapter (has adapter_config.json)
    is_peft_adapter = (sft_ckpt and (Path(sft_ckpt) / "adapter_config.json").exists())
    model_id = base_id if is_peft_adapter else (sft_ckpt or base_id)

    print(f"\n  DPO {exp_id}: base={base_id}")
    if sft_ckpt:
        print(f"  SFT checkpoint: {sft_ckpt} ({'PEFT adapter' if is_peft_adapter else 'full model'})")
    print(f"  Beta={exp_cfg['beta']}, LR={exp_cfg['lr']}")

    tok = AutoTokenizer.from_pretrained(sft_ckpt if sft_ckpt else base_id,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )

    # If SFT checkpoint is a PEFT adapter, merge it into the base model first
    if is_peft_adapter:
        from peft import PeftModel
        print("  Loading and merging SFT PEFT adapter...")
        m = PeftModel.from_pretrained(m, sft_ckpt)
        m = m.merge_and_unload()
        print("  Adapter merged into base model")

    mem_load = torch.cuda.memory_allocated() / 1e9

    # LoRA for all DPO experiments — saves memory for both policy model gradients and optimizer state
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05, bias="none",
    )
    m = get_peft_model(m, lora_cfg)
    m.print_trainable_parameters()

    print("  Loading dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # DPOTrainer expects: prompt, chosen, rejected columns
    def format_dpo(ex):
        return {
            "prompt": ex["prompt"],
            "chosen": ex["chosen"][-1]["content"] if isinstance(ex["chosen"], list) else ex["chosen"],
            "rejected": ex["rejected"][-1]["content"] if isinstance(ex["rejected"], list) else ex["rejected"],
        }
    ds = ds.map(format_dpo)

    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=1 if args.smoke_test else 1,
        max_steps=args.max_steps,
        per_device_train_batch_size=exp_cfg["batch_size"],
        gradient_accumulation_steps=exp_cfg["grad_accum"],
        learning_rate=exp_cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=exp_cfg["beta"],
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        max_length=1024,
    )

    trainer = DPOTrainer(
        model=m,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tok,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    logs = trainer.state.log_history
    final_loss = next((x.get("train_loss") or x.get("loss")
                       for x in reversed(logs) if "loss" in str(x)), None)
    # Extract reward margins if logged
    reward_margin = next(
        (x.get("rewards/margins") for x in reversed(logs) if "rewards/margins" in x),
        None
    )

    result = {
        "experiment": exp_id,
        "model": exp_cfg["model"],
        "method": "dpo",
        "beta": exp_cfg["beta"],
        "elapsed_min": round(elapsed / 60, 1),
        "steps": trainer.state.global_step,
        "final_loss": final_loss,
        "reward_margin": reward_margin,
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_gb": round(mem_load, 2),
        "output_dir": str(out_dir),
        "timestamp": datetime.now().isoformat(),
    }

    out = RESULTS / f"{exp_id}_dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    print(f"  {exp_id}: loss={final_loss}  margin={reward_margin}  "
          f"time={result['elapsed_min']}min  mem={peak_mem:.1f}GB")

    import gc; gc.collect(); torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="T20", choices=["T20", "T22", "all"])
    parser.add_argument("--max-steps", type=int, default=-1, dest="max_steps")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test")
    # Allow overriding SFT checkpoint
    parser.add_argument("--sft-checkpoint", default=None, dest="sft_checkpoint",
                        help="Path to SFT checkpoint (e.g., T3 or T6 output)")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_steps = 20
        args.max_samples = args.max_samples or 200

    if args.sft_checkpoint:
        for exp_id in EXPERIMENTS:
            EXPERIMENTS[exp_id]["sft_checkpoint"] = args.sft_checkpoint

    experiments = (["T20", "T22"] if args.experiment == "all"
                   else [args.experiment])

    all_results = []
    for exp_id in experiments:
        print(f"\n{'='*60}")
        print(f"  DPO Experiment {exp_id}")
        result = run_dpo(exp_id, args)
        all_results.append(result)

    print(f"\n{'='*60}")
    for r in all_results:
        print(f"  {r.get('experiment')}: loss={r.get('final_loss')}  "
              f"margin={r.get('reward_margin')}  "
              f"time={r.get('elapsed_min')}min")


if __name__ == "__main__":
    main()
