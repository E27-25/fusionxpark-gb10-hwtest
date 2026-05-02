#!/usr/bin/env python3
"""
Phase 2 — GRPO Training (T21: Qwen3-8B, GSM8K math)
Group Relative Policy Optimization with rule-based math rewards
Group size G=8, clip ratio eps=0.2
"""
import json, time, re, argparse, subprocess
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/training"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
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
                _i(parts[1]) if len(parts) > 1 else 0)
    except:
        return 0.0, 0


def extract_gsm8k_answer(text):
    """Extract the final numeric answer from GSM8K-style response."""
    # Look for #### answer pattern (GSM8K standard)
    match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1).strip()
    # Fallback: last number in response
    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def math_correctness_reward(completions, solution, **kwargs):
    """
    Rule-based reward: 1.0 if final answer matches ground truth, 0.0 otherwise.
    TRL calls reward functions with completions and solution both as lists of length B*G.
    """
    rewards = []
    for comp, sol in zip(completions, solution):
        ref = extract_gsm8k_answer(sol)
        pred = extract_gsm8k_answer(comp)
        if pred is not None and ref is not None and pred == ref:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    """Bonus reward for following #### format."""
    rewards = []
    for comp in completions:
        if "####" in comp:
            rewards.append(0.1)
        else:
            rewards.append(-0.05)
    return rewards


# ─────────────────────────────────────────────────────────────
def run_grpo(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset

    model_id = "Qwen/Qwen3-8B"
    out_dir  = MODELS_DIR / f"T21_{model_id.split('/')[-1]}_grpo"

    print(f"\n  GRPO T21: {model_id}")
    print(f"  Group size=8, eps=0.2, LR=1e-6")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    mem_load = torch.cuda.memory_allocated() / 1e9

    # LoRA to fit training in memory
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05, bias="none",
    )
    m = get_peft_model(m, lora_cfg)
    m.print_trainable_parameters()

    print("  Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def format_gsm8k(ex):
        return {
            "prompt": (f"Solve this math problem step by step. "
                       f"End with '#### <answer>' on the last line.\n\n"
                       f"Problem: {ex['question']}\n\nSolution:"),
            "solution": ex["answer"],
        }
    ds = ds.map(format_gsm8k)

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        # GRPO-specific
        num_generations=8,          # group size G
        max_completion_length=512,
        temperature=0.7,
        epsilon=0.2,                # clip ratio
        beta=0.001,                 # KL penalty (light for GRPO)
    )

    trainer = GRPOTrainer(
        model=m,
        args=grpo_config,
        train_dataset=ds,
        processing_class=tok,
        reward_funcs=[math_correctness_reward, format_reward],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    logs = trainer.state.log_history
    final_loss   = next((x.get("loss") for x in reversed(logs) if "loss" in x), None)
    mean_reward  = next((x.get("reward") for x in reversed(logs) if "reward" in x), None)

    result = {
        "experiment": "T21",
        "model": model_id,
        "method": "grpo",
        "group_size": 8,
        "epsilon": 0.2,
        "elapsed_min": round(elapsed / 60, 1),
        "steps": trainer.state.global_step,
        "final_loss": final_loss,
        "mean_reward": mean_reward,
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_gb": round(mem_load, 2),
        "output_dir": str(out_dir),
        "timestamp": datetime.now().isoformat(),
    }

    out = RESULTS / f"T21_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    print(f"  T21: loss={final_loss}  reward={mean_reward}  "
          f"time={result['elapsed_min']}min  mem={peak_mem:.1f}GB")

    import gc; gc.collect(); torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1, dest="max_steps")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_steps = 20
        args.max_samples = args.max_samples or 100

    run_grpo(args)


if __name__ == "__main__":
    main()
