#!/usr/bin/env python3
"""
Phase 2 — Full FT / LoRA / QLoRA Training
Covers T1-T13 (LLM experiments) and T18/T19 (Embedding/Reranker)
Usage:
  python3 phase2_train.py --experiment T3   # Qwen3-8B LoRA
  python3 phase2_train.py --experiment T1   # Qwen3-8B Full FT
  python3 phase2_train.py --experiment all  # Run all in order
"""
import os, sys, json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/training"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

EXPERIMENTS = {
    "T1":  {"model": "Qwen/Qwen3-8B",              "method": "full_ft",  "dataset": "tatsu-lab/alpaca"},
    "T2":  {"model": "mistralai/Mistral-7B-Instruct-v0.3", "method": "full_ft", "dataset": "tatsu-lab/alpaca"},
    "T3":  {"model": "Qwen/Qwen3-8B",              "method": "lora_r16", "dataset": "tatsu-lab/alpaca"},
    "T4":  {"model": "mistralai/Mistral-7B-Instruct-v0.3", "method": "lora_r16", "dataset": "tatsu-lab/alpaca"},
    "T5":  {"model": "Qwen/Qwen3-8B",              "method": "lora_r64", "dataset": "teknium/OpenHermes-2.5"},
    "T6":  {"model": "Qwen/Qwen3-32B",             "method": "lora_r32", "dataset": "tatsu-lab/alpaca"},
    "T7":  {"model": "Qwen/Qwen3-30B-A3B",         "method": "lora_r16", "dataset": "tatsu-lab/alpaca", "moe": True},
    "T8":  {"model": "Qwen/Qwen2.5-72B-Instruct",  "method": "qlora_nf4", "dataset": "tatsu-lab/alpaca"},
    "T9":  {"model": "Qwen/Qwen2.5-VL-7B-Instruct","method": "vlm_lora", "dataset": "liuhaotian/LLaVA-Instruct-150K"},
    "T10": {"model": "Qwen/Qwen2.5-VL-7B-Instruct","method": "vlm_lora", "dataset": "liuhaotian/LLaVA-Instruct-150K"},
    "T11": {"model": "deepseek-ai/DeepSeek-V2-Lite","method": "lora_r32", "dataset": "tatsu-lab/alpaca",
             "moe": True, "lora_targets": ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]},
    "T12": {"model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "method": "qlora_r16", "dataset": "tatsu-lab/alpaca", "moe": True},
    "T24": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "method": "lora_r16", "dataset": "tatsu-lab/alpaca"},
}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]

PEAK_TFLOPS = 67.0   # GB10 BF16


def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        parts = [x.strip() for x in r.stdout.strip().split(",")]
        def safe_float(s, default=0.0):
            try: return float(s)
            except: return default
        def safe_int(s, default=0):
            try: return int(float(s))
            except: return default
        p = safe_float(parts[0]) if len(parts) > 0 else 0.0
        t = safe_int(parts[1]) if len(parts) > 1 else 0
        m = safe_int(parts[2]) if len(parts) > 2 else 0
        return p, t, m
    except:
        return 0.0, 0, 0


def compute_train_mfu(n_params, tokens_per_sec, method="full_ft"):
    """MFU for training.
    Full FT: 6N FLOPs/token (2N fwd + 4N bwd/optimizer).
    LoRA/QLoRA: 2N FLOPs/token — frozen-layer backward is not computed; only
    the forward pass (2N) dominates. Using 6N would overstate compute and give
    MFU > 100%, which is meaningless.
    """
    try:
        flop_mul = 6 if method == "full_ft" else 2
        achieved_tflops = flop_mul * n_params * tokens_per_sec / 1e12
        return round(achieved_tflops, 3), round(achieved_tflops / PEAK_TFLOPS * 100, 2)
    except:
        return None, None


class ThroughputCallback:
    """Lightweight callback to record training throughput and hardware stats."""
    def __init__(self, batch_size, seq_len, n_params, grad_accum=1):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_params = n_params
        self.grad_accum = grad_accum
        self.step_times = []
        self.loss_curve = []
        self._t_step = None

    def on_step_begin(self):
        self._t_step = time.perf_counter()

    def on_step_end(self, logs=None):
        if self._t_step is not None:
            dt = time.perf_counter() - self._t_step
            self.step_times.append(dt)
            if logs and "loss" in logs:
                self.loss_curve.append(round(logs["loss"], 5))

    def summary(self, method="full_ft"):
        if not self.step_times:
            return {}
        import statistics
        # Each optimizer step processes batch_size * seq_len * grad_accum tokens
        tokens_per_step = self.batch_size * self.seq_len * self.grad_accum
        tps_list = [tokens_per_step / t for t in self.step_times[-50:]]
        tps_mean = statistics.mean(tps_list)
        tflops, mfu = compute_train_mfu(self.n_params, tps_mean, method=method)
        power, temp, _ = get_gpu_stats()
        return {
            "train_tokens_per_sec": round(tps_mean, 1),
            "train_tps_stdev": round(statistics.stdev(tps_list) if len(tps_list) > 1 else 0, 1),
            "train_mfu_tflops": tflops,
            "train_mfu_pct": mfu,
            "power_w": round(power, 1),
            "temp_c": temp,
            "loss_curve": self.loss_curve,
        }


def format_alpaca(example):
    if example.get("input", ""):
        text = (f"Below is an instruction that describes a task, paired with an input. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}")
    else:
        text = (f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}")
    return {"text": text}


def load_dataset_for_experiment(exp_cfg, max_samples=None):
    from datasets import load_dataset
    ds_name = exp_cfg["dataset"]
    method  = exp_cfg["method"]

    if "alpaca" in ds_name:
        ds = load_dataset(ds_name, split="train")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        ds = ds.map(format_alpaca, remove_columns=ds.column_names)
        return ds

    if "OpenHermes" in ds_name:
        ds = load_dataset(ds_name, split="train")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        def fmt_openhermes(ex):
            convs = ex.get("conversations", [])
            text = ""
            for c in convs:
                role = "Human" if c["from"] == "human" else "Assistant"
                text += f"{role}: {c['value']}\n"
            return {"text": text.strip()}
        ds = ds.map(fmt_openhermes, remove_columns=ds.column_names)
        return ds

    if "LLaVA" in ds_name:
        ds = load_dataset(ds_name, split="train")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        def fmt_llava(ex):
            convs = ex.get("conversations", [])
            text = ""
            for c in convs:
                role = "Human" if c["from"] == "human" else "Assistant"
                text += f"{role}: {c['value']}\n"
            return {"text": text.strip()}
        ds = ds.map(fmt_llava, remove_columns=ds.column_names)
        return ds

    raise ValueError(f"Unknown dataset: {ds_name}")


# ─────────────────────────────────────────────────────────────
def run_full_ft(exp_id, exp_cfg, args):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               TrainingArguments, Trainer,
                               DataCollatorForLanguageModeling)
    import bitsandbytes as bnb

    model_id = exp_cfg["model"]
    out_dir  = MODELS_DIR / f"{exp_id}_{model_id.split('/')[-1]}_full_ft"
    print(f"\n  Full FT: {model_id}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    m.gradient_checkpointing_enable()
    mem_load = torch.cuda.memory_allocated() / 1e9

    dataset = load_dataset_for_experiment(exp_cfg, max_samples=args.max_samples)

    def tokenize(example):
        return tok(example["text"], truncation=True, max_length=args.max_seq_len,
                   padding=False)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)

    n_params = sum(p.numel() for p in m.parameters())
    cb = ThroughputCallback(args.batch_size, args.max_seq_len, n_params,
                            grad_accum=args.grad_accum)
    optimizer = bnb.optim.AdamW8bit(m.parameters(), lr=2e-5)

    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            cb.on_step_begin()
        def on_step_end(self, args, state, control, **kw):
            logs = {k: v for k, v in (state.log_history[-1] if state.log_history else {}).items()}
            cb.on_step_end(logs)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1 if args.smoke_test else 3,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(
        model=m, args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        optimizers=(optimizer, None),
        callbacks=[_CB()],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    logs = trainer.state.log_history
    final_loss = next((x["loss"] for x in reversed(logs) if "loss" in x), None)

    result = {
        "experiment": exp_id, "model": model_id, "method": "full_ft",
        "elapsed_min": round(elapsed / 60, 1),
        "steps": trainer.state.global_step,
        "final_loss": final_loss,
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_gb":  round(mem_load, 2),
        "n_params": n_params,
        "output_dir": str(out_dir),
    }
    result.update(cb.summary(method="full_ft"))
    return result


def run_lora(exp_id, exp_cfg, args):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    model_id = exp_cfg["model"]
    method   = exp_cfg["method"]
    is_moe   = exp_cfg.get("moe", False)
    rank     = int(method.split("r")[-1]) if "r" in method else 16
    out_dir  = MODELS_DIR / f"{exp_id}_{model_id.split('/')[-1]}_lora_r{rank}"
    print(f"\n  LoRA r={rank}: {model_id}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    )
    mem_load = torch.cuda.memory_allocated() / 1e9

    lora_target = exp_cfg.get("lora_targets", LORA_TARGETS)
    if is_moe and "lora_targets" not in exp_cfg:
        lora_target = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=lora_target,
        lora_dropout=0.05,
        bias="none",
    )
    m = get_peft_model(m, lora_cfg)
    m.print_trainable_parameters()

    # Use PEFT's method — m.parameters() on PeftModel only returns adapter params
    n_params, n_params_all = m.get_nb_trainable_parameters()
    cb = ThroughputCallback(args.batch_size, args.max_seq_len, n_params_all,
                            grad_accum=args.grad_accum)

    from transformers import TrainerCallback
    _seen_lora = set()
    class _CB(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            cb.on_step_begin()
        def on_step_end(self, args, state, control, **kw):
            step = state.global_step
            if state.log_history and step not in _seen_lora:
                last = state.log_history[-1]
                if "loss" in last:
                    _seen_lora.add(step)
                    cb.on_step_end({"loss": float(last["loss"])})
                    return
            cb.on_step_end({})

    dataset = load_dataset_for_experiment(exp_cfg, max_samples=args.max_samples)

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=1 if args.smoke_test else 3,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        max_length=args.max_seq_len,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=m, args=training_args,
        train_dataset=dataset,
        processing_class=tok,
        callbacks=[_CB()],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    trainer.model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    logs = trainer.state.log_history
    final_loss = next((x["loss"] for x in reversed(logs) if "loss" in x), None)

    result = {
        "experiment": exp_id, "model": model_id, "method": f"lora_r{rank}",
        "elapsed_min": round(elapsed / 60, 1),
        "steps": trainer.state.global_step,
        "final_loss": final_loss,
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_gb":  round(mem_load, 2),
        "n_params": n_params_all,
        "n_params_trainable": n_params,
        "output_dir": str(out_dir),
    }
    result.update(cb.summary(method="lora"))
    return result


def run_qlora(exp_id, exp_cfg, args):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    model_id = exp_cfg["model"]
    out_dir  = MODELS_DIR / f"{exp_id}_{model_id.split('/')[-1]}_qlora"
    print(f"\n  QLoRA NF4: {model_id}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_cfg,
        device_map={"": 0}, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    m = prepare_model_for_kbit_training(m)
    mem_load = torch.cuda.memory_allocated() / 1e9

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05, bias="none",
    )
    m = get_peft_model(m, lora_cfg)
    m.print_trainable_parameters()

    n_params, n_params_all = m.get_nb_trainable_parameters()
    bs_qlora = max(1, args.batch_size // 2)
    ga_qlora  = args.grad_accum * 2
    cb = ThroughputCallback(bs_qlora, min(args.max_seq_len, 2048), n_params_all,
                            grad_accum=ga_qlora)

    from transformers import TrainerCallback
    class _CB(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            cb.on_step_begin()
        def on_step_end(self, args, state, control, **kw):
            logs = {k: v for k, v in (state.log_history[-1] if state.log_history else {}).items()}
            cb.on_step_end(logs)

    dataset = load_dataset_for_experiment(exp_cfg, max_samples=args.max_samples)

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=1 if args.smoke_test else 3,
        max_steps=args.max_steps,
        per_device_train_batch_size=bs_qlora,
        gradient_accumulation_steps=args.grad_accum * 2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        max_length=min(args.max_seq_len, 2048),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=m, args=training_args,
        train_dataset=dataset,
        processing_class=tok,
        callbacks=[_CB()],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    trainer.model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    logs = trainer.state.log_history
    final_loss = next((x["loss"] for x in reversed(logs) if "loss" in x), None)

    result = {
        "experiment": exp_id, "model": model_id, "method": "qlora_nf4",
        "elapsed_min": round(elapsed / 60, 1),
        "steps": trainer.state.global_step,
        "final_loss": final_loss,
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_gb":  round(mem_load, 2),
        "n_params_trainable": n_params,
        "output_dir": str(out_dir),
    }
    result.update(cb.summary(method="qlora"))
    return result


# ─────────────────────────────────────────────────────────────
def run_experiment(exp_id, args):
    if exp_id not in EXPERIMENTS:
        print(f"  Unknown experiment: {exp_id}")
        return None

    exp_cfg = EXPERIMENTS[exp_id]
    method  = exp_cfg["method"]

    import torch
    torch.cuda.empty_cache()

    try:
        if method == "full_ft":
            result = run_full_ft(exp_id, exp_cfg, args)
        elif method.startswith("lora"):
            result = run_lora(exp_id, exp_cfg, args)
        elif method.startswith("qlora"):
            result = run_qlora(exp_id, exp_cfg, args)
        elif method == "vlm_lora":
            result = run_lora(exp_id, exp_cfg, args)
        else:
            result = {"experiment": exp_id, "error": f"Unknown method: {method}"}
    except Exception as e:
        result = {"experiment": exp_id, "model": exp_cfg["model"],
                  "method": method, "error": str(e)}
        print(f"  ERROR in {exp_id}: {e}")

    result["timestamp"] = datetime.now().isoformat()
    out = RESULTS / f"{exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    if "error" not in result:
        print(f"  {exp_id}: loss={result.get('final_loss'):.4f}  "
              f"time={result.get('elapsed_min')}min  "
              f"mem={result.get('peak_memory_gb')}GB")

    import gc, torch; gc.collect(); torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────
def main():
    # Order: cheapest experiments first
    default_order = ["T3", "T4", "T24", "T1", "T8", "T9", "T10",
                     "T11", "T6", "T7", "T12", "T2", "T5"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="T3",
                        help="Experiment ID (T1-T12,T24) or 'all'")
    parser.add_argument("--batch-size", type=int, default=2, dest="batch_size")
    parser.add_argument("--grad-accum", type=int, default=8, dest="grad_accum")
    parser.add_argument("--max-seq-len", type=int, default=2048, dest="max_seq_len")
    parser.add_argument("--max-steps", type=int, default=-1, dest="max_steps")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                        help="Run 20 steps only to verify setup")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_steps = 20
        args.max_samples = args.max_samples or 200

    if args.experiment == "all":
        experiments = default_order
    else:
        experiments = [args.experiment]

    all_results = []
    for exp_id in experiments:
        print(f"\n{'='*60}")
        print(f"  Running Experiment {exp_id}")
        print(f"{'='*60}")
        result = run_experiment(exp_id, args)
        if result:
            all_results.append(result)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  {'ID':<6} {'Model':<30} {'Method':<12} {'Loss':>7} {'Time':>8} {'Mem':>6}")
    print(f"  {'-'*70}")
    for r in all_results:
        if "error" not in r:
            print(f"  {r.get('experiment',''):<6} "
                  f"{str(r.get('model',''))[-30:]:<30} "
                  f"{r.get('method',''):<12} "
                  f"{r.get('final_loss',0):>7.4f} "
                  f"{r.get('elapsed_min',0):>7.1f}m "
                  f"{r.get('peak_memory_gb',0):>5.1f}G")
        else:
            print(f"  {r.get('experiment',''):<6} ERROR: {r.get('error','')[:50]}")


if __name__ == "__main__":
    main()
