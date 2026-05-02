#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into the base model to produce a full-weight checkpoint.
Usage: python3 merge_lora_to_full.py --adapter <path> --output <path>
"""
import argparse, json, torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to PEFT adapter dir")
    parser.add_argument("--output",  required=True, help="Output dir for merged model")
    parser.add_argument("--base",    default=None,  help="Override base model ID")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    adapter_cfg = json.loads(Path(args.adapter, "adapter_config.json").read_text())
    base_model_id = args.base or adapter_cfg["base_model_name_or_path"]

    print(f"  Base model: {base_model_id}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Output: {args.output}")

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    m = PeftModel.from_pretrained(m, args.adapter)
    print("  Merging LoRA weights...")
    m = m.merge_and_unload()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"  Saved merged model to: {out}")

if __name__ == "__main__":
    main()
