"""
Quick Qwen3-8B think vs no_think accuracy comparison on GSM8K (50 samples).
Saves results to results/evaluation/think_vs_nothink.json
"""
import json, re, time
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-8B"
N_SAMPLES = 50
MAX_NEW_TOKENS_THINK = 2048
MAX_NEW_TOKENS_NOTHINK = 512
OUT = Path("/home/student/Desktop/Test/results/evaluation/think_vs_nothink.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def extract_answer(text):
    # Pull last number from model output (GSM8K style)
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None

def extract_gt(solution):
    # GSM8K ground truth is after ####
    m = re.search(r"####\s*([-\d,]+)", solution)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"\d+", solution)
    return nums[-1] if nums else None

print("Loading dataset...")
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N_SAMPLES))

print(f"Loading model {MODEL_ID}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True,
)
model.eval()

results = {"think": [], "nothink": []}

for mode in ["nothink", "think"]:
    print(f"\n=== Mode: {mode} ===")
    correct = 0
    think_token_counts = []
    t_start = time.time()

    for i, ex in enumerate(ds):
        question = ex["question"]
        gt = extract_gt(ex["answer"])

        if mode == "think":
            sys_content = "You are a helpful assistant."
            # Qwen3 thinking: add /think in user turn
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": question + "\n/think"},
            ]
            max_new = MAX_NEW_TOKENS_THINK
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. /no_think"},
                {"role": "user", "content": question},
            ]
            max_new = MAX_NEW_TOKENS_NOTHINK

        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tok(text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )

        generated = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Count thinking tokens (text inside <think>...</think>)
        think_match = re.search(r"<think>(.*?)</think>", generated, re.DOTALL)
        think_text = think_match.group(1) if think_match else ""
        think_toks = len(tok.encode(think_text)) if think_text else 0
        think_token_counts.append(think_toks)

        # Extract answer (after </think> if present, else full output)
        answer_text = generated[think_match.end():] if think_match else generated
        pred = extract_answer(answer_text)

        is_correct = (pred == gt) if pred and gt else False
        if is_correct:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N_SAMPLES}] acc={correct/(i+1):.2f} avg_think_toks={sum(think_token_counts)/(i+1):.0f}")

    elapsed = time.time() - t_start
    acc = correct / N_SAMPLES
    avg_think = sum(think_token_counts) / N_SAMPLES
    print(f"  Final: acc={acc:.3f}, avg_think_tokens={avg_think:.0f}, time={elapsed:.0f}s")

    results[mode] = {
        "accuracy": acc,
        "correct": correct,
        "total": N_SAMPLES,
        "avg_think_tokens": avg_think,
        "elapsed_sec": elapsed,
    }

print("\n=== Summary ===")
print(f"No-think:  acc={results['nothink']['accuracy']:.3f}")
print(f"Think:     acc={results['think']['accuracy']:.3f}")
print(f"Delta:     {results['think']['accuracy'] - results['nothink']['accuracy']:+.3f}")
print(f"Think tokens (think mode): {results['think']['avg_think_tokens']:.0f} avg")

results["model"] = MODEL_ID
results["n_samples"] = N_SAMPLES
results["dataset"] = "openai/gsm8k test"
OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUT}")
