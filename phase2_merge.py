#!/usr/bin/env python3
"""
Phase 2 — Model Merging (M1, M2, M3) without mergekit
Uses pure PyTorch SLERP / TIES for model weight interpolation.

M1: SLERP(t=0.5) — Qwen3-8B-base + T3-SFT → blend instruction + base
M2: TIES merge — T3-SFT + T20-DPO → combine SFT and preference-aligned
M3: DARE+TIES — Qwen3-8B + Qwen2.5-Coder-7B → general + code specialist
"""
import json, time, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

BASE    = Path("/home/student/Desktop/Test")
MODELS  = BASE / "models"
RESULTS = BASE / "results/training"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)


# ─── SLERP ───────────────────────────────────────────────────────────────────

def slerp_tensor(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    orig_shape = v0.shape
    v0f = v0.float().flatten()
    v1f = v1.float().flatten()
    n0 = F.normalize(v0f, dim=0)
    n1 = F.normalize(v1f, dim=0)
    dot = torch.clamp((n0 * n1).sum(), -1.0, 1.0)
    omega = torch.acos(dot.abs())  # abs to handle anti-parallel
    sin_omega = torch.sin(omega)
    if sin_omega.item() < 1e-6:
        # Nearly parallel — fall back to linear
        return ((1 - t) * v0f + t * v1f).reshape(orig_shape).to(v0.dtype)
    # If dot is negative, flip v1 so we take the short arc
    if dot.item() < 0:
        v1f = -v1f
    result = (torch.sin((1 - t) * omega) / sin_omega) * v0f + \
             (torch.sin(t * omega) / sin_omega) * v1f
    return result.reshape(orig_shape).to(v0.dtype)


def slerp_models(state_a: dict, state_b: dict, t: float = 0.5) -> dict:
    """SLERP between two model state dicts at interpolation factor t."""
    merged = {}
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    for k in keys_a & keys_b:
        pa, pb = state_a[k], state_b[k]
        if pa.dtype in (torch.float32, torch.float16, torch.bfloat16):
            merged[k] = slerp_tensor(pa, pb, t)
        else:
            merged[k] = pa  # non-float params (int, bool) — keep from model A
    for k in keys_a - keys_b:
        merged[k] = state_a[k]
    for k in keys_b - keys_a:
        merged[k] = state_b[k]
    return merged


# ─── TIES merge ──────────────────────────────────────────────────────────────

def ties_merge(state_base: dict, state_a: dict, state_b: dict,
               density: float = 0.5, weight_a: float = 0.5, weight_b: float = 0.5) -> dict:
    """
    TIES (Task-specific Intervention Ensemble) merge:
    - Compute task vectors: delta_a = state_a - base, delta_b = state_b - base
    - Prune low-magnitude deltas (keep top `density` fraction)
    - Resolve sign conflicts by majority vote
    - Scale and add back to base
    """
    merged = {}
    for k in state_base:
        pb = state_base[k]
        if pb.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            merged[k] = pb
            continue
        pa_a = state_a.get(k, pb)
        pa_b = state_b.get(k, pb)

        delta_a = (pa_a - pb).float()
        delta_b = (pa_b - pb).float()

        # Prune: keep top `density` fraction by magnitude for each delta
        def prune(delta, d):
            if delta.numel() == 0:
                return delta
            threshold = delta.abs().flatten().kthvalue(
                max(1, int((1 - d) * delta.numel()))).values
            return torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))

        da = prune(delta_a, density)
        db = prune(delta_b, density)

        # Sign resolution: where signs conflict, use sign of larger magnitude
        sign_a = da.sign()
        sign_b = db.sign()
        # Combined sign = sign of sum if both nonzero; else whichever is nonzero
        combined = da + db
        sign_combined = combined.sign()
        # Zero out parameters where signs conflict
        mask = (sign_a * sign_b) >= 0  # same sign or one is zero
        da_masked = torch.where(mask, da, torch.zeros_like(da))
        db_masked = torch.where(mask, db, torch.zeros_like(db))

        merged_delta = weight_a * da_masked + weight_b * db_masked
        merged[k] = (pb.float() + merged_delta).to(pb.dtype)
    return merged


# ─── DARE (random pruning of deltas before TIES) ─────────────────────────────

def dare_prune(delta: torch.Tensor, density: float) -> torch.Tensor:
    """DARE: randomly prune task vector, rescale by 1/density."""
    if density >= 1.0:
        return delta
    mask = torch.bernoulli(torch.full_like(delta.float(), density))
    return (delta * mask) / max(density, 1e-6)


# ─── Merge LoRA adapter into base model ──────────────────────────────────────

def merge_peft_adapter(base_model_id: str, adapter_dir: str,
                       save_dir: str, device: str = "cpu") -> None:
    """Load base + LoRA adapter, merge, save full model weights."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    print(f"  Loading base: {base_model_id}")
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True
    )
    print(f"  Loading adapter: {adapter_dir}")
    m = PeftModel.from_pretrained(m, adapter_dir)
    print("  Merging adapter into base...")
    m = m.merge_and_unload()
    print(f"  Saving merged model to {save_dir}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    m.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    del m
    torch.cuda.empty_cache()


# ─── Validate merged model ───────────────────────────────────────────────────

def validate_model(model_path: str, exp_id: str) -> dict:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import gc
    print(f"  Validating {exp_id} at {model_path}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )
        m.eval()
        mem = torch.cuda.memory_allocated() / 1e9
        prompt = "What is the difference between supervised and unsupervised learning?"
        inp = tok(prompt, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=80, do_sample=False)
        elapsed = time.perf_counter() - t0
        gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"    Memory: {mem:.1f} GB  Time: {elapsed:.1f}s")
        print(f"    Output: {gen[:150]}...")
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return {"status": "ok", "memory_gb": round(mem, 2),
                "gen_sec": round(elapsed, 2), "sample_output": gen[:300]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── M1: SLERP(base, SFT) ────────────────────────────────────────────────────

def run_m1(args):
    from transformers import AutoModelForCausalLM
    import gc

    base_model_id = "Qwen/Qwen3-8B"
    sft_adapter   = str(MODELS / "T3_Qwen3-8B_lora_r16")
    sft_merged    = str(MODELS / "T3_Qwen3-8B_sft_merged")
    out_dir       = str(MODELS / "M1_Qwen3-8B_slerp_sft")

    print(f"\n{'='*60}")
    print(f"  M1: SLERP(t=0.5) — Qwen3-8B-base + T3-SFT")

    # Step 1: Merge T3 LoRA adapter into base model (if not done yet)
    if not Path(sft_merged, "config.json").exists():
        print("  Step 1: Merging T3 adapter into base model...")
        merge_peft_adapter(base_model_id, sft_adapter, sft_merged, device="cpu")
    else:
        print(f"  Step 1: Using cached merged SFT at {sft_merged}")

    # Step 2: Load both models as state dicts on CPU
    print("  Step 2: Loading base model state dict...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True
    )
    state_base = {k: v.clone() for k, v in base.state_dict().items()}
    del base; gc.collect()

    print("  Step 2b: Loading SFT merged state dict...")
    sft = AutoModelForCausalLM.from_pretrained(
        sft_merged, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True
    )
    state_sft = {k: v.clone() for k, v in sft.state_dict().items()}
    cfg = sft.config
    del sft; gc.collect()

    # Step 3: SLERP
    t = args.slerp_t if hasattr(args, "slerp_t") else 0.5
    print(f"  Step 3: SLERP at t={t}...")
    t0 = time.time()
    merged_state = slerp_models(state_base, state_sft, t=t)
    elapsed = time.time() - t0
    print(f"  SLERP done in {elapsed:.1f}s")

    # Step 4: Save merged model
    print(f"  Step 4: Saving to {out_dir}...")
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_config(cfg)
    m.load_state_dict(merged_state)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out_dir)
    del m, merged_state, state_base, state_sft; gc.collect()

    # Copy tokenizer from SFT merged
    import shutil
    for f in Path(sft_merged).glob("token*"):
        shutil.copy(f, out_dir)

    print(f"  Saved M1 merged model to {out_dir}")

    # Step 5: Validate
    val = validate_model(out_dir, "M1")
    result = {
        "experiment": "M1",
        "method": "slerp",
        "slerp_t": t,
        "base_model": base_model_id,
        "sft_model": sft_merged,
        "output_dir": out_dir,
        "slerp_time_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        **val,
    }
    out = RESULTS / f"M1_slerp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    return result


# ─── M2: TIES(SFT, DPO) ──────────────────────────────────────────────────────

def run_m2(args):
    from transformers import AutoModelForCausalLM
    import gc

    base_model_id = "Qwen/Qwen3-8B"
    sft_merged    = str(MODELS / "T3_Qwen3-8B_sft_merged")
    dpo_dir       = str(MODELS / "T20_Qwen3-8B_dpo")
    out_dir       = str(MODELS / "M2_Qwen3-8B_ties_sft_dpo")

    print(f"\n{'='*60}")
    print(f"  M2: TIES — T3-SFT + T20-DPO")

    dpo_exists = (Path(dpo_dir, "config.json").exists() or
                  Path(dpo_dir, "adapter_config.json").exists())
    if not dpo_exists:
        print(f"  ERROR: T20 DPO checkpoint not found at {dpo_dir}")
        print("  Run T20 first. Skipping M2.")
        return {"experiment": "M2", "error": "T20 checkpoint missing"}

    print("  Loading base model state dict...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True
    )
    state_base = {k: v.clone() for k, v in base.state_dict().items()}
    del base; gc.collect()

    print("  Loading SFT merged state dict...")
    sft = AutoModelForCausalLM.from_pretrained(
        sft_merged, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True
    )
    state_sft = {k: v.clone() for k, v in sft.state_dict().items()}
    cfg = sft.config
    del sft; gc.collect()

    print("  Loading DPO model state dict...")
    # DPO output might be PEFT adapter or full model
    dpo_adapter = Path(dpo_dir) / "adapter_config.json"
    if dpo_adapter.exists():
        from peft import PeftModel
        m = AutoModelForCausalLM.from_pretrained(
            base_model_id, dtype=torch.bfloat16,
            device_map="cpu", trust_remote_code=True
        )
        m = PeftModel.from_pretrained(m, dpo_dir).merge_and_unload()
        state_dpo = {k: v.clone() for k, v in m.state_dict().items()}
        del m; gc.collect()
    else:
        dpo_m = AutoModelForCausalLM.from_pretrained(
            dpo_dir, dtype=torch.bfloat16,
            device_map="cpu", trust_remote_code=True
        )
        state_dpo = {k: v.clone() for k, v in dpo_m.state_dict().items()}
        del dpo_m; gc.collect()

    print("  Running TIES merge...")
    t0 = time.time()
    merged_state = ties_merge(state_base, state_sft, state_dpo, density=0.5)
    elapsed = time.time() - t0
    print(f"  TIES done in {elapsed:.1f}s")

    print(f"  Saving to {out_dir}...")
    m = AutoModelForCausalLM.from_config(cfg)
    m.load_state_dict(merged_state)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out_dir)
    del m, merged_state, state_base, state_sft, state_dpo; gc.collect()
    import shutil
    for f in Path(sft_merged).glob("token*"):
        shutil.copy(f, out_dir)

    val = validate_model(out_dir, "M2")
    result = {
        "experiment": "M2", "method": "ties",
        "output_dir": out_dir, "ties_time_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(), **val,
    }
    out = RESULTS / f"M2_ties_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    return result


# ─── M3: DARE+TIES(Qwen3-8B, Qwen2.5-Coder-7B) ──────────────────────────────

def run_m3(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gc

    model_a_id = "Qwen/Qwen3-8B"
    model_b_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    out_dir    = str(MODELS / "M3_Qwen3-8B_dare_coder")

    print(f"\n{'='*60}")
    print(f"  M3: DARE+TIES — Qwen3-8B + Qwen2.5-Coder-7B")

    print(f"  Loading {model_a_id}...")
    ma = AutoModelForCausalLM.from_pretrained(
        model_a_id, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True
    )
    state_a = {k: v.clone() for k, v in ma.state_dict().items()}
    cfg_a = ma.config
    del ma; gc.collect()

    print(f"  Loading {model_b_id}...")
    mb = AutoModelForCausalLM.from_pretrained(
        model_b_id, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True
    )
    state_b = {k: v.clone() for k, v in mb.state_dict().items()}
    tok_b = AutoTokenizer.from_pretrained(model_b_id, trust_remote_code=True)
    del mb; gc.collect()

    # For M3: use model A as "base" (Qwen3-8B), merge Coder task vector into it
    # DARE prune coder deltas then add to Qwen3-8B
    print("  Applying DARE+TIES merge...")
    t0 = time.time()
    density = 0.5
    weight_coder = 0.4  # Less weight to coder to preserve general capability

    merged = {}
    for k in state_a:
        pa = state_a[k]
        if pa.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            merged[k] = pa
            continue
        pb = state_b.get(k, pa)
        if pa.shape != pb.shape:
            merged[k] = pa  # shape mismatch — keep model A
            continue
        delta = (pb - pa).float()
        dare_delta = dare_prune(delta, density)
        merged[k] = (pa.float() + weight_coder * dare_delta).to(pa.dtype)

    elapsed = time.time() - t0
    print(f"  DARE+TIES done in {elapsed:.1f}s")

    print(f"  Saving to {out_dir}...")
    m = AutoModelForCausalLM.from_config(cfg_a)
    m.load_state_dict(merged)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out_dir)
    del m, merged, state_a, state_b; gc.collect()
    tok_a = AutoTokenizer.from_pretrained(model_a_id, trust_remote_code=True)
    tok_a.save_pretrained(out_dir)

    val = validate_model(out_dir, "M3")
    result = {
        "experiment": "M3", "method": "dare_ties",
        "model_a": model_a_id, "model_b": model_b_id,
        "dare_density": density, "coder_weight": weight_coder,
        "output_dir": out_dir, "merge_time_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(), **val,
    }
    out = RESULTS / f"M3_dare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="M1", choices=["M1", "M2", "M3", "all"])
    parser.add_argument("--slerp-t", type=float, default=0.5, dest="slerp_t")
    args = parser.parse_args()

    experiments = ["M1", "M2", "M3"] if args.experiment == "all" else [args.experiment]
    results = []
    for exp in experiments:
        fn = {"M1": run_m1, "M2": run_m2, "M3": run_m3}[exp]
        try:
            r = fn(args)
            results.append(r)
        except Exception as e:
            print(f"  {exp} failed: {e}")
            results.append({"experiment": exp, "error": str(e)})

    print(f"\n{'='*60}")
    for r in results:
        print(f"  {r.get('experiment')}: {r.get('status', 'done')}  {r.get('error', '')}")


if __name__ == "__main__":
    main()
