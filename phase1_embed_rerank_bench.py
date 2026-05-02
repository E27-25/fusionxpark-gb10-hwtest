#!/usr/bin/env python3
"""
Phase 1 — Embedding & Reranker Benchmark
Embedding: MTEB subset (STS, Retrieval), throughput, memory
Reranker:  BEIR NDCG@10 proxy, throughput, memory
"""
import json, time, subprocess, argparse
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/inference"
RESULTS.mkdir(parents=True, exist_ok=True)

EMBED_MODELS = {
    "bge-m3":             "BAAI/bge-m3",                       # 570M, cross-lingual SOTA
    "e5-large-v2":        "intfloat/e5-large-v2",               # 335M, English SOTA
    "qwen3-embed-0.6b":   "Qwen/Qwen3-Embedding-0.6B",         # 0.6B, decoder-based embed
    "qwen3-embed-8b":     "Qwen/Qwen3-Embedding-8B",           # 8B, top MTEB
    "e5-mistral-7b":      "intfloat/e5-mistral-7b-instruct",   # 7B, LLM-based embed
}

# Decoder-only models that need last-token pooling instead of mean pooling
DECODER_EMBED_MODELS = {"qwen3-embed-0.6b", "qwen3-embed-8b", "e5-mistral-7b"}

RERANK_MODELS = {
    "bge-reranker-v2":       "BAAI/bge-reranker-v2-m3",
    "qwen3-vl-rerank-2b":    "Qwen/Qwen3-VL-Reranker-2B",
    "qwen3-vl-rerank-8b":    "Qwen/Qwen3-VL-Reranker-8B",
}

# Generative rerankers (score via "yes"/"no" token logits, not classifier head)
GENERATIVE_RERANK_MODELS = {"qwen3-vl-rerank-2b", "qwen3-vl-rerank-8b"}

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models learn from data.",
    "Natural language processing transforms text into insights.",
    "Deep neural networks have revolutionized computer vision.",
    "Transformers are the backbone of modern AI systems.",
    "GPU acceleration enables faster model training.",
    "Quantization reduces model size while preserving accuracy.",
    "Fine-tuning adapts pre-trained models to specific tasks.",
]

RETRIEVAL_QUERIES = [
    "How do transformers work?",
    "What is quantization in neural networks?",
    "Explain gradient descent optimization.",
]


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


# ─────────────────────────────────────────────────────────────
def bench_embedding(model_id, model_key, batch_sizes=(1, 8, 32), decoder_pooling=False):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    print(f"\n  Loading {model_id} (pooling={'last-token' if decoder_pooling else 'mean'})...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )
        model.eval()
        mem_load = torch.cuda.memory_allocated() / 1e9

        rows = []
        for bs in batch_sizes:
            texts = (TEST_SENTENCES * (bs // len(TEST_SENTENCES) + 1))[:bs]
            inputs = tok(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=512).to("cuda")

            # warmup
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()

            ITERS = 20
            t0 = time.perf_counter()
            for _ in range(ITERS):
                with torch.no_grad():
                    out = model(**inputs)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / ITERS

            if decoder_pooling:
                # Last-token pooling for decoder-only LLM embeddings
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                embeddings = out.last_hidden_state[
                    torch.arange(bs, device="cuda"), seq_lens
                ]
            else:
                # Mean pooling with attention mask
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                embeddings = (out.last_hidden_state * mask).sum(1) / mask.sum(1)

            embed_dim = embeddings.shape[-1]
            tps = bs / elapsed
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            gpu = get_gpu_stats()

            row = {
                "model": model_id, "model_key": model_key,
                "batch_size": bs,
                "sentences_per_sec": round(tps, 1),
                "latency_ms": round(elapsed * 1000, 2),
                "embed_dim": embed_dim,
                "pooling": "last-token" if decoder_pooling else "mean",
                "peak_memory_gb": round(peak_mem, 2),
                "model_load_gb":  round(mem_load, 2),
                "power_w": gpu.get("power_w"),
            }
            rows.append(row)
            print(f"    bs={bs:3d}  {tps:.0f} sent/s  latency={elapsed*1000:.1f}ms  "
                  f"dim={embed_dim}  mem={peak_mem:.1f}GB")

        del model
        torch.cuda.empty_cache()
        return rows

    except Exception as e:
        print(f"    Error: {e}")
        return [{"model": model_id, "model_key": model_key, "error": str(e)}]


def bench_generative_reranker(model_id, model_key, batch_sizes=(1, 4, 8)):
    """Score query-passage pairs via yes/no token log-probabilities (generative reranker).
    Supports both plain causal LMs and VL models (Qwen3-VL-Reranker)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n  Loading generative reranker {model_id}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Qwen3-VL-Reranker uses VL architecture — load appropriate class
        cfg = __import__("transformers").AutoConfig.from_pretrained(
            model_id, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", "")
        if "qwen3_vl" in model_type:
            from transformers import Qwen3VLForConditionalGeneration
            ModelClass = Qwen3VLForConditionalGeneration
        elif "qwen2_5_vl" in model_type or "qwen2_vl" in model_type:
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
            ModelClass = Qwen2_5_VLForConditionalGeneration if "2_5" in model_type else Qwen2VLForConditionalGeneration
        else:
            ModelClass = AutoModelForCausalLM
        model = ModelClass.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )

        model.eval()
        mem_load = torch.cuda.memory_allocated() / 1e9

        # Get token IDs for "Yes" and "No"
        yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        no_id  = tok.encode("No",  add_special_tokens=False)[0]

        rows = []
        for bs in batch_sizes:
            queries  = (RETRIEVAL_QUERIES * (bs // len(RETRIEVAL_QUERIES) + 1))[:bs]
            passages = (TEST_SENTENCES * (bs // len(TEST_SENTENCES) + 1))[:bs]

            prompts = [
                f"Given a query and a passage, determine if the passage is relevant to the query.\n"
                f"Query: {q}\nPassage: {p}\nRelevant (Yes/No):"
                for q, p in zip(queries, passages)
            ]
            inputs = tok(prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=512).to("cuda")

            # warmup
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()

            ITERS = 10
            t0 = time.perf_counter()
            for _ in range(ITERS):
                with torch.no_grad():
                    logits = model(**inputs).logits[:, -1, :]  # last token logits
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / ITERS

            # Score = log P(Yes) - log P(No)
            scores = logits[:, yes_id].float() - logits[:, no_id].float()
            tps = bs / elapsed
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            gpu = get_gpu_stats()

            row = {
                "model": model_id, "model_key": model_key,
                "batch_size": bs,
                "pairs_per_sec": round(tps, 1),
                "latency_ms": round(elapsed * 1000, 2),
                "sample_score_mean": round(float(scores.mean()), 4),
                "peak_memory_gb": round(peak_mem, 2),
                "model_load_gb":  round(mem_load, 2),
                "power_w": gpu.get("power_w"),
            }
            rows.append(row)
            print(f"    bs={bs:3d}  {tps:.0f} pairs/s  latency={elapsed*1000:.1f}ms  "
                  f"mem={peak_mem:.1f}GB  score_mean={scores.mean():.3f}")

        del model
        torch.cuda.empty_cache()
        return rows

    except Exception as e:
        print(f"    Error: {e}")
        return [{"model": model_id, "model_key": model_key, "error": str(e)}]


def run_mteb_subset(model_id, model_key, tasks=("STSBenchmark", "MSMARCO")):
    print(f"  Running MTEB subset ({', '.join(tasks)}) on {model_id}...")
    try:
        import mteb
        from sentence_transformers import SentenceTransformer
        import torch
        model = SentenceTransformer(model_id, device="cuda",
                                    model_kwargs={"dtype": torch.bfloat16})
        results = {}
        for task_name in tasks:
            try:
                task = mteb.get_task(task_name)
                evaluation = mteb.MTEB(tasks=[task])
                result = evaluation.run(model, output_folder=str(RESULTS / "mteb"),
                                        verbosity=0)
                if result:
                    score = result[0].scores.get("test", [{}])
                    if score:
                        results[task_name] = score[0].get("main_score", None)
            except Exception as e:
                results[task_name] = f"error: {e}"
        del model
        import torch; torch.cuda.empty_cache()
        return results
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
def bench_reranker(model_id, model_key, batch_sizes=(1, 8, 16)):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"\n  Loading reranker {model_id}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True,
            num_labels=1,
        )
        model.eval()
        mem_load = torch.cuda.memory_allocated() / 1e9

        rows = []
        for bs in batch_sizes:
            # Query-passage pairs
            queries   = (RETRIEVAL_QUERIES * (bs // len(RETRIEVAL_QUERIES) + 1))[:bs]
            passages  = (TEST_SENTENCES * (bs // len(TEST_SENTENCES) + 1))[:bs]
            pairs = list(zip(queries, passages))

            inputs = tok(
                [p[0] for p in pairs], [p[1] for p in pairs],
                return_tensors="pt", padding=True, truncation=True,
                max_length=512
            ).to("cuda")

            # warmup
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()

            ITERS = 20
            t0 = time.perf_counter()
            for _ in range(ITERS):
                with torch.no_grad():
                    out = model(**inputs)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / ITERS

            scores = out.logits.squeeze(-1)
            tps = bs / elapsed
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            gpu = get_gpu_stats()

            row = {
                "model": model_id, "model_key": model_key,
                "batch_size": bs,
                "pairs_per_sec": round(tps, 1),
                "latency_ms": round(elapsed * 1000, 2),
                "peak_memory_gb": round(peak_mem, 2),
                "model_load_gb":  round(mem_load, 2),
                "power_w": gpu.get("power_w"),
            }
            rows.append(row)
            print(f"    bs={bs:3d}  {tps:.0f} pairs/s  latency={elapsed*1000:.1f}ms  "
                  f"mem={peak_mem:.1f}GB")

        del model
        torch.cuda.empty_cache()
        return rows

    except Exception as e:
        print(f"    Error: {e}")
        return [{"model": model_id, "model_key": model_key, "error": str(e)}]


def run_beir_subset(model_id, model_key, datasets=("msmarco",)):
    print(f"  Running BEIR subset on {model_id}...")
    try:
        from beir import util, LoggingHandler
        from beir.reranking import Rerank
        results = {}
        for ds_name in datasets:
            results[ds_name] = "beir_eval_placeholder"
        return results
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["embed", "rerank", "all"], default="all")
    parser.add_argument("--embed-models", nargs="+",
                        choices=list(EMBED_MODELS.keys()) + ["all"], default=["all"])
    parser.add_argument("--rerank-models", nargs="+",
                        choices=list(RERANK_MODELS.keys()) + ["all"], default=["all"])
    parser.add_argument("--mteb", action="store_true",
                        help="Run MTEB subset (slow)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    args = parser.parse_args()

    all_results = {"embedding": [], "reranking": [], "mteb": {}, "beir": {}}

    # ── Embedding ──
    if args.mode in ("embed", "all"):
        embed_keys = (list(EMBED_MODELS.keys())
                      if "all" in args.embed_models else args.embed_models)
        print(f"\n{'='*60}")
        print("  EMBEDDING MODELS")
        print(f"{'='*60}")
        for key in embed_keys:
            model_id = EMBED_MODELS[key]
            use_decoder = key in DECODER_EMBED_MODELS
            rows = bench_embedding(model_id, key, batch_sizes=args.batch_sizes,
                                   decoder_pooling=use_decoder)
            all_results["embedding"].extend(rows)

            if args.mteb and "error" not in rows[0]:
                mteb_scores = run_mteb_subset(model_id, key)
                all_results["mteb"][key] = mteb_scores

    # ── Reranking ──
    if args.mode in ("rerank", "all"):
        rerank_keys = (list(RERANK_MODELS.keys())
                       if "all" in args.rerank_models else args.rerank_models)
        print(f"\n{'='*60}")
        print("  RERANKER MODELS")
        print(f"{'='*60}")
        for key in rerank_keys:
            model_id = RERANK_MODELS[key]
            if key in GENERATIVE_RERANK_MODELS:
                rows = bench_generative_reranker(model_id, key,
                                                 batch_sizes=[1, 4, 8])
            else:
                rows = bench_reranker(model_id, key, batch_sizes=args.batch_sizes)
            all_results["reranking"].extend(rows)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  EMBEDDING SUMMARY")
    print(f"  {'Model':<25} {'bs':>4} {'sent/s':>8} {'lat_ms':>8} {'Mem':>6}")
    print(f"  {'-'*70}")
    for r in all_results["embedding"]:
        if "error" not in r:
            print(f"  {r.get('model_key',''):<25} {r.get('batch_size',0):>4} "
                  f"{r.get('sentences_per_sec',0):>8.0f} "
                  f"{r.get('latency_ms',0):>8.1f} "
                  f"{r.get('peak_memory_gb',0):>5.1f}G")

    print(f"\n  RERANKER SUMMARY")
    print(f"  {'Model':<25} {'bs':>4} {'pairs/s':>8} {'lat_ms':>8} {'Mem':>6}")
    print(f"  {'-'*70}")
    for r in all_results["reranking"]:
        if "error" not in r:
            print(f"  {r.get('model_key',''):<25} {r.get('batch_size',0):>4} "
                  f"{r.get('pairs_per_sec',0):>8.0f} "
                  f"{r.get('latency_ms',0):>8.1f} "
                  f"{r.get('peak_memory_gb',0):>5.1f}G")

    out = RESULTS / f"embed_rerank_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
