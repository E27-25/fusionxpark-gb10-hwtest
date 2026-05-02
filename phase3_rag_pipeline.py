#!/usr/bin/env python3
"""
Phase 3B — End-to-End RAG Pipeline
Architecture: Embed → FAISS → Reranker → LLM
Configs: RAG-base, RAG-qwen3, RAG-finetuned, No-RAG
"""
import json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime

BASE    = Path("/home/student/Desktop/Test")
RESULTS = BASE / "results/rag"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE / "models"

RAG_CONFIGS = {
    "rag-base": {
        "embed_model":   "BAAI/bge-m3",
        "rerank_model":  "BAAI/bge-reranker-v2-m3",
        "llm_model":     "Qwen/Qwen3-8B",
    },
    "rag-qwen3": {
        "embed_model":   "Qwen/Qwen3-Embedding-4B",
        "rerank_model":  "Qwen/Qwen3-VL-Reranker-8B",
        "llm_model":     "Qwen/Qwen3-32B",
    },
    "no-rag": {
        "embed_model":   None,
        "rerank_model":  None,
        "llm_model":     "Qwen/Qwen3-32B",
    },
}

TEST_QUESTIONS = [
    "What is the capital of France?",
    "How does gradient descent work in machine learning?",
    "What are the main causes of climate change?",
    "Explain how neural networks learn.",
    "What is the difference between supervised and unsupervised learning?",
]


def get_gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        p, t, m = r.stdout.strip().split(",")
        return {"power_w": float(p.strip()), "temp_c": int(t.strip()),
                "mem_mb": int(m.strip())}
    except:
        return {}


# ─────────────────────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )
        self.model.eval()

    def embed(self, texts, batch_size=32):
        import torch
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tok(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                out = self.model(**inputs)
            emb = out.last_hidden_state.mean(dim=1)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_emb.append(emb.cpu().float().numpy())
        import numpy as np
        return np.concatenate(all_emb, axis=0)


class RerankerModel:
    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True, num_labels=1
        )
        self.model.eval()

    def rerank(self, query, passages):
        import torch
        pairs = [(query, p) for p in passages]
        inputs = self.tok(
            [p[0] for p in pairs], [p[1] for p in pairs],
            return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to("cuda")
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
        return scores.cpu().float().tolist()


class LLMModel:
    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True
        )
        self.model.eval()

    def generate(self, prompt, max_new_tokens=256):
        import torch
        inputs = self.tok(prompt, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        elapsed = time.perf_counter() - t0
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        return text, elapsed, n_new


# ─────────────────────────────────────────────────────────────
def build_faiss_index(corpus_texts, embed_model):
    import faiss
    import numpy as np

    print(f"  Building FAISS index for {len(corpus_texts)} passages...")
    embeddings = embed_model.embed(corpus_texts)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"  Index built: {index.ntotal} vectors, dim={dim}")
    return index


def retrieve_top_k(query, index, embed_model, corpus_texts, k=100):
    import numpy as np
    q_emb = embed_model.embed([query])
    import faiss
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)
    return [corpus_texts[i] for i in indices[0]], distances[0].tolist()


def load_corpus(max_passages=1000):
    """Load a small corpus from NQ or similar dataset."""
    try:
        from datasets import load_dataset
        print(f"  Loading NQ corpus ({max_passages} passages)...")
        ds = load_dataset("natural_questions", split="train",
                          streaming=True, trust_remote_code=True)
        corpus = []
        for item in ds:
            if len(corpus) >= max_passages:
                break
            text = item.get("document", {}).get("text", "")
            if text and len(text) > 100:
                corpus.append(text[:500])
        return corpus
    except Exception as e:
        print(f"  NQ load failed: {e}. Using synthetic corpus.")
        return [
            f"This is a sample passage about topic {i}. "
            f"Machine learning involves {i} key principles." * 3
            for i in range(max_passages)
        ]


# ─────────────────────────────────────────────────────────────
def run_rag_config(config_name, cfg, corpus, questions, args):
    import torch
    import gc

    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")

    embed_model  = None
    rerank_model = None
    faiss_index  = None
    llm_model    = None

    results = []
    mem_snapshots = []

    try:
        # Load embedding model
        if cfg["embed_model"]:
            print(f"  Loading embed: {cfg['embed_model']}")
            embed_model = EmbeddingModel(cfg["embed_model"])
            mem_snapshots.append(("after_embed_load",
                                  torch.cuda.memory_allocated() / 1e9))
            faiss_index = build_faiss_index(corpus, embed_model)

        # Load reranker
        if cfg["rerank_model"]:
            print(f"  Loading reranker: {cfg['rerank_model']}")
            rerank_model = RerankerModel(cfg["rerank_model"])
            mem_snapshots.append(("after_rerank_load",
                                  torch.cuda.memory_allocated() / 1e9))

        # Load LLM
        print(f"  Loading LLM: {cfg['llm_model']}")
        llm_model = LLMModel(cfg["llm_model"])
        mem_snapshots.append(("after_llm_load",
                              torch.cuda.memory_allocated() / 1e9))
        gpu = get_gpu_stats()
        total_mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  All models loaded: {total_mem_gb:.1f} GB GPU memory")

        # Run RAG pipeline for each question
        for q_idx, question in enumerate(questions):
            result = {
                "config": config_name, "question": question,
                "q_idx": q_idx,
            }
            t_total_start = time.perf_counter()

            # Retrieval
            retrieved_passages = []
            if faiss_index and embed_model:
                t_ret_start = time.perf_counter()
                candidates, scores = retrieve_top_k(
                    question, faiss_index, embed_model, corpus, k=100
                )
                result["retrieval_ms"] = round((time.perf_counter() - t_ret_start) * 1000, 1)
                result["n_candidates"] = len(candidates)

                # Reranking
                if rerank_model:
                    t_rer_start = time.perf_counter()
                    rerank_scores = rerank_model.rerank(question, candidates[:50])
                    top_indices = sorted(range(len(rerank_scores)),
                                        key=lambda i: rerank_scores[i], reverse=True)
                    retrieved_passages = [candidates[i] for i in top_indices[:5]]
                    result["rerank_ms"] = round((time.perf_counter() - t_rer_start) * 1000, 1)
                else:
                    retrieved_passages = candidates[:5]
            else:
                result["retrieval_ms"] = 0
                result["rerank_ms"]    = 0

            # Generation
            if retrieved_passages:
                context = "\n\n".join(retrieved_passages[:5])
                prompt = (f"Context:\n{context}\n\n"
                          f"Question: {question}\n\nAnswer:")
            else:
                prompt = f"Question: {question}\n\nAnswer:"

            t_gen_start = time.perf_counter()
            answer, gen_elapsed, n_new = llm_model.generate(prompt, max_new_tokens=128)
            result["generation_ms"] = round(gen_elapsed * 1000, 1)
            result["output_tokens"] = n_new
            result["tps"] = round(n_new / gen_elapsed, 1)
            result["answer"] = answer[:300]
            result["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 1)
            result["has_context"] = len(retrieved_passages) > 0

            print(f"  [{q_idx}] {question[:50]}... "
                  f"→ {result['total_ms']}ms  "
                  f"({result.get('retrieval_ms',0)}+{result.get('rerank_ms',0)}+"
                  f"{result['generation_ms']}ms)")
            results.append(result)

        # Summary
        mean_total = sum(r["total_ms"] for r in results) / len(results)
        mean_tps   = sum(r["tps"] for r in results) / len(results)
        print(f"  Mean total latency: {mean_total:.0f}ms  Mean TPS: {mean_tps:.0f}")

    except Exception as e:
        print(f"  ERROR in {config_name}: {e}")
        results.append({"config": config_name, "error": str(e)})

    finally:
        del embed_model, rerank_model, faiss_index, llm_model
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "config": config_name,
        "timestamp": datetime.now().isoformat(),
        "memory_snapshots": {k: round(v, 2) for k, v in mem_snapshots},
        "results": results,
    }


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+",
                        choices=list(RAG_CONFIGS.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--corpus-size", type=int, default=500,
                        dest="corpus_size")
    parser.add_argument("--n-questions", type=int, default=5,
                        dest="n_questions")
    args = parser.parse_args()

    configs = (list(RAG_CONFIGS.keys()) if "all" in args.configs
               else args.configs)
    questions = TEST_QUESTIONS[:args.n_questions]

    print("  Loading corpus...")
    corpus = load_corpus(max_passages=args.corpus_size)
    print(f"  Corpus: {len(corpus)} passages")

    all_results = {}
    for cfg_name in configs:
        cfg = RAG_CONFIGS[cfg_name]
        result = run_rag_config(cfg_name, cfg, corpus, questions, args)
        all_results[cfg_name] = result

    # Summary comparison
    print(f"\n{'='*70}")
    print("  RAG PIPELINE COMPARISON")
    print(f"  {'Config':<20} {'Mean Latency':>14} {'Mean TPS':>10} {'Retrieval':>10}")
    print(f"  {'-'*70}")
    for cfg_name, res in all_results.items():
        rows = [r for r in res.get("results", []) if "error" not in r]
        if rows:
            mean_lat = sum(r["total_ms"] for r in rows) / len(rows)
            mean_tps = sum(r["tps"] for r in rows) / len(rows)
            mean_ret = sum(r.get("retrieval_ms", 0) for r in rows) / len(rows)
            print(f"  {cfg_name:<20} {mean_lat:>13.0f}ms {mean_tps:>10.0f} "
                  f"{mean_ret:>9.0f}ms")

    out = RESULTS / f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
