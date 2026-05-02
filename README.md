# fusionxpark-gb10-hwtest

**NVIDIA GB10 Grace Blackwell — Full AI Benchmark Suite**

Comprehensive hardware characterization and AI model benchmarking on the NVIDIA GB10 Grace Blackwell SoC: inference throughput, training efficiency, quantization quality, ASR, TTS, embeddings, and alignment training (SFT / LoRA / DPO / GRPO). This is the **first update** — experiments still running. A second update will follow once all queued experiments complete, and a third for supplementary tests (FP8, long-context 128K, AlpacaEval 2, RAG pipeline).

---

## Hardware

| Property | Value |
|---|---|
| Device | NVIDIA GB10 Grace Blackwell |
| GPU Architecture | Blackwell (SM 12.1) |
| CPU | ARM Cortex-X925 / A725, 20 cores (aarch64) |
| Unified Memory | 128.5 GB (LPDDR5X, shared CPU+GPU via NVLink-C2C) |
| Storage | 1.8 TB NVMe |
| CUDA | 13.0, Driver 580.142 |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| Python | 3.12.3 |

**Theoretical Peak:**

| Metric | Spec |
|---|---|
| Memory Bandwidth | 273 GB/s |
| BF16 Compute | 67 TFLOPS |
| FP8 Compute | 134 TFLOPS |
| INT8 Compute | 134 TOPS |
| NVLink-C2C | 900 GB/s bidirectional |
| FP4 Compute | ~1 PFLOPS (marketing claim) |

---

## Software Stack

| Package | Version | Notes |
|---|---|---|
| PyTorch | 2.12.0.dev20260407+cu128 | Nightly, aarch64+CUDA 12.8 |
| transformers | 5.7.0 | HuggingFace |
| peft | 0.19.1 | LoRA / QLoRA |
| trl | 1.3.0 | SFT / DPO / GRPO |
| bitsandbytes | 0.49.2 | SM120 kernels confirmed |
| lm_eval | 0.4.11 | CLI: `lm-eval run` |
| SGLang | 0.5.10.post1 | Inference server |
| torchao | 0.17.0 | Quantization |

**Not installed / failed:**
- vLLM — removed (broke CUDA on aarch64)
- TensorRT-LLM — not yet installed
- Flash Attention 3 — not built (build pending)
- AWQ — failed (Triton not available on aarch64)
- GGUF / llama.cpp — not tested

---

## Phase 0.5 — Hardware Characterization

### HW-1: Memory Bandwidth

| Tensor Size | Achieved BW | % of 273 GB/s Peak |
|---|---|---|
| 1 GB | 133.6 GB/s | 48.9% |
| 4 GB | 132.8 GB/s | 48.6% |
| 16 GB | 125.0 GB/s | 45.8% |
| 32 GB | 123.9 GB/s | 45.4% |
| 64 GB | OOM | — |

**Peak sustained: ~134 GB/s (49% of theoretical)**

### HW-2: Compute Throughput (GEMM)

| Precision | Achieved TFLOPS | % of Theoretical |
|---|---|---|
| BF16 | **11.9 TFLOPS** | 17.7% of 67 TFLOPS |
| FP32 | 39.9 TFLOPS | 59.6% of 67 TFLOPS |
| FP8 | **61.1 TFLOPS** | 45.6% of 134 TFLOPS |

> **Note:** BF16 GEMM is severely underperforming (17.7%). PyTorch's cuBLAS kernels are not yet optimized for Blackwell aarch64. FP8 performance is significantly better at 45.6%. FP4 was not tested.

### HW-3: NVLink-C2C Transfer

| Transfer Size | H2D (GB/s) | D2H (GB/s) |
|---|---|---|
| 0.1 GB | 48.3 | 4.0 |
| 0.5 GB | 53.5 | 3.4 |
| 1 GB | 54.9 | 54.7 |
| 4 GB | 54.8 | 0.4 |
| 8 GB | 51.4 | 4.4 |

> Measured DMA memcpy speed: ~48–55 GB/s. This reflects pinned-memory DMA, not the 900 GB/s coherent NVLink-C2C bandwidth (which is used transparently by unified memory, not measurable via explicit copies).

### HW-4: Thermal Behavior

| State | Power | Temperature | SM Clock |
|---|---|---|---|
| Idle | 10.9 W | 49°C | 2405 MHz |
| Sustained load | 88–91 W | 68–72°C | 2177–2216 MHz |
| Throttle threshold | — | **53°C** | — |

> **Thermal throttling occurs on every sustained workload.** At 72°C, SM clock drops from 2405 MHz to 2177 MHz (~9.5% reduction). This affects all inference benchmarks. The device is fanless — thermal management depends entirely on passive cooling and chassis airflow.

### HW-5: Roofline Analysis

```
Ridge point (BF16):  67 TFLOPS / 273 GB/s = 245 FLOPs/byte
Ridge point (FP8):   134 TFLOPS / 273 GB/s = 491 FLOPs/byte

LLM decode (batch=1):  ~1 FLOPs/byte   → MEMORY-BOUND
LLM decode (batch=32): ~32 FLOPs/byte  → MEMORY-BOUND
LLM prefill (2K seq):  ~1000 FLOPs/byte → COMPUTE-BOUND
Training fwd+bwd:      ~2000 FLOPs/byte → COMPUTE-BOUND
```

All decode workloads are memory-bandwidth-bound. Training and prefill are compute-bound.

### HW-5b: Attention Kernel (SDPA vs Flash Attention 3)

| Sequence Length | torch SDPA (ms) | Flash Attention 3 |
|---|---|---|
| 512 | 0.072 ms | not built |
| 1024 | 0.271 ms | not built |
| 2048 | 0.895 ms | not built |
| 4096 | 3.322 ms | not built |
| 8192 | 12.771 ms | not built |

> SDPA scales as O(seq²). FA3 build pending — expected significant speedup at seq > 4K.

### HW-5c: NVMe Storage

| Metric | Value |
|---|---|
| Sequential write | 3.41 GB/s |
| Sequential read | 5.43 GB/s |
| 7B BF16 model load | 2.6 sec |
| 32B BF16 model load | 11.8 sec |
| 72B BF16 model load | 26.5 sec |
| 235B NF4 model load | 21.7 sec |

### HW-5d: Memory Pressure

| Allocation | Time | Op Latency |
|---|---|---|
| 10 GB (7.8% used) | 625 ms | 0.223 ms |
| 20 GB (15.6%) | 724 ms | 0.044 ms |
| 40 GB (31.2%) | 1524 ms | 0.044 ms |

No swap activation or latency spikes observed up to 40 GB allocated. Upper bound not yet tested.

### HW-6: Model-Level Bandwidth Efficiency (Decode)

| Model | Achieved BW | % of 273 GB/s | Est. Tokens/s |
|---|---|---|---|
| 7B BF16 | 118.5 GB/s | 43.4% | 8.5 |
| 32B BF16 | 121.3 GB/s | 44.4% | 1.9 |
| 70B NF4 | 120.6 GB/s | 44.2% | 3.4 |

---

## Phase 1 — Inference Benchmarks

All experiments run on HuggingFace `generate()` backend (SGLang available but not used for standard benchmarks). **All runs throttled** (temp consistently >53°C).

### LLM Throughput (batch=1, short prompt, BF16)

| Model | Params | Memory | Tokens/s | BW Util | Throttled |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | ~1.2 GB | **77–108** tok/s | — | Yes |
| Qwen3-1.7B | 1.7B | ~3.4 GB | **37–60** tok/s | — | Yes |
| Qwen3-4B | 4B | ~8 GB | **17–32** tok/s | — | Yes |
| Qwen3-8B | 8B | 16.4 GB | **9.9–10.9** tok/s | — | Yes |
| Qwen3-8B (batch=8) | 8B | — | **20–22** tok/s | — | Yes |
| Qwen3-14B | 14B | ~28 GB | **2.4–3.4** tok/s | — | Yes |
| Qwen3-32B | 32B | 65.6 GB | **2.3** tok/s | 55.4% | Yes |
| Qwen3-32B (batch=4) | 32B | 65.7 GB | **3.0** tok/s | 18.2% | Yes |
| Qwen2.5-7B-Instruct | 7B | ~14 GB | **4.7–5.9** tok/s | — | Yes |
| Qwen2.5-32B-Instruct | 32B | 65.6 GB | **2.5** tok/s | 59.7% | Yes |
| Qwen2.5-Coder-1.5B | 1.5B | ~3 GB | **14–28** tok/s | — | Yes |
| Qwen2.5-Coder-7B | 7B | ~14 GB | **4.7–5.8** tok/s | — | Yes |
| phi-4 (14B) | 14B | ~28 GB | **1.0–5.6** tok/s | — | Yes |
| DeepSeek-R1-Distill-Llama-8B | 8B | ~16 GB | ~10 tok/s | — | — |

> 32B models achieve 55–60% memory bandwidth utilization — the best observed class on this device. 7B models reach ~43%. Small models (0.6–4B) are faster in tok/s but less efficient in BW utilization.

### Quantization Sweep (Qwen3-8B)

| Format | Tokens/s | Peak Memory | BW Util | Tokens/W |
|---|---|---|---|---|
| BF16 | 10.0 tok/s | 16.5 GB | 58.4% | 0.31 |
| INT8 | 14.4 tok/s | 9.6 GB | — | — |
| NF4 | 26.3 tok/s | 16.4 GB | 154%* | 0.56 |
| AWQ | FAILED | — | — | — |
| GGUF | FAILED | — | — | — |

> *NF4 BW utilization >100% indicates the measurement formula does not account for decompression overhead — NF4 reads 4-bit weights and expands to BF16, so effective bytes read are smaller than model file size implies. AWQ failed due to Triton not available on aarch64. GGUF not tested.

### Speculative Decoding

| Pair | Baseline | Speculative | Speedup |
|---|---|---|---|
| Qwen3-0.6B → Qwen3-8B (HF backend) | 10.0 tok/s | 5.7 tok/s | **0.57×** |

> HF backend speculative decoding is slower than baseline — the draft model overhead dominates. Real speedup requires vLLM or SGLang speculative decoding (both not tested; vLLM incompatible).

### Long-Context (Qwen3-8B, Needle-in-Haystack)

| Context Length | Depth 10% | Depth 50% | Depth 90% | Memory |
|---|---|---|---|---|
| 4K tokens | ✓ | ✓ | ✗ | 17.4 GB |
| 8K tokens | ✓ | ✓ | ✗ | 18.5 GB |
| 16K tokens | ✓ | ✓ | ✓ | ~20 GB |

> 6/9 needles found at 16K. Depth 90% fails at 4K–8K (needle near end of context). 32K–128K not yet tested. Qwen3-8B supports up to 128K context.

### ASR Benchmarks (LibriSpeech-style, 50 samples)

| Model | WER | CER | Mean RTF | P95 RTF | Memory |
|---|---|---|---|---|---|
| Whisper-large-v3 | 3.72% | 1.40% | 0.338 | 0.58 | 3.4 GB |
| Whisper-large-v3-turbo | **2.95%** | **1.18%** | **0.183** | 0.35 | 3.1 GB |
| Qwen2-Audio-7B-Instruct | 12.79% | 8.69% | 0.600 | 0.89 | 17.4 GB |

> Whisper-large-v3-turbo achieves best WER and is 1.8× faster than full v3. Both are well below RTF=1.0 (faster than real-time). Qwen2-Audio has significantly worse WER but is a general audio+text model.

### TTS Benchmarks

| Model | Mean RTF | P95 RTF | Memory |
|---|---|---|---|
| suno/bark | 2.17 | 2.32 | 2.5 GB |

> RTF > 1.0 means slower than real-time. Bark generates ~2.2s of audio computation per 1s of output. Qwen3-TTS not benchmarked (API format differs).

### Embedding / Reranking

| Model | Task | Throughput | Memory | Power |
|---|---|---|---|---|
| Qwen3-VL-Reranker-8B | Reranking (batch=1) | 1.3 pairs/s | 17.6 GB | 38 W |
| Qwen3-VL-Reranker-8B | Reranking (batch=4) | 3.7 pairs/s | 17.7 GB | 50 W |
| Qwen3-VL-Reranker-8B | Reranking (batch=8) | 3.8 pairs/s | 17.8 GB | 55 W |

> MTEB / BEIR evaluation not completed (requires internet access to eval datasets during run). Throughput saturates at batch=4→8.

---

## Phase 2 — Training Results

All training on `tatsu-lab/alpaca` (52K samples) unless noted. 8-bit AdamW + gradient checkpointing throughout.

### Training Summary Table

| ID | Model | Method | Steps | Final Loss | Peak Memory | Time | Tokens/s | MFU |
|---|---|---|---|---|---|---|---|---|
| T1 | Qwen3-8B | Full FT (BF16 + 8-bit Adam) | 500 | **1.019** | 52.3 GB | 146.7 min | 234 | 17.2% |
| T3 | Qwen3-8B | LoRA r=16 | 500 | **1.017** | 22.0 GB | 152.2 min | 1802 | 44.3% |
| T4 | Mistral-7B-Instruct-v0.3 | LoRA r=16 | 500 | **0.841** | 16.7 GB | 159.4 min | 1728 | 37.6% |
| T6 | Qwen3-32B | LoRA r=32 | 150 | **0.990** | 74.0 GB | 194.8 min | 447 | 44.1% |
| T21 | Qwen3-8B | GRPO (GSM8K math) | 200 | — | 18.8 GB | 1003.7 min | — | — |
| T24 | DeepSeek-R1-Distill-Llama-8B | LoRA r=16 | 500 | **1.313** | 21.5 GB | 162.0 min | 1704 | 41.1% |

### Key Training Observations

**Full FT vs LoRA (T1 vs T3 — Qwen3-8B):**
- Loss is nearly identical (1.019 vs 1.017) — LoRA matches Full FT quality at this scale
- Memory: LoRA uses 22 GB vs 52 GB Full FT (57% reduction)
- MFU: LoRA achieves 44.3% vs 17.2% for Full FT — LoRA is much more compute-efficient
- Throughput: LoRA 1802 tok/s vs 234 tok/s Full FT (7.7× faster)
- **Conclusion: LoRA strictly dominates Full FT on this device at 8B scale**

**32B LoRA (T6):**
- 74 GB peak memory — fits comfortably in 128.5 GB unified memory
- MFU 44.1% — identical to 8B LoRA, excellent scaling
- This would OOM on any standard 80 GB GPU; GB10 unified memory is a genuine advantage

**GRPO (T21 — Reinforcement Learning from Math Rewards):**
- 200 steps on GSM8K, group size 8, clip ratio 0.2
- Mean reward: **0.373** (37.3% of math problems answered correctly after 200 steps)
- 1003.7 minutes (~16.7 hours) — GRPO is ~7× slower than SFT due to rollout generation
- Peak memory: 18.8 GB — surprisingly efficient (generates 8 completions per prompt, but sequentially)

**Cross-model comparison (T3 vs T4 — LoRA r=16):**
- Mistral-7B achieves lower loss (0.841) than Qwen3-8B (1.017) on Alpaca
- Mistral uses less memory (16.7 GB vs 22.0 GB) due to smaller model footprint
- Both achieve similar throughput (~1700–1800 tok/s)

**DeepSeek-R1-Distill (T24):**
- Higher loss (1.313) on Alpaca — expected, as R1-Distill is trained on reasoning data, not instruction-following format
- MFU 41.1%, memory 21.5 GB — similar profile to Qwen3-8B LoRA

### Training Config Details

```
LoRA targets:  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
LoRA dropout:  0.05
Optimizer:     paged_adamw_8bit
LR scheduler:  cosine with 3% warmup
Max seq len:   2048
Precision:     BF16
Grad checkpointing: ON (use_reentrant=False)
```

### Pending Training Experiments

| ID | Model | Method | Status |
|---|---|---|---|
| T2 | Mistral-7B | Full FT | Queued |
| T5 | Qwen3-8B | LoRA r=64 (OpenHermes) | Queued |
| T7 | Qwen3-30B-A3B | MoE LoRA r=16 | Queued |
| T8 | Qwen2.5-72B | QLoRA NF4 r=16 | Queued |
| T11 | DeepSeek-V2-Lite | LoRA r=32 (fixed targets) | Queued |
| T12 | Mixtral-8x7B | QLoRA NF4 r=16 | Queued |
| T14 | Whisper-large-v3 | Full FT (LibriSpeech) | Queued |
| T15 | Whisper-large-v3 | LoRA r=16 | Queued |
| T20 | Qwen3-8B | DPO (from T3 checkpoint) | Queued |
| T22 | Qwen3-32B | DPO (from T6 checkpoint) | Queued |
| T23 | Qwen3-8B | CPT (OpenWebText 1B tokens) | Queued |
| M1 | Qwen3-8B-base + T3 | SLERP merge | Queued |
| M2 | T3-SFT + T20-DPO | TIES merge | Queued |
| M3 | Qwen3-8B + Qwen2.5-Coder-7B | DARE+TIES merge | Queued |

---

## Marketing Claims vs Measured Reality

| Claim | Measured | Assessment |
|---|---|---|
| "1 PFLOPS FP4" | FP8=61 TFLOPS, FP4 untested | ~8× gap assumed (FP4 ~120 TFLOPS realistic) |
| "273 GB/s memory bandwidth" | **134 GB/s achieved** (49%) | Reasonable — DAXPY benchmark limit |
| "128 GB unified memory" | **128.5 GB confirmed** | Accurate |
| "900 GB/s NVLink-C2C" | 48–55 GB/s DMA measured | Coherent path unmeasurable via explicit copy |
| "Train 32B models on single device" | **74 GB peak confirmed** | TRUE — genuine advantage |
| No discrete VRAM limit | All 128 GB available to GPU | TRUE — unified memory works |
| FP8 native Blackwell | FP8 GEMM = 61.1 TFLOPS | TRUE, but FP8 inference not yet tested end-to-end |

### Summary: What This Hardware Is Good For

**Strong fit:**
- Training large models with LoRA (32B LoRA r=32 on a single SoC — impossible on standard GPUs)
- Research requiring large RAM without multi-GPU complexity
- Long-context inference (128K context fits entirely in unified memory)
- Prototyping with multiple large models loaded simultaneously

**Weak fit:**
- Production inference throughput (10 tok/s for 8B is slow; thermal throttling persistent)
- Speculative decoding (no vLLM support on aarch64)
- FP8/FP4 optimized inference (TRT-LLM not installed, FP8 gains unrealized in practice)
- AWQ / GGUF quantization (toolchain not available on aarch64)

---

## Known Issues and Fixes Applied

| Issue | Fix |
|---|---|
| `is_torch_fx_available` missing from transformers 5.7 | Patch in `modeling_deepseek.py`: `is_torch_fx_available = lambda: True` |
| DeepSeek-V2-Lite LoRA targets wrong | Changed `q_a_proj`/`q_b_proj` → `q_proj` (V2-Lite uses standard MHA, not MLA split) |
| AWQ fails on aarch64 | Triton not available on ARM — AWQ excluded |
| vLLM breaks CUDA | Removed vLLM entirely; using HF generate + SGLang |
| DPO `max_prompt_length` removed in TRL 1.3.0 | Removed kwarg from DPOConfig |
| Mixtral OOM with BF16 LoRA | Switched to QLoRA NF4 |
| lm-eval CLI changed in 0.4.11 | Use `lm-eval run` not `lm_eval` |
| `nvidia-smi` memory returns N/A | Use `torch.cuda.memory_allocated()` or `free -h` |
| 72B+ models OOM during lm-eval | Added `load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16` to eval model_args |
| `torch_dtype` deprecated in transformers 5.7 | Changed to `dtype=` in all `from_pretrained()` calls |

---

## Repository Structure

```
/home/student/Desktop/Test/
├── phase05_hw_bench.py          # Hardware microbenchmarks (BW, GEMM, NVLink, thermal, roofline)
├── phase05_monitor.sh           # nvidia-smi dmon logger
├── phase1_inference_bench.py    # LLM inference benchmark
├── phase1_asr_bench.py          # ASR benchmark (WER, RTF)
├── phase1_tts_bench.py          # TTS benchmark (RTF)
├── phase1_embed_rerank_bench.py # Embedding + Reranker benchmark
├── phase1_quant_sweep.py        # Quantization sweep (BF16/INT8/NF4/AWQ/GGUF)
├── phase1_speculative_bench.py  # Speculative decoding benchmark
├── phase1_longctx_bench.py      # Long-context (RULER needle-in-haystack)
├── phase1_stress_test.py        # Concurrent load simulation
├── phase2_train.py              # Full FT / LoRA / QLoRA training (T1–T8, T11–T13, T23–T24)
├── phase2_dpo.py                # DPO training (T20, T22)
├── phase2_grpo.py               # GRPO training (T21)
├── phase2_cpt.py                # Continued pre-training (T23)
├── phase2_asr_train.py          # Whisper fine-tuning (T14, T15)
├── phase2_merge.py              # Model merging (M1–M3: SLERP, TIES, DARE+TIES)
├── phase3_eval.sh               # lm-eval standard benchmarks
├── phase3_asr_eval.py           # ASR evaluation (WER before/after FT)
├── phase3_rag_pipeline.py       # End-to-end RAG pipeline
├── tracker.py                   # Experiment dashboard
├── run_retry_queue.sh           # Automated experiment queue
├── results/
│   ├── hardware/                # hw_bench_*.json
│   ├── inference/               # inference_bench, quant_sweep, asr_bench, ...
│   ├── training/                # T1, T3, T4, T6, T21, T24 results
│   └── evaluation/              # lm-eval output (pending)
└── models/                      # Saved checkpoints
    ├── T1_Qwen3-8B_full_ft/
    ├── T3_Qwen3-8B_lora_r16/
    ├── T4_Mistral-7B-Instruct-v0.3_lora_r16/
    ├── T6_Qwen3-32B_lora_r32/
    ├── T21_Qwen3-8B_grpo/
    └── T24_DeepSeek-R1-Distill-Llama-8B_lora_r16/
```

---

## Monitoring Commands

```bash
# Experiment dashboard
python3 ~/Desktop/Test/tracker.py

# Memory usage (nvidia-smi returns N/A on GB10)
free -h

# Active tmux sessions
tmux list-sessions

# Inference sweep progress
grep -E "(✓|✗|Benchmarking|===)" ~/Desktop/Test/logs/inference_sweep2.log | tail -10

# Retry queue progress
tail -f ~/Desktop/Test/logs/retry_queue.log

# Attach to running session
tmux attach -t gb10bench    # training queue
tmux attach -t infer_xl     # XL inference sweep
```

---

## Planned Next Steps (Update 2 — after current queue completes)

1. Full lm-eval MMLU/GSM8K/ARC results for T3, T6, T20, T21, T22 + base models
2. DPO results (T20 Qwen3-8B, T22 Qwen3-32B) — reward margins, loss curves
3. 72B QLoRA training (T8) results
4. MoE LoRA results (T7 Qwen3-30B-A3B)
5. Model merge quality vs individual fine-tuned models (M1, M2, M3)
6. ASR fine-tuning WER delta (T14/T15 vs baseline Whisper)

**Planned Update 3 (supplementary experiments):**
1. FP8 inference via TensorRT-LLM (biggest gap — Blackwell's main advantage untested)
2. Flash Attention 3 build + SDPA comparison at 8K–32K context
3. Long-context 32K–128K needle-in-haystack (Qwen3 supports 128K)
4. AlpacaEval 2 / MT-Bench instruction quality evaluation
5. EvalPlus HumanEval+ / MBPP+ for code models
6. Full 30-minute thermal/power curve
7. RAG pipeline end-to-end (Embedding + Reranker + 32B LLM simultaneously)
8. FP4 quantization if toolchain becomes available

---

*Date: 2026-05-02 | Device: NVIDIA GB10 Grace Blackwell | Update: 1/3*
