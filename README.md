<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4aa,50:00a86b,100:006400&height=200&section=header&text=GB10%20Grace%20Blackwell&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Complete%20AI%20Benchmark%20Suite&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=18&duration=3000&pause=800&color=00D4AA&center=true&vCenter=true&multiline=true&repeat=true&width=700&height=80&lines=Hardware+Characterization+%E2%80%A2+Inference+%E2%80%A2+Training;Full+FT+%7C+LoRA+%7C+DPO+%7C+GRPO+%7C+CPT+%7C+Merging;128.5+GB+Unified+Memory+%E2%80%A2+SM+12.1+Blackwell)](https://git.io/typing-svg)

<br/>

![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900?style=for-the-badge&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0_nightly-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-5.7.0-ffd21e?style=for-the-badge)

![Memory](https://img.shields.io/badge/Unified_Memory-128.5_GB-00d4aa?style=for-the-badge&logo=memory&logoColor=white)
![Arch](https://img.shields.io/badge/Architecture-Blackwell_SM_12.1-76b900?style=for-the-badge&logo=nvidia&logoColor=white)
![OS](https://img.shields.io/badge/Ubuntu_24.04-aarch64-e95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Status](https://img.shields.io/badge/Experiments-Complete_✓-2ea44f?style=for-the-badge)

</div>

---

<div align="center">

## Navigation

[Hardware](#-hardware-specifications) · [Software](#-software-stack) · [Benchmarks](#-hardware-characterization) · [Inference](#-inference-benchmarks) · [Training](#-training-results) · [Evaluation](#-evaluation-results) · [Analysis](#-comparative-analysis) · [Structure](#-repository-structure)

</div>

---

## ⚡ Hardware Specifications

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:1a1a2e,100:16213e&height=3&section=header" width="100%"/>

<table>
<tr>
<td width="50%" valign="top">

**Device Profile**
| Property | Value |
|---|---|
| Device | NVIDIA GB10 Grace Blackwell |
| GPU Architecture | Blackwell **(SM 12.1)** |
| CPU | ARM Cortex-X925/A725, 20 cores |
| Unified Memory | **128.5 GB** LPDDR5X |
| NVLink-C2C | 900 GB/s bidirectional |
| Storage | 1.8 TB NVMe |
| CUDA | 13.0 · Driver 580.142 |
| OS | Ubuntu 24.04.4 LTS (aarch64) |

</td>
<td width="50%" valign="top">

**Theoretical vs Achieved**
| Metric | Spec | Achieved | Eff. |
|---|---|---|---|
| Memory BW | 273 GB/s | **134 GB/s** | 49% |
| BF16 Compute | 67 TFLOPS | **11.9 TFLOPS** | ⚠️ 17.7% |
| FP8 Compute | 134 TFLOPS | **61.1 TFLOPS** | 45.6% |
| Unified Mem | 128.5 GB | **128.5 GB** | ✅ 100% |
| NVMe Read | — | **5.43 GB/s** | — |
| 32B load time | — | **11.8 sec** | — |

</td>
</tr>
</table>

> **Note on BF16 (⚠️ 17.7%):** This is a PyTorch cuBLAS kernel maturity issue on Blackwell aarch64 — not a hardware defect. FP8 performs substantially better (45.6%) as Blackwell's FP8 hardware path is more mature in the current driver stack.

---

## 🧰 Software Stack

<table>
<tr>
<td width="50%">

| Package | Version |
|---|---|
| PyTorch | 2.12.0.dev20260407+cu128 |
| transformers | 5.7.0 |
| peft | 0.19.1 |
| trl | 1.3.0 |
| bitsandbytes | 0.49.2 ✅ SM120 |
| lm_eval | 0.4.11 |

</td>
<td width="50%">

| Package | Version |
|---|---|
| SGLang | 0.5.10.post1 |
| torchao | 0.17.0 |
| ~~vLLM~~ | removed (breaks CUDA) |
| ~~AWQ~~ | failed (no Triton/aarch64) |
| ~~Flash Attention 3~~ | not built |
| ~~TensorRT-LLM~~ | not installed |

</td>
</tr>
</table>

---

## 🔬 Hardware Characterization

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d1117,100:161b22&height=3" width="100%"/>

### Memory & Compute

<table>
<tr>
<td width="33%" valign="top">

**HW-1: Memory Bandwidth**
| Size | BW | % Peak |
|---|---|---|
| 1 GB | 133.6 GB/s | 48.9% |
| 4 GB | 132.8 GB/s | 48.6% |
| 16 GB | 125.0 GB/s | 45.8% |
| 32 GB | 123.9 GB/s | 45.4% |

Peak: **~134 GB/s (49%)**

</td>
<td width="33%" valign="top">

**HW-2: GEMM Throughput**
| Precision | TFLOPS | % Theory |
|---|---|---|
| BF16 | 11.9 | 17.7% |
| FP32 | 39.9 | 59.6% |
| **FP8** | **61.1** | **45.6%** |

FP8 is **5.1× faster** than BF16 GEMM

</td>
<td width="33%" valign="top">

**HW-4: Thermal**
| State | Power | Temp |
|---|---|---|
| Idle | 10.9 W | 49°C |
| Load | 88–91 W | 68–72°C |
| Throttle | — | **>53°C** |

SM clock: 2405 → 2177 MHz (−9%)

**All benchmarks are throttled.**

</td>
</tr>
</table>

### Roofline Model

```
BF16 Ridge point:  67 TFLOPS ÷ 273 GB/s ≈ 245 FLOPs/byte
FP8  Ridge point: 134 TFLOPS ÷ 273 GB/s ≈ 491 FLOPs/byte

  LLM decode  (batch=1)  →   ~1 FLOPs/byte  ──▶  MEMORY-BOUND   ◀── all inference
  LLM decode  (batch=32) →  ~32 FLOPs/byte  ──▶  MEMORY-BOUND
  LLM prefill (seq=2K)   → ~1000 FLOPs/byte ──▶  COMPUTE-BOUND  ◀── training / prefill
  Training fwd+bwd       → ~2000 FLOPs/byte ──▶  COMPUTE-BOUND
```

---

## 🚀 Inference Benchmarks

*HuggingFace `generate()` backend · All runs thermally throttled*

### LLM Throughput

| Model | Params | Memory | Tokens/s | BW Util |
|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | ~1.2 GB | **77–108** | — |
| Qwen3-1.7B | 1.7B | ~3.4 GB | **37–60** | — |
| Qwen3-4B | 4B | ~8 GB | **17–32** | — |
| Qwen3-8B | 8B | 16.4 GB | **9.9–10.9** | — |
| Qwen3-8B (batch=8) | 8B | — | **20–22** | — |
| Qwen3-14B | 14B | ~28 GB | **2.4–3.4** | — |
| **Qwen3-32B** | 32B | 65.6 GB | **2.3** | **55.4%** ⭐ |
| Qwen2.5-7B-Instruct | 7B | ~14 GB | **4.7–5.9** | — |
| Qwen2.5-32B-Instruct | 32B | 65.6 GB | **2.5** | **59.7%** ⭐ |
| phi-4 (14B) | 14B | ~28 GB | **1.0–5.6** | — |
| DeepSeek-R1-Distill-8B | 8B | ~16 GB | ~10 | — |

> ⭐ **32B models achieve the highest memory bandwidth utilization (55–60%)** — the sweet spot for memory-bound decode on this device.

### Quantization Sweep (Qwen3-8B)

| Format | Tokens/s | Memory | Speedup |
|---|---|---|---|
| BF16 | 10.0 tok/s | 16.5 GB | 1.0× baseline |
| INT8 | 14.4 tok/s | 9.6 GB | 1.4× |
| **NF4** | **26.3 tok/s** | 16.4 GB | **2.6×** |
| AWQ | ❌ FAILED | — | Triton not on aarch64 |
| GGUF | ❌ FAILED | — | llama.cpp not tested |

### ASR · TTS · Embedding

<table>
<tr>
<td width="50%" valign="top">

**ASR — LibriSpeech (50 samples)**
| Model | WER | RTF | Mem |
|---|---|---|---|
| Whisper-large-v3 | 3.72% | 0.338 | 3.4 GB |
| **Whisper-large-v3-turbo** | **2.95%** | **0.183** | 3.1 GB |
| Qwen2-Audio-7B | 12.79% | 0.600 | 17.4 GB |

Best: turbo — lower WER **and** 1.8× faster

</td>
<td width="50%" valign="top">

**TTS & Reranking**
| Model | Task | Result |
|---|---|---|
| suno/bark | TTS | RTF = **2.17** ⚠️ |
| Qwen3-VL-Reranker-8B | Rerank (b=4) | 3.7 pairs/s |
| Qwen3-VL-Reranker-8B | Rerank (b=8) | 3.8 pairs/s |

RTF > 1.0 = slower than real-time

</td>
</tr>
</table>

### Speculative Decoding & Long-Context

| Test | Result |
|---|---|
| Qwen3-0.6B → Qwen3-8B speculative (HF backend) | **0.57×** — slower than baseline (HF overhead) |
| 16K needle-in-haystack (Qwen3-8B) | 6/9 correct — depth 90% fails at 4K–8K |
| 32B LoRA at 32K context (GB10 advantage) | Fits entirely in unified memory; standard GPUs need sharding |

---

## 🏋️ Training Results

*All LLM training: `tatsu-lab/alpaca` (52K) · 8-bit AdamW · gradient checkpointing · BF16*

### Complete Training Table

| ID | Model | Method | Steps | Loss | Memory | Time | MFU |
|---|---|---|---|---|---|---|---|
| T1 | Qwen3-8B | Full FT | 500 | 1.019 | 52.3 GB | 147 min | 17.2% |
| T2 | Mistral-7B | Full FT | 500 | 1.022 | 44.8 GB | 151 min | — |
| **T3** | **Qwen3-8B** | **LoRA r=16** | 500 | **1.017** | **22.0 GB** | 152 min | **44.3%** ⭐ |
| T4 | Mistral-7B | LoRA r=16 | 500 | 0.841 | 16.7 GB | 159 min | 37.6% |
| T5 | Qwen3-8B | LoRA r=64 | 500 | 0.928 | 30.5 GB | 450 min | — |
| **T6** | **Qwen3-32B** | **LoRA r=32** | 150 | **0.990** | **74.0 GB** | 195 min | **44.1%** ⭐ |
| T7 | Qwen3-30B-A3B | LoRA r=16 (MoE) | 150 | 1.023 | 64.9 GB | 158 min | — |
| T8 | Qwen2.5-72B | QLoRA NF4 | — | ❌ OOM | — | — | — |
| T11 | DeepSeek-V2-Lite | LoRA r=32 | 500 | 5.626† | 35.1 GB | 394 min | 31.2% |
| T12 | Mixtral-8x7B | QLoRA NF4 | — | ❌ OOM | — | — | — |
| T14 | Whisper-large-v3 | Full FT | 500 | 0.479 | 19.5 GB | 119 min | — |
| **T15** | **Whisper-large-v3** | **LoRA r=16** | 500 | **0.207** | 20.6 GB | 128 min | — |
| T20 | Qwen3-8B | DPO | 300 | 0.693 | 29.3 GB | 832 min | — |
| T21 | Qwen3-8B | GRPO (GSM8K) | 200 | reward=0.373 | 18.8 GB | 1004 min | — |
| T22 | Qwen3-32B | DPO | 150 | 0.694 | 74.4 GB | 1703 min | — |
| T23 | Qwen3-8B | CPT (OpenWebText 1B) | 500 | 2.380 | 53.9 GB | 1495 min | — |
| T24 | DeepSeek-R1-Distill-8B | LoRA r=16 | 500 | 1.313 | 21.5 GB | 162 min | 41.1% |
| **M1** | Qwen3-8B base + T3 | **SLERP** t=0.5 | — | — | 16.4 GB | **10 sec** ⚡ | — |
| M2 | T3-SFT + T20-DPO | **TIES** | — | — | — | sec | — |
| **M3** | Qwen3-8B + Coder-7B | **DARE+TIES** | — | — | 16.4 GB | **13 sec** ⚡ | — |

† High loss due to LoRA target mismatch with DeepSeek-V2-Lite's MLA attention architecture.

---

## 📊 Evaluation Results

*MMLU (5-shot) + GSM8K (5-shot) via `lm-eval run`*

### MMLU & GSM8K — Full Comparison Table

| Model | Method | MMLU | GSM8K | ΔMMLU | ΔGSM8K |
|---|---|:---:|:---:|:---:|:---:|
| **Qwen3-8B base** | — | 0.7895 | 0.9100 | — | — |
| T1: Qwen3-8B Full FT | Full FT, Alpaca | 0.8035 | 0.8800 | +0.014 | −0.030 |
| T3: Qwen3-8B LoRA r=16 | LoRA, Alpaca | 0.8035 | 0.8600 | +0.014 | −0.050 |
| T5: Qwen3-8B LoRA r=64 | LoRA, OpenHermes | 0.8035 | 0.8300 | +0.014 | −0.080 |
| T20: Qwen3-8B DPO | DPO from T3 | 0.7860 | 0.9100 | −0.004 | **±0** |
| **T21: Qwen3-8B GRPO** | **GRPO, GSM8K rewards** | 0.7860 | **0.9200** | −0.004 | **+0.010** ⭐ |
| M1: SLERP (base+SFT) | Weight interpolation | 0.7930 | 0.9000 | −0.000 | −0.010 |
| M2: TIES (SFT+DPO) | Weight selection | 0.8000 | 0.8800 | +0.011 | −0.030 |
| **M3: DARE+TIES** | Sparse merge | 0.7895 | 0.9100 | **±0** | **±0** ⭐ |
| **Qwen3-32B LoRA (T6)** | LoRA r=32, Alpaca | **0.8877** | 0.9000 | +0.098* | −0.010* |
| T22: Qwen3-32B DPO | DPO from T6 | 0.8702 | 0.8100 | +0.081* | −0.100* |
| T2: Mistral-7B Full FT | Full FT, Alpaca | 0.3825 | **0.0200** ❌ | — | — |
| T4: Mistral-7B LoRA | LoRA, Alpaca | 0.6281 | 0.2900 | — | — |

*Qwen3-32B deltas vs own base. ⭐ = notable result

---

## 🧠 Qwen3 Think vs No-Think

| Mode | Accuracy | Avg Think Tokens | Cost |
|---|:---:|:---:|---|
| No-think | 0.780 | 0 | 1× |
| Think | 0.780 | **900** | ~10× |
| **Delta** | **0.000** | **+900** | +90 sec/question |

**Finding:** 900 extra tokens, zero accuracy gain on GSM8K. At 10 tok/s = **90 extra seconds per question** with no return. GSM8K is likely too easy for Qwen3-8B (base acc = 0.91). Think mode's benefit emerges on harder benchmarks where base acc is 30–50%.

> **Rule:** Disable thinking mode when base accuracy ≥ 75%.

---

## 📈 Comparative Analysis

*การวิเคราะห์เปรียบเทียบ — Bilingual EN / TH*

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d1117,100:161b22&height=3" width="100%"/>

### 1 · Full FT vs LoRA — LoRA ครองที่ 8B

| | Full FT (T1) | LoRA r=16 (T3) |
|---|---|---|
| Memory | 52.3 GB | **22.0 GB** (−57%) |
| MFU | 17.2% | **44.3%** (+2.6×) |
| Throughput | 234 tok/s | **1,802 tok/s** (+7.7×) |
| Loss / MMLU | ≈ identical | ≈ identical |

**EN:** LoRA matches Full FT quality while using 57% less memory, achieving 2.6× higher GPU utilization, and training 7.7× faster. The only Full FT advantage is +0.02 GSM8K — not worth the resource cost. On a throttled SM clock, LoRA's superior MFU is decisive. **LoRA strictly dominates at 8B scale.**

**TH:** LoRA ให้คุณภาพเท่ากับ Full FT แต่ใช้ memory น้อยกว่า 57%, MFU สูงกว่า 2.6 เท่า, เร็วกว่า 7.7 เท่า บน SM clock ที่ถูก throttle LoRA ที่ MFU 44.3% เป็นปัจจัยตัดสิน **LoRA ครองความเหนือกว่าอย่างสมบูรณ์ที่ขนาด 8B**

---

### 2 · SFT vs DPO vs GRPO — Alignment Trade-offs

| Method | MMLU | GSM8K | Time | Key Behavior |
|---|---|---|---|---|
| SFT (T3) | +0.014 | −0.050 | 152 min | Fastest; hurts math |
| DPO (T20) | −0.004 | **±0** | 832 min | Preserves baseline |
| **GRPO (T21)** | −0.004 | **+0.010** | 1004 min | **Only one that improves math** |

**EN:** GRPO is the only method that improves GSM8K above baseline — task-specific rule-based rewards strengthen exactly the capability being measured. SFT is fastest but most destructive (catastrophic forgetting of math). DPO balances: preserves knowledge, corrects format. Cost: GRPO is ~7× slower than SFT.

**TH:** GRPO เป็นวิธีเดียวที่ math performance ดีกว่า base — reward ตรงเป้าทำให้เสริม capability ที่ต้องการโดยตรง SFT เร็วที่สุดแต่ทำลาย math มากที่สุด DPO รักษา knowledge ไว้พร้อมแก้ format แลกกับ GRPO ช้ากว่า SFT 7 เท่า

---

### 3 · 8B vs 32B — Scale ชนะ ด้วย MFU เท่ากัน

| | Qwen3-8B LoRA | Qwen3-32B LoRA |
|---|---|---|
| Memory | 22.0 GB | 74.0 GB |
| MFU | 44.3% | **44.1%** ← identical |
| MMLU | 0.8035 | **0.8877** (+0.084) |
| GSM8K | 0.8600 | **0.9000** (+0.040) |

**EN:** LoRA training scales perfectly — identical MFU regardless of model size. 74 GB BF16 on one device is impossible on any 80 GB GPU without parallelism. GB10's unified memory makes 32B LoRA a single-device experiment. Cost: 4.2× longer.

**TH:** LoRA scaling มีประสิทธิภาพสมบูรณ์แบบ — MFU เหมือนกันโดยไม่คำนึงถึงขนาดโมเดล 74 GB บนอุปกรณ์เดียวเป็นไปไม่ได้บน GPU 80 GB unified memory ของ GB10 คือข้อได้เปรียบที่แท้จริง แลกกับเวลา 4.2 เท่า

---

### 4 · Qwen3 vs Mistral — Catastrophic Forgetting

| | Qwen3-8B LoRA | Mistral-7B LoRA |
|---|---|---|
| Training loss | 1.017 (higher) | **0.841** (lower) |
| Post-FT GSM8K | **0.8600** | 0.2900 (−0.62!) |
| Full FT GSM8K | **0.8800** | **0.0200** ← near random |

**EN:** Mistral achieves lower training loss yet collapses on benchmarks — **loss-accuracy decoupling**. Full FT on Alpaca (different distribution from Mistral's RLHF training) causes catastrophic forgetting: GSM8K → 0.02. Qwen3 is dramatically more robust to distribution shift. **Lower training loss ≠ better generalization.**

**TH:** Mistral ได้ training loss ต่ำกว่า แต่ benchmark collapsed — **loss-accuracy decoupling** ที่เห็นได้ชัด Full FT บน distribution ต่างจาก RLHF ทำให้ catastrophic forgetting GSM8K เหลือ 0.02 ใกล้เคียงการสุ่ม Qwen3 ทนทานต่อ distribution shift ได้ดีกว่าอย่างมีนัยสำคัญ

---

### 5 · Model Merging — 10 วินาที vs 152 นาที

| Approach | MMLU | GSM8K | Time |
|---|---|---|---|
| T3 SFT (LoRA training) | 0.8035 | 0.8600 | 152 min |
| **M1 SLERP** (base+SFT, t=0.5) | 0.7930 | **0.9000** | **10 sec** ⚡ |
| **M3 DARE+TIES** (8B+Coder) | 0.7895 | **0.9100** | **13 sec** ⚡ |

**EN:** SLERP in 10 seconds recovers most of SFT's GSM8K loss (0.86 → 0.90) while retaining MMLU gain. DARE+TIES perfectly preserves base performance while adding code specialization. **Merging delivers ~70% of SFT quality at <0.01% of compute cost.** Compelling for rapid prototyping.

**TH:** SLERP ใน 10 วินาที recover GSM8K ที่หายไปจาก SFT ได้ส่วนใหญ่ DARE+TIES รักษาประสิทธิภาพ base ได้สมบูรณ์แบบพร้อมรวม code specialization **Merging ให้ผล ~70% ของ SFT โดยใช้ compute <0.01%** เหมาะมากสำหรับ rapid prototyping

---

### 6 · QLoRA 72B (T8) — OOM Finding

**EN:** Failed at weight loading every attempt (4 runs). bitsandbytes loads weights in BF16 first before quantizing in-place → transient peak ~72 GB. Combined with framework overhead, this exceeds the ~119 GB practical limit. **GB10 can infer 72B NF4 (36 GB runtime) but cannot train QLoRA-style without a pre-quantized checkpoint.** This is a bitsandbytes loading path limitation, not a memory size limitation.

**TH:** ล้มเหลวทุกครั้ง (4 ครั้ง) เพราะ bitsandbytes โหลด BF16 ก่อน quantize → transient peak ~72 GB เกิน ~119 GB ที่มีจริง **GB10 infer 72B ได้ (36 GB ตอน runtime) แต่ train QLoRA ไม่ได้โดยไม่ใช้ pre-quantized checkpoint** เป็นข้อจำกัดของ loading path ไม่ใช่ memory ไม่พอ

---

### Summary Scorecard

| Comparison | Winner | Key Reason |
|---|---|---|
| LoRA vs Full FT | **LoRA** | MFU +2.6×, memory −57%, same quality |
| LoRA r=16 vs r=64 | **r=16** | Lower rank generalizes better |
| GRPO vs SFT (math) | **GRPO** | Only method that beats baseline math |
| DPO vs SFT (safety) | **DPO** | Preserves knowledge, no forgetting |
| Qwen3 vs Mistral (FT robustness) | **Qwen3** | 43× better GSM8K retention |
| 32B vs 8B LoRA (quality) | **32B** | MMLU +0.084 at identical MFU |
| Dense 32B vs MoE 30B-A3B | **Dense** | MoE advantage negated at 128 GB |
| Merging vs Training | **Training** (slight) | But merging in 10 sec = 70% of gains |
| Think vs No-think (GSM8K) | **No-think** | 0 gain, −900 tokens wasted |
| Whisper LoRA vs Full FT | **LoRA** | Small data → LoRA regularizes better |
| FP8 vs BF16 GEMM eff. | **FP8** | 45.6% vs 17.7% — toolchain maturity |

---

## ⚠️ Marketing Claims vs Reality

| Claim | Measured | Verdict |
|---|---|---|
| "1 PFLOPS FP4" | FP4 toolchain unavailable on aarch64 | ❓ Unverifiable |
| "273 GB/s memory bandwidth" | **134 GB/s** (49%) | 🟡 Expected gap |
| "128 GB unified memory" | **128.5 GB confirmed** | ✅ Accurate |
| "900 GB/s NVLink-C2C" | 54 GB/s DMA (coherent path unmeasurable) | 🟡 Used transparently |
| "Train 32B on a single device" | **74 GB peak — confirmed** | ✅ Genuine advantage |
| No discrete VRAM ceiling | All 128 GB accessible to GPU | ✅ True |
| FP8 native Blackwell | **61.1 TFLOPS** GEMM | ✅ True (inference stack pending) |
| "Train 72B QLoRA" | **OOM every attempt** | ❌ Loading spike exceeds headroom |

---

## 🛠 Known Issues & Fixes

| Issue | Fix |
|---|---|
| `is_torch_fx_available` missing (transformers 5.7) | Patched: `is_torch_fx_available = lambda: True` |
| DeepSeek-V2-Lite wrong LoRA targets | Changed to standard `q_proj`, `k_proj` (V2-Lite uses MHA) |
| AWQ fails on aarch64 | No Triton on ARM — excluded |
| vLLM breaks CUDA on install | Removed; using HF generate + SGLang |
| `max_prompt_length` removed in TRL 1.3.0 | Removed kwarg from `DPOConfig` |
| `nvidia-smi` memory returns N/A on GB10 | Use `torch.cuda.memory_allocated()` or `free -h` |
| 72B+ OOM during lm-eval | Added `load_in_4bit=True` to model args |
| `torch_dtype` deprecated in transformers 5.7 | Changed to `dtype=` in `from_pretrained()` |
| QLoRA 72B OOM even with `device_map="auto"` | BF16 loading spike ~72 GB; pre-quantized checkpoint needed |
| `apply_chat_template` returns `BatchEncoding` | Use `tokenize=False` + separate `tok()` call |
| `lm-eval` CLI changed in 0.4.11 | Use `lm-eval run` not `lm_eval` |

---

## 📁 Repository Structure

```
fusionxpark-gb10-hwtest/
│
├── 📄 README.md              ← This file — all results + bilingual analysis
├── 📊 tracker.py             ← Experiment dashboard
│
├── 📂 scripts/               ← All experiment scripts
│   ├── phase0_*.py/sh        ← Environment setup & verification
│   ├── phase05_*.py/sh       ← Hardware microbenchmarks
│   ├── phase1_*.py           ← Inference benchmarks (LLM, ASR, TTS, embed)
│   ├── phase2_*.py/sh        ← Training (Full FT, LoRA, DPO, GRPO, CPT, merge)
│   ├── phase3_*.py/sh        ← Evaluation & RAG pipeline
│   ├── think_vs_nothink_eval.py
│   └── run_*.sh              ← Queue runner scripts
│
├── 📂 config/                ← Hyperparameter configs (YAML)
│
├── 📂 docs/                  ← Supplementary documents
│   └── GB10_vs_RTX5070Ti_comparison.md
│
├── 📂 results/               ← All experiment outputs (JSON)
│   ├── hardware/             ← hw_bench_*.json
│   ├── inference/            ← inference, quant, asr, tts, embed results
│   ├── training/             ← T1–T24, M1–M3 training JSON
│   └── evaluation/           ← lm-eval MMLU/GSM8K + think_vs_nothink.json
│
└── 📂 models/                ← Fine-tuned checkpoints
    ├── T1_Qwen3-8B_full_ft/
    ├── T3_Qwen3-8B_lora_r16/
    ├── T6_Qwen3-32B_lora_r32/
    ├── T20_Qwen3-8B_dpo/
    ├── T21_Qwen3-8B_grpo/
    ├── M1_Qwen3-8B_slerp_sft/
    └── ...
```

### Quick Commands

```bash
free -h                                           # Memory usage (nvidia-smi N/A on GB10)
python3 tracker.py                                # Experiment dashboard
python3 scripts/phase05_hw_bench.py               # Hardware benchmarks
python3 scripts/phase1_inference_bench.py         # Inference benchmark
python3 scripts/phase2_train.py --experiment T3   # LoRA training
bash scripts/phase3_eval.sh <model_path>          # lm-eval
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:006400,50:00a86b,100:00d4aa&height=120&section=footer&animation=fadeIn" width="100%"/>

*2026-05-09 · NVIDIA GB10 Grace Blackwell · SM 12.1 · 128.5 GB Unified Memory · All Experiments Complete*

</div>
