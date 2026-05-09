# GB10 Grace Blackwell — Comparative Analysis
# GB10 Grace Blackwell — การวิเคราะห์เปรียบเทียบ

> Bilingual analytical summary of all experiments conducted on the NVIDIA GB10 Grace Blackwell SoC.
> สรุปการวิเคราะห์เปรียบเทียบแบบสองภาษาสำหรับทุกการทดลองบน NVIDIA GB10 Grace Blackwell SoC

---

## 1. Training Methods: Full FT vs LoRA vs QLoRA
## 1. วิธีการเทรน: Full FT vs LoRA vs QLoRA

### 1A. Full FT vs LoRA (Qwen3-8B, T1 vs T3)

| Metric | Full FT (T1) | LoRA r=16 (T3) | Winner |
|---|---|---|---|
| Final Loss | 1.019 | 1.017 | ≈ Tie |
| Peak Memory | 52.3 GB | 22.0 GB | **LoRA** (−57%) |
| MFU | 17.2% | 44.3% | **LoRA** (+2.6×) |
| Throughput | 234 tok/s | 1,802 tok/s | **LoRA** (+7.7×) |
| Training Time | 147 min | 152 min | ≈ Tie |
| Post-FT MMLU | 0.8035 | 0.8035 | Tie |
| Post-FT GSM8K | 0.8800 | 0.8600 | **Full FT** (+0.02) |

**EN:** Full Fine-Tuning and LoRA produce nearly identical loss and downstream scores at 8B scale, but LoRA is dramatically more efficient: 57% less memory, 2.6× higher GPU utilization (MFU), and 7.7× higher training throughput. The only measurable advantage of Full FT is a marginal +0.02 GSM8K edge — which is easily offset by LoRA's resource savings. On GB10, where memory is abundant but compute efficiency is constrained by the throttled SM clock, LoRA's superior MFU (44.3% vs 17.2%) is the decisive factor. **LoRA strictly dominates Full FT at 8B scale on this device.**

**TH:** Full FT และ LoRA ให้ loss และคะแนน downstream ใกล้เคียงกันมากที่ขนาด 8B แต่ LoRA มีประสิทธิภาพสูงกว่าอย่างชัดเจน: ใช้ memory น้อยกว่า 57%, MFU สูงกว่า 2.6 เท่า และ throughput สูงกว่า 7.7 เท่า ข้อได้เปรียบเพียงอย่างเดียวของ Full FT คือ GSM8K สูงกว่าเล็กน้อย +0.02 ซึ่งไม่คุ้มกับทรัพยากรที่ต้องใช้เพิ่ม บน GB10 ที่ memory มีเหลือเฟือแต่ compute efficiency ถูกจำกัดจาก SM clock throttle LoRA ที่มี MFU 44.3% (vs 17.2%) เป็นปัจจัยตัดสินใจ **LoRA ครองความเหนือกว่า Full FT ที่ขนาด 8B อย่างสมบูรณ์บนอุปกรณ์นี้**

---

### 1B. LoRA Rank Scaling (Qwen3-8B: r=16 vs r=64, T3 vs T5)

| Metric | LoRA r=16 (T3) | LoRA r=64 (T5) | Delta |
|---|---|---|---|
| Final Loss | 1.017 | 0.928 | r=64 better (−0.089) |
| Peak Memory | 22.0 GB | 30.5 GB | r=64 costs +8.5 GB |
| Training Time | 152 min | 450 min | r=64 costs +3× time |
| Post-FT GSM8K | 0.8600 | 0.8300 | r=64 **worse** (−0.030) |
| Post-FT MMLU | 0.8035 | 0.8035 | Tie |

**EN:** Higher LoRA rank (r=64) achieves lower training loss (0.928 vs 1.017) but this does not translate to better downstream performance — GSM8K actually drops by 0.03. The r=64 run used a larger dataset (OpenHermes 100K vs Alpaca 52K), which confounds direct comparison, but the key finding is clear: **lower training loss does not guarantee better generalization**. r=64 costs 3× the compute time and 8.5 GB more memory for no benchmark gain. r=16 is the practical sweet spot for 8B instruction tuning.

**TH:** LoRA rank ที่สูงกว่า (r=64) ได้ training loss ต่ำกว่า (0.928 vs 1.017) แต่ไม่ได้หมายความว่าโมเดลจะดีขึ้นใน downstream tasks — GSM8K กลับแย่ลง 0.03 ต้องหมายเหตุว่า r=64 ใช้ dataset ใหญ่กว่า (OpenHermes 100K vs Alpaca 52K) ทำให้เปรียบเทียบตรงๆ ไม่ได้ 100% แต่ข้อสรุปสำคัญยังชัดเจน: **loss ต่ำกว่าใน training ไม่ได้รับประกันว่า generalization จะดีกว่า** r=64 ใช้เวลา 3 เท่าและ memory เพิ่มอีก 8.5 GB โดยไม่ได้ benchmark ที่ดีขึ้น r=16 คือจุดสมดุลที่ดีที่สุดสำหรับ 8B instruction tuning

---

### 1C. QLoRA 72B (T8) — OOM Finding

**EN:** QLoRA NF4 on Qwen2.5-72B failed at weight loading every attempt (4 runs). Root cause: while NF4 reduces the stored weights to ~36 GB, bitsandbytes loads weights in BF16 first before quantizing in-place, creating a transient peak of ~72 GB. Combined with framework overhead, activation buffers, and optimizer state, this exceeds the 119 GB practical limit (128.5 GB total minus OS, framework, and gradient buffers). **GB10 can infer 72B NF4 (36 GB at runtime) but cannot train it QLoRA-style without a pre-quantized checkpoint.** This is a fundamental limitation of the current bitsandbytes loading path, not the device's memory size.

**TH:** QLoRA NF4 บน Qwen2.5-72B ล้มเหลวทุกครั้งที่โหลด weights (4 ครั้ง) สาเหตุหลัก: แม้ NF4 จะเก็บ weights ไว้ที่ ~36 GB แต่ bitsandbytes โหลด weights เป็น BF16 ก่อนแล้วค่อย quantize in-place ทำให้มี transient peak ~72 GB รวมกับ framework overhead, activation buffers และ optimizer state ทำให้เกินขีดจำกัดจริง ~119 GB **GB10 สามารถ infer 72B NF4 ได้ (36 GB ตอน runtime) แต่ train แบบ QLoRA ไม่ได้โดยไม่ใช้ pre-quantized checkpoint** นี่คือข้อจำกัดของ bitsandbytes loading path ไม่ใช่ข้อจำกัดของขนาด memory ของอุปกรณ์

---

## 2. Alignment Methods: SFT vs DPO vs GRPO
## 2. วิธี Alignment: SFT vs DPO vs GRPO

| Method | Experiment | MMLU | GSM8K | Time | Memory |
|---|---|---|---|---|---|
| Base (no FT) | — | 0.7895 | 0.9100 | — | — |
| SFT (LoRA r=16) | T3 | 0.8035 | 0.8600 | 152 min | 22 GB |
| DPO (from T3) | T20 | 0.7860 | 0.9100 | 832 min | 29 GB |
| GRPO (GSM8K rewards) | T21 | 0.7860 | **0.9200** | 1004 min | 19 GB |

**EN:** The three alignment methods produce fundamentally different trade-off profiles:

- **SFT** improves MMLU (+0.014) by teaching instruction format but hurts GSM8K (−0.05) through catastrophic forgetting of math reasoning. It is the fastest (152 min) but most destructive to existing capabilities.
- **DPO** preserves GSM8K at the baseline level (0.91) while slightly reducing MMLU (−0.004). The preference-based training signal corrects format and tone without overwriting underlying knowledge. However, it requires a completed SFT checkpoint first and takes 5.5× longer than SFT.
- **GRPO** is the only method that improves GSM8K above baseline (+0.01). By using rule-based math rewards rather than next-token prediction, it strengthens exactly the capability being measured. The cost is severe: ~7× slower than SFT, requiring 200 rollout steps with 8 completions per prompt. MMLU drops slightly (−0.004) — an acceptable trade for math tasks.

**Key insight:** If the goal is math performance, GRPO is the only correct choice. If the goal is general instruction quality with preserved reasoning, DPO from a SFT checkpoint is preferable. Pure SFT is fastest but most harmful to reasoning benchmarks.

**TH:** วิธี alignment ทั้งสามให้ trade-off profile ที่แตกต่างกันอย่างสิ้นเชิง:

- **SFT** ช่วย MMLU (+0.014) โดยสอน instruction format แต่ทำลาย GSM8K (−0.05) จาก catastrophic forgetting ของ math reasoning เป็นวิธีเร็วที่สุด (152 นาที) แต่ทำลาย capability เดิมมากที่สุด
- **DPO** รักษา GSM8K ไว้ที่ระดับ baseline (0.91) ขณะที่ MMLU ลดลงเล็กน้อย (−0.004) สัญญาณ preference-based training แก้ไข format และ tone โดยไม่เขียนทับ knowledge พื้นฐาน อย่างไรก็ตามต้องมี SFT checkpoint ก่อนและใช้เวลานานกว่า SFT 5.5 เท่า
- **GRPO** เป็นวิธีเดียวที่ทำให้ GSM8K ดีกว่า baseline (+0.01) การใช้ rule-based math rewards แทน next-token prediction ทำให้เสริมความสามารถที่ต้องการวัดโดยตรง แลกกับความช้า ~7 เท่าเทียบกับ SFT และต้องทำ rollout 200 steps

**ข้อสรุปสำคัญ:** ถ้าเป้าหมายคือ math performance GRPO เป็นทางเลือกที่ถูกต้องเพียงอย่างเดียว ถ้าเป้าหมายคือ instruction quality ทั่วไปพร้อมรักษา reasoning ให้ใช้ DPO ต่อจาก SFT checkpoint SFT เพียงอย่างเดียวเร็วที่สุดแต่ทำร้าย reasoning benchmarks มากที่สุด

---

## 3. Model Size Scaling: 8B vs 32B
## 3. การ Scale ขนาดโมเดล: 8B vs 32B

| Metric | Qwen3-8B LoRA (T3) | Qwen3-32B LoRA (T6) | Delta |
|---|---|---|---|
| Peak Memory | 22.0 GB | 74.0 GB | +52 GB |
| MFU | 44.3% | 44.1% | ≈ Identical |
| Tokens/s | 1,802 | 447 | −75% |
| Time (150 steps) | ~46 min | 195 min | 4.2× slower |
| Final Loss | 1.017 | 0.990 | 32B slightly better |
| MMLU | 0.8035 | **0.8877** | +0.084 |
| GSM8K | 0.8600 | 0.9000 | +0.040 |

**EN:** Scaling from 8B to 32B LoRA delivers a substantial quality improvement: +0.084 MMLU and +0.040 GSM8K. Critically, MFU is nearly identical (44.1% vs 44.3%), confirming that the LoRA training regime scales efficiently — the hardware is utilized equally well regardless of model size. The 32B model fits in 74 GB, which is impossible on any standard 80 GB GPU without quantization, making GB10's unified memory a genuine differentiator. The cost is 4.2× longer wall-clock time. For researchers who need the best quality and are not time-constrained, 32B LoRA is clearly superior. For iteration speed, 8B LoRA at 44% MFU is the practical choice.

**TH:** การ scale จาก 8B เป็น 32B LoRA ให้คุณภาพที่ดีขึ้นอย่างมีนัยสำคัญ: MMLU +0.084 และ GSM8K +0.040 สิ่งที่น่าสังเกตคือ MFU แทบเหมือนกัน (44.1% vs 44.3%) ยืนยันว่า LoRA training regime scale ได้อย่างมีประสิทธิภาพ — hardware ถูกใช้งานอย่างมีประสิทธิภาพเท่ากันโดยไม่คำนึงถึงขนาดโมเดล โมเดล 32B ใช้ 74 GB ซึ่งเป็นไปไม่ได้บน GPU มาตรฐาน 80 GB โดยไม่ quantize ทำให้ unified memory ของ GB10 เป็นข้อได้เปรียบที่แท้จริง แลกกับเวลาที่นานกว่า 4.2 เท่า สำหรับนักวิจัยที่ต้องการคุณภาพสูงสุดและไม่จำกัดเวลา 32B LoRA ดีกว่าอย่างชัดเจน สำหรับ iteration speed 8B LoRA ที่ 44% MFU เป็นทางเลือกที่ใช้งานได้จริง

---

## 4. Cross-Family Comparison: Qwen3 vs Mistral
## 4. เปรียบเทียบข้ามตระกูล: Qwen3 vs Mistral

### 4A. Full FT (T1 Qwen3-8B vs T2 Mistral-7B)

| Metric | Qwen3-8B Full FT (T1) | Mistral-7B Full FT (T2) |
|---|---|---|
| Peak Memory | 52.3 GB | 44.8 GB |
| Training Loss | 1.019 | 1.022 |
| Post-FT MMLU | 0.8035 | **0.3825** |
| Post-FT GSM8K | 0.8800 | **0.0200** |

### 4B. LoRA r=16 (T3 Qwen3-8B vs T4 Mistral-7B)

| Metric | Qwen3-8B LoRA (T3) | Mistral-7B LoRA (T4) |
|---|---|---|
| Training Loss | 1.017 | **0.841** |
| Memory | 22.0 GB | 16.7 GB |
| Post-FT MMLU | **0.8035** | 0.6281 |
| Post-FT GSM8K | **0.8600** | 0.2900 |

**EN:** These results reveal a critical warning about Full Fine-Tuning on RLHF-trained instruction models. Mistral-7B-Instruct-v0.3 was trained with RLHF on a specific distribution. Full FT on Alpaca (a different distribution) causes **catastrophic forgetting**: MMLU collapses from its original ~0.6 to 0.38, and GSM8K drops to 0.02 — near random. This is not a hardware issue; it is a training stability issue. Even Mistral LoRA shows significant degradation (MMLU=0.63, GSM8K=0.29), whereas Qwen3-8B LoRA retains strong performance (MMLU=0.80, GSM8K=0.86).

The training loss tells an opposite story: Mistral achieves lower loss (0.841 vs 1.017). This demonstrates the **loss-accuracy decoupling problem** — lower training loss on the fine-tuning dataset does not correlate with better generalization on held-out benchmarks. Qwen3's architecture and pre-training apparently make it significantly more robust to distribution shift during fine-tuning.

**TH:** ผลลัพธ์นี้เปิดเผยคำเตือนสำคัญเกี่ยวกับ Full FT บนโมเดล instruction ที่ผ่าน RLHF มาแล้ว Mistral-7B-Instruct-v0.3 ถูกเทรนด้วย RLHF บน distribution เฉพาะ การทำ Full FT บน Alpaca (distribution ต่างออกไป) ทำให้เกิด **catastrophic forgetting**: MMLU ร่วงจาก ~0.6 เดิมเหลือ 0.38 และ GSM8K ร่วงเหลือ 0.02 ใกล้เคียงกับการตอบแบบสุ่ม นี่ไม่ใช่ปัญหาของ hardware แต่เป็นปัญหาความเสถียรของการเทรน แม้แต่ Mistral LoRA ก็ยังแสดงการเสื่อมถอยอย่างมีนัยสำคัญ (MMLU=0.63, GSM8K=0.29) ขณะที่ Qwen3-8B LoRA ยังคงประสิทธิภาพสูง

Training loss บอกเรื่องตรงข้าม: Mistral ได้ loss ต่ำกว่า (0.841 vs 1.017) นี่แสดงให้เห็น **ปัญหา loss-accuracy decoupling** — loss ที่ต่ำกว่าใน fine-tuning dataset ไม่สัมพันธ์กับ generalization ที่ดีกว่าบน benchmark Architecture และ pre-training ของ Qwen3 ทำให้มันทนทานต่อ distribution shift ระหว่าง fine-tuning ได้ดีกว่าอย่างมีนัยสำคัญ

---

## 5. Dense vs MoE Architecture
## 5. สถาปัตยกรรม Dense vs MoE

| Model | Architecture | Params (total/active) | Memory | Loss | Time (150 steps) |
|---|---|---|---|---|---|
| Qwen3-32B LoRA (T6) | Dense | 32B / 32B | 74.0 GB | 0.990 | 195 min |
| Qwen3-30B-A3B LoRA (T7) | MoE | 30B / 3B | 64.9 GB | 1.023 | 158 min |

**EN:** The MoE model (Qwen3-30B-A3B) activates only 3B parameters per token but still requires all 30B expert weights in memory. This means memory savings over the dense 32B model are modest (64.9 GB vs 74 GB, −12%). Training speed is slightly faster (158 vs 195 min for 150 steps) due to fewer active FLOPs per forward pass, but the loss is higher (1.023 vs 0.990). The 32B dense model's advantage is consistent quality at the cost of slightly more memory and compute. For GB10's memory profile, the MoE advantage is largely negated — the device can hold both equally well. MoE becomes more compelling at 235B+ scale where only a fraction of weights can fit in memory at all.

**TH:** โมเดล MoE (Qwen3-30B-A3B) activate เพียง 3B parameters ต่อ token แต่ยังต้องเก็บ weights ของ expert ทั้งหมด 30B ใน memory หมายความว่าประหยัด memory ได้น้อยมากเมื่อเทียบกับ dense 32B (64.9 GB vs 74 GB ประหยัด 12%) ความเร็วเทรนเร็วกว่าเล็กน้อย (158 vs 195 นาที สำหรับ 150 steps) เพราะใช้ FLOPs ต่อ forward pass น้อยกว่า แต่ loss สูงกว่า (1.023 vs 0.990) สำหรับ memory profile ของ GB10 ข้อได้เปรียบของ MoE ถูกลดทอนลงมาก — อุปกรณ์นี้รองรับทั้งคู่ได้สบาย MoE จะน่าสนใจกว่ามากเมื่อ scale ถึง 235B+ ที่ต้องการโหลด weights เพียงส่วนหนึ่งเข้า memory

---

## 6. Model Merging vs Fine-Tuning
## 6. Model Merging vs Fine-Tuning

| Approach | Method | MMLU | GSM8K | Cost |
|---|---|---|---|---|
| Qwen3-8B base | — | 0.7895 | 0.9100 | — |
| T3 SFT (LoRA) | Gradient training | 0.8035 | 0.8600 | 152 min |
| M1 SLERP (base+SFT) | Weight interpolation | 0.7930 | 0.9000 | **10 sec** |
| M2 TIES (SFT+DPO) | Weight selection | 0.8000 | 0.8800 | seconds |
| M3 DARE+TIES (8B+Coder) | Sparse merge | 0.7895 | 0.9100 | **13 sec** |

**EN:** Model merging produces remarkably competitive results at essentially zero compute cost:

- **M1 SLERP** (base + SFT at t=0.5) blends both models and achieves MMLU=0.793, GSM8K=0.900 in 10 seconds. This recovers most of the GSM8K loss from SFT (0.860 → 0.900) while keeping MMLU gain (+0.004 vs base). SLERP is the cheapest possible "undo the forgetting" operation.
- **M3 DARE+TIES** (Qwen3-8B + Qwen2.5-Coder-7B) perfectly preserves base performance (0.7895 / 0.9100) while integrating code specialization — suggesting the code model's knowledge was additive without overwriting general knowledge.
- **M2 TIES** (SFT + DPO merge) gives MMLU=0.800, GSM8K=0.880 — better than SFT alone on MMLU but slightly worse on GSM8K. Merging two fine-tuned models averages their strengths and weaknesses.

**Key finding:** For tasks where training cost is prohibitive, SLERP between base and SFT delivers ~70% of the SFT quality gain at <0.01% of the compute. This is a compelling strategy for rapid prototyping.

**TH:** Model merging ให้ผลลัพธ์ที่แข่งขันได้อย่างน่าประหลาดใจโดยแทบไม่ใช้ compute เลย:

- **M1 SLERP** (base + SFT ที่ t=0.5) ผสม weights ทั้งสองและได้ MMLU=0.793, GSM8K=0.900 ใน 10 วินาที ช่วย recover GSM8K ที่เสียไปจาก SFT ได้ส่วนใหญ่ (0.860 → 0.900) SLERP คือ operation "แก้การลืม" ที่ถูกที่สุดเท่าที่มี
- **M3 DARE+TIES** (Qwen3-8B + Qwen2.5-Coder-7B) รักษาประสิทธิภาพ base ได้สมบูรณ์แบบ (0.7895/0.9100) ขณะที่รวม code specialization เข้าไป แสดงให้เห็นว่า knowledge ของ code model เป็น additive โดยไม่เขียนทับ knowledge ทั่วไป
- **M2 TIES** (SFT + DPO merge) ให้ MMLU=0.800, GSM8K=0.880 ดีกว่า SFT เพียงอย่างเดียวใน MMLU แต่แย่กว่าเล็กน้อยใน GSM8K

**ข้อสรุปสำคัญ:** สำหรับงานที่ต้นทุนการเทรนสูงเกินไป SLERP ระหว่าง base และ SFT ให้ผลดีประมาณ 70% ของ SFT โดยใช้ compute น้อยกว่า 0.01% เป็นกลยุทธ์ที่น่าสนใจมากสำหรับการทดลองอย่างรวดเร็ว

---

## 7. Qwen3 Think vs No-Think Mode
## 7. Qwen3 โหมด Think vs No-Think

| Mode | GSM8K Accuracy | Avg Think Tokens | Compute Cost |
|---|---|---|---|
| No-think | 0.780 | 0 tokens | 1× |
| Think | 0.780 | **900 tokens** | **~10×** |
| Delta | **0.000** | +900 | +900 sec/50 questions |

**EN:** Qwen3-8B's thinking mode generates ~900 tokens of chain-of-thought reasoning per question but produces zero accuracy improvement on GSM8K at 50-sample scale. This result has two interpretations:

1. **GSM8K may be too easy for Qwen3-8B.** Base accuracy is already 0.91 (lm-eval 5-shot), meaning the model solves most problems without extended reasoning. Thinking mode's benefit is expected to appear on harder benchmarks (MATH competition, AIME, AMC) where base accuracy is in the 30–50% range.

2. **900 extra tokens = 90 seconds per question at 10 tok/s.** For a 50-question evaluation, thinking mode adds ~75 minutes of compute with no measurable return. This has direct implications for deployment cost: thinking mode should be reserved for problems where no-think accuracy is demonstrably insufficient.

**Practical guidance:** Disable thinking mode for tasks where base accuracy is already ≥75%. Enable it only for genuinely hard problems (math olympiad, complex multi-step reasoning) where the reasoning budget can meaningfully change the outcome.

**TH:** Thinking mode ของ Qwen3-8B สร้าง chain-of-thought reasoning ~900 tokens ต่อคำถาม แต่ให้ accuracy ที่ดีขึ้น 0% บน GSM8K ในการทดสอบ 50 ตัวอย่าง ผลลัพธ์นี้มีการตีความได้สองแบบ:

1. **GSM8K อาจง่ายเกินไปสำหรับ Qwen3-8B** ซึ่ง base accuracy อยู่ที่ 0.91 แล้ว หมายความว่าโมเดลแก้ปัญหาส่วนใหญ่ได้โดยไม่ต้องใช้ extended reasoning ประโยชน์ของ thinking mode น่าจะเห็นได้ชัดบน benchmark ที่ยากกว่า (MATH competition, AIME, AMC) ที่ base accuracy อยู่ที่ 30–50%

2. **900 extra tokens = 90 วินาทีต่อคำถาม ที่ 10 tok/s** สำหรับ evaluation 50 ข้อ thinking mode เพิ่มเวลา ~75 นาทีโดยไม่ได้ผลที่วัดได้ มีผลโดยตรงต่อต้นทุน deployment: thinking mode ควรสงวนไว้สำหรับปัญหาที่ no-think accuracy ไม่เพียงพออย่างชัดเจน

**คำแนะนำเชิงปฏิบัติ:** ปิด thinking mode สำหรับงานที่ base accuracy ≥75% แล้ว เปิดใช้เฉพาะกับปัญหายากจริงๆ (คณิตศาสตร์โอลิมปิก, multi-step reasoning ซับซ้อน) ที่ reasoning budget สามารถเปลี่ยนผลลัพธ์ได้

---

## 8. ASR: Full FT vs LoRA (Whisper)
## 8. ASR: Full FT vs LoRA (Whisper)

| Method | Experiment | Final Loss | Memory | Time |
|---|---|---|---|---|
| Full FT | T14 | 0.479 | 19.5 GB | 119 min |
| LoRA r=16 | T15 | **0.207** | 20.6 GB | 128 min |

**EN:** For Whisper fine-tuning on speech data, LoRA (r=16) achieves significantly lower loss (0.207 vs 0.479) than Full FT at nearly identical memory and time cost. This is the opposite of the LLM result (where FT and LoRA were equivalent). The explanation: Whisper's encoder-decoder architecture benefits from LoRA's regularization effect — full weight updates overfit more readily to the small CommonVoice subset (10K clips), while LoRA's reduced parameter count acts as an implicit regularizer. **For ASR fine-tuning on limited data, LoRA is superior in both efficiency and quality.**

**TH:** สำหรับการ fine-tune Whisper บนข้อมูลเสียง LoRA (r=16) ได้ loss ต่ำกว่า Full FT อย่างมีนัยสำคัญ (0.207 vs 0.479) โดยใช้ memory และเวลาใกล้เคียงกัน นี่คือผลตรงข้ามกับ LLM (ที่ FT และ LoRA เทียบกันได้) คำอธิบาย: สถาปัตยกรรม encoder-decoder ของ Whisper ได้ประโยชน์จาก regularization effect ของ LoRA — การอัปเดต weights ทั้งหมด overfit กับ CommonVoice subset ขนาดเล็ก (10K clips) ได้ง่ายกว่า ขณะที่ parameter count ที่น้อยกว่าของ LoRA ทำหน้าที่เป็น implicit regularizer **สำหรับการ fine-tune ASR ด้วยข้อมูลจำกัด LoRA เหนือกว่าทั้งด้านประสิทธิภาพและคุณภาพ**

---

## 9. Hardware: Theoretical vs Achieved
## 9. Hardware: ค่าทฤษฎี vs ค่าที่วัดได้จริง

| Metric | Theoretical | Achieved | Efficiency |
|---|---|---|---|
| Memory Bandwidth | 273 GB/s | **134 GB/s** | 49% |
| BF16 GEMM | 67 TFLOPS | **11.9 TFLOPS** | 17.7% |
| FP8 GEMM | 134 TFLOPS | **61.1 TFLOPS** | 45.6% |
| NVLink-C2C | 900 GB/s | **54 GB/s** | ~6% (DMA path) |
| Unified Memory | 128.5 GB | **128.5 GB** | 100% ✓ |

**EN:** The most striking hardware finding is the BF16 GEMM underperformance: only 11.9 TFLOPS against a 67 TFLOPS theoretical peak (17.7%). This is not a hardware defect — it reflects immature PyTorch cuBLAS kernel optimization for the Blackwell aarch64 target. FP8 performs substantially better (45.6% efficiency) because Blackwell's FP8 hardware path is more mature in the current driver stack. Memory bandwidth achieves 49% (134 GB/s of 273 GB/s), which is reasonable for a memory-bandwidth-bound workload using 32-bit tensors in a DAXPY-style benchmark.

The NVLink-C2C measurement (54 GB/s) reflects the DMA memcpy path, not the coherent shared memory path. The actual 900 GB/s coherent bandwidth is used transparently when GPU code accesses CPU-allocated memory — it cannot be measured via explicit host↔device copies.

**Thermal throttling is persistent and universal:** every benchmark exceeds the 53°C throttle threshold within minutes, reducing the SM clock from 2405 MHz to ~2200 MHz (~9% reduction). This affects all reported throughput numbers — real-world sustained performance is lower than single-query peaks.

**TH:** ผลที่น่าตกใจที่สุดด้านฮาร์ดแวร์คือ BF16 GEMM ที่ต่ำกว่าที่ควร: ได้เพียง 11.9 TFLOPS จาก peak ทางทฤษฎี 67 TFLOPS (17.7%) นี่ไม่ใช่ข้อบกพร่องของ hardware แต่สะท้อนถึงการ optimize PyTorch cuBLAS kernels สำหรับ Blackwell aarch64 target ที่ยังไม่สมบูรณ์ FP8 ทำงานได้ดีกว่ามาก (45.6% efficiency) เพราะ FP8 hardware path ของ Blackwell มีความสมบูรณ์กว่าใน driver stack ปัจจุบัน Memory bandwidth ได้ 49% (134 GB/s จาก 273 GB/s) ซึ่งสมเหตุสมผลสำหรับ memory-bandwidth-bound workload

การวัด NVLink-C2C (54 GB/s) สะท้อน DMA memcpy path ไม่ใช่ coherent shared memory path bandwidth 900 GB/s ที่แท้จริงถูกใช้โดยอัตโนมัติเมื่อ GPU code เข้าถึง CPU-allocated memory — ไม่สามารถวัดผ่าน explicit host↔device copies ได้

**Thermal throttling เกิดขึ้นตลอดและทั่วถ้วน:** ทุก benchmark เกิน threshold 53°C ภายในไม่กี่นาที ลด SM clock จาก 2405 MHz เหลือ ~2200 MHz (~9%) ตัวเลข throughput ทั้งหมดในรายงานได้รับผลกระทบนี้

---

## 10. Overall Device Assessment
## 10. การประเมินอุปกรณ์โดยรวม

### Strengths / จุดแข็ง

**EN:**
- **Unified 128 GB memory** is the single biggest advantage. Training 32B LoRA (74 GB peak) on one device is impossible on any standard GPU without model parallelism. Running inference + embedding + reranker simultaneously is trivial.
- **No VRAM ceiling.** All memory is equally accessible to both CPU and GPU, eliminating the PCIe bottleneck for large model transfers.
- **Fast NVMe** (5.4 GB/s read) allows a 32B model to be loaded from disk in 11.8 seconds — critical for experiment iteration speed.
- **FP8 compute is viable** (61 TFLOPS) — once TensorRT-LLM or an optimized FP8 inference stack is installed, significant inference speedup is achievable.

**TH:**
- **Unified memory 128 GB** คือข้อได้เปรียบที่ใหญ่ที่สุด การเทรน 32B LoRA (74 GB peak) บนอุปกรณ์เดียวเป็นไปไม่ได้บน GPU มาตรฐานใดๆ โดยไม่ใช้ model parallelism
- **ไม่มีเพดาน VRAM** memory ทั้งหมดเข้าถึงได้เท่ากันทั้ง CPU และ GPU กำจัด PCIe bottleneck
- **NVMe เร็ว** (5.4 GB/s read) โหลดโมเดล 32B จาก disk ใน 11.8 วินาที สำคัญมากสำหรับ iteration speed ในการทดลอง
- **FP8 compute ใช้งานได้** (61 TFLOPS) — เมื่อติดตั้ง TensorRT-LLM หรือ FP8 inference stack ที่ optimize แล้วจะได้ inference ที่เร็วขึ้นอย่างมีนัยสำคัญ

### Weaknesses / จุดอ่อน

**EN:**
- **Persistent thermal throttling** reduces sustained SM clock by ~9%. All throughput numbers are throttled-state numbers. Performance in a properly cooled chassis would be higher.
- **BF16 compute underperformance** (17.7% of theoretical). PyTorch cuBLAS is not yet optimized for Blackwell aarch64. MFU of 44% for LoRA training sounds acceptable but means 56% of FLOPs are wasted — largely due to this gap.
- **aarch64 ecosystem gaps**: No vLLM, no AWQ, no GGUF, no Triton, no Flash Attention 3. The open-source AI toolchain primarily targets x86 + CUDA, and aarch64 support lags by months to years.
- **Inference throughput is modest** (10 tok/s for 8B BF16). This device is not competitive with production inference accelerators. It is a research/training device that can also serve models.

**TH:**
- **Thermal throttling ตลอดเวลา** ลด SM clock ~9% ตัวเลข throughput ทั้งหมดเป็นค่าในสภาวะ throttle chassis ที่มีการระบายความร้อนที่ดีกว่าจะให้ประสิทธิภาพสูงกว่า
- **BF16 compute ต่ำกว่าที่ควร** (17.7% ของทฤษฎี) PyTorch cuBLAS ยังไม่ optimize สำหรับ Blackwell aarch64 MFU 44% สำหรับ LoRA training ดูโอเค แต่หมายความว่า 56% ของ FLOPs สูญเปล่า ส่วนใหญ่มาจากช่องว่างนี้
- **ช่องว่างของ aarch64 ecosystem**: ไม่มี vLLM, AWQ, GGUF, Triton, Flash Attention 3 open-source AI toolchain มุ่งเป้าที่ x86 + CUDA เป็นหลัก และ aarch64 support ล้าหลังเป็นเดือนถึงปี
- **Inference throughput ปานกลาง** (10 tok/s สำหรับ 8B BF16) อุปกรณ์นี้ไม่สามารถแข่งขันกับ production inference accelerator ได้ มันคืออุปกรณ์วิจัย/training ที่ยังสามารถ serve โมเดลได้

---

## Summary Table: Key Differentiators
## ตารางสรุป: ปัจจัยที่แตกต่าง

| Comparison | Winner | Margin | Key Reason |
|---|---|---|---|
| LoRA vs Full FT (8B) | **LoRA** | MFU +2.6×, memory −57% | Same quality, far better efficiency |
| LoRA r=16 vs r=64 | **r=16** | GSM8K +0.03 | Overfitting with high rank + large dataset |
| GRPO vs SFT (math) | **GRPO** | GSM8K +0.06 | Task-specific rewards vs distribution shift |
| DPO vs SFT (preservation) | **DPO** | GSM8K +0.05 | Preference signal preserves base knowledge |
| Qwen3 vs Mistral (robustness) | **Qwen3** | GSM8K gap ×43 | Qwen3 resists distribution shift |
| 32B vs 8B LoRA (quality) | **32B** | MMLU +0.084 | Scale wins, equal MFU efficiency |
| Dense 32B vs MoE 30B-A3B | **Dense** | Lower loss | MoE advantage negated at 128 GB memory |
| Merging vs Training | **Training** (slight) | MMLU +0.01 | Merging is 99.99% cheaper |
| Think vs No-think (GSM8K) | **No-think** | 0× tokens | Thinking wastes 900 tokens, no gain |
| LoRA vs Full FT (Whisper ASR) | **LoRA** | Loss −57% | LoRA regularizes small ASR dataset |
| FP8 vs BF16 GEMM efficiency | **FP8** | 45% vs 18% | Blackwell FP8 hardware more mature |

---

*Analysis Date: 2026-05-09 | Device: NVIDIA GB10 Grace Blackwell | All experiments run on device with SM 12.1, 128.5 GB unified memory*
