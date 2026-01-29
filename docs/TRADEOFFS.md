# Quality vs Efficiency Tradeoffs

Analysis of the tradeoffs between model quality, inference speed, and memory usage.

## The Efficient Frontier

Small language models (2-3B parameters) occupy an interesting position: large enough to be useful, small enough to run on consumer hardware. The key question is: **which model gives the best results for your constraints?**

```
Quality
  ^
  |     * Phi-2 (best quality)
  |
  |        * Gemma-2B (balanced)
  |
  |           * SmolLM2 (most efficient)
  |
  +-------------------------> Efficiency
           (speed, memory)
```

## Quality Analysis

### Benchmark Results (Expected)

| Model | MMLU | GSM8K | HumanEval | Overall |
|-------|------|-------|-----------|---------|
| Phi-2 | ~57% | ~45% | ~35% | Best |
| Gemma-2B | ~52% | ~38% | ~30% | Middle |
| SmolLM2 | ~49% | ~35% | ~28% | Lowest |

### Quality by Task Type

| Task | Best Model | Runner-up | Notes |
|------|------------|-----------|-------|
| Reasoning | Phi-2 | Gemma-2B | Phi-2's training shows here |
| Math | Phi-2 | SmolLM2 | Step-by-step helps |
| Code | Phi-2 | Gemma-2B | All decent |
| Conversation | Gemma-2B | SmolLM2 | Instruction tuning matters |
| Long context | Gemma-2B | SmolLM2 | Phi-2 limited to 2k |
| Multilingual | Gemma-2B | - | Only Gemma trained for this |

## Speed Analysis

### Throughput (tokens/second)

| Model | CPU | GPU | INT4 |
|-------|-----|-----|------|
| SmolLM2 | ~18 | ~95 | ~110 |
| Gemma-2B | ~15 | ~85 | ~100 |
| Phi-2 | ~12 | ~80 | ~95 |

**Insight:** Smaller models are faster. SmolLM2 is ~50% faster than Phi-2 on CPU.

### First Token Latency

| Model | CPU | GPU |
|-------|-----|-----|
| SmolLM2 | ~350ms | ~80ms |
| Gemma-2B | ~400ms | ~100ms |
| Phi-2 | ~450ms | ~120ms |

**Insight:** For interactive applications, first token latency matters. SmolLM2 feels snappier.

## Memory Analysis

### Model Size

| Model | FP16 | INT4 | Reduction |
|-------|------|------|-----------|
| Phi-2 | 5.4GB | 1.8GB | 3.0x |
| Gemma-2B | 4.2GB | 1.4GB | 3.0x |
| SmolLM2 | 3.4GB | 1.1GB | 3.1x |

### Peak VRAM During Inference

| Model | FP16 | INT4 |
|-------|------|------|
| Phi-2 | ~6.2GB | ~2.1GB |
| Gemma-2B | ~5.0GB | ~1.7GB |
| SmolLM2 | ~4.0GB | ~1.4GB |

**Insight:** INT4 quantization enables running on 4GB GPUs with minimal quality loss.

## Decision Framework

### By Primary Constraint

**Memory constrained (<4GB VRAM)?**
→ SmolLM2 + INT4 quantization

**Need best quality regardless of speed?**
→ Phi-2

**Need long context (>2k tokens)?**
→ Gemma-2B or SmolLM2

**Building a chatbot?**
→ Gemma-2B (instruction-tuned, conversational)

**Need multilingual support?**
→ Gemma-2B (only real option)

**Maximizing throughput?**
→ SmolLM2 (smallest, fastest)

### By Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Code assistant | Phi-2 | Best code quality |
| Customer support bot | Gemma-2B | Conversational, safe |
| Mobile app | SmolLM2 + INT4 | Smallest footprint |
| Educational tutor | Phi-2 | Best reasoning |
| Document Q&A | Gemma-2B | Long context |
| Real-time suggestions | SmolLM2 | Lowest latency |
| Multilingual chat | Gemma-2B | Only option |

## Quantization Impact

### Quality Retention with INT4

| Model | FP16 MMLU | INT4 MMLU | Retention |
|-------|-----------|-----------|-----------|
| Phi-2 | ~57% | ~55% | 96% |
| Gemma-2B | ~52% | ~50% | 96% |
| SmolLM2 | ~49% | ~48% | 98% |

**Recommendation:** Use INT4 quantization unless you need maximum quality. The 3-4% quality loss is worth the 3x memory reduction for most applications.

## Cost-Benefit Summary

### Quality per GB of Memory

| Model | Quality Score | Memory (FP16) | Quality/GB |
|-------|---------------|---------------|------------|
| SmolLM2 | 49 | 3.4GB | 14.4 |
| Gemma-2B | 52 | 4.2GB | 12.4 |
| Phi-2 | 57 | 5.4GB | 10.6 |

**Insight:** SmolLM2 offers the best "bang for buck" if memory is your constraint.

### Quality per Token/Second (CPU)

| Model | Quality Score | Speed | Quality/Speed |
|-------|---------------|-------|---------------|
| SmolLM2 | 49 | 18 | 2.7 |
| Gemma-2B | 52 | 15 | 3.5 |
| Phi-2 | 57 | 12 | 4.75 |

**Insight:** Phi-2 offers the best quality relative to its speed. If you can accept slower inference, Phi-2 delivers more.

## Recommendations

### General Purpose
**Gemma-2B** - Best balance of quality, features, and efficiency. Instruction-tuned, long context, reasonable speed.

### Quality-First
**Phi-2** - When you need the best answers and can tolerate slower inference and shorter context.

### Efficiency-First
**SmolLM2** - When speed and memory matter more than raw quality. Perfect for edge deployment.

### The "It Depends" Answer

There is no universally best model. The right choice depends on:
1. Your hardware constraints
2. Your latency requirements
3. Your quality threshold
4. Your specific task type

Run the benchmarks on your hardware with your prompts to make an informed decision.
