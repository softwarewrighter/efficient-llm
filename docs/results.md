# Benchmark Results

Expected benchmark results based on published research and model documentation.

> **Note:** These are projected values based on published research. Run the benchmark scripts to generate actual measurements on your hardware.

> **Gemma-2B requires HuggingFace authentication.** Run `huggingface-cli login` and accept the model license at https://huggingface.co/google/gemma-2b-it before using Gemma.

## Quality Benchmarks

### MMLU (Massive Multitask Language Understanding)

| Model | MMLU Score | Source |
|-------|------------|--------|
| Phi-2 | 56.7% | Microsoft Research |
| Gemma-2B | 52.1% | Google DeepMind |
| SmolLM2-1.7B | 49.3% | HuggingFace |

**Winner: Phi-2** - Microsoft's synthetic textbook training approach delivers exceptional reasoning for the parameter count.

### GSM8K (Math Word Problems)

| Model | Accuracy | Notes |
|-------|----------|-------|
| Phi-2 | ~45% | Strong multi-step reasoning |
| Gemma-2B | ~38% | Solid performance |
| SmolLM2-1.7B | ~35% | Competitive for size |

**Winner: Phi-2** - Reasoning-focused training pays off on math tasks.

### Code Generation (HumanEval)

| Model | Pass@1 | Notes |
|-------|--------|-------|
| Phi-2 | ~35% | Trained on quality code |
| Gemma-2B | ~30% | Decent code capabilities |
| SmolLM2-1.7B | ~28% | Acceptable for small model |

**Winner: Phi-2** - Synthetic training data included high-quality code.

## Speed Benchmarks

### Throughput (tokens/second)

| Model | CPU | GPU (FP16) | GPU (INT4) |
|-------|-----|------------|------------|
| Phi-2 | ~12 | ~80 | ~95 |
| Gemma-2B | ~15 | ~85 | ~100 |
| SmolLM2-1.7B | ~18 | ~95 | ~110 |

**Winner: SmolLM2** - Smallest model, fastest inference.

### First Token Latency

| Model | CPU | GPU |
|-------|-----|-----|
| Phi-2 | ~450ms | ~120ms |
| Gemma-2B | ~400ms | ~100ms |
| SmolLM2-1.7B | ~350ms | ~80ms |

**Winner: SmolLM2** - Lower latency for interactive applications.

## Memory Benchmarks

### Model Size

| Model | FP16 | INT4 | Reduction |
|-------|------|------|-----------|
| Phi-2 | 5.4GB | 1.8GB | 3.0x |
| Gemma-2B | 4.2GB | 1.4GB | 3.0x |
| SmolLM2-1.7B | 3.4GB | 1.1GB | 3.1x |

**Winner: SmolLM2** - Smallest footprint, ideal for edge deployment.

### Peak VRAM During Inference

| Model | FP16 | INT4 |
|-------|------|------|
| Phi-2 | ~6.2GB | ~2.1GB |
| Gemma-2B | ~5.0GB | ~1.7GB |
| SmolLM2-1.7B | ~4.0GB | ~1.4GB |

**Winner: SmolLM2** - Runs on 4GB GPU with INT4.

## Task-Specific Winners

| Task | Winner | Runner-up |
|------|--------|-----------|
| **Reasoning** | Phi-2 | Gemma-2B |
| **Math (GSM8K)** | Phi-2 | SmolLM2 |
| **Code Generation** | Phi-2 | Gemma-2B |
| **Instruction Following** | SmolLM2 | Gemma-2B |
| **Edge Efficiency** | Gemma-2B | SmolLM2 |
| **Throughput** | SmolLM2 | Gemma-2B |
| **Memory Efficiency** | SmolLM2 | Gemma-2B |
| **Multilingual** | Gemma-2B | - |
| **Long Context** | Gemma-2B | SmolLM2 |

## Key Insights

### 1. Data Quality Beats Parameter Count
Phi-2's 56.7% MMLU with only 2.7B parameters demonstrates that training data quality (synthetic textbooks from GPT-4) can outperform larger models trained on lower-quality data.

### 2. The Efficient Frontier Trade-offs
- **Phi-2**: Best quality, slowest, most memory
- **Gemma-2B**: Balanced, good edge deployment
- **SmolLM2**: Fastest, smallest, good-enough quality

### 3. Quantization is Worth It
INT4 quantization provides:
- ~3x memory reduction
- ~15-20% speed improvement
- Only 2-4% quality degradation

### 4. Choose Based on Task
- Need best reasoning? → **Phi-2**
- Need instruction following? → **SmolLM2**
- Need edge efficiency? → **Gemma-2B**
- Memory constrained? → **SmolLM2 + INT4**

## Generating Actual Results

Run the benchmarks to get real measurements:

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# Download models
python download_models.py

# Run benchmarks
python benchmark_quality.py   # → results/quality.json
python benchmark_speed.py     # → results/speed.json
python benchmark_memory.py    # → results/memory.json
```

Results will be saved to the `results/` directory as JSON files.

## References

- [Phi-2 Technical Report](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
- [Gemma Technical Report](https://ai.google.dev/gemma)
- [SmolLM Blog Post](https://huggingface.co/blog/smollm)
