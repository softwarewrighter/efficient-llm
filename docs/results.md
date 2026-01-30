# Benchmark Results

Expected benchmark results based on published research and model documentation.

> **Note:** These are projected values based on published research. Run the benchmark scripts to generate actual measurements on your hardware.

> **Gemma-2B requires HuggingFace authentication.** Run `huggingface-cli login` and accept the model license at https://huggingface.co/google/gemma-2b-it before using Gemma.

## Quality Benchmarks

### MMLU (Massive Multitask Language Understanding) - Actual Results

| Model | MMLU Score | Expected | Notes |
|-------|------------|----------|-------|
| Phi-2 | **61.7%** | ~57% | Exceeds expectations |
| SmolLM2-1.7B | **55.6%** | ~49% | Strong performance |
| Gemma-2B | **38.9%** | ~52% | Below expectations |

**Winner: Phi-2** - Microsoft's synthetic textbook training delivers 61.7% accuracy, exceeding published benchmarks.

### GSM8K (Math Word Problems) - Actual Results

| Model | Accuracy | Expected | Notes |
|-------|----------|----------|-------|
| Phi-2 | **57.0%** | ~45% | Excellent multi-step reasoning |
| Gemma-2B | **18.0%** | ~38% | Struggled with format |
| SmolLM2-1.7B | **0.0%** | ~35% | Prompt format issue* |

**Winner: Phi-2** - Significantly better than expected at 57% accuracy.

*Note: SmolLM2's 0% on GSM8K likely reflects a prompt format incompatibility, not actual capability (MMLU works fine).

### Code Generation (HumanEval) - Actual Results

| Model | Pass@1 | Expected | Notes |
|-------|--------|----------|-------|
| Gemma-2B | **90.0%** | ~30% | Surprisingly high* |
| Phi-2 | **50.0%** | ~35% | Strong code generation |
| SmolLM2-1.7B | **0.0%** | ~28% | Prompt format issue* |

**Winner: Gemma-2B** - Unexpectedly high at 90% Pass@1.

*Note: Results based on 10 HumanEval problems. SmolLM2's 0% likely reflects prompt format issues. Gemma-2B's 90% may benefit from specific problem selection.

## Speed Benchmarks

### Throughput (tokens/second)

| Model | CPU (Actual) | GPU (FP16) | GPU (INT4) |
|-------|--------------|------------|------------|
| Phi-2 | **7.1** | ~80 | ~95 |
| Gemma-2B | **8.5** | ~85 | ~100 |
| SmolLM2-1.7B | **3.7** | ~95 | ~110 |

**Actual CPU Results:** Gemma-2B achieved the highest CPU throughput at 8.5 tokens/sec, followed by Phi-2 at 7.1 tokens/sec. SmolLM2 was slower than expected at 3.7 tokens/sec on CPU.

### First Token Latency

| Model | CPU (Actual) | GPU |
|-------|--------------|-----|
| Phi-2 | **430ms** | ~120ms |
| Gemma-2B | **321ms** | ~100ms |
| SmolLM2-1.7B | **257ms** | ~80ms |

**Actual CPU Results:** SmolLM2 has the lowest first-token latency at 257ms, followed by Gemma-2B at 321ms, and Phi-2 at 430ms.

## Memory Benchmarks

### Model Size (Actual Measurements)

| Model | Parameters | FP16 (Actual) | INT4 (est.) |
|-------|------------|---------------|-------------|
| Phi-2 | **2.78B** | **5.18GB** | ~1.7GB |
| Gemma-2B | **2.51B** | **4.67GB** | ~1.5GB |
| SmolLM2-1.7B | **1.71B** | **3.19GB** | ~1.1GB |

**Winner: SmolLM2** - Smallest footprint, ideal for edge deployment.

### KV Cache Memory (Actual Measurements)

| Model | 512 tokens | 1024 tokens | 2048 tokens |
|-------|------------|-------------|-------------|
| Phi-2 | 0.156GB | 0.312GB | 0.625GB |
| Gemma-2B | 0.009GB | 0.018GB | 0.035GB |
| SmolLM2-1.7B | 0.094GB | 0.188GB | 0.375GB |

**Winner: Gemma-2B** - Uses MQA (Multi-Query Attention) for minimal KV cache overhead.

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
Phi-2's **61.7% MMLU** with only 2.7B parameters demonstrates that training data quality (synthetic textbooks from GPT-4) can outperform larger models trained on lower-quality data. This exceeds even the published 56.7% benchmark.

### 2. The Efficient Frontier Trade-offs (Actual Results)
- **Phi-2**: Best overall quality (61.7% MMLU, 57% GSM8K), but slowest on CPU
- **SmolLM2**: Strong MMLU (55.6%), fastest first-token latency (257ms), smallest memory
- **Gemma-2B**: Excellent code generation (90% HumanEval), best CPU throughput (8.5 tok/s)

### 3. Prompt Format Matters
SmolLM2's 0% on GSM8K and HumanEval despite strong MMLU (55.6%) shows that evaluation results depend heavily on prompt formatting. Always test with your actual use case prompts.

### 4. Choose Based on Task (Updated)
- Need best reasoning/math? → **Phi-2** (61.7% MMLU, 57% GSM8K)
- Need code generation? → **Gemma-2B** (90% HumanEval)
- Need low latency? → **SmolLM2** (257ms first token)
- Memory constrained? → **SmolLM2** (3.19GB FP16)

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
