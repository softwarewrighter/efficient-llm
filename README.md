# efficient-llm

Compare 2-3B parameter models on the efficient frontier.

## Overview

This repository benchmarks small but capable language models to help you choose the right model for your use case. We compare quality, speed, and memory across:

| Model | Parameters | Source |
|-------|------------|--------|
| Phi-2 | 2.7B | Microsoft |
| Gemma-2B | 2B | Google |
| SmolLM2-1.7B | 1.7B | HuggingFace |

## Quick Start

```bash
# Clone repo
git clone https://github.com/softwarewrighter/efficient-llm
cd efficient-llm

# Setup with uv
uv venv && source .venv/bin/activate
uv pip install -e .

# Download models
python download_models.py

# Run benchmarks
python benchmark_quality.py
python benchmark_speed.py
python benchmark_memory.py

# Try demos
python demo_reasoning.py
python demo_chat.py
```

## Results Summary

| Model | MMLU | GSM8K | Speed (tok/s) | Memory |
|-------|------|-------|---------------|--------|
| Phi-2 | ~57% | ~45% | ~12 (CPU) | 5.4GB |
| Gemma-2B | ~52% | ~38% | ~15 (CPU) | 4.2GB |
| SmolLM2 | ~49% | ~35% | ~18 (CPU) | 3.4GB |

*Results will be updated after benchmarking.*

## Which Model Should I Use?

```
├── Need best reasoning?      → Phi-2
├── Need instruction following? → SmolLM2
├── Need multilingual?        → Gemma
├── Memory constrained (<4GB)? → SmolLM2 + INT4
└── General purpose?          → Any, they're all good!
```

## Hardware Requirements

- **Minimum:** 8GB RAM, CPU only
- **Recommended:** 16GB RAM, NVIDIA GPU with 8GB VRAM
- **Quantized mode:** Works on 4GB VRAM

## Repository Structure

```
efficient-llm/
├── download_models.py      # Download all models
├── benchmark_quality.py    # MMLU, GSM8K, HumanEval
├── benchmark_speed.py      # Throughput, latency
├── benchmark_memory.py     # Peak memory, model sizes
├── demo_reasoning.py       # Reasoning comparison
├── demo_code.py            # Code generation comparison
├── demo_chat.py            # Interactive chat
├── results/                # Benchmark outputs
└── docs/                   # Documentation
```

## License

MIT
