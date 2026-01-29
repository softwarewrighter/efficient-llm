# efficient-llm

Compare 2-3B parameter models on the efficient frontier.

## Overview

This repository benchmarks small but capable language models to help you choose the right model for your use case. We compare quality, speed, and memory across:

| Model | Parameters | Source | Strengths |
|-------|------------|--------|-----------|
| Phi-2 | 2.7B | Microsoft | Reasoning, knowledge |
| Gemma-2B | 2B | Google | Multilingual, edge deployment |
| SmolLM2-1.7B | 1.7B | HuggingFace | Instruction following, small footprint |

**Key insight:** Data quality beats parameter count. Phi-2's synthetic textbook training achieves 56.7% MMLU with only 2.7B parameters.

## Quick Start

```bash
# Clone repo
git clone https://github.com/softwarewrighter/efficient-llm
cd efficient-llm

# Setup with uv
uv venv && source .venv/bin/activate
uv pip install torch transformers accelerate bitsandbytes datasets tqdm

# Download models (Gemma requires: huggingface-cli login)
python download_models.py

# Run benchmarks
python benchmark_quality.py
python benchmark_speed.py
python benchmark_memory.py

# Try demos
python demo_reasoning.py
python demo_code.py
python demo_chat.py
```

## Expected Results

Based on published research (run benchmarks to verify on your hardware):

| Model | MMLU | GSM8K | Speed (CPU) | Memory |
|-------|------|-------|-------------|--------|
| Phi-2 | ~57% | ~45% | ~12 tok/s | 5.4GB |
| Gemma-2B | ~52% | ~38% | ~15 tok/s | 4.2GB |
| SmolLM2 | ~49% | ~35% | ~18 tok/s | 3.4GB |

See [docs/results.md](docs/results.md) for detailed benchmark methodology and analysis.

## Which Model Should I Use?

```
├── Need best reasoning?       → Phi-2
├── Need instruction following? → SmolLM2
├── Need multilingual?         → Gemma
├── Memory constrained (<4GB)? → SmolLM2 + INT4
└── General purpose?           → Any, they're all good!
```

See [docs/TRADEOFFS.md](docs/TRADEOFFS.md) for detailed decision framework.

## Hardware Requirements

- **Minimum:** 8GB RAM, CPU only
- **Recommended:** 16GB RAM, NVIDIA GPU with 8GB VRAM
- **Quantized mode:** Works on 4GB VRAM

## Documentation

| Document | Description |
|----------|-------------|
| [docs/results.md](docs/results.md) | Benchmark results and methodology |
| [docs/MODELS.md](docs/MODELS.md) | Detailed model cards |
| [docs/TRADEOFFS.md](docs/TRADEOFFS.md) | Quality vs efficiency analysis |
| [docs/architecture.md](docs/architecture.md) | System architecture |
| [docs/design.md](docs/design.md) | Technical design |
| [docs/prd.md](docs/prd.md) | Product requirements |
| [docs/plan.md](docs/plan.md) | Implementation plan |
| [docs/status.md](docs/status.md) | Project status |

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
├── results/                # Benchmark outputs (JSON)
└── docs/                   # Documentation
```

## License

MIT
