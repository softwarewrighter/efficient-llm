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

# Authenticate with HuggingFace (required for Gemma-2B)
huggingface-cli login
# Then accept the Gemma license at: https://huggingface.co/google/gemma-2b-it

# Download models
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

## Benchmark Results

Actual measurements from CPU benchmarks (GPU results depend on hardware):

| Model | MMLU | GSM8K | Speed (CPU) | First Token | Memory |
|-------|------|-------|-------------|-------------|--------|
| Phi-2 | ~57% | ~45% | 7.1 tok/s | 430ms | 5.2GB |
| Gemma-2B | ~52% | ~38% | 8.5 tok/s | 321ms | 4.7GB |
| SmolLM2 | ~49% | ~35% | 3.7 tok/s | 257ms | 3.2GB |

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

## Prerequisites

### HuggingFace Authentication (Required for Gemma-2B)

Gemma-2B is a gated model that requires HuggingFace authentication:

1. Create a HuggingFace account at https://huggingface.co
2. Generate an access token at https://huggingface.co/settings/tokens
3. Accept the Gemma license at https://huggingface.co/google/gemma-2b-it
4. Login via CLI: `huggingface-cli login`

Without authentication, Gemma-2B will fail to download. Phi-2 and SmolLM2 work without authentication.

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
