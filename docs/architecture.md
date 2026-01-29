# Architecture

## System Overview

The efficient-llm project is a benchmarking and comparison framework for 2-3B parameter language models. It provides a modular architecture for downloading, benchmarking, and demonstrating small but capable LLMs.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  (CLI scripts: demo_chat.py, demo_reasoning.py, demo_code.py)  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Benchmark Layer                            │
│  benchmark_quality.py │ benchmark_speed.py │ benchmark_memory.py│
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Management                           │
│                      download_models.py                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Dependencies                        │
│    HuggingFace Transformers │ PyTorch │ bitsandbytes           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Hardware Layer                           │
│           CPU (fallback) │ CUDA GPU │ Apple MPS                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Model Management (`download_models.py`)

Responsible for:
- Downloading models from HuggingFace Hub
- Managing local model cache
- Validating model availability
- Reporting model configurations

**Target Models:**
| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| Phi-2 | 2.7B | `microsoft/phi-2` |
| Gemma-2B | 2B | `google/gemma-2b-it` |
| SmolLM2-1.7B | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| SmolLM3-3B | 3B | `HuggingFaceTB/SmolLM3-3B` (optional) |

### 2. Benchmark Layer

#### Quality Benchmarks (`benchmark_quality.py`)
- MMLU subset (10 categories, 5-shot)
- GSM8K math problems (100 problems)
- HumanEval code generation (20 problems)

#### Speed Benchmarks (`benchmark_speed.py`)
- Tokens per second measurement
- First token latency
- Batch throughput testing
- Multi-configuration testing (CPU, GPU, INT4)

#### Memory Benchmarks (`benchmark_memory.py`)
- Peak RAM/VRAM usage
- Model size on disk (FP16 vs INT4)
- KV cache size at various sequence lengths

### 3. Demo Layer

Interactive demonstrations for hands-on comparison:
- `demo_reasoning.py` - Logic and reasoning tasks
- `demo_code.py` - Code generation comparison
- `demo_chat.py` - Interactive multi-model chat

## Data Flow

```
Models (HuggingFace) ──► download_models.py ──► Local Cache (~/.cache/huggingface)
                                                       │
                                                       ▼
                              ┌─────────────────────────────────────┐
                              │         Benchmark Scripts           │
                              │  (load models, run evaluations)     │
                              └─────────────────────────────────────┘
                                                       │
                                                       ▼
                              ┌─────────────────────────────────────┐
                              │           results/ directory        │
                              │  quality.json, speed.json,          │
                              │  memory.json, summary.md            │
                              └─────────────────────────────────────┘
```

## Hardware Abstraction

The system automatically detects and uses available hardware:

```python
# Device selection priority
1. CUDA GPU (if available)
2. Apple MPS (if on macOS with Apple Silicon)
3. CPU (fallback)
```

## Quantization Support

INT4 quantization via bitsandbytes enables:
- 3-4x memory reduction
- Faster inference on compatible hardware
- Maintains acceptable quality for most tasks

## Directory Structure

```
efficient-llm/
├── download_models.py      # Model acquisition
├── benchmark_quality.py    # Quality evaluation
├── benchmark_speed.py      # Performance testing
├── benchmark_memory.py     # Memory profiling
├── demo_reasoning.py       # Reasoning demos
├── demo_code.py            # Code generation demos
├── demo_chat.py            # Interactive chat
├── results/                # Benchmark outputs
│   ├── quality.json
│   ├── speed.json
│   ├── memory.json
│   └── summary.md
├── docs/                   # Documentation
│   ├── architecture.md
│   ├── prd.md
│   ├── design.md
│   ├── plan.md
│   └── status.md
├── pyproject.toml          # Package config (uv)
└── .python-version         # Python version pin
```

## Key Design Decisions

1. **Standalone Scripts**: Each benchmark is an independent script for flexibility
2. **JSON Output**: Machine-readable results for automation and visualization
3. **CPU Fallback**: All benchmarks work without GPU
4. **UV Package Manager**: Modern, fast dependency management
5. **No Framework**: Direct HuggingFace Transformers usage, no additional abstraction
