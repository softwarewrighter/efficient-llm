# Product Requirements Document (PRD)

## Executive Summary

**Project:** efficient-llm
**Purpose:** Compare and benchmark 2-3B parameter language models to identify the best model for various tasks and constraints.

## Problem Statement

Developers and researchers face difficulty choosing between small language models (2-3B parameters) because:
1. Published benchmarks use different methodologies
2. Real-world performance varies from reported metrics
3. Trade-offs between quality, speed, and memory are unclear
4. No unified comparison framework exists

## Goals

### Primary Goals
1. Provide reproducible benchmarks for 2-3B parameter models
2. Measure quality, speed, and memory characteristics
3. Create interactive demos for hands-on comparison
4. Generate actionable recommendations for model selection

### Success Metrics
- All target models successfully benchmarked
- Results reproducible on standard hardware
- Clear documentation of methodology
- Decision tree for model selection

## Target Users

1. **Application Developers** - Building products with local LLMs
2. **Researchers** - Comparing model architectures
3. **Hobbyists** - Running LLMs on consumer hardware
4. **Educators** - Teaching about efficient ML

## Requirements

### Functional Requirements

#### FR1: Model Download
- Download and cache all target models
- Support offline operation after initial download
- Report model sizes and configurations
- Handle unavailable models gracefully

#### FR2: Quality Benchmarks
- Run MMLU evaluation (10 category subset)
- Run GSM8K math reasoning (100 problems)
- Run HumanEval code generation (20 problems)
- Output structured JSON results

#### FR3: Speed Benchmarks
- Measure tokens per second
- Measure first token latency
- Test CPU, GPU, and quantized configurations
- Support batch throughput testing

#### FR4: Memory Benchmarks
- Measure peak RAM/VRAM usage
- Compare FP16 vs INT4 model sizes
- Profile KV cache at various sequence lengths

#### FR5: Interactive Demos
- Reasoning comparison demo
- Code generation comparison demo
- Interactive chat with model selection
- Side-by-side output comparison

#### FR6: Results Reporting
- JSON output for all benchmarks
- Human-readable summary markdown
- Comparison tables and visualizations

### Non-Functional Requirements

#### NFR1: Hardware Compatibility
- Minimum: 8GB RAM, CPU-only operation
- Recommended: 16GB RAM, 8GB VRAM GPU
- Quantized: Works on 4GB VRAM

#### NFR2: Platform Support
- Linux with CUDA
- macOS with Apple Silicon (MPS)
- Windows with CUDA (best effort)

#### NFR3: Performance
- Benchmarks complete within reasonable time
- Progress indicators for long operations
- Graceful handling of resource constraints

#### NFR4: Usability
- Single-command setup with uv
- Clear error messages
- Comprehensive documentation

## Target Models

| Model | Parameters | Source | Strengths |
|-------|------------|--------|-----------|
| Phi-2 | 2.7B | Microsoft | Reasoning, knowledge |
| Gemma-2B | 2B | Google | Multilingual, edge deployment |
| SmolLM2-1.7B | 1.7B | HuggingFace | Instruction following, small footprint |
| SmolLM3-3B | 3B | HuggingFace | Long context (optional) |

## Out of Scope

- Models larger than 3B parameters
- Fine-tuning or training
- Cloud deployment
- API-based models
- Comprehensive MMLU (all 57 categories)

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers 4.36+
- bitsandbytes for quantization
- datasets for benchmark data
- uv for package management

## Deliverables

1. **Source Code**
   - download_models.py
   - benchmark_quality.py
   - benchmark_speed.py
   - benchmark_memory.py
   - demo_reasoning.py
   - demo_code.py
   - demo_chat.py

2. **Results**
   - quality.json
   - speed.json
   - memory.json
   - summary.md

3. **Documentation**
   - README.md with quick start
   - MODELS.md with detailed model cards
   - TRADEOFFS.md with analysis

## Timeline

| Phase | Deliverables |
|-------|--------------|
| Phase 1 | Project setup, model download |
| Phase 2 | Quality benchmarks |
| Phase 3 | Speed benchmarks |
| Phase 4 | Memory benchmarks |
| Phase 5 | Interactive demos |
| Phase 6 | Documentation and polish |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model unavailable | Medium | Graceful fallback, skip unavailable |
| Hardware limitations | Medium | CPU fallback, quantization |
| Benchmark reproducibility | High | Pin versions, document methodology |
| Long runtime | Low | Progress bars, partial results |
