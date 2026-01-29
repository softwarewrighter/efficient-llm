# Implementation Plan

## Phase Overview

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Project Setup | Not Started |
| 2 | Model Download | Not Started |
| 3 | Quality Benchmarks | Not Started |
| 4 | Speed Benchmarks | Not Started |
| 5 | Memory Benchmarks | Not Started |
| 6 | Interactive Demos | Not Started |
| 7 | Documentation | Not Started |

## Phase 1: Project Setup

### Tasks
- [ ] Create pyproject.toml with dependencies
- [ ] Create .python-version file (3.10)
- [ ] Set up directory structure
- [ ] Create empty results/ directory
- [ ] Initialize README.md skeleton

### Dependencies (pyproject.toml)
```toml
[project]
name = "efficient-llm"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "transformers>=4.36",
    "accelerate>=0.25",
    "bitsandbytes>=0.41",
    "datasets>=2.16",
    "evaluate>=0.4",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]
```

### Verification
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
python -c "import torch; print(torch.__version__)"
```

## Phase 2: Model Download

### Tasks
- [ ] Implement download_models.py
- [ ] Add model list configuration
- [ ] Implement download with progress
- [ ] Add error handling for unavailable models
- [ ] Print model metadata after download

### Implementation Approach
1. Define MODELS dict with names and HuggingFace IDs
2. Use AutoModelForCausalLM.from_pretrained()
3. Also download tokenizer
4. Print success/failure summary

### Verification
```bash
python download_models.py
# Verify models in ~/.cache/huggingface
```

## Phase 3: Quality Benchmarks

### Tasks
- [ ] Implement benchmark_quality.py
- [ ] Add MMLU evaluation (10 categories, 5-shot)
- [ ] Add GSM8K evaluation (100 problems)
- [ ] Add HumanEval evaluation (20 problems)
- [ ] Save results to results/quality.json

### Implementation Approach

#### MMLU
1. Load MMLU dataset from HuggingFace
2. Select 10 categories
3. Format 5-shot prompts
4. Generate completions
5. Parse answers and compute accuracy

#### GSM8K
1. Load GSM8K dataset
2. Sample 100 problems
3. Generate solutions
4. Extract numerical answers
5. Compare to ground truth

#### HumanEval
1. Load HumanEval problems
2. Select 20 problems
3. Generate code completions
4. Execute in sandbox
5. Compute pass@1

### Verification
```bash
python benchmark_quality.py
cat results/quality.json
```

## Phase 4: Speed Benchmarks

### Tasks
- [ ] Implement benchmark_speed.py
- [ ] Add tokens/second measurement
- [ ] Add first token latency
- [ ] Test CPU configuration
- [ ] Test GPU configuration (if available)
- [ ] Test INT4 quantization
- [ ] Save results to results/speed.json

### Implementation Approach
1. Define test prompts
2. For each model and config:
   - Load model with appropriate settings
   - Warm-up run
   - Timed generation (100 tokens)
   - Record metrics
3. Aggregate results

### Verification
```bash
python benchmark_speed.py
cat results/speed.json
```

## Phase 5: Memory Benchmarks

### Tasks
- [ ] Implement benchmark_memory.py
- [ ] Measure model size on disk
- [ ] Measure peak VRAM/RAM
- [ ] Compare FP16 vs INT4
- [ ] Save results to results/memory.json

### Implementation Approach
1. For each model:
   - Load FP16 version, measure
   - Load INT4 version, measure
   - Run inference, track peak memory
2. Save comparison

### Verification
```bash
python benchmark_memory.py
cat results/memory.json
```

## Phase 6: Interactive Demos

### Tasks
- [ ] Implement demo_reasoning.py
- [ ] Implement demo_code.py
- [ ] Implement demo_chat.py
- [ ] Add timing display
- [ ] Add model comparison mode

### demo_reasoning.py
- Classic reasoning puzzles
- Side-by-side model comparison
- Show timing for each

### demo_code.py
- Code generation prompts
- Optional code execution
- Syntax highlighting

### demo_chat.py
- Model selection menu
- Interactive prompt loop
- Compare mode (run all models)

### Verification
```bash
python demo_reasoning.py
python demo_code.py
python demo_chat.py
```

## Phase 7: Documentation

### Tasks
- [ ] Complete README.md
  - Overview
  - Quick start guide
  - Results summary table
  - Decision tree
- [ ] Create docs/MODELS.md
  - Detailed model cards
  - Architecture notes
  - Recommended use cases
- [ ] Create docs/TRADEOFFS.md
  - Quality vs speed analysis
  - Memory vs quality analysis
  - Recommendations by use case
- [ ] Generate results/summary.md
  - Key findings
  - Recommendations
  - Methodology notes

### README Structure
```markdown
# efficient-llm

Compare 2-3B parameter models on the efficient frontier.

## Quick Start
...

## Results Summary
| Model | MMLU | Speed | Memory |
...

## Decision Tree
...

## Running Benchmarks
...
```

### Verification
- Review all documentation
- Test quick start instructions
- Verify links work

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Model download fails | Graceful skip, continue with available models |
| Out of memory | Reduce batch size, use quantization |
| Slow benchmarks | Add progress bars, save partial results |
| HumanEval sandbox issues | Make code execution optional |

## Hardware Requirements

| Configuration | RAM | VRAM | Notes |
|--------------|-----|------|-------|
| Minimum | 8GB | - | CPU only, slow |
| Recommended | 16GB | 8GB | Full benchmarks |
| Quantized | 8GB | 4GB | INT4 mode |

## Commands Quick Reference

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# Download models
python download_models.py

# Run benchmarks
python benchmark_quality.py
python benchmark_speed.py
python benchmark_memory.py

# Interactive demos
python demo_reasoning.py
python demo_code.py
python demo_chat.py
```
