# Design Document

## Overview

This document describes the technical design for the efficient-llm benchmarking framework.

## Design Principles

1. **Simplicity** - Standalone scripts over complex frameworks
2. **Reproducibility** - Deterministic results with documented methodology
3. **Flexibility** - Each component works independently
4. **Accessibility** - Works on consumer hardware

## Module Design

### 1. download_models.py

**Purpose:** Download and cache all target models

```python
MODELS = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",  # optional
}

def download_model(model_id: str) -> dict:
    """Download model and return metadata."""
    # Uses AutoModelForCausalLM.from_pretrained()
    # Returns: model_size, config, success status

def main():
    for name, model_id in MODELS.items():
        try:
            result = download_model(model_id)
            print(f"Downloaded {name}: {result}")
        except Exception as e:
            print(f"Failed {name}: {e}")
```

**Key Design Decisions:**
- Use HuggingFace cache directory (no custom caching)
- Download tokenizer alongside model
- Print config summary for verification

### 2. benchmark_quality.py

**Purpose:** Evaluate model quality on standard benchmarks

```python
BENCHMARKS = {
    "mmlu": {
        "categories": [
            "abstract_algebra", "anatomy", "astronomy",
            "business_ethics", "clinical_knowledge",
            "college_biology", "college_chemistry",
            "college_math", "computer_security", "conceptual_physics"
        ],
        "n_shot": 5,
    },
    "gsm8k": {
        "n_samples": 100,
    },
    "humaneval": {
        "n_samples": 20,
    },
}

def evaluate_mmlu(model, tokenizer, categories, n_shot) -> dict:
    """Run MMLU evaluation."""
    # Load dataset from HuggingFace datasets
    # Format few-shot prompts
    # Measure accuracy per category
    # Return overall and per-category scores

def evaluate_gsm8k(model, tokenizer, n_samples) -> dict:
    """Run GSM8K math evaluation."""
    # Load GSM8K dataset
    # Generate answers
    # Parse and verify numerical answers
    # Return accuracy

def evaluate_humaneval(model, tokenizer, n_samples) -> dict:
    """Run HumanEval code evaluation."""
    # Load HumanEval problems
    # Generate code completions
    # Execute tests (sandboxed)
    # Return pass@1 score
```

**Output Format:**
```json
{
  "phi-2": {
    "mmlu": {"overall": 0.567, "categories": {...}},
    "gsm8k": {"accuracy": 0.45},
    "humaneval": {"pass@1": 0.35}
  }
}
```

### 3. benchmark_speed.py

**Purpose:** Measure inference performance

```python
CONFIGS = ["cpu", "gpu", "int4"]

PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Explain quantum computing to a 5-year-old.",
    # ... more diverse prompts
]

def measure_speed(model, tokenizer, config) -> dict:
    """Measure inference speed for a configuration."""
    results = {
        "tokens_per_sec": [],
        "first_token_ms": [],
    }

    for prompt in PROMPTS:
        # Warm-up run
        # Timed generation (100 tokens)
        # Record metrics

    return {
        "tokens_per_sec": mean(results["tokens_per_sec"]),
        "first_token_ms": mean(results["first_token_ms"]),
    }

def load_model_for_config(model_id, config):
    """Load model with appropriate settings."""
    if config == "cpu":
        return load_model(device="cpu")
    elif config == "gpu":
        return load_model(device="cuda")
    elif config == "int4":
        return load_model(quantization="int4")
```

**Output Format:**
```json
{
  "phi-2": {
    "cpu": {"tokens_per_sec": 12.3, "first_token_ms": 450},
    "gpu": {"tokens_per_sec": 85.2, "first_token_ms": 120},
    "int4": {"tokens_per_sec": 95.1, "first_token_ms": 100}
  }
}
```

### 4. benchmark_memory.py

**Purpose:** Profile memory usage

```python
def measure_memory(model_id) -> dict:
    """Measure memory characteristics."""

    # FP16 model
    model = load_model(model_id, dtype=torch.float16)
    fp16_size = get_model_size(model)
    fp16_peak = measure_peak_memory(model)

    # INT4 model
    model_int4 = load_model(model_id, quantization="int4")
    int4_size = get_model_size(model_int4)
    int4_peak = measure_peak_memory(model_int4)

    return {
        "model_size_gb": fp16_size,
        "peak_vram_gb": fp16_peak,
        "int4_size_gb": int4_size,
        "int4_peak_vram_gb": int4_peak,
    }

def measure_peak_memory(model):
    """Run inference and measure peak memory."""
    torch.cuda.reset_peak_memory_stats()
    # Generate tokens
    return torch.cuda.max_memory_allocated() / 1e9
```

### 5. Demo Scripts

#### demo_reasoning.py
```python
REASONING_PROMPTS = [
    "A bat and a ball cost $1.10 in total...",
    "If all roses are flowers...",
    "Sally has 3 brothers...",
]

def compare_reasoning(models: list, prompts: list):
    """Run prompts through all models and display comparison."""
    for prompt in prompts:
        print(f"\nPrompt: {prompt}\n")
        for model_name, model, tokenizer in models:
            start = time.time()
            response = generate(model, tokenizer, prompt)
            elapsed = time.time() - start
            print(f"{model_name} ({elapsed:.2f}s):\n{response}\n")
```

#### demo_code.py
```python
CODE_PROMPTS = [
    "Write a Python function to check if a string is a palindrome.",
    "Implement binary search in Python.",
    "Write a function to find the nth Fibonacci number using memoization.",
]

def compare_code(models, prompts):
    """Generate and optionally execute code."""
    # Similar to reasoning but with code formatting
    # Optional: execute generated code to verify correctness
```

#### demo_chat.py
```python
def interactive_chat():
    """Interactive multi-model chat."""
    print("Available models:", list(MODELS.keys()))
    print("Commands: /switch <model>, /compare, /quit")

    while True:
        user_input = input("You: ")
        if user_input.startswith("/"):
            handle_command(user_input)
        else:
            response = generate(current_model, user_input)
            print(f"Assistant: {response}")
```

## Error Handling

```python
class ModelUnavailableError(Exception):
    """Raised when a model cannot be loaded."""
    pass

def safe_load_model(model_id):
    """Load model with graceful error handling."""
    try:
        return AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        print(f"Warning: Could not load {model_id}: {e}")
        return None
```

## Configuration Management

All configuration via constants at module top:
- No external config files
- Clear defaults
- Easy to modify

## Results Storage

```python
def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    output_path = Path("results") / filename
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
```

## Progress Reporting

```python
from tqdm import tqdm

def run_benchmark(models, test_fn):
    """Run benchmark with progress bar."""
    results = {}
    for model_name in tqdm(models, desc="Benchmarking"):
        results[model_name] = test_fn(model_name)
    return results
```

## Device Detection

```python
def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

## Testing Strategy

- Manual testing of each script
- Verify results JSON format
- Test on multiple hardware configs
- Document expected outputs
