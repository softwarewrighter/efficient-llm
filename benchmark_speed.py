#!/usr/bin/env python3
"""Benchmark model inference speed across different configurations."""

import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

TEST_PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Explain quantum computing in simple terms.",
    "What are the benefits of exercise for mental health?",
    "Describe the process of photosynthesis.",
    "Write a haiku about the ocean.",
]

RESULTS_PATH = Path("results/speed.json")
WARMUP_RUNS = 2
TOKENS_TO_GENERATE = 100


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_config(model_id: str):
    """Load config with pad_token_id fix for Phi-2."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = getattr(config, "eos_token_id", 0)
    return config


def load_model_cpu(model_id: str):
    """Load model for CPU inference."""
    config = _get_config(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def load_model_gpu(model_id: str):
    """Load model for GPU inference."""
    config = _get_config(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def load_model_int4(model_id: str):
    """Load model with INT4 quantization."""
    config = _get_config(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def measure_inference(model, tokenizer, prompt: str, num_tokens: int) -> dict:
    """Measure inference speed for a single prompt."""
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Measure first token latency
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    first_token_time = time.perf_counter() - start_time

    # Measure full generation
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time

    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = generated_tokens / total_time if total_time > 0 else 0

    return {
        "first_token_ms": first_token_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "generated_tokens": generated_tokens,
        "total_time_sec": total_time,
    }


def benchmark_config(model, tokenizer, config_name: str) -> dict:
    """Run benchmarks for a specific configuration."""
    results = {
        "first_token_ms": [],
        "tokens_per_sec": [],
    }

    # Warmup
    for _ in range(WARMUP_RUNS):
        measure_inference(model, tokenizer, TEST_PROMPTS[0], 20)

    # Benchmark
    for prompt in tqdm(TEST_PROMPTS, desc=f"  {config_name}", leave=False):
        metrics = measure_inference(model, tokenizer, prompt, TOKENS_TO_GENERATE)
        results["first_token_ms"].append(metrics["first_token_ms"])
        results["tokens_per_sec"].append(metrics["tokens_per_sec"])

    return {
        "first_token_ms": sum(results["first_token_ms"]) / len(results["first_token_ms"]),
        "tokens_per_sec": sum(results["tokens_per_sec"]) / len(results["tokens_per_sec"]),
    }


def main():
    device = get_device()
    print(f"Primary device: {device}")
    print("=" * 60)

    results = {}
    has_gpu = torch.cuda.is_available()
    has_int4 = has_gpu  # bitsandbytes requires CUDA

    for name, model_id in MODELS.items():
        print(f"\nBenchmarking {name}...")
        model_results = {}

        # CPU benchmark
        print("  Loading for CPU...")
        try:
            model, tokenizer = load_model_cpu(model_id)
            model_results["cpu"] = benchmark_config(model, tokenizer, "CPU")
            print(f"    {model_results['cpu']['tokens_per_sec']:.1f} tok/s, {model_results['cpu']['first_token_ms']:.0f}ms first token")
            del model, tokenizer
        except Exception as e:
            print(f"    CPU benchmark failed: {e}")

        # GPU benchmark
        if has_gpu:
            print("  Loading for GPU...")
            try:
                model, tokenizer = load_model_gpu(model_id)
                model_results["gpu"] = benchmark_config(model, tokenizer, "GPU")
                print(f"    {model_results['gpu']['tokens_per_sec']:.1f} tok/s, {model_results['gpu']['first_token_ms']:.0f}ms first token")
                del model, tokenizer
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    GPU benchmark failed: {e}")

        # INT4 benchmark
        if has_int4:
            print("  Loading for INT4...")
            try:
                model, tokenizer = load_model_int4(model_id)
                model_results["int4"] = benchmark_config(model, tokenizer, "INT4")
                print(f"    {model_results['int4']['tokens_per_sec']:.1f} tok/s, {model_results['int4']['first_token_ms']:.0f}ms first token")
                del model, tokenizer
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    INT4 benchmark failed: {e}")

        results[name] = model_results

    # Save results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary (tokens/sec)")
    print("=" * 60)

    configs = ["cpu", "gpu", "int4"]
    available_configs = [c for c in configs if any(c in r for r in results.values())]

    header = f"{'Model':<12}" + "".join(f"{c:>10}" for c in available_configs)
    print(header)
    print("-" * len(header))

    for name, r in results.items():
        row = f"{name:<12}"
        for config in available_configs:
            if config in r:
                row += f"{r[config]['tokens_per_sec']:>9.1f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
