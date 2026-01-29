#!/usr/bin/env python3
"""Benchmark model memory usage across different configurations."""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _get_config(model_id: str):
    """Load config with pad_token_id fix for Phi-2."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = getattr(config, "eos_token_id", 0)
    return config

MODELS = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

RESULTS_PATH = Path("results/memory.json")


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_size_gb(model) -> float:
    """Calculate model size in GB from parameters."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024**3)


def get_model_param_count(model) -> float:
    """Get parameter count in billions."""
    return sum(p.numel() for p in model.parameters()) / 1e9


def measure_peak_memory_cuda(model, tokenizer, prompt: str, max_tokens: int = 100) -> float:
    """Measure peak VRAM during inference (CUDA only)."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    return peak_memory


def measure_fp16(model_id: str, device: torch.device) -> dict:
    """Measure FP16 model characteristics."""
    config = _get_config(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()

    model_size = get_model_size_gb(model)
    param_count = get_model_param_count(model)

    # Measure peak memory during inference
    peak_vram = 0.0
    if device.type == "cuda":
        peak_vram = measure_peak_memory_cuda(
            model, tokenizer, "Write a short story about a robot.", max_tokens=100
        )

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_size_gb": round(model_size, 2),
        "parameters_b": round(param_count, 2),
        "peak_vram_gb": round(peak_vram, 2) if peak_vram > 0 else None,
    }


def measure_int4(model_id: str) -> dict:
    """Measure INT4 quantized model characteristics."""
    if not torch.cuda.is_available():
        return {"error": "INT4 quantization requires CUDA"}

    config = _get_config(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

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

    # For quantized models, calculate effective size
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    model_size = total_size / (1024**3)

    # Measure peak memory during inference
    peak_vram = measure_peak_memory_cuda(
        model, tokenizer, "Write a short story about a robot.", max_tokens=100
    )

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model_size_gb": round(model_size, 2),
        "peak_vram_gb": round(peak_vram, 2),
    }


def measure_kv_cache_size(model_id: str, seq_lengths: list = None) -> dict:
    """Estimate KV cache size at various sequence lengths."""
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048]

    config = _get_config(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Get model config for KV cache calculation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    num_layers = getattr(config, "num_hidden_layers", 32)
    num_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32))
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    del model, tokenizer

    results = {}
    for seq_len in seq_lengths:
        # KV cache size = 2 (K and V) * num_layers * num_heads * head_dim * seq_len * dtype_size
        # FP16 = 2 bytes per element
        kv_cache_bytes = 2 * num_layers * num_heads * head_dim * seq_len * 2
        kv_cache_gb = kv_cache_bytes / (1024**3)
        results[f"seq_{seq_len}"] = round(kv_cache_gb, 3)

    return results


def main():
    device = get_device()
    print(f"Primary device: {device}")
    print("=" * 60)

    results = {}
    has_gpu = torch.cuda.is_available()

    for name, model_id in tqdm(MODELS.items(), desc="Models"):
        print(f"\nMeasuring {name}...")
        model_results = {}

        # FP16 measurements
        print("  Measuring FP16...")
        try:
            fp16_results = measure_fp16(model_id, device)
            model_results["fp16"] = fp16_results
            print(f"    Size: {fp16_results['model_size_gb']}GB, Params: {fp16_results['parameters_b']}B")
            if fp16_results.get("peak_vram_gb"):
                print(f"    Peak VRAM: {fp16_results['peak_vram_gb']}GB")
        except Exception as e:
            print(f"    FP16 measurement failed: {e}")

        # INT4 measurements
        if has_gpu:
            print("  Measuring INT4...")
            try:
                int4_results = measure_int4(model_id)
                model_results["int4"] = int4_results
                print(f"    Size: {int4_results['model_size_gb']}GB, Peak VRAM: {int4_results['peak_vram_gb']}GB")
            except Exception as e:
                print(f"    INT4 measurement failed: {e}")

        # KV cache estimates
        print("  Estimating KV cache...")
        try:
            kv_cache = measure_kv_cache_size(model_id)
            model_results["kv_cache_gb"] = kv_cache
            print(f"    KV cache at 2048 tokens: {kv_cache.get('seq_2048', 'N/A')}GB")
        except Exception as e:
            print(f"    KV cache estimation failed: {e}")

        results[name] = model_results

    # Save results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<12} {'FP16 Size':>12} {'INT4 Size':>12} {'FP16 VRAM':>12} {'INT4 VRAM':>12}")
    print("-" * 60)

    for name, r in results.items():
        fp16_size = r.get("fp16", {}).get("model_size_gb", "N/A")
        int4_size = r.get("int4", {}).get("model_size_gb", "N/A")
        fp16_vram = r.get("fp16", {}).get("peak_vram_gb", "N/A")
        int4_vram = r.get("int4", {}).get("peak_vram_gb", "N/A")

        fp16_size_str = f"{fp16_size}GB" if isinstance(fp16_size, (int, float)) else fp16_size
        int4_size_str = f"{int4_size}GB" if isinstance(int4_size, (int, float)) else int4_size
        fp16_vram_str = f"{fp16_vram}GB" if isinstance(fp16_vram, (int, float)) else fp16_vram
        int4_vram_str = f"{int4_vram}GB" if isinstance(int4_vram, (int, float)) else int4_vram

        print(f"{name:<12} {fp16_size_str:>12} {int4_size_str:>12} {fp16_vram_str:>12} {int4_vram_str:>12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
