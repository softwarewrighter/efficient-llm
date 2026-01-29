#!/usr/bin/env python3
"""Download all models for local benchmarking."""

import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",  # Requires HuggingFace login
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

# Models that require authentication
GATED_MODELS = {"gemma-2b"}


def get_model_size_gb(model) -> float:
    """Calculate model size in GB from parameters."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024**3)


def download_model(name: str, model_id: str) -> dict:
    """Download model and tokenizer, return metadata."""
    print(f"\nDownloading {name} ({model_id})...")

    # Load config first and fix missing pad_token_id (needed for Phi-2 on transformers 5.x)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = getattr(config, "eos_token_id", 0)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    size_gb = get_model_size_gb(model)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9

    # Safely get max context length
    max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is None:
        max_len = getattr(model.config, "n_positions", "unknown")

    return {
        "model_id": model_id,
        "parameters_b": round(num_params, 2),
        "size_gb": round(size_gb, 2),
        "vocab_size": tokenizer.vocab_size,
        "max_length": max_len,
    }


def main():
    print("=" * 60)
    print("Efficient-LLM Model Downloader")
    print("=" * 60)

    results = {}
    failed = []

    for name, model_id in tqdm(MODELS.items(), desc="Models"):
        try:
            metadata = download_model(name, model_id)
            results[name] = metadata
            print(f"  ✓ {name}: {metadata['parameters_b']}B params, {metadata['size_gb']}GB")
        except Exception as e:
            print(f"  ✗ {name}: Failed - {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\nSuccessfully downloaded: {len(results)}/{len(MODELS)}")
    for name, meta in results.items():
        print(f"  • {name}: {meta['parameters_b']}B params, {meta['size_gb']}GB")

    if failed:
        print(f"\nFailed to download: {', '.join(failed)}")
        print("Note: Some models may require authentication (huggingface-cli login)")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
