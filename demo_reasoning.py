#!/usr/bin/env python3
"""Interactive demo comparing reasoning capabilities across models."""

import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

REASONING_PROMPTS = [
    {
        "name": "Bat and Ball",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step.",
        "answer": "$0.05",
    },
    {
        "name": "Logical Deduction",
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
        "answer": "No (invalid syllogism)",
    },
    {
        "name": "Sally's Sisters",
        "prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Think carefully.",
        "answer": "1 sister",
    },
    {
        "name": "Lily Pad",
        "prompt": "A patch of lily pads doubles in size every day. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake?",
        "answer": "47 days",
    },
    {
        "name": "Counterfactual",
        "prompt": "If a plane crashes on the border of the United States and Canada, where do they bury the survivors?",
        "answer": "You don't bury survivors",
    },
]


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str, device: torch.device):
    """Load model and tokenizer."""
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
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if device.type not in ["cuda"]:
        model = model.to(device)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 200) -> tuple[str, float]:
    """Generate response and return (text, time_taken)."""
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed = time.perf_counter() - start_time

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip(), elapsed


def print_separator(char: str = "-", width: int = 70):
    """Print a separator line."""
    print(char * width)


def run_comparison(models_loaded: dict, prompt_data: dict):
    """Run a single prompt through all models and display results."""
    print(f"\n{'='*70}")
    print(f"PUZZLE: {prompt_data['name']}")
    print(f"{'='*70}")
    print(f"\nPrompt: {prompt_data['prompt']}")
    print(f"Expected: {prompt_data['answer']}")
    print_separator()

    for name, (model, tokenizer) in models_loaded.items():
        response, elapsed = generate_response(model, tokenizer, prompt_data["prompt"])
        print(f"\n[{name}] ({elapsed:.2f}s)")
        print_separator(".")
        print(response[:500])  # Truncate long responses
        if len(response) > 500:
            print("... (truncated)")
        print_separator(".")


def interactive_mode(models_loaded: dict):
    """Run custom prompts through all models."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter your own reasoning questions. Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("Your prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if prompt.lower() in ["quit", "exit", "q"]:
            break

        if not prompt:
            continue

        print_separator()

        for name, (model, tokenizer) in models_loaded.items():
            response, elapsed = generate_response(model, tokenizer, prompt)
            print(f"\n[{name}] ({elapsed:.2f}s)")
            print_separator(".")
            print(response[:500])
            if len(response) > 500:
                print("... (truncated)")
            print_separator(".")


def main():
    device = get_device()
    print(f"Using device: {device}")
    print("=" * 70)
    print("REASONING COMPARISON DEMO")
    print("=" * 70)

    # Load all models
    print("\nLoading models...")
    models_loaded = {}

    for name, model_id in MODELS.items():
        print(f"  Loading {name}...", end=" ", flush=True)
        try:
            model, tokenizer = load_model(model_id, device)
            models_loaded[name] = (model, tokenizer)
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")

    if not models_loaded:
        print("No models loaded. Exiting.")
        return 1

    print(f"\nLoaded {len(models_loaded)} models: {', '.join(models_loaded.keys())}")

    # Run predefined prompts
    print("\nRunning reasoning puzzles...")

    for prompt_data in REASONING_PROMPTS:
        run_comparison(models_loaded, prompt_data)
        print()

    # Interactive mode
    print("\n" + "=" * 70)
    response = input("Enter interactive mode? [y/N]: ").strip().lower()
    if response in ["y", "yes"]:
        interactive_mode(models_loaded)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
