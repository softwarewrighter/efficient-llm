#!/usr/bin/env python3
"""Interactive demo comparing code generation capabilities across models."""

import sys
import time
import traceback

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

CODE_PROMPTS = [
    {
        "name": "Palindrome Check",
        "prompt": "Write a Python function called `is_palindrome` that checks if a string is a palindrome. Return True or False.",
        "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('A man a plan a canal Panama'.replace(' ', '').lower()) == True",
    },
    {
        "name": "Binary Search",
        "prompt": "Write a Python function called `binary_search` that takes a sorted list and a target value. Return the index if found, or -1 if not found.",
        "test": "assert binary_search([1, 2, 3, 4, 5], 3) == 2\nassert binary_search([1, 2, 3, 4, 5], 6) == -1\nassert binary_search([], 1) == -1",
    },
    {
        "name": "Fibonacci with Memoization",
        "prompt": "Write a Python function called `fibonacci` that returns the nth Fibonacci number using memoization. fibonacci(0) = 0, fibonacci(1) = 1.",
        "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\nassert fibonacci(20) == 6765",
    },
    {
        "name": "FizzBuzz",
        "prompt": "Write a Python function called `fizzbuzz` that takes a number n and returns a list of strings from 1 to n, where multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', and multiples of both are 'FizzBuzz'.",
        "test": "result = fizzbuzz(15)\nassert result[2] == 'Fizz'\nassert result[4] == 'Buzz'\nassert result[14] == 'FizzBuzz'\nassert result[0] == '1'",
    },
    {
        "name": "Merge Sorted Lists",
        "prompt": "Write a Python function called `merge_sorted` that takes two sorted lists and returns a single sorted list containing all elements.",
        "test": "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]\nassert merge_sorted([1], []) == [1]",
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


def generate_code(model, tokenizer, prompt: str, max_tokens: int = 300) -> tuple[str, float]:
    """Generate code and return (text, time_taken)."""
    device = next(model.parameters()).device

    full_prompt = f"Write Python code for the following task. Only output the code, no explanations.\n\n{prompt}\n\n```python\n"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start_time

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip(), elapsed


def extract_code(response: str) -> str:
    """Extract Python code from response."""
    # Try to find code between triple backticks
    if "```" in response:
        parts = response.split("```")
        for part in parts:
            if part.startswith("python"):
                return part[6:].strip()
            elif "def " in part:
                return part.strip()

    # Look for function definition
    lines = response.split("\n")
    code_lines = []
    in_function = False

    for line in lines:
        if line.strip().startswith("def "):
            in_function = True
        if in_function:
            code_lines.append(line)
            # Stop at empty line after function or new non-indented line
            if code_lines and line.strip() == "" and len(code_lines) > 2:
                break

    if code_lines:
        return "\n".join(code_lines)

    return response


def test_code(code: str, test: str) -> tuple[bool, str]:
    """Test generated code. Returns (passed, error_message)."""
    try:
        # Create a clean namespace
        namespace = {}
        exec(code, namespace)
        exec(test, namespace)
        return True, ""
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def print_separator(char: str = "-", width: int = 70):
    """Print a separator line."""
    print(char * width)


def run_comparison(models_loaded: dict, prompt_data: dict):
    """Run a single prompt through all models and display results."""
    print(f"\n{'='*70}")
    print(f"TASK: {prompt_data['name']}")
    print(f"{'='*70}")
    print(f"\nPrompt: {prompt_data['prompt']}")
    print_separator()

    results = {}

    for name, (model, tokenizer) in models_loaded.items():
        response, elapsed = generate_code(model, tokenizer, prompt_data["prompt"])
        code = extract_code(response)
        passed, error = test_code(code, prompt_data["test"])

        results[name] = {"passed": passed, "time": elapsed}

        status = "PASS" if passed else "FAIL"
        print(f"\n[{name}] ({elapsed:.2f}s) - {status}")
        print_separator(".")
        print(code[:400])  # Truncate long code
        if len(code) > 400:
            print("... (truncated)")
        if not passed:
            print(f"\nError: {error}")
        print_separator(".")

    # Summary for this task
    print(f"\nResults: ", end="")
    for name, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{name}={status}", end="  ")
    print()


def interactive_mode(models_loaded: dict):
    """Run custom code generation prompts."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter your own code generation prompts. Type 'quit' to exit.\n")

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
            response, elapsed = generate_code(model, tokenizer, prompt)
            code = extract_code(response)

            print(f"\n[{name}] ({elapsed:.2f}s)")
            print_separator(".")
            print(code[:500])
            if len(code) > 500:
                print("... (truncated)")
            print_separator(".")


def main():
    device = get_device()
    print(f"Using device: {device}")
    print("=" * 70)
    print("CODE GENERATION COMPARISON DEMO")
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
    print("\nRunning code generation tasks...")

    pass_counts = {name: 0 for name in models_loaded}

    for prompt_data in CODE_PROMPTS:
        run_comparison(models_loaded, prompt_data)

    # Interactive mode
    print("\n" + "=" * 70)
    response = input("Enter interactive mode? [y/N]: ").strip().lower()
    if response in ["y", "yes"]:
        interactive_mode(models_loaded)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
