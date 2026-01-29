#!/usr/bin/env python3
"""Benchmark model quality on MMLU, GSM8K, and HumanEval subsets."""

import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

MMLU_CATEGORIES = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_math",
    "computer_security",
    "conceptual_physics",
]

RESULTS_PATH = Path("results/quality.json")


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str, device: torch.device):
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
        low_cpu_mem_usage=True,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# --- MMLU Evaluation ---

def format_mmlu_prompt(question: str, choices: list, few_shot_examples: list = None) -> str:
    """Format MMLU question with few-shot examples."""
    prompt = "Answer the following multiple choice question by responding with A, B, C, or D.\n\n"

    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Question: {ex['question']}\n"
            for i, choice in enumerate(ex['choices']):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += f"Answer: {chr(65 + ex['answer'])}\n\n"

    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"

    return prompt


def evaluate_mmlu(model, tokenizer, categories: list, n_shot: int = 5, max_per_category: int = 20) -> dict:
    """Evaluate model on MMLU subset."""
    results = {"categories": {}, "correct": 0, "total": 0}

    for category in tqdm(categories, desc="MMLU categories", leave=False):
        try:
            dataset = load_dataset("cais/mmlu", category, split="test", trust_remote_code=True)
            dev_dataset = load_dataset("cais/mmlu", category, split="dev", trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: Could not load {category}: {e}")
            continue

        # Get few-shot examples from dev set
        few_shot = []
        for i in range(min(n_shot, len(dev_dataset))):
            few_shot.append({
                "question": dev_dataset[i]["question"],
                "choices": dev_dataset[i]["choices"],
                "answer": dev_dataset[i]["answer"],
            })

        correct = 0
        total = 0

        for item in list(dataset)[:max_per_category]:
            prompt = format_mmlu_prompt(item["question"], item["choices"], few_shot)
            response = generate_text(model, tokenizer, prompt, max_new_tokens=5)

            # Extract answer letter
            response = response.strip().upper()
            predicted = None
            for char in response:
                if char in "ABCD":
                    predicted = ord(char) - ord("A")
                    break

            if predicted == item["answer"]:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        results["categories"][category] = {"accuracy": accuracy, "correct": correct, "total": total}
        results["correct"] += correct
        results["total"] += total

    results["overall"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    return results


# --- GSM8K Evaluation ---

def format_gsm8k_prompt(question: str) -> str:
    """Format GSM8K question."""
    return f"""Solve this math problem step by step. Give your final answer as a number after "The answer is".

Question: {question}

Solution:"""


def extract_number(text: str) -> float | None:
    """Extract the final numerical answer from text."""
    # Look for "the answer is X" pattern
    patterns = [
        r"[Tt]he answer is[:\s]*\$?([\d,]+(?:\.\d+)?)",
        r"[Aa]nswer[:\s]*\$?([\d,]+(?:\.\d+)?)",
        r"=\s*\$?([\d,]+(?:\.\d+)?)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num_str = match.group(1).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                continue

    # Fallback: find last number in text
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def evaluate_gsm8k(model, tokenizer, n_samples: int = 100) -> dict:
    """Evaluate model on GSM8K subset."""
    dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)

    correct = 0
    total = 0

    for item in tqdm(list(dataset)[:n_samples], desc="GSM8K", leave=False):
        prompt = format_gsm8k_prompt(item["question"])
        response = generate_text(model, tokenizer, prompt, max_new_tokens=256)

        # Extract ground truth answer
        answer_text = item["answer"].split("####")[-1].strip()
        try:
            ground_truth = float(answer_text.replace(",", ""))
        except ValueError:
            continue

        predicted = extract_number(response)

        if predicted is not None and abs(predicted - ground_truth) < 0.01:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}


# --- HumanEval Evaluation (Simplified) ---

HUMANEVAL_PROBLEMS = [
    {
        "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome.\"\"\"\n",
        "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('') == True",
    },
    {
        "prompt": "def factorial(n: int) -> int:\n    \"\"\"Return the factorial of n.\"\"\"\n",
        "test": "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(3) == 6",
    },
    {
        "prompt": "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number (0-indexed).\"\"\"\n",
        "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
    },
    {
        "prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Check if n is a prime number.\"\"\"\n",
        "test": "assert is_prime(2) == True\nassert is_prime(4) == False\nassert is_prime(17) == True",
    },
    {
        "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Reverse a string.\"\"\"\n",
        "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''",
    },
    {
        "prompt": "def max_element(lst: list) -> int:\n    \"\"\"Return the maximum element in a non-empty list.\"\"\"\n",
        "test": "assert max_element([1, 2, 3]) == 3\nassert max_element([-1, -2, -3]) == -1",
    },
    {
        "prompt": "def sum_list(lst: list) -> int:\n    \"\"\"Return the sum of all elements in a list.\"\"\"\n",
        "test": "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0",
    },
    {
        "prompt": "def count_vowels(s: str) -> int:\n    \"\"\"Count the number of vowels in a string.\"\"\"\n",
        "test": "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0",
    },
    {
        "prompt": "def gcd(a: int, b: int) -> int:\n    \"\"\"Return the greatest common divisor of a and b.\"\"\"\n",
        "test": "assert gcd(12, 8) == 4\nassert gcd(17, 13) == 1",
    },
    {
        "prompt": "def remove_duplicates(lst: list) -> list:\n    \"\"\"Remove duplicates from a list while preserving order.\"\"\"\n",
        "test": "assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]",
    },
]


def evaluate_humaneval(model, tokenizer, problems: list = None) -> dict:
    """Evaluate model on code generation problems."""
    if problems is None:
        problems = HUMANEVAL_PROBLEMS

    correct = 0
    total = 0

    for problem in tqdm(problems, desc="HumanEval", leave=False):
        prompt = f"Complete this Python function:\n\n{problem['prompt']}"
        response = generate_text(model, tokenizer, prompt, max_new_tokens=150)

        # Combine prompt with generated code
        full_code = problem["prompt"] + response

        # Try to extract just the function
        lines = full_code.split("\n")
        func_lines = []
        in_function = False
        for line in lines:
            if line.startswith("def "):
                in_function = True
            if in_function:
                func_lines.append(line)
                if line.strip() and not line.startswith(" ") and not line.startswith("def "):
                    break

        code_to_test = "\n".join(func_lines) + "\n" + problem["test"]

        try:
            exec(code_to_test, {})
            correct += 1
        except Exception:
            pass

        total += 1

    return {"pass_at_1": correct / total if total > 0 else 0, "correct": correct, "total": total}


def main():
    device = get_device()
    print(f"Using device: {device}")
    print("=" * 60)

    results = {}

    for name, model_id in MODELS.items():
        print(f"\nEvaluating {name}...")

        try:
            model, tokenizer = load_model_and_tokenizer(model_id, device)
        except Exception as e:
            print(f"  Failed to load model: {e}")
            continue

        model_results = {}

        # MMLU
        print("  Running MMLU...")
        model_results["mmlu"] = evaluate_mmlu(model, tokenizer, MMLU_CATEGORIES)
        print(f"    Overall: {model_results['mmlu']['overall']:.1%}")

        # GSM8K
        print("  Running GSM8K...")
        model_results["gsm8k"] = evaluate_gsm8k(model, tokenizer, n_samples=100)
        print(f"    Accuracy: {model_results['gsm8k']['accuracy']:.1%}")

        # HumanEval
        print("  Running HumanEval...")
        model_results["humaneval"] = evaluate_humaneval(model, tokenizer)
        print(f"    Pass@1: {model_results['humaneval']['pass_at_1']:.1%}")

        results[name] = model_results

        # Free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<12} {'MMLU':>10} {'GSM8K':>10} {'HumanEval':>10}")
    print("-" * 44)
    for name, r in results.items():
        mmlu = r.get("mmlu", {}).get("overall", 0)
        gsm8k = r.get("gsm8k", {}).get("accuracy", 0)
        humaneval = r.get("humaneval", {}).get("pass_at_1", 0)
        print(f"{name:<12} {mmlu:>9.1%} {gsm8k:>9.1%} {humaneval:>9.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
