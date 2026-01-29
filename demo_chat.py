#!/usr/bin/env python3
"""Interactive chat demo with model selection and comparison mode."""

import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

HELP_TEXT = """
Commands:
  /models          - List available models
  /switch <name>   - Switch to a different model
  /compare         - Toggle compare mode (run all models)
  /clear           - Clear conversation history
  /help            - Show this help message
  /quit            - Exit the chat

Current shortcuts:
  Ctrl+C           - Exit the chat
"""


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


def format_prompt(user_message: str, history: list = None, model_name: str = None) -> str:
    """Format prompt with conversation history."""
    # Simple format that works across models
    prompt = ""

    if history:
        for msg in history[-4:]:  # Keep last 4 turns for context
            prompt += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"

    prompt += f"User: {user_message}\nAssistant:"

    return prompt


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 256) -> tuple[str, float]:
    """Generate response and return (text, time_taken)."""
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start_time

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    # Clean up response - stop at "User:" if present
    if "User:" in response:
        response = response.split("User:")[0]

    return response.strip(), elapsed


def print_separator(char: str = "-", width: int = 60):
    """Print a separator line."""
    print(char * width)


class ChatSession:
    def __init__(self, models_loaded: dict):
        self.models_loaded = models_loaded
        self.model_names = list(models_loaded.keys())
        self.current_model = self.model_names[0] if self.model_names else None
        self.compare_mode = False
        self.history = []

    def handle_command(self, command: str) -> bool:
        """Handle chat commands. Returns True if should continue."""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/quit" or cmd == "/exit":
            return False

        elif cmd == "/help":
            print(HELP_TEXT)

        elif cmd == "/models":
            print("\nAvailable models:")
            for name in self.model_names:
                marker = " *" if name == self.current_model else ""
                print(f"  - {name}{marker}")
            print()

        elif cmd == "/switch":
            if len(parts) < 2:
                print("Usage: /switch <model_name>")
            else:
                name = parts[1]
                if name in self.models_loaded:
                    self.current_model = name
                    print(f"Switched to {name}")
                else:
                    print(f"Unknown model: {name}")
                    print(f"Available: {', '.join(self.model_names)}")

        elif cmd == "/compare":
            self.compare_mode = not self.compare_mode
            status = "ON" if self.compare_mode else "OFF"
            print(f"Compare mode: {status}")

        elif cmd == "/clear":
            self.history = []
            print("Conversation history cleared.")

        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands.")

        return True

    def chat(self, user_message: str):
        """Process a chat message."""
        if self.compare_mode:
            # Run through all models
            print()
            responses = {}

            for name in self.model_names:
                model, tokenizer = self.models_loaded[name]
                prompt = format_prompt(user_message, self.history, name)
                response, elapsed = generate_response(model, tokenizer, prompt)
                responses[name] = response

                print(f"[{name}] ({elapsed:.2f}s)")
                print_separator(".")
                print(response)
                print_separator(".")
                print()

            # Use first model's response for history
            self.history.append({
                "user": user_message,
                "assistant": responses[self.model_names[0]],
            })

        else:
            # Single model
            model, tokenizer = self.models_loaded[self.current_model]
            prompt = format_prompt(user_message, self.history, self.current_model)
            response, elapsed = generate_response(model, tokenizer, prompt)

            print(f"\n[{self.current_model}] ({elapsed:.2f}s)")
            print(response)
            print()

            self.history.append({
                "user": user_message,
                "assistant": response,
            })

    def run(self):
        """Main chat loop."""
        print("\n" + "=" * 60)
        print("INTERACTIVE CHAT")
        print("=" * 60)
        print(f"Current model: {self.current_model}")
        print("Type /help for commands, /quit to exit.")
        print_separator()

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not self.handle_command(user_input):
                    print("Goodbye!")
                    break
            else:
                self.chat(user_input)


def main():
    device = get_device()
    print(f"Using device: {device}")
    print("=" * 60)
    print("MULTI-MODEL CHAT DEMO")
    print("=" * 60)

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

    # Start chat session
    session = ChatSession(models_loaded)
    session.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
