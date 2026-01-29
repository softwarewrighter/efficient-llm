# Model Cards

Detailed information about each model in the efficient-llm benchmark suite.

## Phi-2

| Property | Value |
|----------|-------|
| **Developer** | Microsoft Research |
| **Parameters** | 2.7B |
| **Architecture** | Transformer decoder |
| **Context Length** | 2048 tokens |
| **Training Data** | Textbooks, synthetic data |
| **License** | MIT |
| **HuggingFace ID** | `microsoft/phi-2` |

### Strengths
- Exceptional reasoning for its size
- Strong performance on knowledge benchmarks (MMLU)
- Good at multi-step problem solving
- Efficient "textbook quality" training approach

### Weaknesses
- Shorter context window (2048)
- Not instruction-tuned (base model)
- Can be verbose
- May struggle with conversational tasks

### Best Use Cases
- Reasoning and logic puzzles
- Knowledge-intensive tasks
- Code generation
- Educational applications

### Architecture Notes
Phi-2 uses a standard transformer decoder architecture with:
- 32 layers
- 32 attention heads
- 2560 hidden dimension
- Rotary position embeddings (RoPE)

---

## Gemma-2B

| Property | Value |
|----------|-------|
| **Developer** | Google DeepMind |
| **Parameters** | 2B |
| **Architecture** | Transformer decoder |
| **Context Length** | 8192 tokens |
| **Training Data** | Web, code, multilingual |
| **License** | Gemma license (permissive) |
| **HuggingFace ID** | `google/gemma-2b-it` |

### Strengths
- Instruction-tuned for conversations
- Longer context window (8192)
- Multilingual capabilities
- Good balance of size and capability

### Weaknesses
- Slightly lower benchmark scores than Phi-2
- Requires agreement to license terms
- Can be overly cautious/safe

### Best Use Cases
- Conversational AI
- Multilingual applications
- Tasks requiring longer context
- Edge deployment

### Architecture Notes
Gemma uses a refined transformer architecture with:
- 18 layers
- 8 attention heads (with GQA)
- 2048 hidden dimension
- RoPE with extended context

---

## SmolLM2-1.7B-Instruct

| Property | Value |
|----------|-------|
| **Developer** | HuggingFace |
| **Parameters** | 1.7B |
| **Architecture** | Transformer decoder |
| **Context Length** | 8192 tokens |
| **Training Data** | FineWeb-Edu, synthetic |
| **License** | Apache 2.0 |
| **HuggingFace ID** | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |

### Strengths
- Smallest footprint (1.7B)
- Instruction-tuned
- Excellent efficiency
- Long context support
- Fully open source

### Weaknesses
- Lower absolute performance
- Less world knowledge
- May struggle with complex reasoning

### Best Use Cases
- Resource-constrained environments
- Mobile/edge deployment
- Simple instruction following
- Applications where speed matters

### Architecture Notes
SmolLM2 uses an efficient architecture with:
- 24 layers
- 16 attention heads
- 2048 hidden dimension
- Grouped query attention (GQA)

---

## Model Comparison Matrix

| Feature | Phi-2 | Gemma-2B | SmolLM2 |
|---------|-------|----------|---------|
| Parameters | 2.7B | 2B | 1.7B |
| Context | 2048 | 8192 | 8192 |
| Instruction-tuned | No | Yes | Yes |
| Multilingual | Limited | Yes | Limited |
| License | MIT | Gemma | Apache 2.0 |
| Memory (FP16) | ~5.4GB | ~4.2GB | ~3.4GB |
| Memory (INT4) | ~1.8GB | ~1.4GB | ~1.1GB |

## Quantization Notes

All models support INT4 quantization via bitsandbytes:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

**Expected quality impact:** 1-3% degradation on benchmarks, acceptable for most applications.

## Hardware Recommendations

| Model | Minimum RAM | Recommended | GPU VRAM |
|-------|-------------|-------------|----------|
| Phi-2 | 8GB | 16GB | 6GB |
| Gemma-2B | 8GB | 12GB | 5GB |
| SmolLM2 | 6GB | 8GB | 4GB |

With INT4 quantization, all models can run on 4GB VRAM.

## Loading Examples

### Basic Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
```

### CPU-Only Loading
```python
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
```

### INT4 Quantized Loading
```python
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=quantization_config,
    device_map="auto",
)
```
