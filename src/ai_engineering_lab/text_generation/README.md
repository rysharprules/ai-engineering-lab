# Text generation

## First text generation

Generate a response to a prompt locally using a pretrained instruction model and Hugging Face Transformers.

### Reference

Jay Alammar and Maarten Grootendorst, [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/), Chapter 1, **Generating Your First Text**.

### How it works

The tokenizer converts the chat-formatted prompt into token IDs. The causal language model predicts new tokens autoregressively, and the tokenizer decodes them into text. The Transformers pipeline coordinates these steps. With sampling disabled, generation selects the most probable next token at each step (greedy decoding).

### How to run

Complete the [shared setup](../../../README.md#shared-setup), including PyTorch, then from the repository root:

```bash
python -m ai_engineering_lab.text_generation.first_generation
python -m ai_engineering_lab.text_generation.first_generation --prompt "Explain embeddings in one sentence."
python -m ai_engineering_lab.text_generation.first_generation --show-tokens
```

| Argument | Purpose |
| --- | --- |
| `--prompt` | Change the prompt (default: a joke about chickens). |
| `--model` | Select a compatible Hugging Face instruction model. |
| `--max-new-tokens` | Limit newly generated tokens (default: 100). |
| `--show-tokens` | Display token IDs and tokens for the raw prompt, excluding chat-template tokens. |

The first run downloads model weights; later runs reuse the cache. No paid API key is required. In PyCharm, configure a Python run configuration using module name `ai_engineering_lab.text_generation.first_generation` and the repository root as working directory.

### Hardware

CPU execution is the simplest option for Windows laptops with integrated AMD graphics. NVIDIA CUDA can be used on a desktop with a supported NVIDIA GPU and the appropriate PyTorch build. The program automatically selects CUDA when available; otherwise it uses CPU. The small default model requires system RAM and disk space even when running on CPU. Actual inference speed depends on hardware.

### Differences from the book

| Book | This experiment |
| --- | --- |
| Microsoft Phi-3-mini (3.8B) | Qwen2.5-0.5B-Instruct, a smaller model for easier local execution. |
| Hardcoded CUDA | Automatic CUDA/CPU selection. |
| `torch_dtype` | `dtype` parameter in recent Transformers versions. |
| `trust_remote_code=True` | Omitted; the default model is supported directly. |
| 500 new tokens | 100 by default, configurable. |
| Inline example | Runnable module, CLI options and lightweight unit tests. |

The underlying approach remains the same: explicitly load a model and tokenizer, then use the text-generation pipeline. Model output differs from the book because the model differs. Unit tests cover helper functions, not actual inference or output quality.

### Try next

Compare two prompts, inspect their token IDs, and change the output token limit. Model execution has not yet been validated on the target machines.
