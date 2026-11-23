"""Generate text with a Hugging Face model.

Run:
    python -m ai_engineering_lab.text_generation.first_generation
    python -m ai_engineering_lab.text_generation.first_generation --show-tokens

The model is downloaded from Hugging Face on first run.
"""

import argparse
import sys

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "Create a funny joke about chickens."


def choose_device(cuda_available: bool) -> str:
    """Select NVIDIA CUDA when available, otherwise CPU."""
    return "cuda" if cuda_available else "cpu"


def extract_generated_text(result: list[dict]) -> str:
    """Unwrap the generated text from Transformers' pipeline response.

    When a chat-style prompt is supplied, Transformers may return generated
    chat messages rather than a plain string. Handle either representation.
    """
    generated = result[0]["generated_text"]
    if isinstance(generated, str):
        return generated
    if isinstance(generated, list):
        for message in reversed(generated):
            if message.get("role") == "assistant":
                return str(message["content"])
    raise ValueError("Unexpected text-generation pipeline response")


def generate_text(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 100,
    show_tokens: bool = False,
) -> str:
    """Load a tokenizer/model and generate a response to a chat prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    device = choose_device(torch.cuda.is_available())
    print(f"Device: {device}")
    print(f"Model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if show_tokens:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        print(f"Input token IDs ({len(ids)}): {ids}")
        print(f"Input tokens: {tokenizer.convert_ids_to_tokens(ids)}")

    # float32 is broadly supported on CPU; model dtype is selected by the
    # checkpoint on CUDA where reduced precision may be available.
    dtype = "auto" if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    messages = [{"role": "user", "content": prompt}]
    result = generator(
        messages,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return extract_generated_text(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate your first text")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to give the model")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--show-tokens", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    return args


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    response = generate_text(
        prompt=args.prompt,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        show_tokens=args.show_tokens,
    )
    print("\nGenerated text:\n")
    print(response)


if __name__ == "__main__":
    main()
