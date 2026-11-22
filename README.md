# AI Engineering Lab

A growing set of hands-on Python experiments exploring language models, tokenization, embeddings, prompting, retrieval, agents, and evaluation.

The examples start with *Hands-On Large Language Models* (Jay Alammar and Maarten Grootendorst), but the repository is organized by **concept**, not by book, so new experiments can be added independently.

## Chapter 1: Generate your first text

The first experiment uses a small open-weight instruction model, [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), with [Hugging Face Transformers](https://huggingface.co/docs/transformers/). It demonstrates:

- Loading a pretrained causal language model and its tokenizer
- Selecting CPU or NVIDIA CUDA at runtime
- Passing a chat message to a text-generation pipeline
- Controlling maximum output length and sampling
- Looking at the token IDs that the tokenizer produces

The model is downloaded from Hugging Face **on first run**, so an internet connection and enough disk space are needed. Later executions reuse the local cache. Running a model locally does not require a paid API key.

### Windows setup (PyCharm + Python)

1. Install **Python 3.12, 64-bit** from [python.org](https://www.python.org/downloads/windows/). This conservative baseline is compatible with the project's supported Python versions. Verify with `py -3.12 --version`.
2. Install [PyCharm](https://www.jetbrains.com/pycharm/download/). The unified PyCharm includes free core Python IDE functionality.
3. Clone the repository, then open its root folder in PyCharm.
4. Create a virtual environment in the repository root (PyCharm can do this under **Settings → Project → Python Interpreter**), or run:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

   In **Git Bash**, use `source .venv/Scripts/activate` instead of the PowerShell activation command. If PowerShell blocks activation, configure PyCharm to use `.venv\Scripts\python.exe`, or invoke that interpreter directly.

5. Install **PyTorch** for your hardware from the official [PyTorch installer selector](https://pytorch.org/get-started/locally/). Choose **Windows / Pip / Python / CPU** initially if unsure. For an NVIDIA CUDA-compatible GPU, use the selector's supported CUDA build instead. Check `nvidia-smi` to identify your card.
6. Install the project and developer dependencies:

   ```powershell
   python -m pip install -e ".[dev]"
   ```

   `torch` is installed separately in step 5 to allow choosing the correct CPU or CUDA wheel.

7. Run the example and tests from the repository root:

   ```powershell
   python -m ai_engineering_lab.text_generation.first_generation
   python -m pytest
   ```

Use `python -m ai_engineering_lab.text_generation.first_generation --prompt "Explain tokenization in one sentence."` to change the prompt. Use `--show-tokens` to inspect token IDs, or `--max-new-tokens 50` to shorten output.

### PyCharm interpreter

Point PyCharm at the project's virtual environment interpreter:

```text
.venv\Scripts\python.exe
```

For a run configuration, select **Module name** and enter `ai_engineering_lab.text_generation.first_generation`. Set the working directory to the repository root.

### Hardware checks

```powershell
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

If `nvidia-smi` is not recognized, you may not have an NVIDIA GPU, or its drivers may not be installed. CPU execution still works, but can be slow.

## Repository structure

```text
src/
  ai_engineering_lab/
    __init__.py
    text_generation/
      __init__.py
      first_generation.py
tests/
  test_first_generation.py
pyproject.toml
.gitignore
```

Future topics may include embeddings, semantic search, RAG, tool calling, agents, and evaluations. These directories will be added only when required.

## Notes

- **Never commit** downloaded model weights, `.venv/`, cache files, or credentials.
- The first run downloads model files; downloading and loading models can use substantial disk space and memory.
- Example text generation is illustrative, not a production chatbot. Small models may produce poor or repetitive output.
- The unit tests are lightweight and mock inference; **they do not download a model or validate model output quality**.
- Model generation and dependency compatibility can evolve. If an upstream package changes, update and revalidate dependencies rather than assuming old book snippets still work.

## Learning checkpoints

- [ ] Install Python, PyCharm, and a project virtual environment
- [ ] Run the first generation example
- [ ] Inspect token IDs using `--show-tokens`
- [ ] Compare two prompts and their outputs
- [ ] Explain what the tokenizer, model, and pipeline each do
