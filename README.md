# AI Engineering Lab

A collection of hands-on Python experiments in AI engineering, organized by concept rather than learning resource.

## Experiments

| Area | Experiment | Description |
| --- | --- | --- |
| Text generation | [First text generation](src/ai_engineering_lab/text_generation/README.md#first-text-generation) | Run a pretrained language model locally using Hugging Face Transformers. |\n| Embeddings | [Song embeddings and recommendations](src/ai_engineering_lab/song_embeddings/README.md) | Learn song vectors from playlist co-occurrence with Word2Vec and find similar songs. |

## Shared setup


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

5. For experiments that require it, install **PyTorch** for your hardware from the official [PyTorch installer selector](https://pytorch.org/get-started/locally/). Choose **Windows / Pip / Python / CPU** initially if unsure. For an NVIDIA CUDA-compatible GPU, use the selector's supported CUDA build instead. Check `nvidia-smi` to identify your card.
6. Install the project and developer dependencies:

   ```powershell
   python -m pip install -e ".[dev]"
   ```

   `torch` is installed separately in step 5 to allow choosing the correct CPU or CUDA wheel.


Run the unit tests from the repository root:

```bash
python -m pytest
```

See each experiment's README for its run commands and dependencies. Never commit virtual environments, model weights, caches or credentials.
