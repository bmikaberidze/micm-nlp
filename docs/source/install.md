# Install

## Hardware

Training targets NVIDIA GPUs. CPU works for small-scale debugging; there is no
support for non-NVIDIA accelerators.

## Python

Requires **Python 3.10 or newer**.
On an older interpreter `pip` reports `No matching distribution found for micm-nlp`,
which does not say why — check with `python3 --version` first.
Any Python 3.10+ environment with pip works (`venv`, `uv`, `conda`, `pyenv`):

```bash
python3 -m venv .venv && source .venv/bin/activate
```

## From PyPI

```bash
pip install micm-nlp
```

Installing pulls the full training stack — `torch`, `transformers`, `peft`, `datasets`, `spacy` and others.
Two of them are tightly constrained (`peft==0.14.0`, `transformers>=4.48,<4.50`) and will pin whatever
environment they land in, which is the virtual environment above earning its keep.

## From source

For an unreleased change, or a platform with no wheel:

```bash
git clone https://github.com/bmikaberidze/micm-nlp.git
cd micm-nlp
pip install -e .
```

Add the `dev` extra — `pip install -e ".[dev]"` — if you intend to work *on* the
package rather than with it; it pulls `pytest` and `ruff`.

## Docker

Recommended for reproducibility on GPU machines. The image is built from the
repository, so clone first:

```bash
git clone https://github.com/bmikaberidze/micm-nlp.git
cd micm-nlp
docker build -t micm-nlp .
docker run --gpus all -it --rm -v $(pwd):/app -w /app micm-nlp bash
```

## Environment

Read from the environment, or from a `.env` file — `.env.example` in the
repository is the starting point for a clone:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT_PATH` | Workspace directory; `artefacts/` is created under it. The fallback when `init()` gets no `root_path`. |
| `WANDB_API_KEY` | Only needed to log to the W&B service. The example configs set `WANDB_MODE: offline`, so they need no account.  |
| `HF_TOKEN` | Required only for gated HuggingFace models or datasets. |
