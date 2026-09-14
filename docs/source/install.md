# Install

## PyPi

```bash
pip install micm-nlp
```

Requires **Python 3.10 or newer** — on an older interpreter `pip` reports `No matching distribution found for micm-nlp`, which does not say why.  

> [!NOTE]
> Training targets NVIDIA GPUs; CPU works for small-scale debugging, and there is no support for other accelerators.

Installing pulls the full training stack — `torch`, `transformers`, `peft`, `datasets`, `spacy` and others.
Two are tightly constrained (`peft==0.14.0`, `transformers>=4.48,<4.50`) and will pin whatever environment they land in, so give it one of its own:

```bash
python3 -m venv .venv && source .venv/bin/activate
```


## From source

For an unreleased change, or a platform with no wheel:

```bash
git clone https://github.com/bmikaberidze/micm-nlp.git
cd micm-nlp
pip install -e .
```

Add the `dev` extra — `pip install -e ".[dev]"` — to work *on* the package rather than with it; it pulls `pytest` and `ruff`.

> [!NOTE]
> An editable install links the clone into the active environment and edits apply immediately.
> So you can keep `micm-nlp/` beside your project and work on both together.
> What counts is the environment you install into, not the directory you run `pip install -e path/to/micm-nlp` from.

## Docker

Recommended for reproducibility on GPU machines. The image is built from the repository, so clone first:

```bash
git clone https://github.com/bmikaberidze/micm-nlp.git
cd micm-nlp
docker build -t micm-nlp .
docker run --gpus all -it --rm -v $(pwd):/app -w /app micm-nlp bash
```

## Environment

Read from the environment, or from a `.env` file — `.env.example` is the starting point for a clone:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT_PATH` | workspace directory; `artefacts/` is created under it, and it is what `init()` and `--root-path` fall back to |
| `WANDB_API_KEY` | only to log to the W&B service; the example configs set `WANDB_MODE: offline` and need no account |
| `HF_TOKEN` | only for gated HuggingFace models or datasets |
