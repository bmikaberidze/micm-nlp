# Installation

Requires **Python 3.10 or newer**.

## From PyPI

```bash
pip install micm-nlp
```

## From source

```bash
git clone https://github.com/bmikaberidze/micm-nlp.git
cd micm-nlp
pip install -e ".[dev]"
```

The `dev` extra adds `pytest` and `ruff`.

## Docker

Recommended for reproducibility on GPU machines:

```bash
docker build -t micm-nlp .
docker run --gpus all -it --rm -v $(pwd):/app -w /app micm-nlp bash
```

## Environment

Credentials and the workspace root come from a `.env` file:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT_PATH` | Workspace directory; `artefacts/` (datasets, models, evals, wandb) is created under it. Used as the fallback when `init()` is called without `root_path`. |
| `WANDB_API_KEY` | Required only if `training_args.args.report_to` includes `wandb`. |
| `HF_TOKEN` | Required only for gated HuggingFace models or datasets. |

## Hardware

Training targets NVIDIA GPUs. CPU works for small-scale debugging; there is no
support for non-NVIDIA accelerators.

`peft` is pinned to `0.14.0`: the Cross-Prompt Encoder subclasses stock PEFT
internals, so raising that pin is a breaking-change review rather than a routine
upgrade.
