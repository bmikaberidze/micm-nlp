# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nlpka** is an NLP research toolkit built on HuggingFace Transformers, supporting tokenization, pretraining, fine-tuning, and PEFT across encoder-only, decoder-only, and encoder-decoder architectures. It has been used in two peer-reviewed publications covering BERT-like pretraining/fine-tuning for text and token classification, and parameter-efficient soft prompt tuning across 200 languages (SIB-200 topic classification) with cross-prompt encoders on XLM-RoBERTa-large. The most recent work — **Cross-Prompt Encoder (XPE)** — is accepted at *Findings of IJCNLP–AACL 2025* ([arXiv:2508.10352](https://arxiv.org/abs/2508.10352)). The toolkit is expanding to decoder-only models and includes T5 encoder-decoder pipelines for fine-tuning and evaluation.

## Setup

```bash
cp .env.example .env  # add WANDB_API_KEY to .env

# Docker (recommended)
docker build -t xpe .
docker run --gpus all -it --rm -v $(pwd):/xpe_runner -w /xpe_runner xpe bash

# Local
pip install -e ".[dev]"
```

## Common Commands

**Download dataset:**
```bash
python -m micm_nlp.datasets.scripts.sib200.download_tokenized
```

**Run experiment (main entrypoint):**
```bash
python -m micm_nlp.models.scripts.peft.xpe.run \
  --config xlmr/finetune/peft/sib200_hybrid.xpe \
  --supervision_regime=<0|1> <source_dataset> <setup_id>
```

- `--supervision_regime`: `0` = Zero-Shot XLT, `1` = Fully Supervised XLT
- `<source_dataset>`: `sib200_enarzho`, `sib200_joshi5`, `sib200_xlmr_seen` (zero-shot) or `sib200_joshi5_divers_24` (supervised)
- `<setup_id>`: `1`=SPT, `2`=D30 (30% XPE hybrid), `3`=D70 (70% XPE hybrid), `4`=XPE

**Collect hidden states:**
```bash
python -m micm_nlp.models.scripts.peft.xpe.collect_hs_v
```

**Evaluate AYA on Belebele:**
```bash
python -m micm_nlp.evaluations.scripts.eval_aya_belebele
```

**Visualize / quantitative analysis:**
```bash
python -m micm_nlp.evaluations.scripts.plot
python -m micm_nlp.evaluations.scripts.quant
```

**Lint & format:**
```bash
ruff check src/
ruff format src/
```

## Project Structure

```
ExpXPE/
├── pyproject.toml              # Package metadata, dependencies, ruff config
├── dockerfile
├── .env.example
├── config/                     # YAML experiment configs
├── examples/                   # Simple usage demos for the package API
├── experiments/                # Research-specific code (XPE, SIB-200, evals)
│   ├── config/                 # Experiment config utilities (xpe_utils, etc.)
│   ├── datasets/               # Dataset prep (SIB-200, Belebele, xStory, etc.)
│   ├── models/                 # Experiment runners (XPE train, collect HS, etc.)
│   └── evals/                  # Plots, quantitative analysis, eval scripts
├── artefacts/                  # Generated outputs
└── src/micm_nlp/               # Package source (standard src layout)
    ├── pipeline.py             # High-level wiring: load_dataset, load_model, run
    ├── env.py                  # Loads .env, exposes os.environ as `env`
    ├── setup.py                # Runtime init (Rich pretty-printing + traceback)
    ├── utils.py                # Pure helpers, JSON/YAML/pickle I/O, SimpleNamespace utils
    ├── path.py                 # Project path resolution, directory traversal
    ├── enums.py                # StrEnum definitions for all categorical choices
    ├── config.py               # Config loader (YAML → SimpleNamespace)
    ├── datasets/               # Dataset loading/preprocessing
    ├── tokenizers/             # Tokenizer factory (XLM-R, BERT, T5, etc.)
    ├── models/                 # Model, PEFT, XPE, trainers, callbacks
    └── evals/                  # Metrics, confusion matrices, plots
```

## Architecture

All core logic follows a class-based, config-driven pattern:

```
CONFIG (YAML) → TOKENIZER → DATASET → MODEL → PEFT → Trainer → EVALUATE
```

| Class | File | Role |
|-------|------|------|
| `CONFIG` | `src/nlpka/config/config.py` | Loads YAML configs from `config/language_model/` |
| `TOKENIZER` | `src/nlpka/tokenizers/tokenizer.py` | Tokenizer factory for XLM-R, BERT, T5, etc. |
| `DATASET` | `src/nlpka/datasets/dataset.py` | Loads/preprocesses HuggingFace, CSV, or TXT datasets |
| `MODEL` | `src/nlpka/models/model.py` | Wraps HuggingFace model + Trainer, WandB logging |
| `PEFT` | `src/nlpka/models/peft.py` | Attaches LoRA / Prefix / P-Tuning / XPE to base model |
| `CrossPromptEncoder` | `src/nlpka/models/xpe.py` | The XPE module (based on NeMo's prompt encoder) |
| `EVALUATE` | `src/nlpka/evals/eval.py` | Metrics (accuracy, F1), confusion matrices, t-SNE plots |

**PEFT dispatch**: `PEFT.setup_model()` checks `is_xpe_config()` to route between standard PEFT methods and the custom XPE path (`get_xpe_model()`, which instantiates `XPEPeftModelForSequenceClassification`). `PEFT.from_pretrained()` peeks at `adapter_config.json` via `is_xpe_adapter_dir()` and dispatches to `load_xpe_pretrained()` — this preserves loading of paper-era checkpoints saved with `peft_type='P_TUNING' + encoder_ratio`.

**Enums** in `src/nlpka/enums.py` define all categorical choices (`ModelArchSE`, `TaskCatSE`, etc.) using `StrEnum` — check here before adding new method/task types.

**Foundational modules** (no circular dependencies):

| Module | Role | Depends on |
|--------|------|------------|
| `env.py` | Loads `.env`, exposes `env` | stdlib only |
| `utils.py` | Pure helpers, file I/O | stdlib + numpy/rich/tqdm/yaml |
| `path.py` | Path resolution, directory ops | `env` |
| `setup.py` | Runtime init (`init()`): Rich pretty + traceback | stdlib + rich |
| `enums.py` | All `StrEnum` types | stdlib only |

**Runtime init**: Call `micm_nlp.setup.init()` at the top of entrypoint scripts to activate Rich pretty-printing and tracebacks. This is not triggered on import.

## Key Conventions

- All experiments are tracked via **WandB** (`WANDB_API_KEY` in `.env`).
- HuggingFace token (`HF_TOKEN`) is needed for gated models/datasets.
- Model/dataset/tokenizer artifacts are cached under their respective `storage/` subdirectories.
- GPU training only (CPU supported for small-scale debugging); no non-NVIDIA GPU support.
- Package uses standard `src` layout — `pip install -e .` makes `from micm_nlp.X import Y` work.
