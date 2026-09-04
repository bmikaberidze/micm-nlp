# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**micm-nlp** (import name `micm_nlp`) is an NLP research toolkit built on HuggingFace
Transformers, covering tokenization, pretraining, fine-tuning and PEFT across
encoder-only, decoder-only and encoder-decoder architectures. It is the shared core
that experiment repos build on — **not** a frozen third-party dependency; it is
maintained alongside them.

It has backed two peer-reviewed publications: BERT-like pretraining/fine-tuning for
text and token classification, and parameter-efficient soft-prompt tuning across 200
languages (SIB-200 topic classification) with cross-prompt encoders on
XLM-RoBERTa-large. The most recent work — **Cross-Prompt Encoder (XPE)** — is in
*Findings of IJCNLP–AACL 2025* ([arXiv:2508.10352](https://arxiv.org/abs/2508.10352)).
Current work extends it to decoder-only backbones (Aya, BLOOMZ) and to further
encoder backbones on SIB-200.

Predecessor: the package was formerly called `nlpka`. That name and its module paths
(`src/nlpka/...`, `nlpka.models.scripts.*`) survive only in the legacy repo at
`/fscratch/bmikaberidze/XPE` — nothing here uses them.

## Setup

```bash
cp .env.example .env  # add WANDB_API_KEY to .env

# Docker (recommended)
docker build -t micm-nlp .
docker run --gpus all -it --rm -v $(pwd):/app -w /app micm-nlp bash

# Local
pip install -e ".[dev]"
```

## Running on the SLURM cluster (pegasus)

**NEVER run python on the login node. Every step goes through `sbatch` + an array,
and the container that `run.sh` starts carries the correct environment** — it is the
only interpreter whose output counts. This covers dataset downloads, tokenization,
probes, training, evaluation and pytest. The bare `python -m …` commands documented
below are the **inner** command that goes inside the wrapper's quotes.

```bash
sbatch --array=0 --mem=30G --wait runtime/clusters/pegasus/shell/run.sh --site-packages --no-gpu \
  "python -m pytest tests/ -q"

# a consumer repo's experiment grid, throttled to 10 concurrent tasks
sbatch --array=0-59%10 --mem=30G runtime/clusters/pegasus/shell/run.sh --site-packages \
  "python -m scripts.run_xlt_meta --meta-config ... --test-config ... --source-group ..."
```

### The rules

- **ALWAYS an array — even for one job** (`--array=0`). Never a bare `sbatch`.
- **NEVER loop `sbatch` over runs.** A shell loop submitting one `sbatch` per run puts
  one row per run in `squeue`, floods the cluster and has **no throttle**. One array
  submission is one row and is throttleable. If you are writing
  `for X in ...; do sbatch ...; done` over *runs*, that matrix belongs in a config.
- **ALWAYS throttle with `%10`** — `--array=0-59%10` runs at most 10 tasks at once.
  Unthrottled arrays starve other users and make dead nodes harder to spot.
- `--array`, `--mem`, `--partition` go on the **`sbatch` CLI**, never after `run.sh`:
  the wrapper consumes only `--site-packages` and `--no-gpu` as `$1`, and **anything
  else becomes the command it runs**, so a stray `--mem 30G` silently drops your
  python command. Add `--no-gpu` for CPU-only steps.
- `--mem` stays **≥30G** always — the ~25 GB container image unpacks into the job cgroup.
- Read output from `runtime/clusters/pegasus/shell/logs/sbatch/{jobid}_{task}.{out,err}`,
  not stdout. `--wait` blocks until the job finishes.
- Multi-line inspection snippets: write a `.py` file first, then run
  `"python <path>.py"` through the wrapper — nesting quotes inside `run.sh "…"` is fragile.

## Common Commands

`micm_nlp` is a **library**, not an experiment runner — it ships no experiment
entrypoints of its own. Experiments live in consumer repos and import from here:

- **`/fscratch/bmikaberidze/xpe-exp`** — the current XPE work (Belebele decoders,
  SIB-200 encoders). Its `scripts/run_xlt.py` / `scripts/run_xlt_meta.py` are the
  real entrypoints; they drive `CONFIG → DATASET → MODEL → PEFT → TRAINER`.
- **`/fscratch/bmikaberidze/XPE`** — the legacy `nlpka` repo that produced the
  published IJCNLP-AACL 2025 results. Read-only reference. Its
  `nlpka.models.scripts.peft.xpe.run` entrypoint does **not** exist here.

**Tests** (via the wrapper — see the SLURM section above):
```bash
sbatch --array=0 --mem=30G --wait runtime/clusters/pegasus/shell/run.sh --site-packages --no-gpu \
  "python -m pytest tests/ -q"
```

**Lint & format:**
```bash
ruff check src/
ruff format src/
```

**Install into a consumer repo** (editable, so edits here take effect immediately):
```bash
python -m pip install -e /fscratch/bmikaberidze/micm-nlp
```

## Project Structure

Standard `src` layout; `pip install -e .` makes `from micm_nlp.X import Y` work.

```
micm-nlp/
├── pyproject.toml              # hatchling; name = micm-nlp, requires-python >=3.10
├── dockerfile
├── examples/                   # usage demos for the package API
├── tests/                      # pytest suite (+ tests/golden fixtures)
├── runtime/                    # cluster wrappers (SLURM/pegasus)
├── docs/
└── src/micm_nlp/
    ├── __init__.py             # re-exports `env` and `init` from bootstrap
    ├── bootstrap.py            # .env loading (pydantic-settings), `env`, `init()`, set_root
    ├── config.py               # CONFIG (YAML → validated objects) + PeftConfig/TaskConfig/...
    ├── path.py                 # workspace/artefacts_dir/models_dir/datasets_dir/...
    ├── pipeline.py             # thin wiring: load_dataset, preprocess_dataset, load_model, run
    ├── utils.py                # resolve_cls, timing, get_time_id, JSON/YAML/pickle I/O
    ├── enums.py                # every StrEnum (ModelArchSE, TaskCatSE, TaskNameSE, ...)
    ├── datasets/dataset.py     # DATASET: load, tokenize, concat, splits
    ├── tokenizers/             # tokenizer.py (factory), decoding.py,
    │                           #   architectures.py (BertByT5Tokenizer, CustomXlmRoberta),
    │                           #   ka_sen_tok.py + data/ (Georgian sentence splitter)
    ├── models/
    │   ├── model.py            # MODEL: from_pretrained + task-derived kwargs
    │   ├── peft.py             # PEFT: dispatch to stock PEFT or the XPE path
    │   ├── architectures.py    # CustomT5ForConditionalGeneration (T5 + optional FlashAttention)
    │   └── xpe/                # a PACKAGE, not a module
    │       ├── encoder.py      # CrossPromptEncoder  <- XPE/SPT/DUAL are ALL this one class
    │       ├── config.py       # CrossPromptEncoderConfig
    │       ├── factory.py      # get_xpe_model, is_xpe_config, is_xpe_adapter_dir
    │       ├── peft_models.py  # XPEPeftModelFor{SequenceClassification,CausalLM}
    │       ├── heads.py        # MLP / LSTM / attention reparameterization heads
    │       ├── save_load.py    # XPE-aware state-dict get/set
    │       └── enums.py
    ├── training/
    │   ├── runner.py           # TRAINER: builds the HF Trainer, callbacks, collator
    │   ├── trainers.py         # CustomTrainerMixin, RandomTaskExclusionBatchSampler
    │   ├── callbacks.py        # CustomEarlyStopping, ParamNormLogger, NormalizePromptEncoder...
    │   ├── data_collators.py   # custom collators
    │   ├── batching.py         # TokenBudgetBatchSampler + calibration
    │   └── logits_processors.py
    └── evals/
        ├── eval.py             # get_compute_metrics, preprocess_logits_for_metrics, grouping
        ├── plot.py             # confusion matrices
        └── metrics/            # log_likelihood, multirc, string_f1
```

## Architecture

Class-based and config-driven throughout:

```
CONFIG (YAML) → tokenizer.load() → DATASET → MODEL → PEFT → TRAINER → compute_metrics
```

| Symbol | File | Role |
|--------|------|------|
| `CONFIG` | `src/micm_nlp/config.py` | Loads + validates YAML (`CONFIG.from_yaml`) |
| `tokenizer.load()` | `src/micm_nlp/tokenizers/tokenizer.py` | `AutoTokenizer` factory; `tokenizer.args` is a verbatim passthrough |
| `DATASET` | `src/micm_nlp/datasets/dataset.py` | Loads/preprocesses HF, CSV or TXT datasets; concatenation |
| `MODEL` | `src/micm_nlp/models/model.py` | `from_pretrained` via `model.pretrained.cls`; injects `num_labels` for classification tasks; `model.pretrained.args` is a verbatim passthrough |
| `PEFT` | `src/micm_nlp/models/peft.py` | Routes to stock PEFT or the XPE path |
| `CrossPromptEncoder` | `src/micm_nlp/models/xpe/encoder.py` | The XPE module (based on NeMo's prompt encoder) |
| `TRAINER` | `src/micm_nlp/training/runner.py` | Builds the HF Trainer: args, collator, callbacks, eval |
| `get_compute_metrics` | `src/micm_nlp/evals/eval.py` | Metrics, per-task grouping, logits preprocessing |

**XPE / SPT / DUAL are ALL one class — `CrossPromptEncoder`.** All three set
`peft_type: XPE` and differ only by `encoder_ratio`, the XPE fraction of the virtual
tokens: **SPT = 0** (plain soft prompt, `reparam=NONE`, uses `self.embedding`),
**XPE = 1** (`self.xpe_embedding` → `xpe_head` MLP), **DUAL = 0 < r < 1** (concat of
both; the ratio is a free hyperparameter, not tied to the dataset). Consequence:
anything gated on `isinstance(pe, CrossPromptEncoder)` fires for SPT too.

**PEFT dispatch**: `PEFT.setup_model()` checks `is_xpe_config()` to route between
stock PEFT methods and the XPE path (`get_xpe_model()`, which picks an
`XPEPeftModelFor*` subclass from `task_type`). `PEFT.from_pretrained()` peeks at
`adapter_config.json` via `is_xpe_adapter_dir()` so paper-era checkpoints saved with
`peft_type='P_TUNING' + encoder_ratio` still load.

**Class selection stays in YAML.** `model.pretrained.cls`, `trainer.cls`,
`data_collator.cls` and `training_args.cls` are resolved by name against
`transformers` (and, for collators, `micm_nlp.training.data_collators`). Adding a new
backbone or head should need **no code change here** — if it does, that is a signal
the change belongs in the consumer repo, not in this toolkit.

**Enums** in `src/micm_nlp/enums.py` define the categorical choices (`ModelArchSE`,
`TaskCatSE`, `TaskNameSE`, ...). Note `model.architecture` is *not* validated against
`ModelArchSE` — it is a free-form string used for run-directory naming.

**Runtime init**: `micm_nlp.init()` (from `bootstrap.py`) activates Rich pretty-printing
and tracebacks and sets the project root. Not triggered on import.

## Known defects (deliberately unfixed — see git log before "fixing" either)

None currently. Both former entries are resolved:

- The `training/runner.py` lookup of `self._config.task.peft` — `peft` is a top-level
  config block, so `NormalizePromptEncoderEmbeddings` never registered and the
  embedding clip/unit-norm callback never ran. Fixed in `d3ae7e1`. Note the
  consequence for old results: any run before that commit did not normalise, whatever
  its config said.
- The leftover `print(batch[0])` / `exit()` debug body in
  `DataCollatorTaskIDDecorator.__call__` — both lines are now commented out.

Add an entry here only for a defect that is staying broken *on purpose*, with the
reason. A defect that should be fixed belongs in `docs/internal/roadmap.md`.

## Documentation: the source tree is the source of truth

**Docs are written where the thing they describe lives, and the site is generated
from there.** A page hand-written in `docs/source/` is the exception that has to
justify itself, not the default.

Order of preference, most preferred first:

1. **Docstrings.** Everything about a module, class or function goes in its
   docstring. `sphinx-autoapi` parses `src/` statically (it never imports the
   package) and renders the whole API reference from them. A package's
   `__init__.py` docstring is where its overview and submodule roles belong — see
   `models/xpe/__init__.py` for the shape to copy.
2. **Repo-root files, pulled in with MyST `{include}`.** `docs/source/changelog.md`
   is already nothing but `` ```{include} ../../CHANGELOG.md `` — that is the
   pattern. README sections can be included the same way with `:start-after:` /
   `:end-before:` markers, so README stays the self-contained PyPI landing page and
   the docs never hold a second copy.
3. **A hand-written page in `docs/source/`** — only when the content belongs to no
   module and to no repo-root file. Today that is `config.md`, the YAML schema
   reference: too long for the README, and not owned by any one module.

Two rules that follow:

- **The sidebar mirrors `src/micm_nlp/`.** The API section links autoapi's root and
  nothing else, so the tree *is* the package tree. If a grouping looks wrong in the
  sidebar, the package layout is wrong — move the source, do not hand-list pages in
  a toctree. `conf.py` shortens the sidebar labels to the leaf module name; that is
  presentation, and it is the only intervention allowed there.
- **Never write the same prose twice.** README §Quickstart and
  `docs/source/quickstart.md` were duplicates and had already drifted apart (quote
  style, and one paragraph each that the other lacks). Whichever file owns a passage,
  every other place `{include}`s it.

## Key Conventions

- **Never collapse `models/xpe/` into fewer modules.** Its seven-module split is the
  published method's structure and keeps the pieces separately testable. The docs
  sidebar shows it as a dropdown; that is the intended cost, not a problem to fix.
- Keep this package **general**. Experiment-specific language groups, run-tree
  layouts, result aggregation and dispatch logic belong in the consumer repo
  (`xpe-exp/scripts/`), never here. If a backbone needs a one-off shim, prefer the
  YAML passthroughs (`model.pretrained.args`, `tokenizer.args`) over new code.
- Own git repo with its own branches — commit changes here separately from `xpe-exp`.
- Experiments are tracked via **WandB** (`WANDB_API_KEY` in `.env`); gated
  models/datasets need `HF_TOKEN`.
- GPU training only (CPU works for small-scale debugging); no non-NVIDIA GPU support.
- `peft` is pinned to `0.14.0` in `pyproject.toml` — XPE subclasses stock PEFT
  internals, so bumping it is a breaking-change review, not a routine upgrade.
