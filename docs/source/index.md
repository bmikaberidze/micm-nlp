# micm-nlp

An NLP research toolkit for tokenization, pretraining, fine-tuning and PEFT across
encoder-only, decoder-only and encoder-decoder architectures. Built on HuggingFace
`transformers`, `peft` and `datasets`.

```bash
pip install micm-nlp
```

## What it is

`micm-nlp` is a **config-driven library**, not an experiment runner. It wraps the
HuggingFace stack in a small set of building blocks that compose into reproducible
training and evaluation pipelines:

```
CONFIG (YAML) → tokenizer.load() → DATASET → MODEL → PEFT → TRAINER → compute_metrics
```

Everything above is selected from YAML — including the concrete HuggingFace classes.
`model.pretrained.cls`, `trainer.cls`, `data_collator.cls` and `training_args.cls` are
resolved by name at runtime, so adding a backbone or a head normally needs no code
change. Experiment logic — language groups, run-tree layouts, result aggregation,
cluster dispatch — lives in consumer repositories that import this package, never here.

| Symbol | Role |
|---|---|
| `CONFIG` | Loads and validates YAML (`CONFIG.from_yaml`) |
| `tokenizer.load()` | `AutoTokenizer` factory |
| `DATASET` | Loads and preprocesses HuggingFace, CSV or TXT datasets; concatenation |
| `MODEL` | `from_pretrained` via `model.pretrained.cls`; injects `num_labels` for classification |
| `PEFT` | Routes to stock PEFT methods or the Cross-Prompt Encoder path |
| `TRAINER` | Builds the HuggingFace `Trainer`: arguments, collator, callbacks, evaluation |

## Scope

The core of the package is task- and research-agnostic: configuration, datasets,
models, PEFT dispatch, training and evaluation carry no assumptions about any
particular study.

It also ships two **research modules** — the Cross-Prompt Encoder and a set of
Georgian tokenization utilities — because published work depends on them. Neither is
required to use the core, and both are marked as such in the
[API reference](api.md).

## Links

| | |
|---|---|
| **PyPI** | [pypi.org/project/micm-nlp](https://pypi.org/project/micm-nlp/) |
| **Source** | [github.com/bmikaberidze/micm-nlp](https://github.com/bmikaberidze/micm-nlp) |
| **Issue tracker** | [github.com/bmikaberidze/micm-nlp/issues](https://github.com/bmikaberidze/micm-nlp/issues) |
| **Releases** | [github.com/bmikaberidze/micm-nlp/releases](https://github.com/bmikaberidze/micm-nlp/releases) |
| **Changelog** | [this site](changelog.md) |
| **XPE paper** | [ACL Anthology](https://aclanthology.org/2025.findings-ijcnlp.144/) · [arXiv:2508.10352](https://arxiv.org/abs/2508.10352) |
| **Tokenization paper** | [ACL Anthology](https://aclanthology.org/2024.icnlsp-1.22/) |
| **Contact** | beso.mikaberidze@gmail.com |

## Provenance

`micm-nlp` has backed two peer-reviewed publications:

- *Cross-Prompt Encoder for Low-Performing Languages* — Findings of IJCNLP–AACL 2025
  ([ACL Anthology](https://aclanthology.org/2025.findings-ijcnlp.144/),
  [arXiv](https://arxiv.org/abs/2508.10352))
- *A Comparison of Different Tokenization Methods for the Georgian Language* —
  ICNLSP 2024 ([ACL Anthology](https://aclanthology.org/2024.icnlsp-1.22/))

It was developed at the Muskhelishvili Institute of Computational Mathematics (MICM,
Georgian Technical University). The package was formerly named `nlpka`; that name
survives only in the archived repository behind the IJCNLP–AACL paper.

```{toctree}
:caption: Getting started
:hidden:

Home <self>
install
quickstart
config
```

```{toctree}
:caption: API reference
:hidden:

Overview <api>
```

```{toctree}
:caption: Core
:hidden:

config <autoapi/micm_nlp/config/index>
pipeline <autoapi/micm_nlp/pipeline/index>
bootstrap <autoapi/micm_nlp/bootstrap/index>
path <autoapi/micm_nlp/path/index>
enums <autoapi/micm_nlp/enums/index>
utils <autoapi/micm_nlp/utils/index>
```

```{toctree}
:caption: Data
:hidden:

datasets.dataset <autoapi/micm_nlp/datasets/dataset/index>
tokenizers.tokenizer <autoapi/micm_nlp/tokenizers/tokenizer/index>
tokenizers.xlm_roberta <autoapi/micm_nlp/tokenizers/xlm_roberta/index>
tokenizers.decoding <autoapi/micm_nlp/tokenizers/decoding/index>
```

```{toctree}
:caption: Models
:hidden:

models.model <autoapi/micm_nlp/models/model/index>
models.peft <autoapi/micm_nlp/models/peft/index>
models.architectures <autoapi/micm_nlp/models/architectures/index>
```

```{toctree}
:caption: Training
:hidden:

training.runner <autoapi/micm_nlp/training/runner/index>
training.trainers <autoapi/micm_nlp/training/trainers/index>
training.callbacks <autoapi/micm_nlp/training/callbacks/index>
training.batching <autoapi/micm_nlp/training/batching/index>
training.data_collators <autoapi/micm_nlp/training/data_collators/index>
training.logits_processors <autoapi/micm_nlp/training/logits_processors/index>
```

```{toctree}
:caption: Evaluation
:hidden:

evals.eval <autoapi/micm_nlp/evals/eval/index>
evals.plot <autoapi/micm_nlp/evals/plot/index>
evals.metrics.log_likelihood <autoapi/micm_nlp/evals/metrics/log_likelihood/index>
evals.metrics.multirc <autoapi/micm_nlp/evals/metrics/multirc/index>
evals.metrics.string_f1 <autoapi/micm_nlp/evals/metrics/string_f1/string_f1/index>
```

```{toctree}
:caption: Research modules
:hidden:

models.xpe.encoder <autoapi/micm_nlp/models/xpe/encoder/index>
models.xpe.config <autoapi/micm_nlp/models/xpe/config/index>
models.xpe.factory <autoapi/micm_nlp/models/xpe/factory/index>
models.xpe.peft_models <autoapi/micm_nlp/models/xpe/peft_models/index>
models.xpe.heads <autoapi/micm_nlp/models/xpe/heads/index>
models.xpe.save_load <autoapi/micm_nlp/models/xpe/save_load/index>
models.xpe.enums <autoapi/micm_nlp/models/xpe/enums/index>
tokenizers.bert_byt5 <autoapi/micm_nlp/tokenizers/bert_byt5/index>
tokenizers.lib.sent.ka_sen_tok <autoapi/micm_nlp/tokenizers/lib/sent/ka_sen_tok/index>
```

```{toctree}
:caption: Meta
:hidden:

changelog
PyPI <https://pypi.org/project/micm-nlp/>
GitHub <https://github.com/bmikaberidze/micm-nlp>
Issue tracker <https://github.com/bmikaberidze/micm-nlp/issues>
```
