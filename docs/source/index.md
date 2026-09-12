# micm-nlp

```{include} ../../README.md
:start-after: <!-- start:tagline -->
:end-before: <!-- end:tagline -->
```

```bash
pip install micm-nlp
```

```{include} ../../README.md
:start-after: <!-- start:about -->
:end-before: <!-- end:about -->
```

## What it is

The package contributes three things:

```{include} ../../README.md
:start-after: <!-- start:contributions -->
:end-before: <!-- end:contributions -->
```

Read them in the reverse of the order they are listed in — a feature is written in the vocabulary of the other two, and *per-parameter-group optimizer settings* means nothing until you know a run is a YAML file.  
Start with {doc}`config` for the unit run, then {doc}`groups` for running many of them, and browse {doc}`features` for what you can do inside them.

The building blocks compose into one chain:

```
CONFIG (YAML) → tokenizer.load() → DATASET → MODEL → PEFT → TRAINER → compute_metrics
```

| Symbol | Role |
|---|---|
| `CONFIG` | loads and validates YAML (`CONFIG.from_yaml`) |
| `tokenizer.load()` | `AutoTokenizer` factory |
| `DATASET` | loads and preprocesses local, Hub, CSV or TXT datasets; concatenation |
| `MODEL` | `from_pretrained` via `model.pretrained.cls`; injects `num_labels` for classification |
| `PEFT` | routes to stock PEFT methods or the Cross-Prompt Encoder path |
| `TRAINER` | builds the HuggingFace `Trainer`: arguments, collator, callbacks, evaluation |
| `RunOutput` | the run's output directory: config snapshot, `info.json`, result files, links |
| `run_group()` | expands a group config into runs and picks the one this process runs |
| `RunContext` | what a custom runner receives beside the config: the entry, output dir, extras, `test_config` |
| `cli` | `micm-nlp` / `python -m micm_nlp`: `run`, `run-group`, `init-examples` |

Every link in the chain is named in YAML, the concrete HuggingFace classes included, so a new backbone or head needs no code.  
The package owns how a run is assembled and where it lands; study-specific meaning — what a language group is, what a result table should look like — stays in the repositories that import it.

## Scope

| Architecture | Supported | Covered by a shipped example |
|---|---|---|
| Decoder-only (BLOOM, Aya) | yes | yes |
| Encoder-only (BERT, XLM-R, mDeBERTa) | yes | planned |
| Encoder-decoder (T5) | yes | planned |

PEFT methods: LoRA, Prefix Tuning, P-Tuning / soft prompt tuning, and the Cross-Prompt Encoder — the shipped examples demonstrate the last only.  
Configuration, datasets, models, PEFT dispatch, training and evaluation carry no assumptions about any particular study.

Published work is not partitioned off into a "research" corner. It sits in the package where it belongs, and each module's page cites the paper behind it — the Cross-Prompt Encoder is {doc}`models.xpe </autoapi/micm_nlp/models/xpe/index>`, the Georgian tokenization work is {doc}`tokenizers.architectures </autoapi/micm_nlp/tokenizers/architectures/index>` and {doc}`tokenizers.ka_sen_tok </autoapi/micm_nlp/tokenizers/ka_sen_tok/index>`.  
None of it is required to use the rest.

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
| **MICM** | [micm.edu.ge](https://micm.edu.ge/) |
| **Contact** | beso.mikaberidze@gmail.com |

## Provenance

```{include} ../../README.md
:start-after: <!-- start:acknowledgements -->
:end-before: <!-- end:acknowledgements -->
```

The package was formerly named `nlpka`; that name survives only in the archived
repository behind the IJCNLP–AACL paper.

## Citation

```{include} ../../README.md
:start-after: <!-- start:citation -->
:end-before: <!-- end:citation -->
```

```{toctree}
:caption: Getting started
:hidden:

Home <self>
install
quickstart
```

```{toctree}
:caption: What micm-nlp offers
:hidden:

config
groups
features
```

```{toctree}
:caption: API reference
:hidden:

Core <api/core>
autoapi/micm_nlp/tokenizers/index
autoapi/micm_nlp/datasets/index
autoapi/micm_nlp/models/index
autoapi/micm_nlp/training/index
autoapi/micm_nlp/evals/index
```

```{toctree}
:caption: Meta
:hidden:

contributing
changelog
PyPI <https://pypi.org/project/micm-nlp/>
GitHub <https://github.com/bmikaberidze/micm-nlp>
MICM <https://micm.edu.ge/>
```
