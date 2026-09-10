# micm-nlp

```{include} ../../README.md
:start-after: <!-- start:tagline -->
:end-before: <!-- end:tagline -->
```

```{include} ../../README.md
:start-after: <!-- start:install-pypi -->
:end-before: <!-- end:install-pypi -->
```

```{include} ../../README.md
:start-after: <!-- start:about -->
:end-before: <!-- end:about -->
```

## What it is

A **library with a thin run harness**. The building blocks compose into one chain:

```
CONFIG (YAML) → tokenizer.load() → DATASET → MODEL → PEFT → TRAINER → compute_metrics
```

Everything above is selected from YAML — including the concrete HuggingFace classes.
`model.pretrained.cls`, `trainer.cls`, `data_collator.cls` and `training_args.cls` are
resolved by name at runtime, so adding a backbone or a head normally needs no code
change.

The package owns how a run is assembled and where it lands: one config → one run, one
group config → many. Study-specific meaning — what a language group is, what a result
table should look like — stays in the consumer repositories that import it.

| Symbol | Role |
|---|---|
| `CONFIG` | Loads and validates YAML (`CONFIG.from_yaml`) |
| `tokenizer.load()` | `AutoTokenizer` factory |
| `DATASET` | Loads and preprocesses HuggingFace, CSV or TXT datasets; concatenation |
| `MODEL` | `from_pretrained` via `model.pretrained.cls`; injects `num_labels` for classification |
| `PEFT` | Routes to stock PEFT methods or the Cross-Prompt Encoder path |
| `TRAINER` | Builds the HuggingFace `Trainer`: arguments, collator, callbacks, evaluation |
| `RunOutput` | The run's output directory: config snapshot, `run.json`, result files, links |
| `run_group()` | Expands a group config into runs and picks the one this process runs |
| `RunContext` | What a custom runner receives beside the config: the entry, output dir, extras, `test_config` |
| `cli` | `micm-nlp` / `python -m micm_nlp`: `run`, `run-group`, `init-examples` |

## Scope

Configuration, datasets, models, PEFT dispatch, training and evaluation carry no
assumptions about any particular study.

Published work is not partitioned off into a "research" corner — it sits in the
package where it belongs, and each module's own page cites the paper behind it. The
Cross-Prompt Encoder is {doc}`models.xpe </autoapi/micm_nlp/models/xpe/index>`; the
Georgian tokenization work is
{doc}`tokenizers.architectures </autoapi/micm_nlp/tokenizers/architectures/index>`
and {doc}`tokenizers.ka_sen_tok </autoapi/micm_nlp/tokenizers/ka_sen_tok/index>`.
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
config
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

changelog
PyPI <https://pypi.org/project/micm-nlp/>
GitHub <https://github.com/bmikaberidze/micm-nlp>
MICM <https://micm.edu.ge/>
```
