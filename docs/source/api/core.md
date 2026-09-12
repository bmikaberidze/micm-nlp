# Core

The top-level modules: the pipeline and everything it needs before a tokenizer,
dataset or model is involved. They carry no assumptions about any particular task
or study.

Ordered as the pipeline uses them — `cli` is the command-line entry point, `bootstrap` resolves the workspace, `pipeline` chains the stages, `group` runs many of them from one file, `config` validates the YAML that drives them, and `path`, `enums` and `utils` are the shared vocabulary underneath.

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

```{toctree}
:hidden:

/autoapi/micm_nlp/cli/index
/autoapi/micm_nlp/bootstrap/index
/autoapi/micm_nlp/pipeline/index
/autoapi/micm_nlp/group/index
/autoapi/micm_nlp/config/index
/autoapi/micm_nlp/path/index
/autoapi/micm_nlp/enums/index
/autoapi/micm_nlp/utils/index
```
