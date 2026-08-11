# API reference

Generated from the source tree. Two groups: the **core**, which carries no
assumptions about any particular study, and the **research modules**, which exist
because published work depends on them.

Start at {doc}`autoapi/micm_nlp/index` for the flat, complete module list.

## Core

The config-driven pipeline and everything it needs.

| Module | Role |
|---|---|
| {doc}`config <autoapi/micm_nlp/config/index>` | `CONFIG.from_yaml` and the validated section models |
| {doc}`pipeline <autoapi/micm_nlp/pipeline/index>` | `run()` and the stage-by-stage wiring |
| {doc}`bootstrap <autoapi/micm_nlp/bootstrap/index>` | `.env` loading, `env`, `init()`, workspace root |
| {doc}`path <autoapi/micm_nlp/path/index>` | `artefacts_dir`, `models_dir`, `datasets_dir`, … |
| {doc}`enums <autoapi/micm_nlp/enums/index>` | Categorical choices (`ModelArchSE`, `TaskCatSE`, `TaskNameSE`, …) |
| {doc}`utils <autoapi/micm_nlp/utils/index>` | `resolve_cls`, timing, time ids, JSON/YAML/pickle I/O |

### Datasets and tokenizers

| Module | Role |
|---|---|
| {doc}`datasets.dataset <autoapi/micm_nlp/datasets/dataset/index>` | `DATASET` — load, tokenize, concatenate, split |
| {doc}`tokenizers.tokenizer <autoapi/micm_nlp/tokenizers/tokenizer/index>` | `AutoTokenizer` factory; special tokens and post-processors |
| {doc}`tokenizers.xlm_roberta <autoapi/micm_nlp/tokenizers/xlm_roberta/index>` | XLM-R tokenizer adapted to a target architecture |
| {doc}`tokenizers.decoding <autoapi/micm_nlp/tokenizers/decoding/index>` | Label-aware `decode` / `batch_decode` |

### Models and PEFT

| Module | Role |
|---|---|
| {doc}`models.model <autoapi/micm_nlp/models/model/index>` | `MODEL` — `from_pretrained` plus task-derived kwargs |
| {doc}`models.peft <autoapi/micm_nlp/models/peft/index>` | `PEFT` — dispatch to stock PEFT or the Cross-Prompt Encoder path |
| {doc}`models.architectures <autoapi/micm_nlp/models/architectures/index>` | `CustomT5ForConditionalGeneration` |

### Training

| Module | Role |
|---|---|
| {doc}`training.runner <autoapi/micm_nlp/training/runner/index>` | `TRAINER` — builds the HuggingFace `Trainer`, callbacks, collator |
| {doc}`training.trainers <autoapi/micm_nlp/training/trainers/index>` | `CustomTrainerMixin`, `RandomTaskExclusionBatchSampler` |
| {doc}`training.callbacks <autoapi/micm_nlp/training/callbacks/index>` | `CustomEarlyStoppingCallback`, `ParamNormLogger`, … |
| {doc}`training.batching <autoapi/micm_nlp/training/batching/index>` | `TokenBudgetBatchSampler` and `calibrate_token_budget` |
| {doc}`training.data_collators <autoapi/micm_nlp/training/data_collators/index>` | Custom collators |
| {doc}`training.logits_processors <autoapi/micm_nlp/training/logits_processors/index>` | Generation-time logits processors |

### Evaluation

| Module | Role |
|---|---|
| {doc}`evals.eval <autoapi/micm_nlp/evals/eval/index>` | `get_compute_metrics`, logits preprocessing, per-task grouping |
| {doc}`evals.plot <autoapi/micm_nlp/evals/plot/index>` | Confusion matrices |
| {doc}`evals.metrics.log_likelihood <autoapi/micm_nlp/evals/metrics/log_likelihood/index>` | Log-likelihood scoring |
| {doc}`evals.metrics.multirc <autoapi/micm_nlp/evals/metrics/multirc/index>` | MultiRC metric |
| {doc}`evals.metrics.string_f1 <autoapi/micm_nlp/evals/metrics/string_f1/index>` | String-level F1 |

## Research modules

Shipped because published work depends on them — **not required to use the core**.

:::{note}
The separation here is by purpose, not by packaging: these modules install with
everything else. Moving them behind an optional extra would break existing
consumers and is deferred.
:::

### Cross-Prompt Encoder

The method from *Cross-Prompt Encoder for Low-Performing Languages*
(Findings of IJCNLP–AACL 2025).

XPE, SPT and DUAL are **one class**, `CrossPromptEncoder`, separated only by
`encoder_ratio` — see [the configuration reference](config.md#peft). Anything gated
on `isinstance(pe, CrossPromptEncoder)` therefore fires for plain soft prompt tuning
as well.

| Module | Role |
|---|---|
| {doc}`models.xpe.encoder <autoapi/micm_nlp/models/xpe/encoder/index>` | `CrossPromptEncoder` |
| {doc}`models.xpe.config <autoapi/micm_nlp/models/xpe/config/index>` | `CrossPromptEncoderConfig` |
| {doc}`models.xpe.factory <autoapi/micm_nlp/models/xpe/factory/index>` | `get_xpe_model`, `is_xpe_config`, `is_xpe_adapter_dir` |
| {doc}`models.xpe.peft_models <autoapi/micm_nlp/models/xpe/peft_models/index>` | `XPEPeftModelFor{SequenceClassification,CausalLM}` |
| {doc}`models.xpe.heads <autoapi/micm_nlp/models/xpe/heads/index>` | MLP / LSTM / attention reparameterization heads |
| {doc}`models.xpe.save_load <autoapi/micm_nlp/models/xpe/save_load/index>` | XPE-aware state-dict get and set |
| {doc}`models.xpe.enums <autoapi/micm_nlp/models/xpe/enums/index>` | Reparameterization enums |

### Georgian tokenization

From *A Comparison of Different Tokenization Methods for the Georgian Language*
(ICNLSP 2024).

| Module | Role |
|---|---|
| {doc}`tokenizers.bert_byt5 <autoapi/micm_nlp/tokenizers/bert_byt5/index>` | `BertByT5Tokenizer` — byte-level tokenizer with BERT-style special tokens |
| {doc}`tokenizers.lib.sent.ka_sen_tok <autoapi/micm_nlp/tokenizers/lib/sent/ka_sen_tok/index>` | `KaSenTok` — Georgian sentence tokenizer |

```{toctree}
:hidden:

autoapi/micm_nlp/index
```
