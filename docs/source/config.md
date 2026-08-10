# YAML configuration

Every run is described by a single YAML file loaded with `CONFIG.from_yaml`. The
schema is a set of pydantic models in
{doc}`micm_nlp.config <autoapi/micm_nlp/config/index>`.

Two properties are worth knowing before reading the reference below.

**Every section accepts extra keys.** All config sections inherit from a permissive
base, so a YAML file can carry keys the schema does not declare and runtime code can
attach computed attributes. Validation catches the fields that matter without
blocking the rest.

**Class selection lives in YAML.** `model.pretrained.cls`, `trainer.cls`,
`data_collator.cls` and `training_args.cls` are resolved by name against
`transformers` (and, for collators, `micm_nlp.training.data_collators`). Adding a new
backbone or head should need no change to this package. Where a class needs an
unusual keyword argument, prefer the passthrough dictionaries — `model.pretrained.args`
and `tokenizer.args` are splatted verbatim into the constructor — over new code.

:::{note}
Scientific notation works without a decimal point. PyYAML's `SafeLoader` follows
YAML 1.1, where `5e-5` parses as a *string*; `micm_nlp.config` widens the float
resolver once at import, so `learning_rate: 5e-5` is a float everywhere.
:::

## Top-level sections

| Section | Purpose |
|---|---|
| `mode` | `finetune`, `test`, `train`, … — selects the pipeline path |
| `task` | Task identity, metric groups, prediction post-processing rules |
| `peft` | PEFT method and its hyperparameters |
| `model` | Architecture tag, pretrained source, adapter, or from-scratch init |
| `tokenizer` | Tokenizer source and behaviour |
| `ds` | Dataset location, input/label keys, preprocessing and tokenization rules |
| `eval` | When to evaluate (before/during/after training), per-task grouping |
| `test` | Whether to run the test split, zero-shot behaviour, prediction saving |
| `trainer` | Which HuggingFace `Trainer` subclass to instantiate |
| `training_args` | Which HuggingFace `TrainingArguments` dataclass, plus its kwargs |
| `data_collator` | Which collator to instantiate, plus its kwargs |
| `custom_training_args` | Behaviour this package adds on top of HuggingFace |
| `cuda` | `empty_cache_steps` |
| `env` | Environment variables set at config-load time |

## `peft`

All Cross-Prompt Encoder variants use `peft_type: XPE` and differ only in
`encoder_ratio` — the fraction of virtual tokens that are cross-prompt encoded.

| `encoder_ratio` | Variant | Behaviour |
|---|---|---|
| `0` | SPT | Plain soft prompt tuning, no reparameterization |
| `1` | XPE | All virtual tokens pass through the encoder head |
| `0 < r < 1` | DUAL | Concatenation of both; the ratio is a free hyperparameter |

```yaml
peft:
    peft_type: XPE
    task_type: CAUSAL_LM
    num_virtual_tokens: 20
    encoder_reparameterization_type: MLP
    encoder_hidden_size: 256
    encoder_num_layers: 2
    encoder_dropout: 0.1
    encoder_ratio: 1
```

`PEFT.setup_model()` routes to the Cross-Prompt Encoder path or to stock PEFT
depending on this block. Checkpoints written before the `XPE` `peft_type` existed
(`P_TUNING` plus an `encoder_ratio`) still load.

## `custom_training_args`

The knobs this package adds beyond HuggingFace's `TrainingArguments`.

| Key | Type | Meaning |
|---|---|---|
| `early_stopping_after` | float | Fraction of training before stopping may trigger |
| `early_stopping_patience` | int | Evaluations without improvement before stopping |
| `early_stopping_threshold` | float | Minimum improvement that counts |
| `early_stopping_metric` | str | Metric to monitor — see below |
| `eval_max_tokens_per_batch` | int \| `'auto'` \| null | Token-budget batching for evaluation |
| `test_max_tokens_per_batch` | int \| `'auto'` \| null | Token-budget batching for the test split |
| `train/eval/test_force_sequential` | bool | Force a sequential sampler for that stage |
| `save_final_model` | bool | Save the final model after training |
| `keep_only_final_model` | bool | Discard intermediate checkpoints |
| `usable_columns` | list[str] | Extra dataset columns to keep past `_remove_unused_columns` |
| `optimizer_grouped_parameters` | list | Per-parameter-group learning rate and weight decay |
| `random_task_exclusion` | bool | Batch sampler that holds out a random task |
| `generation_whitelist` | list[str] | Restrict generation to these strings |

### `early_stopping_metric`

Early stopping is decoupled from best-checkpoint selection.

| Value | Effect |
|---|---|
| `'metric_for_best_model'` | Delegate to `training_args` — same metric and direction used to pick the best checkpoint |
| any literal key, e.g. `'eval_loss'` | Monitor that key directly; direction inferred (a name containing `loss` means lower is better) |
| unset | Defaults to `'eval_loss'` |

This matters when the selection metric and the stopping signal should differ — for
example selecting on accuracy while the evaluation loss is unstable.

### Token-budget batching

Instead of a fixed `per_device_eval_batch_size`, batches can be built to a target
token count. This keeps memory roughly constant across languages whose tokenizations
differ in length by an order of magnitude.

| Value | Behaviour |
|---|---|
| `null` | Fixed `per_device_*_batch_size` (the default) |
| `'auto'` | Probe the GPU at runtime for the largest budget that does not run out of memory |
| an integer | Skip the probe and use this budget exactly |

Two constraints are validated at config load:

- Mutually exclusive with the matching `*_force_sequential` flag — token-budget mode
  needs length-sorted batching, which a sequential sampler overrides.
- Mutually exclusive with HuggingFace's `LengthGroupedSampler`; the token-budget
  sampler already sorts by length.

Booleans are rejected, and integers must be positive.

:::{warning}
The token-budget sampler yields samples in globally length-sorted order, not dataset
order. Anything zipping predictions against a dataset split must use the sampler's
`order` permutation. The package does this internally for per-task grouping and
prediction saving; custom consumers of raw predictions should be aware of it.
:::

## `optimizer_grouped_parameters`

Assigns a different learning rate and weight decay to parameters whose names contain
given substrings — the mechanism behind giving prompt embeddings their own schedule:

```yaml
custom_training_args:
    optimizer_grouped_parameters:
    - param_name_parts:
        - dedicated_embeddings
      lr: 5.0e-5
      weight_decay: 0.01
```

Parameters that match no group fall back to the global `learning_rate` and
`weight_decay` from `training_args`.

## `task.preproc_rules`

Post-processing applied to predictions before metrics.

| Key | Meaning |
|---|---|
| `flatten` | Flatten predictions and labels before metric computation |
| `filter_padded` | Drop padded positions |
| `label_id_to_name` / `label_name_to_id` | Convert between label ids and names |
| `label_name_strip_lower` | Normalise label names before comparison |
| `verify_labels_match` | Assert predictions and labels line up |
| `calc_confusion_matrix` | Produce a confusion matrix |
| `prediction_axis` | Axis for the argmax (default `-1`) |
| `label_restricted_likelihood` | Restrict the answer-slot argmax to the candidate tokens in `ds.label.names` |

`label_restricted_likelihood` implements lm-eval-harness `multiple_choice` scoring
for `mcqa_ftp`: rather than taking a full-vocabulary argmax at the answer position,
only the configured label tokens compete. It is opt-in and off by default.

## A complete example

`examples/configs/xsc_finetune.yml` fine-tunes BLOOM-560M with the Cross-Prompt
Encoder on the Arabic split of FTP-reframed XStoryCloze:

```yaml
mode: finetune

task:
    category: text_generation
    name: mcqa_ftp
    metric_groups:
    - metrics:
        - accuracy
    preproc_rules:
        flatten: true
        filter_padded: true
        verify_labels_match: true

peft:
    peft_type: XPE
    task_type: CAUSAL_LM
    num_virtual_tokens: 20
    encoder_reparameterization_type: MLP
    encoder_hidden_size: 256
    encoder_num_layers: 2
    encoder_dropout: 0.1
    encoder_ratio: 1

model:
    architecture: bloom
    pretrained:
        cls: AutoModelForCausalLM
        name: bigscience/bloom-560m
        source: huggingface

tokenizer:
    source: huggingface
    name: bigscience/bloom-560m
    args:
        padding_side: right

ds:
    category: benchmarks
    dirs: mikaberidze/xstory-cloze-ftp
    name: ar
    type: huggingface
    comes_with_splits:
        train: eval
        test: false
        validation: train
    input:
        key: text
        standardize_key: true
    label:
        key: answer_label
        standardize_key: true

trainer:
    cls: Trainer

training_args:
    cls: TrainingArguments
    args:
        num_train_epochs: 10
        learning_rate: 5.0e-5
        metric_for_best_model: accuracy
        greater_is_better: true
        load_best_model_at_end: true
        bf16: true
```

The full file, including tokenization rules, evaluation schedule and collator
settings, is in the repository.
