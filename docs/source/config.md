# Pipeline Unification

*unit config → unit run*

> How do I describe a whole run in one place, and make it reproducible?

```{include} ../../README.md
:start-after: <!-- start:blocks -->
:end-before: <!-- end:blocks -->
```

## Output Dir

```{include} ../../README.md
:start-after: <!-- start:run-dir -->
:end-before: <!-- end:run-dir -->
```

- The trainer is the only writer of the run directory.
- `info.json` holds what the config cannot: `started` / `finished`, every `SLURM*` variable, host, Python, `CUDA_VISIBLE_DEVICES`, the package versions, wandb id / url / dir, the resolved seed and `metric_for_best_model`, `paths.best_checkpoint`.
- Metric rows are the `compute_metrics` dict verbatim, plus `metric_group`, `step`, `time_id`, `uuid4` and `output.columns`.
- Predictions are saved after the same preprocessing the metric used, so any metric can be recomputed from the file; the `sample` column is the example's position in the test split.
- `<stage>` is `before_train` or `after_train`; a run that never trains has no suffix: `test.csv`, `predictions.csv`.
- Each `evaluate()` or `predict()` call is an event and writes its file once.
- The same rows are in memory as `output.results` and `output.predictions`, keyed by event name.

## Config

| Section | Purpose |
|---|---|
| `mode` | `preprocess`, `train`, `finetune`, `evaluate`, `test` — selects the pipeline path |
| `task` | Task identity, metric groups, prediction post-processing rules |
| `peft` | PEFT method and its hyperparameters |
| `model` | Architecture tag, pretrained source, adapter, or from-scratch init |
| `tokenizer` | Tokenizer source and behaviour |
| `ds` | Dataset location, input/label keys, preprocessing and tokenization rules |
| `eval` | When to evaluate (before/during/after training), per-task grouping |
| `test` | Whether to run the test split, and zero-shot behaviour |
| `trainer` | Which HuggingFace `Trainer` subclass to instantiate |
| `training_args` | Which HuggingFace `TrainingArguments` dataclass, plus its kwargs |
| `data_collator` | Which collator to instantiate, plus its kwargs |
| `custom_training_args` | Behaviour this package adds on top of HuggingFace |
| `cuda` | `empty_cache_steps` |
| `env` | Environment variables set at config-load time |
| `output` | Where the run writes and what every result row carries — see [`output`](#output) |
| `generation_config` | Kwargs for the HuggingFace `GenerationConfig`, for generative evaluation |

Each section is a pydantic model in {doc}`micm_nlp.config <autoapi/micm_nlp/config/index>` — the full schema.

**Every section accepts extra keys.**   
Undeclared keys pass through; declared ones are still validated.

:::{note}
PyYAML follows YAML 1.1, where `5e-5` (no decimal point) is a string.  
`micm_nlp.config` extends its float resolver at import, so `learning_rate: 5e-5` is a float everywhere.
:::

### `task.preproc_rules`

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

`label_restricted_likelihood` is lm-eval-harness `multiple_choice` scoring for `mcqa_ftp`: only the label tokens compete at the answer position, not the full vocabulary. Off by default.

### `peft`

All Cross-Prompt Encoder variants use `peft_type: XPE` and differ only in `encoder_ratio`, the fraction of virtual tokens that are cross-prompt encoded.

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

Any other `peft_type` goes to stock PEFT.

### `custom_training_args`

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


#### `early_stopping_metric`

Early stopping is decoupled from best-checkpoint selection.

| Value | Effect |
|---|---|
| `'metric_for_best_model'` | Delegate to `training_args` — same metric and direction used to pick the best checkpoint |
| any literal key, e.g. `'eval_loss'` | Monitor that key directly; direction inferred (a name containing `loss` means lower is better) |
| unset | Defaults to `'eval_loss'` |

Use it to select on accuracy while the evaluation loss is too unstable to stop on.

#### `eval/test_max_tokens_per_batch`

Token-budget batching: batches are built to a token count, keeping memory steady across languages whose tokenizations differ in length by an order of magnitude.

| Value | Behaviour |
|---|---|
| `null` | Fixed `per_device_*_batch_size` (the default) |
| `'auto'` | Probe the GPU at runtime for the largest budget that does not run out of memory |
| an integer | Skip the probe and use this budget exactly |

| Rule, checked at config load | Why |
|---|---|
| Excludes the matching `*_force_sequential` | Token budgets need length-sorted batches; a sequential sampler overrides that |
| Booleans rejected, integers must be positive | — |
| `training_args.group_by_length` is ignored, not rejected | The token-budget sampler length-sorts anyway |

:::{warning}
Samples come out length-sorted, not in dataset order.  
Zip predictions against a split through the sampler's `order` permutation — the package already does, for per-task grouping and saved predictions.
:::

#### `optimizer_grouped_parameters`

Gives parameters their own learning rate and weight decay — the mechanism behind a separate schedule for prompt embeddings.

| Key | Meaning |
|---|---|
| `param_name_parts` | A trainable parameter joins the group if its name contains any of these substrings |
| `lr` | The group's learning rate |
| `weight_decay` | The group's weight decay |

```yaml
custom_training_args:
    optimizer_grouped_parameters:
    - param_name_parts:
        - dedicated_embeddings
      lr: 5.0e-5
      weight_decay: 0.01
```

Parameters that match no group use `learning_rate` and `weight_decay` from `training_args`.

### `output`

`output` is optional; the group runner fills it for you.  
Set it by hand only to change `dir`, add `columns`, or set a `prefix` in a custom runner.

## Config Example

`xsc_finetune.yml` (from `micm-nlp init-examples`) preprocesses and fine-tunes BLOOM-560M with the Cross-Prompt
Encoder on the Arabic split of FTP-reframed XStoryCloze:

```{literalinclude} ../../src/micm_nlp/configs/xsc_finetune.yml
:language: yaml
```
