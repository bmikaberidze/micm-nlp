# Changelog

All notable changes to micm-nlp will be documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-10

### Added
- Token-budget batching: `TokenBudgetBatchSampler` builds batches to a target token
  count instead of a fixed sample count, with a `calibrate_token_budget` probe that
  binary-searches the largest budget that fits in memory.
- `eval_max_tokens_per_batch` / `test_max_tokens_per_batch` config keys (validated),
  wiring token-budget batching into the eval and test dataloaders. When unset, the
  previous fixed-batch path is used unchanged.
- `early_stopping_metric` on `CustomTrainingArgsConfig`, decoupling early stopping
  from best-checkpoint selection: `'metric_for_best_model'` delegates to
  `training_args`, any literal key (e.g. `'eval_loss'`) is monitored directly with
  its direction inferred. Defaults to `'eval_loss'`, preserving prior behaviour.
- A seed configured in `training_args` is now honoured without enabling
  `full_determinism`; the seed is randomised only when none is configured. Lets
  callers share fixed seeds across methods for paired comparison at no determinism
  overhead.
- Label-restricted likelihood for `mcqa_ftp` via
  `preproc_rules.label_restricted_likelihood` (opt-in, lm-eval-harness
  `multiple_choice` style): restricts the answer-slot argmax to the candidate label
  tokens in `ds.label.names` rather than the full vocabulary.
- Documentation site (Sphinx + Furo, API reference generated from source),
  published on Read the Docs.

### Changed
- **BREAKING** `LossEarlyStoppingCallback` is now `CustomEarlyStoppingCallback`.
  The callback monitors any metric, not only loss, so the old name was misleading.
- **BREAKING** `env.py` and `setup.py` have been consolidated into `bootstrap.py`;
  both were small and conceptually overlapping, and `setup.py` collided with
  packaging tooling. `PROJECT_ROOT_PATH` is relaxed to `Path | None` so `Env()` can
  be imported before `init()` runs.
- `get_preprocess_logits_for_metrics` moved from `training/callbacks.py` to
  `evals/eval.py`, next to `get_compute_metrics` — the two share the
  prediction-shape contract, and the hook was never a `TrainerCallback`.
- Token-budget calibration merged into the `training/batching` module.
- Sequence lengths now fall back to `input_ids` in the HuggingFace style instead of
  requiring an explicit length-column override.
- Token-budget `HEADROOM` lowered from 0.85 to 0.80; the 15% margin was consumed by
  cumulative memory fragmentation during long evaluation sweeps.
- `calibrate_token_budget`'s `tolerance` parameter is deprecated and ignored — the
  search now always runs to convergence.

### Fixed
- `calibrate_token_budget` ended its binary search once the window shrank below
  `tolerance` and then probed only the window's top edge, so any true fitting batch
  size in the coarse-halving dead zone was skipped and the budget collapsed to a
  single sequence. Every heavily-tokenised language was silently batched at ~1, and
  those whose shortest sequence fell below the floor were skipped entirely.
- Predictions were zipped against `ds_split` in dataset order while
  `TokenBudgetBatchSampler` yields globally length-sorted samples, so under
  `eval_per_task` grouping every per-task metric was attributed to the wrong task.
  Batch samplers now expose an `order` permutation and alignment is applied
  sampler-agnostically; the `SequentialSampler` path is unaffected.
- Calibration probe correctness: binary-search over sorted lengths for a
  deterministic, shape-correct probe; post-loop `hi` probe so datasets smaller than
  the tolerance get the right budget; probe with `labels=` so the cross-entropy
  logits cost is included; probe `hard_cap` directly and guard misconfigured
  `start`/`hard_cap`.
- `NormalizePromptEncoderEmbeddings` hooked `on_optimizer_step`, which does not
  receive `model` under transformers 4.48, making the callback a no-op.
- The length column was stripped by `_remove_unused_columns` before the token-budget
  and length-grouped samplers could read it.
- Boolean values were accepted where a token budget was expected.
- Single-process runs on multi-GPU partitions inherited SLURM environment variables
  that pushed accelerate into `MULTI_GPU` mode and aborted during NCCL init;
  `init()` now strips them when `WORLD_SIZE=1`.
- `pad_to_multiple_of` is sourced from the data collator instead of a duplicated
  parameter.

## [0.1.0] - 2026-04-30

### Added
- Initial public release of the micm-nlp toolkit.
- Config-driven pipeline (tokenization → preprocessing → training → evaluation).
- Example: HuggingFace Hub dataset loading + decoder-only tokenization (`examples/preprocess_dataset.py` + `examples/configs/xsc_preprocess.yml`).
- Example: PEFT fine-tuning + evaluation using Cross-Prompt Encoder (XPE) on a decoder-only LM (`examples/run_model.py` + `examples/configs/xsc_finetune.yml`).
- WandB experiment tracking integration.
