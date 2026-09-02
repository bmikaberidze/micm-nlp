# Changelog

All notable changes to micm-nlp will be documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking)
- The `tokenizers` subpackage is flat. Two module paths moved:
  - `micm_nlp.tokenizers.bert_byt5.BertByT5Tokenizer` and
    `micm_nlp.tokenizers.xlm_roberta.CustomXlmRoberta` are now both in
    `micm_nlp.tokenizers.architectures`, mirroring `micm_nlp.models.architectures`.
    Each module held one wrapper class and nothing else; the split bought no
    isolation.
  - `micm_nlp.tokenizers.lib.sent.ka_sen_tok.KaSenTok` is now
    `micm_nlp.tokenizers.ka_sen_tok.KaSenTok`, and its two data files moved from
    `lib/sent/data/` to `tokenizers/data/`. The `lib` and `lib.sent` packages held
    a docstring each and no code — three levels of nesting around one module.

  No shims: the old paths are gone. Nothing in this repo or in `xpe-exp` imported
  either class outside of tests and one lazy import, both updated. Update
  `tokenizer.cls` values in any consumer config that names them.


### Fixed
- `CrossPromptEncoderConfig.encoder_embedding_normalize` defaulted to `'unit'`
  (max_norm `1.0`). Because `_filtered_kwargs` strips `None`-valued kwargs at the
  factory boundary, a YAML `encoder_embedding_normalize: null` never reached the
  dataclass — so **every** saved `adapter_config.json` recorded `"unit"` regardless
  of what the run actually did. The defaults are now `None`/`None`, so a saved
  adapter config records what happened. Behaviour is unchanged: normalisation is
  driven by the callback, whose registration reads the top-level `peft` block.
  Adapter configs written before this change misreport the field — do not read a
  normalisation claim out of them.
- `CrossPromptEncoder.__init__` now validates the normalisation settings: an
  unknown mode raises, and `'clip'` without a `max_norm` raises rather than
  silently doing nothing (`Tensor.clamp(max=None)` is a no-op, which the new
  `None` default would otherwise have turned into a silent non-normalising clip).

## [0.2.1] - 2026-08-11

### Fixed
- `NormalizePromptEncoderEmbeddings` was never registered: the trainer read its
  settings from `task.peft`, but `peft` is a top-level config block, so the lookup
  always returned `None`. Registration is now additionally gated on
  `peft.encoder_embedding_normalize` being set — without it `normalize_embeddings()`
  is a no-op that would still log a `0.0` norm to W&B on every step of every XPE run.
  **Consequence for existing results:** any run that set `encoder_embedding_normalize`
  did not in fact normalise, and measured the unnormalised model.
- `DataCollatorTaskIDDecorator.__call__` opened with a leftover `print()` / `exit()`
  debug pair, which made the rest of the method dead code.
- `tokenize_sentences()` defaulted to `SentTokTypeSE.KA`, whose branch was commented
  out, so calling it without an explicit method raised `ValueError`. The default is
  now `SentTokTypeSE.NLTK`, and the `KA` branch works again.
- `micm_nlp.tokenizers.lib.sent.ka_sen_tok` could not be imported at all: it read its
  abbreviation lists from `micm_nlp.datasets.storage.collections.abbreviations`, a
  package that does not exist here. The two data files (885 Georgian abbreviations,
  379 abbreviation endings) now ship inside the package at
  `tokenizers/lib/sent/data/` and are loaded through `importlib.resources`, so they
  survive installation from a wheel. `nltk.download('punkt')` no longer runs at import
  time — the models are checked first and only fetched when genuinely missing.

### Changed
- Every module now carries a module-level docstring, so the generated API reference
  explains what each module is for instead of listing bare symbols. Three known
  defects are now documented where they live: the unregistered
  `NormalizePromptEncoderEmbeddings` callback, the leftover debug body in
  `DataCollatorTaskIDDecorator.__call__`, and `ka_sen_tok`'s missing abbreviation
  data (it cannot be imported as shipped).

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
- `micm_nlp.evals.metrics` had no `__init__.py`, making it an implicit namespace
  package unlike every other subpackage. It is now a regular package.

## [0.1.0] - 2026-04-30

### Added
- Initial public release of the micm-nlp toolkit.
- Config-driven pipeline (tokenization → preprocessing → training → evaluation).
- Example: HuggingFace Hub dataset loading + decoder-only tokenization (`examples/preprocess_dataset.py` + `examples/configs/xsc_preprocess.yml`).
- Example: PEFT fine-tuning + evaluation using Cross-Prompt Encoder (XPE) on a decoder-only LM (`examples/run_model.py` + `examples/configs/xsc_finetune.yml`).
- WandB experiment tracking integration.
