# Changelog

All notable changes to micm-nlp will be documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-19

Runs on transformers 5 and peft 0.21 (was 4.49 / 0.14). Adapters saved by 0.4.x load
unchanged; replaying a 0.4-era Aya adapter reproduces its stored per-language test
accuracies to within one item in 300 on a few languages (bf16 numerics).

### Changed
- Requires `transformers>=5.5,<6` and `peft==0.21.0`.
- Configs must use the transformers 5 `TrainingArguments`: `warmup_ratio` →
  `warmup_steps` (a value below 1 is a ratio of total steps, computed the same way),
  `group_by_length: true` → `train_sampling_strategy: group_by_length`, and no
  `save_safetensors`, `overwrite_output_dir` or `logging_dir`. In
  `model.pretrained.args`, `torch_dtype` is now `dtype`.
- transformers 5 loads a checkpoint in its own dtype (`"auto"`, e.g. bf16 for Aya) and
  defaults `optim` to `adamw_torch_fused`; transformers 4 loaded float32 and used
  `adamw_torch`. To keep earlier numbers, set `model.pretrained.args.dtype` and
  `training_args.args.optim` explicitly.
- XPE registers through `register_peft_method`, which peft now needs for its config,
  tuner and prefix mappings.
- Tokenized datasets are saved as `tokenized--{org}--{model}`, not `tokenized|…|…`:
  datasets ≥ 4 reads a cache path as a regex, where `|` is alternation.
- `requirements-lock.txt` records the transformers 5 environment; the previous record
  is `requirements-lock.pre-micm-nlp-0.4.txt`.

### Added
- `info.json` records `versions.micm_nlp_commit`: the commit of the package's own
  checkout, suffixed `-dirty` when its tree has uncommitted changes, and `None` for an
  installed copy. An editable install follows a working tree that moves between
  releases, so the version string alone does not identify the code a run used.
- Every metric group records `n`, the number of predictions it was scored on, so
  per-group results can be pooled by weight.

### Fixed
- XPE adapters for `CAUSAL_LM` saved no prompt-encoder weights under peft ≥ 0.17,
  which stopped wrapping a prompt encoder's parts in `modules_to_save`; the encoder's
  weights are now read and written directly. Loading raises if one is missing.
- A model with no length limit in its config (Bloom) crashed the collator setup;
  transformers 5 no longer backs `max_length` with the generation default of 20.
- `dataloader_num_workers > 0` crashed: transformers 5 gave `seed_worker` two more
  arguments.

## [0.4.1] - 2026-09-16

### Added
- `@micm_plugin` makes a class in your workspace selectable by name in a
  config's `cls`. On the first lookup the workspace is scanned and only files that
  declare a plugin are imported; resolve order is plugins → micm-nlp → `transformers`.

### Changed
- Trainer and data-collator names resolve in micm-nlp before `transformers`.

## [0.4.0] - 2026-09-14

### Added
- `python -m micm_nlp` reaches the CLI (`__main__.py`), so the commands work on
  `--target` installs that have no `bin/`.
- `run --config` and `run-group --group-config`: a group config names unit configs
  and lists runs over them (reserved keys `config`, `overrides`, `seed`, `name`,
  `separate_test`; every other scalar key becomes a result column, except the
  framework's own column names — `group`, `index`, `time_id`, `uuid4`,
  `metric_group`, `step` — which an entry may not use). The entry is
  picked by `SLURM_ARRAY_TASK_ID`, then `--run-index`, else every entry runs;
  `--seed` supplies a seed for entries that do not set their own, and `--root-path`
  gives the workspace root.
  `--runner module:attr` or `--runner path/to/script.py[:fn]` supplies the
  science (the attribute defaults to `run`); unknown flags reach it as
  `ctx.extras`. Example shipped as `groups/xsc_tune_across_seeds.yml`.
- Every run writes its resolved `config.yml`, `info.json`, one metrics file per
  evaluation event and always-on predictions into its run directory, through
  `training/run_output.py` (`RunOutput`) and `evals/results.py`
  (`save_metrics` / `save_predictions`). `info.json` carries `started`/`finished`,
  every `SLURM*` variable, host, Python, `CUDA_VISIBLE_DEVICES`, package
  versions and the wandb id/url/dir, and the directory also gets
  `model` / `wandb` symlinks to the checkpoint and the wandb run. An `output:`
  config block (`dir`, `config_file`, `prefix`, `columns`) redirects the
  directory, renames the saved config, sets a filename prefix and stamps static
  columns onto every row.
- A `separate_test` entry resolves a second config into the same directory as
  `test_config.yml` with `output.prefix` set to `separate_`, so its files sit
  beside the primary's rather than over them. The runner receives it as
  `ctx.test_config`, and its resolved values land under `separate_resolved` in
  `info.json`.

### Changed
- The run directory is `artefacts/runs/groups/{group}/{time_id}_{name}`;
  a run outside any group lands in `artefacts/runs/units/{name}`, keeping the
  generated model name. No segment above the group varies with the run, so a
  group that varies the backbone stays one directory.
  Re-dispatching one entry within the same second raises `FileExistsError`
  rather than merging into the existing directory. This replaces
  `evals/runs/{model name}`, which a test run typically left empty.
- **Breaking:** the shipped-config package `micm_nlp.example_configs` is now
  `micm_nlp.configs`, with no compatibility shim.
- **Breaking:** `get_compute_metrics` and `calc_confusion_matrix` take
  `output_dir` where they took `eval_path`. Keyword callers must be updated;
  positional callers are unaffected.
- `pipeline.run(config, ctx=None)` accepts and ignores a run context, so it is the
  default runner.
- **Breaking:** `pipeline.run()` returns the run's `RunOutput` instead of
  `(model, test_output)`, and `None` instead of `(None, None)` for a
  `mode: preprocess` config. One object whose shape is the run directory:
  `output.dir`, plus `output.results` and `output.predictions` holding the rows
  of every file written, keyed by event name (`test_after_train`,
  `eval_validation_before_train`, …) and in the same shape the CSV holds. The old
  pair named its two results `full_shot` / `zero_shot` — a second vocabulary for
  the same events — and dropped the four `_evaluate` results entirely, which were
  reachable only by reading the files back.
- **Breaking:** `TRAINER.run()` returns the same `RunOutput` — `output = trainer.run()` —
  instead of a `SimpleNamespace(full_shot=…, zero_shot=…)` of raw HuggingFace
  prediction outputs. The metrics those carried are in `output.results`, one row
  per metric group.
- **Breaking:** `micm_nlp.init()` takes the workspace root positionally —
  `init('/path/to/workspace', pretty_output=True)`. The dict and `MicmNlpConfig`
  forms still work, so `init({'root_path': ...})` is unaffected.
- `micm_nlp` re-exports `CONFIG` and `run`, so a script needs one import line.
  `run` resolves on first access (PEP 562) rather than at import, keeping
  `import micm_nlp` — and therefore `init()` — clear of torch.
- `pipeline.run()` accepts a path to a config's YAML as well as a `CONFIG`, so a
  script needs no separate load step.
- `pipeline.run()` calls the core classes directly instead of the single-stage
  functions beside it, so its body is the pipeline written out and a consumer
  intervening between two stages copies it and changes one line. The single-stage
  functions are unchanged and still exported. `tests/test_pipeline_stages.py`
  pins the docs' copy of the sequence against `run`'s source.
- **Breaking:** `ds.comes_with_splits` is now `ds.splits`. A config still using the
  old name raises rather than being kept as an unknown extra while `splits`
  silently takes its default.
- **Breaking:** `group.run_solo()` is `group.run_unit()`, and `RunContext.group` is
  `None` for such a run rather than the sentinel `_solo`. The vocabulary is unit
  config / unit run / group config throughout.
- **Breaking:** the run-info file is `info.json`, not `run.json` — it pairs with
  `output.info`, and in a directory where everything describes the run, "info" is
  what distinguishes it.
- `RunOutput.info` holds the run details as written, alongside `results` and
  `predictions`, so `output` answers for the whole run directory in memory.
- `micm-nlp run` / `run-group` take `--root-path`, the workspace root the Python
  API takes as `init()`'s first argument; `PROJECT_ROOT_PATH` stays the default.
  With neither, the command exits naming both — it never falls back to the working
  directory, which would scatter one experiment across as many trees as the
  directories it was launched from.
- `micm_nlp.example(name)` returns the path of a config shipped inside the package,
  so a script can run one without `init-examples` copying it out first.
- The CLI no longer accepts abbreviated options (`allow_abbrev=False`), so an
  unknown flag is forwarded to the runner instead of being matched to a prefix.
- Predictions are always written, `predictions_<stage>.csv` in the run
  directory, one row per sample after the metric's preprocessing. The old
  writer produced a real file only for token classification, an empty
  three-column one for text classification, and none at all otherwise.
- The config is read-only for the trainer — what the run resolved is in
  `info.json`.
- `eval_validation_after_train.csv` is written from the after-training
  evaluation of the best checkpoint, and its `step` column is the final
  training step, not the best checkpoint's own step (that is in HF's
  `trainer_state.json`, under the checkpoint directory); the best checkpoint's
  path is recorded as `info.json` → `paths.best_checkpoint`.
- Event files carry `before_train` / `after_train`, or no stage for a
  `test` / `evaluate` run, which has one pass per event.
- The dockerfile builds on `python:3.12-slim`; the CUDA runtime comes from
  torch's own `nvidia-*` wheels rather than a CUDA base image.

### Fixed
- `pipeline.run` stops after tokenising for a `mode: preprocess` config and
  returns `None`. It previously went on to build a model and a trainer, so the
  shipped `xsc_preprocess.yml` could not be run through it.

### Removed
- **Breaking:** `init-examples` writes to `configs/examples/` instead of
  `micm-nlp-examples/` — beside where real configs live, not a package-named
  directory at the repo root. A destination argument still overrides it.
- **Breaking:** the `examples/` scripts are removed. `run_model.py` was
  `micm-nlp run --config …` and `preprocess_dataset.py` was the same command on a
  `mode: preprocess` config, so both were a third path to what the CLI already does
  and `init-examples` already prints. The configs they ran still ship.
- `test.save_predictions` is removed — predictions are always written.
- **Breaking:** `MODEL.eval_path` and `MODEL.logs_path` are removed. The trainer
  owns the output directory (`RunOutput`) and derives the logging directory from
  it; `MODEL` keeps `path` (the checkpoint location) only.

### Documentation
- The README and the docs site are organised around the three contributions —
  Pipeline Unification (unit config → unit run), Experiment Orchestration (group
  config → many unit runs) and Features — each page opening with its tagline and
  the question it answers. A section shared by both lives once, in the README, and
  the site includes it.
- The runner contract is written out: `run(config: CONFIG, ctx: RunContext) ->
  RunOutput`, ending in `return trainer.run()`.
- The stage-by-stage listing in the quickstart runs as copied: it calls `init()`
  and loads the shipped example config.
- Notes use GitHub alert syntax (`> [!NOTE]`), which renders on GitHub and, through
  a local Sphinx extension, as note cards on the site.
- The site restyle: VS Code One Dark Pro Darker / GitHub Light palettes, code
  blocks coloured by role (calls, constants, keyword arguments, shell commands and
  flags) at build time, Inter and JetBrains Mono, and a readable "On this page"
  menu.

## [0.3.1] - 2026-09-04

### Fixed
- `micm_nlp.init()` with no arguments raised `TypeError`, though both the README and
  the Quickstart present it as the form to use when `PROJECT_ROOT_PATH` is already
  set. `config` is now optional.
- The two example scripts never called `init()`, so both died at the first pipeline
  stage with `RuntimeError: Call micm_nlp.path.set_root(...) first` — an error naming
  an API neither page documents. Both call it now.
- `micm-nlp init-examples` exited 1 when it skipped an existing file, which breaks
  running it twice from a script. Skipping is a normal outcome; it exits 0.
- `torch>=2.4` made a source install impossible on macOS-Intel, where 2.2.2 is the
  last available wheel. The floor is now 2.2.

### Changed
- The example configs set `WANDB_MODE: offline` in their `env:` block, so they log to
  `artefacts/wandb/` and run without a W&B account. Both were previously unrunnable
  without one, and the installation page described the key as optional.

### Added
- `micm-nlp init-examples` writes the example configurations into a directory you can
  edit (`micm-nlp-examples/` by default). The configs now ship inside the package, so
  the copy you get always matches the version installed — nothing is downloaded, which
  keeps it working offline and on a cluster node. An existing file is reported and left
  alone unless `--force` is given.

### Changed
- Dependency bounds instead of exact pins. `transformers` was pinned to `==4.49.0`,
  which both blocked co-installation with anything wanting a different version and
  was already violated by the project's own container (4.48.2); it is now
  `>=4.48,<4.50`. `torch` was **not declared at all** despite being imported
  directly, so it arrived transitively and a fresh install resolved 2.14 — it is now
  declared as `>=2.4,<3`. `numpy<3` and `datasets<6` gained upper bounds for the same
  reason. `peft==0.14.0` stays exact: the Cross-Prompt Encoder subclasses its
  internals.

### Added
- `requirements-lock.txt`, recording the container the published results were
  produced on. It is a record rather than an installable file — 76 of its entries
  are wheels baked into the NGC image, torch among them.

### Fixed
- The documented `docker build -t micm-nlp .` could never have worked. `pyproject.toml`
  declares `readme = "README.md"`, but the dockerfile copied only `pyproject.toml` and
  `src/`, so the build backend failed with
  `OSError: Readme file does not exist: README.md`.
- `tests/test_parity.py` compared snapshots by exact equality, including floats rounded
  to six decimals and `repr()` of a module tree. On torch 2.14 that failed six tests
  while every number that matters -- loss, logits, state-dict keys, parameter counts --
  was identical: a weight norm read `3.372187` instead of `3.372186`, and
  `LayerNorm.__repr__` gained a `bias=True` field. Floats now compare with a tolerance
  and the repr is recorded but not compared.

### Documentation
- The Quickstart's `CONFIG.from_yaml('examples/configs/xsc_finetune.yml')` raised
  `FileNotFoundError` for anyone who followed the `pip install micm-nlp` directly
  above it, because `examples/` ships in the sdist but not the wheel. The configs now
  ship in the package and the snippet points at `micm-nlp init-examples` output. The
  example *scripts* stay in the repository — each is four lines, both are reproduced
  in the Quickstart, and a runnable script tree does not belong inside an installed
  package.


## [0.3.0] - 2026-09-03

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

### Added
- Docstrings for every published class and function. 220 objects had none, so each
  API page opened with a module description and then listed bare signatures; the
  generated reference now documents the whole public surface.

### Fixed
- `utils.print_traceback()` imported `micm_nlp.setup`, a module that stopped
  existing in 0.2.0 when it was merged into `bootstrap`, so every call raised
  `ModuleNotFoundError`.
- `TokenizerTrainer.train()` returned silently for a `model.type` it cannot train.
  `TokTypeSE` has six members and only three have a trainer, so a valid config
  could finish a run reporting nothing wrong and leaving no tokenizer behind. It
  now raises `ValueError`.
- `CrossPromptEncoder`'s class docstring used a Markdown code fence, which
  reStructuredText cannot parse, and its example imported
  `micm_nlp.models.cross_prompt_encoder` — a path that no longer exists.
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

### Documentation
- The site's API reference now mirrors `src/micm_nlp/` rather than a hand-written
  taxonomy, and its prose comes from `README.md` through MyST `{include}`, so
  README is the single copy of every passage the two share.

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
