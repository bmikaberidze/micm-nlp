# Experiment Orchestration

*one group config, many unit runs*

> How do I run many variations, and collect their results together?

```{include} ../../README.md
:start-after: <!-- start:groups -->
:end-before: <!-- end:groups -->
```

## What one run leaves behind

Every run — in a group or not — writes one directory, and the trainer is its only writer. Nothing is appended to, nothing overwritten.

```
artefacts/runs/groups/{group}/{time_id}_{name}/
├── config.yml                       # the config as the framework resolved it
├── info.json                        # what the run did (see below)
├── eval_validation_before_train.csv
├── eval_validation_after_train.csv  # …from the best checkpoint
├── test_after_train.csv             # one row per metric group
├── predictions_after_train.csv      # one row per sample, always written
├── model -> …                       # symlink to the checkpoint
└── wandb -> …                       # symlink to the wandb run
```

A run that never trains — `mode: test` or `evaluate` — has one pass per event, so its files carry no stage suffix: `test.csv`, `predictions.csv`.

`info.json` holds what the config cannot: `started` / `finished`, every `SLURM*` variable, host, Python version, `CUDA_VISIBLE_DEVICES`, the versions of the packages that decide numerics, the wandb id / url / dir, the resolved seed and `metric_for_best_model`, and `paths.best_checkpoint`.  
The rule behind the split: **the config is read-only for everything that consumes it** — a fact about the run goes to `info.json`, never back into the config.

## One event, one file

Each `evaluate()` or `predict()` call is an *event*, and each event writes its file once, from the output HuggingFace returned.

Metric rows are the `compute_metrics` dict verbatim, split by metric group, with the framework's identity columns stamped on every row: `group`, `name`, `index`, `config`, `seed`, `time_id`, `uuid4`, plus any other scalar key on the entry.  
Predictions carry the same preprocessing the metric saw, in the dataloader's emit order, so every metric is recomputable from the file.  
There is no row-count column — the count *is* the predictions file's length.

The same rows reach `output.results` and `output.predictions` in memory, keyed by event name, so a runner reads back what it wrote without parsing the files.

```{note}
Reading a whole group back into one table — pooling replicates, comparing methods — is not yet part of the package. Today that lives in the consumer repository. See the roadmap.
```

## Bringing your own science

The default runner is `micm_nlp.pipeline:run`. Replace it with `--runner` when a run needs to do something the pipeline does not: evaluate on many target languages, swap a dataset between stages, train two models and compare them.

A runner is a callable `run(config, ctx)`. The first argument is the resolved config; the second is a {py:class}`~micm_nlp.group.RunContext` carrying everything the framework knows that the config does not — the entry as written, the group and run name, the index, the output directory, unknown CLI flags as `extras` (`--source-group joshi5` arrives as `{'source_group': 'joshi5'}`), and `test_config` when the entry declared a `separate_test`.

`RunContext` is a frozen dataclass rather than keyword arguments, so adding a field never breaks a runner written against an older version.
