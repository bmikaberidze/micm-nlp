# Group run · many configs

A unit config describes one run. A **group config** describes many runs over unit
configs, and is the second half of the framework: the same pipeline, repeated across
the axes an experiment varies — seeds, hyperparameters, methods.

```{include} ../../README.md
:start-after: <!-- start:groups -->
:end-before: <!-- end:groups -->
```

## What one run leaves behind

Every run — solo or grouped — writes one directory, and the trainer is its only
writer. Nothing is appended to, and nothing is overwritten.

```
artefacts/runs/{architecture}/{group}/{time_id}_{name}/
├── config.yml                          # the config as the framework resolved it
├── run.json                            # what the run did (see below)
├── eval_validation_before_train.csv    # one row per metric group
├── eval_validation_after_train.csv     # …from the best checkpoint
├── test_after_train.csv
├── predictions_after_train.csv         # one row per sample, always written
├── model -> …                          # symlink to the checkpoint
└── wandb -> …                          # symlink to the wandb run
```

A run that never trains — `mode: test` or `evaluate` — has one pass per event, so its
files carry no stage suffix: `test.csv`, `predictions.csv`.

`run.json` holds what the config cannot: `started` / `finished`, every `SLURM*`
variable, host, Python version, `CUDA_VISIBLE_DEVICES`, the versions of the packages
that decide numerics, the wandb id / url / dir, the resolved seed and
`metric_for_best_model`, and `paths.best_checkpoint`. The rule behind the split is
that **the config is read-only for everything that consumes it** — a fact about the
run goes to `run.json`, never back into the config.

## One event, one file

Each `evaluate()` or `predict()` call is an *event*, and each event writes its own
file once, from the output HuggingFace returned. Metric rows are the `compute_metrics`
dict verbatim, split by metric group, with the framework's identity columns stamped on
every row: `group`, `name`, `index`, `config`, `seed`, `time_id`, `uuid4`, plus any
other scalar key you put on the entry.

Predictions are written with the same preprocessing the metric saw, in the
dataloader's emit order, so every metric is recomputable from the file. There is no
row count column — the count *is* the predictions file's length.

```{note}
Reading a whole group back into one table — pooling replicates, comparing methods — is
not yet part of the package. Today that lives in the consumer repository. See the
roadmap.
```

## Bringing your own science

The default runner is `micm_nlp.pipeline:run`. Replace it with `--runner` when a run
needs to do something the pipeline does not: evaluate on many target languages, swap a
dataset between stages, train two models and compare them.

A runner is a callable `run(config, ctx)`. The first argument is the resolved config;
the second is a {py:class}`~micm_nlp.group.RunContext` carrying everything the
framework knows that the config does not — the entry as written, the group and run
name, the index, the output directory, unknown CLI flags as `extras`, and
`test_config` when the entry declared a `separate_test`.

Because `RunContext` is a frozen dataclass rather than keyword arguments, adding a
field to it never breaks a runner written against an older version.
