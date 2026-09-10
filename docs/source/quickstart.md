# Quickstart

## Run a whole pipeline

```{include} ../../README.md
:start-after: <!-- start:quickstart -->
:end-before: <!-- end:quickstart -->
```

## Drive the stages yourself

```{include} ../../README.md
:start-after: <!-- start:stages -->
:end-before: <!-- end:stages -->
```

## Run many of them

One YAML describes several runs over your unit configs, and the CLI runs them — the
whole entry, or the one entry a SLURM array task selects:

```bash
python -m micm_nlp run-group --group-config config/groups/lr_sweep.yml
sbatch --array=0-1 my_wrapper.sh "python -m micm_nlp run-group --group-config config/groups/lr_sweep.yml"
```

That is the second half of the framework. See {doc}`groups` for the group config
format, what each run writes, and how to supply your own runner.

## Worked examples

```{include} ../../README.md
:start-after: <!-- start:examples -->
:end-before: <!-- end:examples -->
```

## Supported architectures

```{include} ../../README.md
:start-after: <!-- start:architectures -->
:end-before: <!-- end:architectures -->
```
