# Quickstart

## Run a whole pipeline

```{include} ../../README.md
:start-after: <!-- start:quickstart -->
:end-before: <!-- end:quickstart -->
```

## Drive the stages yourself

This is `run()`'s own body — it calls the core classes directly, so a consumer
intervening between two stages (swap a dataset, concatenate languages, reuse one
tokenizer) copies it and changes one line:

```python
from micm_nlp import CONFIG
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
from micm_nlp.datasets.dataset import DATASET
from micm_nlp.models.model import MODEL
from micm_nlp.training.runner import TRAINER

config = CONFIG.from_yaml('path/to/config.yml')
tokenizer = load_tokenizer(config)
dataset = DATASET(config)
dataset.preprocess(tokenizer)
model = MODEL(config)
trainer = TRAINER(model, dataset, tokenizer)
trainer.run()
output = trainer.output
```

A test pins this listing against `pipeline.run`'s source, so the two cannot drift.

## Run many of them

One YAML describes several runs over your unit configs, and the CLI runs them — the
whole entry, or the one entry a SLURM array task selects:

```bash
python -m micm_nlp run-group --group-config config/groups/lr_sweep.yml
sbatch --array=0-1 my_wrapper.sh "python -m micm_nlp run-group --group-config config/groups/lr_sweep.yml"
```

That is experiment orchestration. See {doc}`groups` for the group config format,
what each run writes, and how to supply your own runner.

## Worked examples

```{include} ../../README.md
:start-after: <!-- start:examples -->
:end-before: <!-- end:examples -->
```
