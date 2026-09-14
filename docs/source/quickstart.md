# Quickstart

## Intro

```{include} ../../README.md
:start-after: <!-- start:quickstart -->
:end-before: <!-- end:quickstart -->
```

## Drive the stages yourself

The chain, unwrapped — this is what `run()` does. Copy it to interact between stages, e.g. add your own dataset processing after `dataset.preprocess(tokenizer)`.

```python
from micm_nlp import CONFIG, init, example
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
from micm_nlp.datasets.dataset import DATASET
from micm_nlp.models.model import MODEL
from micm_nlp.training.runner import TRAINER

# Workspace
init('/path/to/your/workspace', pretty_output=True)

# Config
config = CONFIG.from_yaml(example('xsc_finetune.yml'))

# Tokenizer
tokenizer = load_tokenizer(config)

# Dataset
dataset = DATASET(config)
dataset.preprocess(tokenizer)

# Model
model = MODEL(config)

# Trainer
trainer = TRAINER(model, dataset, tokenizer)
output = trainer.run()
```

## Examples

```{include} ../../README.md
:start-after: <!-- start:examples -->
:end-before: <!-- end:examples -->
```
