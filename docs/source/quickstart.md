# Quickstart

```{include} ../../README.md
:start-after: <!-- start:quickstart -->
:end-before: <!-- end:quickstart -->
```

## Drive the stages yourself

The chain, unwrapped — this is what run() does. Copy it and change the stage you need: swap a dataset, concatenate languages, reuse one tokenizer.

```python
from micm_nlp import CONFIG
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
from micm_nlp.datasets.dataset import DATASET
from micm_nlp.models.model import MODEL
from micm_nlp.training.runner import TRAINER

# Config
config = CONFIG.from_yaml('path/to/config.yml')

# Tokenizer
tokenizer = load_tokenizer(config)

# Dataset
dataset = DATASET(config)
dataset.preprocess(tokenizer)

# Model
model = MODEL(config)

# Trainer
trainer = TRAINER(model, dataset, tokenizer)
trainer.run()

# Output
output = trainer.output
```

## Worked examples

```{include} ../../README.md
:start-after: <!-- start:examples -->
:end-before: <!-- end:examples -->
```
