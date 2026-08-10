# Quickstart

## Run a whole pipeline

```python
import micm_nlp
from micm_nlp.config import CONFIG
from micm_nlp.pipeline import run

# Sets the workspace root (where artefacts/ goes) and, optionally,
# enables Rich pretty-printing and traceback formatting.
micm_nlp.init({'root_path': '/path/to/your/workspace', 'pretty_output': True})
# Or, if PROJECT_ROOT_PATH is set in .env or the environment:
# micm_nlp.init()

config = CONFIG.from_yaml('examples/configs/xsc_finetune.yml')
model, test_output = run(config)
```

`init()` resolves the workspace root from its `root_path` argument, falling back to
`PROJECT_ROOT_PATH` in the environment. Call it once before any pipeline call so
`artefacts/` lands in the right place. It is **not** triggered on import.

`run(config)` chains: load tokenizer → load and preprocess dataset → load model
(with PEFT if configured) → train → evaluate.

## Drive the stages yourself

The same flow, unwrapped — useful when a consumer repository needs to intervene
between stages (swap a dataset, concatenate languages, reuse one tokenizer):

```python
from micm_nlp.config import CONFIG
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
test_output = trainer.run()
```

## Worked examples

Two runnable examples ship with the repository. Together they cover one use case
end to end — preprocessing and decoder-only PEFT fine-tuning on an FTP-reframed
multilingual dataset from the HuggingFace Hub.

| Script | Config | What it does |
|---|---|---|
| `examples/preprocess_dataset.py` | `examples/configs/xsc_preprocess.yml` | Loads FTP-reframed XStoryCloze (English) from the Hub, tokenizes it for BLOOM-560M, saves the result locally. |
| `examples/run_model.py` | `examples/configs/xsc_finetune.yml` | Fine-tunes BLOOM-560M with Cross-Prompt Encoder PEFT on the Arabic split, then evaluates. |

The toolkit's surface is broader than these two demonstrate. Examples for
encoder-only text classification, encoder-decoder seq2seq and MLM pretraining are
planned.

## Supported architectures

| Architecture | Toolkit support | Covered by a shipped example |
|---|---|---|
| Decoder-only (BLOOM, Aya) | yes | yes |
| Encoder-only (BERT, XLM-R, mDeBERTa) | yes | planned |
| Encoder-decoder (T5) | yes | planned |

PEFT methods: LoRA, Prefix Tuning, P-Tuning / soft prompt tuning, and the
Cross-Prompt Encoder. The shipped examples demonstrate the Cross-Prompt Encoder only.
