# micm-nlp

[![PyPI](https://img.shields.io/pypi/v/micm-nlp.svg)](https://pypi.org/project/micm-nlp/)
[![Python](https://img.shields.io/pypi/pyversions/micm-nlp.svg)](https://pypi.org/project/micm-nlp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NLP research toolkit for tokenization, pretraining, fine-tuning, and PEFT across encoder-only, decoder-only, and encoder-decoder architectures. Built on top of HuggingFace `transformers`, `peft`, and `datasets`.

## About

`micm-nlp` is a config-driven research toolkit for multilingual NLP work. It wraps the HuggingFace stack with a small set of high-level building blocks — `CONFIG`, `TOKENIZER`, `DATASET`, `MODEL`, and a unified `TRAINER` — that compose into reproducible training, fine-tuning, and evaluation pipelines. PEFT methods (LoRA, Prefix Tuning, P-Tuning / SPT, and Cross-Prompt Encoder / XPE) are first-class. Datasets load from local paths, local HuggingFace-saved snapshots, or directly from the HuggingFace Hub by repo ID. The toolkit was used in the *Cross-Prompt Encoder for Low-Performing Languages* paper (Findings of IJCNLP–AACL 2025; [arXiv:2508.10352](https://arxiv.org/abs/2508.10352)).

This v0.1.0 release ships **two examples** that exercise a single use case end-to-end: preprocessing and decoder-only PEFT fine-tuning (XPE) on an FTP-reframed multilingual dataset hosted on the HuggingFace Hub. The toolkit's underlying surface is broader than these two examples demonstrate.

Additional examples covering encoder-only text classification, encoder-decoder seq2seq, MLM pretraining, and other PEFT methods will land in subsequent releases. Contributions and issue reports are welcome.

## Install

From PyPI:

```bash
pip install micm-nlp
```

From source (development):

```bash
git clone https://github.com/bmikaberidze/micm-nlp.git
cd micm-nlp
pip install -e ".[dev]"
```

Docker (recommended for reproducibility on GPU machines):

```bash
docker build -t micm-nlp .
docker run --gpus all -it --rm -v $(pwd):/app -w /app micm-nlp bash
```

You will also want a `.env` file for HuggingFace and Weights & Biases credentials:

```bash
cp .env.example .env
# Then add WANDB_API_KEY and (if needed) HF_TOKEN.
```

## Quickstart

```python
import micm_nlp
from micm_nlp.config import CONFIG
from micm_nlp.pipeline import run

micm_nlp.init()  # Rich pretty-printing + traceback formatting

config = CONFIG.from_yaml("examples/configs/xsc_finetune.yml")
model, test_output = run(config)
```

`run(config)` chains: load tokenizer → load and preprocess dataset → load model (with PEFT if configured) → train → evaluate. Every stage is configured by YAML; no plumbing code required.

## Package tour

```
micm_nlp/
├── pipeline.py     # Top-level wiring: load_dataset, preprocess_dataset, load_model, run
├── config.py       # CONFIG.from_yaml; resolves nested namespaces
├── tokenizers/     # Tokenizer factory (XLM-R, BERT, BLOOM, T5, ...)
├── datasets/       # DATASET class — local + HF Hub + HF saved + CSV/TXT/JSON
├── models/         # MODEL wrapper, PEFT dispatch, XPE module, training callbacks
├── training/       # TRAINER — wraps HF Trainer with custom callbacks + WandB
└── evals/          # Metrics, confusion matrices, plotting helpers
```

The five-stage flow:

```python
from micm_nlp.config import CONFIG
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
from micm_nlp.datasets.dataset import DATASET
from micm_nlp.models.model import MODEL
from micm_nlp.training.runner import TRAINER

config = CONFIG.from_yaml("path/to/config.yml")
tokenizer = load_tokenizer(config)
dataset = DATASET(config)
dataset.preprocess(tokenizer)
model = MODEL(config)
trainer = TRAINER(model, dataset, tokenizer)
test_output = trainer.run()
```

## Examples

| Example | Config | Description |
|---|---|---|
| `examples/preprocess_dataset.py` | `examples/configs/xsc_preprocess.yml` | Loads FTP-reframed XStoryCloze (English split) directly from the HuggingFace Hub and tokenizes it for BLOOM-560M; saves tokenized output locally. |
| `examples/run_model.py` | `examples/configs/xsc_finetune.yml` | Fine-tunes BLOOM-560M with XPE PEFT on the Arabic split of FTP-reframed XStoryCloze, then evaluates. |

More examples — encoder-only text classification, encoder-decoder seq2seq, MLM pretraining, additional PEFT methods (LoRA, Prefix, P-Tuning) — are planned for subsequent releases.

## Supported architectures

| Architecture | Toolkit support | Demonstrated by example in v0.1.0 |
|---|---|---|
| Decoder-only (BLOOM, AYA) | ✅ | ✅ |
| Encoder-only (BERT, XLM-R) | ✅ | ⏳ planned |
| Encoder-decoder (T5) | ✅ | ⏳ planned |

PEFT methods supported by the toolkit: LoRA, Prefix Tuning, P-Tuning (SPT), Cross-Prompt Encoder (XPE). v0.1.0 examples demonstrate XPE only.

## Development

```bash
pip install -e ".[dev]"
ruff check src/
ruff format src/
pytest
```

## Contributing

Pull requests are welcome. For non-trivial changes, please open an issue first to discuss the proposed change. A `CONTRIBUTORS.md` will be added with the first external contribution.

## Acknowledgements

`micm-nlp` was developed at the Muskhelishvili Institute of Computational Mathematics (MICM, Georgian Technical University), in close research collaboration with Teimuraz Saghinadze (MICM), Simon Ostermann (DFKI / CERTAIN), and Philipp Müller (Max Planck Institute for Intelligent Systems), whose joint work on the Cross-Prompt Encoder (XPE) drove much of the toolkit's design and validation.

This work was partially supported by the European Union under Horizon Europe project "GAIN" (GA #101078950) and by the German Federal Ministry of Research, Technology and Space (BMFTR) as part of the project TRAILS (01IW24005).

## Citation

If you use `micm-nlp` in your research, please cite the package and (if relevant to your work) the XPE paper that drove its design:

```bibtex
@software{micm_nlp,
  author = {Mikaberidze, Beso},
  title = {micm-nlp: NLP research toolkit for multilingual fine-tuning and PEFT},
  url = {https://github.com/bmikaberidze/micm-nlp},
  version = {0.1.0},
  year = {2026},
}

@misc{mikaberidze2025crosspromptencoderlowperforminglanguages,
  title         = {Cross-Prompt Encoder for Low-Performing Languages},
  author        = {Beso Mikaberidze and Teimuraz Saghinadze and Simon Ostermann and Philipp Muller},
  year          = {2026},
  eprint        = {2508.10352},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2508.10352},
}
```

## Contact

`beso.mikaberidze@gmail.com`
