# micm-nlp

[![PyPI](https://img.shields.io/pypi/v/micm-nlp.svg)](https://pypi.org/project/micm-nlp/)
[![Python](https://img.shields.io/pypi/pyversions/micm-nlp.svg)](https://pypi.org/project/micm-nlp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/micm-nlp/badge/?version=latest)](https://micm-nlp.readthedocs.io/en/latest/)

<!-- start:tagline -->
NLP research toolkit for tokenization, pretraining, fine-tuning, and PEFT across encoder-only, decoder-only, and encoder-decoder architectures. Built on top of HuggingFace `transformers`, `peft`, and `datasets`.
<!-- end:tagline -->

<!--
The blocks between the start/end markers below are pulled into the documentation
site with MyST {include} directives. README is the canonical copy; docs/source/
never holds a second one. Moving or renaming a marker breaks a docs page --
grep for the marker name in docs/source/ before editing.
-->

## About

<!-- start:about -->
`micm-nlp` is a config-driven research toolkit for multilingual NLP work. It wraps the HuggingFace stack with a small set of high-level building blocks — `CONFIG`, `TOKENIZER`, `DATASET`, `MODEL`, and a unified `TRAINER` — that compose into reproducible training, fine-tuning, and evaluation pipelines.

It has backed two peer-reviewed publications: *Cross-Prompt Encoder for Low-Performing Languages* (Findings of IJCNLP–AACL 2025; [ACL Anthology](https://aclanthology.org/2025.findings-ijcnlp.144/)) and *A Comparison of Different Tokenization Methods for the Georgian Language* (ICNLSP 2024; [ACL Anthology](https://aclanthology.org/2024.icnlsp-1.22/)).

The package currently ships **two examples** that exercise a single use case end-to-end: preprocessing and decoder-only PEFT fine-tuning (XPE) on an FTP-reframed multilingual dataset hosted on the HuggingFace Hub. The toolkit's underlying surface is broader than these two examples demonstrate.

Additional examples covering encoder-only text classification, encoder-decoder seq2seq, and MLM pretraining will land in subsequent releases. Contributions and issue reports are welcome.
<!-- end:about -->

📖 **Full documentation: [micm-nlp.readthedocs.io](https://micm-nlp.readthedocs.io/)**

## Install

<!-- start:install-requires -->
Requires **Python 3.10 or newer**.
<!-- end:install-requires -->

From PyPI:

<!-- start:install-pypi -->
```bash
pip install micm-nlp
```
<!-- end:install-pypi -->

From source (development):

<!-- start:install-source -->
```bash
git clone https://github.com/bmikaberidze/micm-nlp.git
cd micm-nlp
pip install -e ".[dev]"
```

The `dev` extra adds `pytest` and `ruff`.
<!-- end:install-source -->

Docker (recommended for reproducibility on GPU machines):

<!-- start:install-docker -->
```bash
docker build -t micm-nlp .
docker run --gpus all -it --rm -v $(pwd):/app -w /app micm-nlp bash
```
<!-- end:install-docker -->

Credentials and the workspace root come from a `.env` file:

<!-- start:install-env -->
```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT_PATH` | Workspace directory; `artefacts/` (datasets, models, evals, wandb) is created under it. Used as the fallback when `init()` is called without `root_path`. |
| `WANDB_API_KEY` | Required only if `training_args.args.report_to` includes `wandb`. |
| `HF_TOKEN` | Required only for gated HuggingFace models or datasets. |
<!-- end:install-env -->

## Quickstart

<!-- start:quickstart -->
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

`init()` resolves the workspace root from its `root_path` argument, falling back to `PROJECT_ROOT_PATH` in the environment. Call it once before any pipeline call so `artefacts/` lands in the right place. It is **not** triggered on import.

`run(config)` chains: load tokenizer → load and preprocess dataset → load model (with PEFT if configured) → train → evaluate. Every stage is configured by YAML; no plumbing code required.
<!-- end:quickstart -->

## Package tour

```
micm_nlp/
├── pipeline.py     # Top-level wiring: load_dataset, preprocess_dataset, load_model, run
├── config.py       # CONFIG.from_yaml; resolves nested namespaces
├── tokenizers/     # Tokenizer factory, custom tokenizer classes, Georgian sentence splitter
├── datasets/       # DATASET class — local + HF Hub + HF saved + CSV/TXT/JSON
├── models/         # MODEL wrapper, PEFT dispatch, Cross-Prompt Encoder
├── training/       # TRAINER — wraps HF Trainer with custom callbacks + WandB
└── evals/          # Metrics, confusion matrices, plotting helpers
```

<!-- start:stages -->
The same flow, unwrapped — useful when a consumer repository needs to intervene between stages (swap a dataset, concatenate languages, reuse one tokenizer):

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
<!-- end:stages -->

## Examples

<!-- start:examples -->
Two runnable examples ship with the repository. Together they cover one use case end to end — preprocessing and decoder-only PEFT fine-tuning on an FTP-reframed multilingual dataset from the HuggingFace Hub.

| Script | Config | What it does |
|---|---|---|
| `examples/preprocess_dataset.py` | `examples/configs/xsc_preprocess.yml` | Loads FTP-reframed XStoryCloze (English) from the Hub, tokenizes it for BLOOM-560M, saves the result locally. |
| `examples/run_model.py` | `examples/configs/xsc_finetune.yml` | Fine-tunes BLOOM-560M with Cross-Prompt Encoder PEFT on the Arabic split, then evaluates. |

The toolkit's surface is broader than these two demonstrate. Examples for encoder-only text classification, encoder-decoder seq2seq and MLM pretraining are planned.
<!-- end:examples -->

## Supported architectures

<!-- start:architectures -->
| Architecture | Toolkit support | Covered by a shipped example |
|---|---|---|
| Decoder-only (BLOOM, Aya) | yes | yes |
| Encoder-only (BERT, XLM-R, mDeBERTa) | yes | planned |
| Encoder-decoder (T5) | yes | planned |

PEFT methods: LoRA, Prefix Tuning, P-Tuning / soft prompt tuning, and the Cross-Prompt Encoder. The shipped examples demonstrate the Cross-Prompt Encoder only.
<!-- end:architectures -->

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

<!-- start:acknowledgements -->
`micm-nlp` was developed at the Muskhelishvili Institute of Computational Mathematics (MICM, Georgian Technical University), in close research collaboration with Teimuraz Saghinadze (MICM), Simon Ostermann (DFKI / CERTAIN), and Philipp Müller (Max Planck Institute for Intelligent Systems), whose joint work on the Cross-Prompt Encoder (XPE) drove much of the toolkit's design and validation.

This work was partially supported by the European Union under Horizon Europe project "GAIN" (GA #101078950) and by the German Federal Ministry of Research, Technology and Space (BMFTR) as part of the project TRAILS (01IW24005).
<!-- end:acknowledgements -->

## Citation

<!-- start:citation -->
If you use `micm-nlp` in your research, please cite the package and (if relevant to your work) the XPE paper that drove its design:

```bibtex
@software{micm_nlp,
  author = {Mikaberidze, Beso},
  title = {micm-nlp: NLP research toolkit for multilingual fine-tuning and PEFT},
  url = {https://github.com/bmikaberidze/micm-nlp},
  version = {0.3.0},
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
<!-- end:citation -->

## Contact

`beso.mikaberidze@gmail.com`
