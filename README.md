# micm-nlp

[![PyPI](https://img.shields.io/pypi/v/micm-nlp.svg)](https://pypi.org/project/micm-nlp/)
[![Docs](https://readthedocs.org/projects/micm-nlp/badge/?version=latest)](https://micm-nlp.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/pypi/pyversions/micm-nlp.svg)](https://pypi.org/project/micm-nlp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


<!-- start:tagline -->
A research framework for NLP — the whole pipeline in a single YAML, run alone or in groups.
Builds on the HuggingFace stack and adds a layer of features of its own.
<!-- end:tagline -->

[micm-nlp.readthedocs.io](https://micm-nlp.readthedocs.io/) — full documentation 📚

<!--
The blocks between the start/end markers below are pulled into the documentation
site with MyST {include} directives. For anything that appears in BOTH places,
README is the canonical copy and docs/source/ never holds a second one -- moving
or renaming a marker breaks a docs page, so grep for the marker name in
docs/source/ before editing.

Detail that belongs only to the docs lives in docs/source/ directly, with no
marker here: install.md, config.md, groups.md, and features.md (an include of
the root FEATURES.md). Keeping it out of README is what stops README growing a
second copy of the documentation.
-->

## About

<!-- start:about -->
`micm-nlp` is developed at the Muskhelishvili Institute of Computational Mathematics (MICM), Georgian Technical University.

It has backed two peer-reviewed publications:
1. **Cross-Prompt Encoder for Low-Performing Languages**  
*Findings of IJCNLP–AACL 2025*; [ACL Anthology](https://aclanthology.org/2025.findings-ijcnlp.144/)  
Beso Mikaberidze, Temo Saghinadze, Simon Ostermann, Philipp Müller  

2. **A Comparison of Different Tokenization Methods for the Georgian Language**  
*ICNLSP 2024*; [ACL Anthology](https://aclanthology.org/2024.icnlsp-1.22/)  
Beso Mikaberidze, Teimuraz Saghinadze, Guram Mikaberidze, Raphael Kalandadze, Konstantine Pkhakadze, Josef van Genabith, Simon Ostermann, Lonneke van der Plas, Philipp Müller  
<!-- end:about -->

## What micm-nlp offers

<!-- start:contributions -->

| Contribution | In short | Answers |
|---|---|---|
| [Pipeline Unification](https://micm-nlp.readthedocs.io/en/latest/config.html) | unit&nbsp;config&nbsp;→&nbsp;unit&nbsp;run | *How do I describe a whole run in one place, and make it reproducible?* |
| [Experiment Orchestration](https://micm-nlp.readthedocs.io/en/latest/groups.html) | group&nbsp;config&nbsp;→&nbsp;many&nbsp;unit&nbsp;runs | *How do I run many variations, and collect their results together?* |
| [Features](https://micm-nlp.readthedocs.io/en/latest/features.html)  | ready-made&nbsp;functionality | *What can I do here that the HuggingFace stack does not already do?* |

<!-- end:contributions -->

## Install

```bash
pip install micm-nlp
```

Requires **Python 3.10 or newer**. For installing from source, with Docker, or setting up
the `.env` file see the [install docs](https://micm-nlp.readthedocs.io/en/latest/install.html).

## Quickstart

<!-- start:quickstart -->
```python
from micm_nlp import init, run, example

init('/path/to/your/workspace')
output = run(example('xsc_finetune.yml'))
```

- `init()` loads `.env` and sets your workspace root. A bare `init()` is enough when `PROJECT_ROOT_PATH` is set in `.env` or the environment.  
- `run()` takes a config, and chains: load tokenizer → load and preprocess dataset → load model, with PEFT if configured → train → evaluate.  
- `example()` resolves the path of a config shipped inside the package.  
- `output` holds info about run, raw predictions, metric results, and the dir path where everything is written.  
<!-- end:quickstart -->

## Pipeline unification

<!-- start:blocks -->
One YAML ***unit config*** describes a whole pipeline that is executed as a ***unit run*** and lands in dedicated output dir.

```yaml
mode:                 finetune # preprocess | train | finetune | evaluate | test
task:                 {category, name, metric_groups, preproc_rules}
tokenizer:            {source, name, args, ...}
ds:                   {category, dirs, name, type, splits, preproc_rules, ...}
model:                {architecture, pretrained: {cls, args, ...}}
peft:                 {peft_type, task_type, ...}
trainer:              {cls, args}
training_args:        {cls, args}
custom_training_args: {...}
data_collator:        {cls, args}
eval:                 {before_training, after_training, ...}
test:                 {run, zero_shot, ...}
generation_config:    {...}
cuda:                 {...}
env:                  {...}
```

- `model`, `tokenizer` and `ds` each can take the HuggingFace Hub slug, or be loaded from local disk.  
- `cls` keys are class names, resolved at runtime from `transformers` and this package — a new backbone or head needs no code.  
`cls` can also name your own class, once decorated with `@micm_plugin` — see [your own classes](https://micm-nlp.readthedocs.io/en/latest/config.html#your-own-classes).  
- `args` keys pass any extra keyword arguments verbatim to the `cls` constructor.  

```bash
python -m micm_nlp run \
    --config        configs/units/tune.lm.aya.ds.bebe.yml \
    --root-path     /path/to/your/workspace
```
- `--root-path` is only needed when `PROJECT_ROOT_PATH` is not set.
<!-- end:blocks -->

<!-- start:run-dir -->
Everything the run writes is reachable from one directory:

```
artefacts/runs/units/{model.name}/
├── config.yml                       # the config as resolved
├── info.json                        # environment, versions, resolved seed and metric, wandb, paths
├── eval_validation_before_train.csv # one row per metric group
├── eval_validation_after_train.csv  # same, from the best checkpoint
├── test_after_train.csv             # same, on the test split
├── predictions_after_train.csv      # one row per sample, always written
├── model -> …                       # symlink to the checkpoint
└── wandb -> …                       # symlink to the wandb run
```
Console output is kept only by an online wandb run, in `wandb/files/output.log`; an offline run writes no such file.
<!-- end:run-dir -->

[The unit run](https://micm-nlp.readthedocs.io/en/latest/config.html) is the full reference; [the stage-by-stage form](https://micm-nlp.readthedocs.io/en/latest/quickstart.html) is the same chain unwrapped.

## Experiment orchestration

<!-- start:groups -->
One YAML ***group config*** varies your unit configs into ***many unit runs***, each landing in its own dir under one group directory.

```yaml
# configs/groups/aya_lr_search.yml
configs:
  tune_aya_bebe: ../units/tune.lm.aya.ds.bebe.yml
runs:
  - {config: tune_aya_bebe, name: lr1e-4, seed: 1, overrides: {training_args.args.learning_rate: 1e-4}}
  - {config: tune_aya_bebe, name: lr5e-5, seed: 1, overrides: {training_args.args.learning_rate: 5e-5}}
```

In a group config each entry in `runs:` resolves one unit config from `configs:` and forms one unit run. The runs land side by side, in `artefacts/runs/groups/{group}/{time_id}_{runs[i].name}/`. Every result row is stamped with the run's identity, so the group's runs concatenate into one table.

```bash
python -m micm_nlp run-group \
    --group-config  configs/groups/aya_lr_search.yml \
    --run-index     0 \
    --root-path     /path/to/your/workspace \
    --runner        /path/to/your/custom/run.py
```
- `--run-index` selects one run; skip it to run everything.  
- `--root-path` is only needed when `PROJECT_ROOT_PATH` is not set.  
- `--runner` allows your own custom `run(config, ctx)`, or can be skipped — resolved config, run entry, and unknown CLI flags are passed. Append `:fn` to name a function other than `run`.

```bash
sbatch --array=0-1 my_wrapper.sh \
    "python -m micm_nlp run-group --group-config configs/groups/aya_lr_search.yml"
```
Under a SLURM array each task picks its own run by `SLURM_ARRAY_TASK_ID`.

<!-- end:groups -->

[The group run](https://micm-nlp.readthedocs.io/en/latest/groups.html) is the full reference: every reserved key, what each run writes, and how to supply a runner.

## Features

Thirty ***features*** on top of the HuggingFace stack, grouped by the area of the package they live in.

| Area | Some of what is there |
|---|---|
| **PEFT** | the Cross-Prompt Encoder — a published method, not a wrapper; soft prompts, the encoder, or any mix of the two, set by one ratio; save and load for adapters stock PEFT cannot serialise |
| **Training** | token-budget batching with an `'auto'` GPU probe; early stopping decoupled from model selection; per-parameter-group optimizer settings from YAML; collators HuggingFace lacks; closed-set generation |
| **Evaluation** | label-restricted likelihood; length-normalised log-likelihood accuracy; a declarative post-processing chain |
| **Data** | length statistics for choosing `max_length` from evidence; a declarative pre-processing chain; splitting, subsetting or concatenation based on config |
| **Tokenizers** | an architecture-aware factory; SentencePiece / WordPiece / byte-level BPE training; a Georgian sentence splitter with 885 abbreviations |

Every entry says what HuggingFace does on its own, so the difference is checkable rather than asserted, and links to the module behind it.

[The feature list](FEATURES.md) is all of them.

## Examples

<!-- start:examples -->
Three configs ship inside the package, covering one use case end to end — preprocessing, decoder-only PEFT fine-tuning on an FTP-reframed multilingual dataset from the HuggingFace Hub, and the same fine-tune as a group.

| Config | What it does |
|---|---|
| `xsc_preprocess.yml` | loads FTP-reframed XStoryCloze (English) from the Hub, tokenizes it for BLOOM-560M, saves the result locally |
| `xsc_finetune.yml` | fine-tunes BLOOM-560M with Cross-Prompt Encoder PEFT on the Arabic split, then evaluates |
| `groups/xsc_tune_across_seeds.yml` | the same fine-tune at two seeds, one run directory each |

```bash
micm-nlp init-examples
python -m micm_nlp run       --config       configs/examples/xsc_preprocess.yml
python -m micm_nlp run       --config       configs/examples/xsc_finetune.yml
python -m micm_nlp run-group --group-config configs/examples/groups/xsc_tune_across_seeds.yml
```

`init-examples` writes editable copies to `configs/examples/`; `example()` reaches the same files in place.  
The package's surface is broader than these three demonstrate. Examples for encoder-only text classification, encoder-decoder seq2seq and MLM pretraining are planned.

> [!NOTE]
> The preprocessing phase can run in every unit run, whatever the mode, but we expose `mode: preprocess` separately for tokenizing once and reusing across many runs.

<!-- end:examples -->

## Contributing

<!-- start:contributing -->
Pull requests are welcome. For non-trivial changes, please open an issue first to discuss the proposed change. A `CONTRIBUTORS.md` will be added with the first external contribution.
<!-- end:contributing -->

<!-- start:development -->
```bash
git clone https://github.com/bmikaberidze/micm-nlp.git
cd micm-nlp
pip install -e ".[dev]"   # the dev extra adds ruff and pytest

ruff check src/
ruff format src/
pytest
```

Where things live:

```
micm_nlp/
├── cli.py          # micm-nlp / python -m micm_nlp: run, run-group, init-examples
├── pipeline.py     # Top-level wiring: load_dataset, preprocess_dataset, load_model, run
├── group.py        # Group config → runs: entry selection, overrides, RunContext
├── config.py       # CONFIG.from_yaml; resolves nested namespaces
├── path.py         # The artefact tree: models, datasets, tokenizers, runs
├── tokenizers/     # Tokenizer factory, custom tokenizer classes, Georgian sentence splitter
├── datasets/       # DATASET class — local + HF Hub + HF saved + CSV/TXT/JSON
├── models/         # MODEL wrapper, PEFT dispatch, Cross-Prompt Encoder
├── training/       # TRAINER — wraps HF Trainer with custom callbacks + WandB;
│                   #   RunOutput — the run's output directory and info.json
└── evals/          # Metrics, confusion matrices, plotting; one result file per event
```
<!-- end:development -->

## Acknowledgements

<!-- start:acknowledgements -->
`micm-nlp` was developed at the Muskhelishvili Institute of Computational Mathematics (MICM, Georgian Technical University), in close research collaboration with Teimuraz Saghinadze (MICM), Simon Ostermann (DFKI / CERTAIN), and Philipp Müller (Max Planck Institute for Intelligent Systems), whose joint work on the Cross-Prompt Encoder (XPE) drove much of the framework's design and validation.

This work was partially supported by the European Union under Horizon Europe project "GAIN" (GA #101078950) and by the German Federal Ministry of Research, Technology and Space (BMFTR) as part of the project TRAILS (01IW24005).
<!-- end:acknowledgements -->

## Citation

<!-- start:citation -->
If you use `micm-nlp` in your research, please cite the package and (if relevant to your work) the XPE paper that drove its design:

```bibtex
@software{micm_nlp,
  author        = {Mikaberidze, Beso},
  title         = {micm-nlp: a research framework for {NLP} built on {HuggingFace}},
  organization  = {Muskhelishvili Institute of Computational Mathematics, Georgian Technical University},
  url           = {https://github.com/bmikaberidze/micm-nlp},
  version       = {0.4.0},
  year          = {2026},
}

@inproceedings{mikaberidze-etal-2025-cross,
  title        = {Cross-Prompt Encoder for Low-Performing Languages},
  author       = {Mikaberidze, Beso and Saghinadze, Temo and Ostermann, Simon and M{\"u}ller, Philipp},
  booktitle    = {Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  month        = dec,
  year         = {2025},
  address      = {Mumbai, India},
  publisher    = {The Asian Federation of Natural Language Processing and The Association for Computational Linguistics},
  url          = {https://aclanthology.org/2025.findings-ijcnlp.144/},
  doi          = {10.18653/v1/2025.findings-ijcnlp.144},
  pages        = {2380--2393},
}

```
<!-- end:citation -->

## Contact

Beso Mikaberidze · `beso.mikaberidze@gmail.com`
