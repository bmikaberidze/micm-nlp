# Changelog

All notable changes to micm-nlp will be documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-30

### Added
- Initial public release of the micm-nlp toolkit.
- Config-driven pipeline (tokenization → preprocessing → training → evaluation).
- Example: HuggingFace Hub dataset loading + decoder-only tokenization (`examples/preprocess_dataset.py` + `examples/configs/xsc_preprocess.yml`).
- Example: PEFT fine-tuning + evaluation using Cross-Prompt Encoder (XPE) on a decoder-only LM (`examples/run_model.py` + `examples/configs/xsc_finetune.yml`).
- WandB experiment tracking integration.
