# micm-nlp

```{include} ../../README.md
:start-after: <!-- start:tagline -->
:end-before: <!-- end:tagline -->
```

```bash
pip install micm-nlp
```

```{include} ../../README.md
:start-after: <!-- start:about -->
:end-before: <!-- end:about -->
```

## What micm-nlp offers

```{include} ../../README.md
:start-after: <!-- start:contributions -->
:end-before: <!-- end:contributions -->
```

{doc}`quickstart` runs one. {doc}`config` is the unit config in full, {doc}`groups` the group config, and {doc}`features` what you can do inside them.

The package owns how a run is assembled and where it lands; study-specific meaning — what a language group is, what a result table should look like — stays in the repositories that import it.

## Scope

| Architecture | Supported | Covered by a shipped example |
|---|---|---|
| Decoder-only (BLOOM, Aya) | yes | yes |
| Encoder-only (BERT, XLM-R, mDeBERTa) | yes | planned |
| Encoder-decoder (T5) | yes | planned |

PEFT methods: LoRA, Prefix Tuning, P-Tuning / soft prompt tuning, and the Cross-Prompt Encoder — the shipped examples demonstrate the last only.  
Configuration, datasets, models, PEFT dispatch, training and evaluation carry no assumptions about any particular study.

Published work is not partitioned off into a "research" corner. It sits in the package where it belongs, and each module's page cites the paper behind it — the Cross-Prompt Encoder is {doc}`models.xpe </autoapi/micm_nlp/models/xpe/index>`, the Georgian tokenization work is {doc}`tokenizers.architectures </autoapi/micm_nlp/tokenizers/architectures/index>` and {doc}`tokenizers.ka_sen_tok </autoapi/micm_nlp/tokenizers/ka_sen_tok/index>`.  
None of it is required to use the rest.

## Links

| | |
|---|---|
| **PyPI** | [pypi.org/project/micm-nlp](https://pypi.org/project/micm-nlp/) |
| **Source** | [github.com/bmikaberidze/micm-nlp](https://github.com/bmikaberidze/micm-nlp) |
| **Issue tracker** | [github.com/bmikaberidze/micm-nlp/issues](https://github.com/bmikaberidze/micm-nlp/issues) |
| **Releases** | [github.com/bmikaberidze/micm-nlp/releases](https://github.com/bmikaberidze/micm-nlp/releases) |
| **Changelog** | [this site](changelog.md) |
| **XPE paper** | [ACL Anthology](https://aclanthology.org/2025.findings-ijcnlp.144/) · [arXiv:2508.10352](https://arxiv.org/abs/2508.10352) |
| **Tokenization paper** | [ACL Anthology](https://aclanthology.org/2024.icnlsp-1.22/) |
| **MICM** | [micm.edu.ge](https://micm.edu.ge/) |
| **Contact** | beso.mikaberidze@gmail.com |

## Provenance

```{include} ../../README.md
:start-after: <!-- start:acknowledgements -->
:end-before: <!-- end:acknowledgements -->
```

The package was formerly named `nlpka`; that name survives only in the archived
repository behind the IJCNLP–AACL paper.

## Citation

```{include} ../../README.md
:start-after: <!-- start:citation -->
:end-before: <!-- end:citation -->
```

```{toctree}
:caption: Getting started
:hidden:

Home <self>
install
quickstart
```

```{toctree}
:caption: What micm-nlp offers
:hidden:

config
groups
features
```

```{toctree}
:caption: API reference
:hidden:

Core <api/core>
autoapi/micm_nlp/tokenizers/index
autoapi/micm_nlp/datasets/index
autoapi/micm_nlp/models/index
autoapi/micm_nlp/training/index
autoapi/micm_nlp/evals/index
```

```{toctree}
:caption: Meta
:hidden:

contributing
changelog
PyPI <https://pypi.org/project/micm-nlp/>
GitHub <https://github.com/bmikaberidze/micm-nlp>
MICM <https://micm.edu.ge/>
```
