# Models

| Module | Role |
|---|---|
| {doc}`model </autoapi/micm_nlp/models/model/index>` | `MODEL` — `from_pretrained` plus task-derived kwargs |
| {doc}`peft </autoapi/micm_nlp/models/peft/index>` | `PEFT` — dispatch to stock PEFT or the Cross-Prompt Encoder path |
| {doc}`architectures </autoapi/micm_nlp/models/architectures/index>` | `CustomT5ForConditionalGeneration` |
| {doc}`xpe </autoapi/micm_nlp/models/xpe/index>` | Cross-Prompt Encoder — seven modules |

`xpe` implements *Cross-Prompt Encoder for Low-Performing Languages*
([Findings of IJCNLP–AACL 2025](https://aclanthology.org/2025.findings-ijcnlp.144/),
[arXiv:2508.10352](https://arxiv.org/abs/2508.10352)). XPE, SPT and DUAL are **one
class**, `CrossPromptEncoder`, separated only by `encoder_ratio` — see
[the configuration reference](../config.md#peft). Anything gated on
`isinstance(pe, CrossPromptEncoder)` therefore fires for plain soft prompt tuning
as well.

```{toctree}
:hidden:

/autoapi/micm_nlp/models/model/index
/autoapi/micm_nlp/models/peft/index
/autoapi/micm_nlp/models/architectures/index
/autoapi/micm_nlp/models/xpe/index
```
