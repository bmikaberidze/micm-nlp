# Training

| Module | Role |
|---|---|
| {doc}`runner </autoapi/micm_nlp/training/runner/index>` | `TRAINER` — builds the HuggingFace `Trainer`, callbacks, collator |
| {doc}`trainers </autoapi/micm_nlp/training/trainers/index>` | `CustomTrainerMixin`, `RandomTaskExclusionBatchSampler` |
| {doc}`callbacks </autoapi/micm_nlp/training/callbacks/index>` | `CustomEarlyStoppingCallback`, `ParamNormLogger`, … |
| {doc}`batching </autoapi/micm_nlp/training/batching/index>` | `TokenBudgetBatchSampler` and `calibrate_token_budget` |
| {doc}`data_collators </autoapi/micm_nlp/training/data_collators/index>` | Custom collators |
| {doc}`logits_processors </autoapi/micm_nlp/training/logits_processors/index>` | Generation-time logits processors |

```{toctree}
:hidden:

/autoapi/micm_nlp/training/runner/index
/autoapi/micm_nlp/training/trainers/index
/autoapi/micm_nlp/training/callbacks/index
/autoapi/micm_nlp/training/batching/index
/autoapi/micm_nlp/training/data_collators/index
/autoapi/micm_nlp/training/logits_processors/index
```
