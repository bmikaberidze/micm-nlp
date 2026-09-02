# Core

The top-level modules: the pipeline and everything it needs before a tokenizer,
dataset or model is involved. They carry no assumptions about any particular task
or study.

Ordered as the pipeline uses them — `bootstrap` resolves the workspace, `pipeline`
chains the stages, `config` validates the YAML that drives them, and `path`,
`enums` and `utils` are the shared vocabulary underneath.

```{toctree}
:hidden:

/autoapi/micm_nlp/bootstrap/index
/autoapi/micm_nlp/pipeline/index
/autoapi/micm_nlp/config/index
/autoapi/micm_nlp/path/index
/autoapi/micm_nlp/enums/index
/autoapi/micm_nlp/utils/index
```
