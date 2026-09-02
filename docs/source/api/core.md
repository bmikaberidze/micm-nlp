# Core

The config-driven pipeline and everything it needs. Task- and research-agnostic:
nothing here assumes any particular study.

| Module | Role |
|---|---|
| {doc}`config </autoapi/micm_nlp/config/index>` | `CONFIG.from_yaml` and the validated section models |
| {doc}`pipeline </autoapi/micm_nlp/pipeline/index>` | `run()` and the stage-by-stage wiring |
| {doc}`bootstrap </autoapi/micm_nlp/bootstrap/index>` | `.env` loading, `env`, `init()`, workspace root |
| {doc}`path </autoapi/micm_nlp/path/index>` | `artefacts_dir`, `models_dir`, `datasets_dir`, … |
| {doc}`enums </autoapi/micm_nlp/enums/index>` | Categorical choices (`ModelArchSE`, `TaskCatSE`, `TaskNameSE`, …) |
| {doc}`utils </autoapi/micm_nlp/utils/index>` | `resolve_cls`, timing, time ids, JSON/YAML/pickle I/O |

```{toctree}
:hidden:

/autoapi/micm_nlp/config/index
/autoapi/micm_nlp/pipeline/index
/autoapi/micm_nlp/bootstrap/index
/autoapi/micm_nlp/path/index
/autoapi/micm_nlp/enums/index
/autoapi/micm_nlp/utils/index
```
