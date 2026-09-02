"""``CONFIG`` — the YAML schema, validated.

``CONFIG.from_yaml`` loads a run description into typed sections: ``task``, ``peft``,
``model``, ``tokenizer``, ``ds``, ``eval``, ``test``, ``trainer``, ``training_args``,
``data_collator``, ``custom_training_args``, ``cuda`` and ``env``. Loading also
applies the ``env`` block to ``os.environ``.

Two design points shape everything here.

**Sections accept extra keys.** Every section inherits from ``_Flex``, which allows
extras, implements the mapping protocol so ``dict(obj)`` and ``**obj`` expose both
declared fields and extras, and recursively wraps nested dicts. YAML may therefore
carry keys the schema does not declare, and runtime code may attach computed
attributes (``uuid4``, ``param_size``). Note ``vars(obj)`` does **not** see extras —
pydantic keeps them in ``__pydantic_extra__``; use ``dict(obj)``.

**Class selection stays in YAML.** ``trainer.cls``, ``training_args.cls``,
``data_collator.cls`` and ``model.pretrained.cls`` are thin shells: the real schema
lives in HuggingFace, and the matching ``args`` block is splatted into the
constructor at runtime.

Importing this module also widens PyYAML's float resolver, so ``5e-5`` parses as a
float rather than a string — YAML 1.1 otherwise requires a decimal point.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from micm_nlp.enums import ModeSE

# PyYAML's SafeLoader follows YAML 1.1, whose float resolver requires a decimal
# point — so `5e-5` parses as a string, not a float. Widen the resolver once,
# here, so every config gets the same treatment without per-key coercion.
_SCI_FLOAT_RE = re.compile(
    r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN)
        )$""",
    re.X,
)
yaml.SafeLoader.add_implicit_resolver('tag:yaml.org,2002:float', _SCI_FLOAT_RE, list('-+0123456789.'))

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _Flex(BaseModel):
    """Base for all config sections. Allows extra fields so YAML configs
    can carry additional keys without breaking validation, and runtime
    code can attach computed attributes (uuid4, param_size, …).

    Implements the mapping protocol (``keys()`` / ``__getitem__``) so
    ``dict(obj)`` and ``**obj`` expose both declared fields and extras.

    Note: ``vars(obj)`` does NOT work as a replacement for
    ``vars(simple_namespace)`` — Pydantic stores extras in
    ``__pydantic_extra__``, not ``__dict__``. Use ``dict(obj)`` instead.
    """

    model_config = ConfigDict(extra='allow')

    def keys(self):
        """Declared fields plus extras -- what ``dict(obj)`` and ``**obj`` see."""
        return list(self.model_fields) + list(self.__pydantic_extra__ or {})

    def __getitem__(self, key: str):
        """Attribute access by key, so a section satisfies the mapping protocol."""
        return getattr(self, key)

    @model_validator(mode='after')
    def _wrap_nested_dicts(self) -> _Flex:
        extras = self.__pydantic_extra__
        if extras:
            for k, v in extras.items():
                extras[k] = _wrap_value(v)
        return self


def _wrap_value(v):
    if isinstance(v, dict):
        return _Flex(**v)
    if isinstance(v, list):
        return [_wrap_value(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Root CONFIG
# ---------------------------------------------------------------------------


class CONFIG(_Flex):
    """One run, fully described.

    Every section is optional except ``mode``, because the same schema covers
    training, evaluation, preprocessing and tokenizer training -- a preprocessing
    config has no ``training_args``, and validation only demands what the mode
    needs. See :meth:`from_yaml` for the entry point.
    """

    mode: ModeSE
    file_path: str | None = None
    task: TaskConfig | None = None
    peft: PeftConfig | None = None
    model: ModelConfig | None = None
    tokenizer: TokenizerConfig | None = None
    ds: DatasetConfig | None = None
    eval: EvalConfig | None = None
    test: TestConfig | None = None
    trainer: TrainerConfig | None = None
    training_args: TrainingArgsConfig | None = None
    data_collator: DataCollatorConfig | None = None
    custom_training_args: CustomTrainingArgsConfig | None = None
    cuda: CudaConfig | None = None
    env: dict[str, str | None] | None = None
    generation_config: _Flex | None = None

    # -- Convenience loaders ------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> CONFIG:
        """Load and validate a YAML file into a :class:`CONFIG`.

        Also records ``file_path`` and applies the ``env`` block to
        ``os.environ`` -- so loading a config has a side effect on the process.

        :param path: path to the YAML file.
        :returns: the validated config.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        config.file_path = str(path)
        config.apply_env_vars()
        return config

    # -- Env-var side-effect ------------------------------------------------

    def apply_env_vars(self) -> None:
        """Copy the ``env`` block into ``os.environ``, skipping null values.

        Called by :meth:`from_yaml`; exposed so a config built in code can do the
        same. Keys with a ``None`` value are left alone rather than cleared.
        """
        if self.env:
            for key, value in self.env.items():
                if value is not None:
                    os.environ[key] = value

    # -- Validation that was previously in CONFIG._validate -----------------

    @model_validator(mode='after')
    def _validate_config(self) -> CONFIG:
        self._validate_model()
        self._validate_labels()
        return self

    def _validate_model(self) -> None:
        if self.model is None:
            return
        if not self.model.architecture:
            raise ValueError('Model architecture must be set')
        if self.mode in (ModeSE.FINETUNE, ModeSE.TEST):
            pretrained = self.model.pretrained
            if not pretrained:
                raise ValueError('Pretrained model must be set')
            if not pretrained.source or (not pretrained.name and not pretrained.time_id):
                raise ValueError('Pretrained model source and (name or time_id) must be set')

    def _validate_labels(self) -> None:
        if self.ds is None or self.ds.label is None:
            return
        Y = getattr(self.ds, 'Y', None)
        postproc = getattr(self.task, 'preproc_rules', None) if self.task else None
        label_id_to_name = getattr(postproc, 'label_id_to_name', False) if postproc else False
        if Y and len(Y.names) != Y.number and (label_id_to_name or Y.name_to_id):
            raise ValueError('Configured label names and number mismatch')


# ---------------------------------------------------------------------------
# task.*
# ---------------------------------------------------------------------------


class PeftConfig(_Flex):
    """The ``peft`` block: which PEFT method, and how it is parameterized.

    A superset of the fields the supported methods need, so ``peft_type: LORA``
    and ``peft_type: XPE`` share one schema; unused fields stay ``None``. The
    ``encoder_*`` fields belong to the Cross-Prompt Encoder -- notably
    ``encoder_ratio``, which is what separates SPT (0), DUAL (between) and XPE (1).
    """

    peft_type: str | None = None
    task_type: str | None = None
    num_virtual_tokens: int | None = None
    encoder_reparameterization_type: str | None = None
    encoder_hidden_size: int | None = None
    encoder_num_layers: int | None = None
    encoder_dropout: float | None = None
    num_tasks: int | None = None
    encoder_input_size: int | None = None
    encoder_init_state_dict_path: str | None = None
    encoder_freeze: bool = False
    encoder_embedding_freeze: bool = False
    encoder_embedding_init_type: str = 'hf_default'
    encoder_embedding_normalize: str | None = None
    encoder_embedding_normalize_max_norm: float | None = None
    encoder_ratio: float | None = None


class PostprocConfig(_Flex):
    """``task.preproc_rules``: what happens to predictions before metrics see them.

    Each flag is a step -- flatten, drop padded positions, decode ids to text,
    map label ids to names, strip and lowercase, coerce to float or back to ids --
    applied in the order ``evals.eval`` runs them. ``label_restricted_likelihood``
    is the opt-in that scores only the candidate label tokens rather than the whole
    vocabulary.
    """

    flatten: bool = False
    filter_padded: bool = False
    label_id_to_name: bool = False
    filter_by_prefixes: bool | list[str] = False
    decode: bool = False
    label_name_strip_lower: bool = False
    label_name_to_float: bool = False
    label_name_to_id: bool = False
    verify_labels_match: bool = False
    calc_confusion_matrix: bool = False
    prediction_axis: int = -1
    # mcqa_ftp opt-in: restrict the answer-token argmax to the candidate label
    # tokens in ds.label.names (lm-eval-harness multiple_choice style) instead of
    # full-vocab argmax. Off keeps the default behaviour unchanged.
    label_restricted_likelihood: bool = False


class TaskConfig(_Flex):
    """The ``task`` block: what is being learned and how it is scored.

    ``metric_groups`` is a list because one run can score several tasks separately;
    ``preproc_rules`` is the post-processing chain applied before scoring.
    """

    id: str | None = None
    category: str | None = None
    name: str | None = None
    metric_groups: list[_Flex] | None = None
    preproc_rules: PostprocConfig | None = None


# ---------------------------------------------------------------------------
# model.*
# ---------------------------------------------------------------------------


class AdapterConfig(_Flex):
    """Locates a saved PEFT adapter to load: by ``name`` or ``uuid4``, optionally
    a specific ``checkpoint``, from ``source``.
    """

    name: str | None = None
    uuid4: str | None = None
    source: str | None = None
    checkpoint: str | None = None


class PretrainedConfig(_Flex):
    """Locates a pretrained model, and names the class to load it with.

    ``cls`` is resolved by name at runtime, so a new backbone usually needs no code
    change. An ``adapter`` here loads PEFT weights on top of the base model.
    """

    cls: str | None = None
    args: _Flex | None = None
    name: str | None = None
    uuid4: str | None = None
    source: str | None = None
    checkpoint: str | None = None
    adapter: AdapterConfig | None = None


class InitModelConfigConfig(_Flex):
    """HF *Config* class + its constructor kwargs. Used in TRAIN mode to build
    the model-config object (e.g. BertConfig) from scratch.
    """

    cls: str | None = None
    args: _Flex | None = None


class InitConfig(_Flex):
    """TRAIN-from-scratch spec: which model class to instantiate and which
    HF config to pass it. Both `cls` fields fall back to arch-derived defaults
    when omitted.
    """

    cls: str | None = None
    config: InitModelConfigConfig | None = None


class ModelConfig(_Flex):
    """The ``model`` block: architecture, and how the model is obtained.

    Exactly one of ``init`` (build from scratch) or ``pretrained`` (load) is used,
    decided by ``mode``. ``architecture`` is a free-form string used for
    run-directory naming -- deliberately *not* validated against
    :class:`~micm_nlp.enums.ModelArchSE`. The ``param_size`` fields are filled in at
    runtime, not by YAML.
    """

    architecture: str
    init: InitConfig | None = None
    pretrained: PretrainedConfig | None = None
    # Runtime-assigned fields (kept optional so YAML doesn't need them)
    uuid4: str | None = None
    param_size: str | None = None
    trainable_param_size: str | None = None
    trainable_param_size_ratio: str | None = None


# ---------------------------------------------------------------------------
# tokenizer.*
# ---------------------------------------------------------------------------


class TokenizerConfig(_Flex):
    """The ``tokenizer`` block: which tokenizer to load, or which to train.

    ``adapt_to_lm`` applies the target architecture's special tokens and
    post-processor to a tokenizer borrowed from elsewhere.
    """

    source: str | None = None
    name: str | None = None
    type: str | None = None
    algorithm: str | None = None
    adapt_to_lm: bool = False
    vocab_size: int | None = None


# ---------------------------------------------------------------------------
# ds.*
# ---------------------------------------------------------------------------


class SplitsConfig(_Flex):
    """Which splits a dataset already ships with.

    Each field is ``False`` when absent, or the split's name when present -- a
    string because the on-disk name is not always ``train``/``test``/``validation``.
    """

    train: bool | str = False
    test: bool | str = False
    validation: bool | str = False


class InputConfig(_Flex):
    """Which dataset column(s) hold the input.

    ``key_2`` and ``key_3`` cover pair and triple inputs (premise/hypothesis,
    context/question/answer). ``standardize_key`` renames the column to the
    canonical name instead of carrying the original through.
    """

    key: str
    key_2: str | None = None
    key_3: str | None = None
    standardize_key: bool = False


class LabelConfig(_Flex):
    """Which column holds the label, and what the label space is.

    ``names`` and ``number`` must agree when the config asks for id-to-name mapping;
    ``CONFIG`` validates that. ``padded`` is the id used to pad label sequences,
    which the loss ignores.
    """

    key: str
    number: int | None = None
    names: list[str] | None = None
    standardize_key: bool = False
    padded: int | None = None


class TaskIdConfig(_Flex):
    """Which column identifies the task, for multi-task runs that score each
    task separately.
    """

    key: str
    standardize_key: bool = False


class DatasetConfig(_Flex):
    """The ``ds`` block: which dataset, where it lives, and how to read it.

    ``dirs`` is a path *template* under ``artefacts/datasets/<category>``; consumer
    repos substitute into it (a language segment, a fold) to assemble a run's data.
    """

    descriptive_name: str | None = None
    category: str | None = None
    dirs: str | None = None
    name: str | None = None
    type: str | None = None
    comes_with_splits: SplitsConfig | None = None
    input: InputConfig | None = None
    label: LabelConfig | None = None
    task_id: TaskIdConfig | None = None
    preproc_rules: _Flex | None = None
    Y: _Flex | None = None


# ---------------------------------------------------------------------------
# eval.*
# ---------------------------------------------------------------------------


class EvalConfig(_Flex):
    """The ``eval`` block: when evaluation runs.

    Before and after training, on the validation split or the test split, and
    during training. ``per_task`` groups metrics by task id.
    """

    before_training: bool = False
    before_training_on_test: bool = False
    during_training: bool | _Flex | None = None
    after_training: bool = False
    after_training_on_test: bool = False
    per_task: _Flex | None = None
    downstream_tasks: bool | _Flex = False


# ---------------------------------------------------------------------------
# test.*
# ---------------------------------------------------------------------------


class TestConfig(_Flex):
    """The ``test`` block: whether and how the held-out test split is scored.

    ``zero_shot`` adds an untrained baseline pass; ``zero_shot_only`` skips training
    altogether, which is how a zero-shot row is produced.
    """

    run: bool = False
    zero_shot: bool = False
    zero_shot_only: bool = False
    save_predictions: bool = False
    report_to_wandb: bool = False


# ---------------------------------------------------------------------------
# training_args.* / data_collator.* / custom_training_args.* / cuda.*
# ---------------------------------------------------------------------------


class TrainerConfig(_Flex):
    """Thin shell. `cls` selects which HF Trainer subclass to instantiate
    (Trainer, Seq2SeqTrainer, …). `args` is reserved for future extra kwargs
    splatted into the trainer ctor (runtime wiring currently fills the rest).
    """

    cls: str | None = None
    args: _Flex | None = None


class TrainingArgsConfig(_Flex):
    """Thin shell. The actual schema lives in HF (TrainingArguments,
    Seq2SeqTrainingArguments, etc.). `cls` selects which HF dataclass to
    instantiate; `args` is splatted into its constructor at runtime.

    For tokenizer-training configs, `cls` may name a non-HF trainer (e.g.
    SentencePieceTrainer) and `args` carries that trainer's kwargs.
    """

    cls: str | None = None
    args: _Flex | None = None


class DataCollatorConfig(_Flex):
    """Thin shell. The actual schema lives in HF (DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq, etc.) or in our custom collators. `cls` selects
    which collator to instantiate; `args` is splatted into its constructor.
    """

    cls: str | None = None
    args: _Flex | None = None


class CustomTrainingArgsConfig(_Flex):
    """Settings this package adds beyond HuggingFace's ``TrainingArguments``.

    Three groups. **Batching**: ``*_force_sequential`` and ``*_max_tokens_per_batch``
    -- mutually exclusive, since token-budget batching needs length-sorted order that
    a sequential sampler would override. **Early stopping**: patience, threshold, a
    floor before stopping is allowed, and a metric that is deliberately separable
    from the one used to pick the best checkpoint. **Everything else**: which columns
    to keep, per-parameter-group optimizer settings, a generation whitelist, and what
    to save at the end.
    """

    train_force_sequential: bool = False
    eval_force_sequential: bool = False
    test_force_sequential: bool = False
    # Token-budget batching: null = use fixed per_device_*_batch_size (legacy);
    # 'auto' = probe the GPU at runtime to find the largest tokens/batch that
    # doesn't OOM; int = skip the probe and use this exact budget.
    # Mutually exclusive with the corresponding *_force_sequential flag.
    # Mutually exclusive with HF's LengthGroupedSampler — when set, the
    # token-budget sampler is used instead (it already sorts by length).
    eval_max_tokens_per_batch: int | Literal['auto'] | None = None
    test_max_tokens_per_batch: int | Literal['auto'] | None = None
    early_stopping_after: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_threshold: float | None = None
    # Metric the early-stopping callback monitors — DECOUPLED from selection.
    # 'metric_for_best_model' = delegate to training_args (same metric used to
    # pick the best checkpoint, with its greater_is_better). Any literal key
    # (e.g. 'eval_loss') = monitor that directly ('loss' in name -> lower better).
    # None defaults to 'eval_loss' (legacy behavior).
    early_stopping_metric: str | None = None
    save_final_model: bool = True
    keep_only_final_model: bool = False
    random_task_exclusion: bool = False
    usable_columns: list[str] | None = None
    optimizer_grouped_parameters: list[_Flex] | None = None
    generation_whitelist: list[str] | None = None

    @field_validator('eval_max_tokens_per_batch', 'test_max_tokens_per_batch', mode='before')
    @classmethod
    def _reject_bool_tokens_per_batch(cls, v):
        if isinstance(v, bool):
            raise ValueError(
                f'Token budget must be a positive integer, \'auto\', or null; '
                f'bool is not allowed.'
            )
        return v

    @model_validator(mode='after')
    def _validate_token_budget(self) -> 'CustomTrainingArgsConfig':
        for stage in ('eval', 'test'):
            budget = getattr(self, f'{stage}_max_tokens_per_batch')
            force_seq = getattr(self, f'{stage}_force_sequential')
            if budget is not None and force_seq:
                raise ValueError(
                    f'{stage}_max_tokens_per_batch={budget!r} is incompatible '
                    f'with {stage}_force_sequential=True — token-budget mode '
                    f'requires length-sorted batching, which the sequential '
                    f'sampler overrides. Set {stage}_force_sequential=False or '
                    f'unset {stage}_max_tokens_per_batch.'
                )
            if isinstance(budget, int) and budget <= 0:
                raise ValueError(
                    f'{stage}_max_tokens_per_batch={budget!r} must be a positive '
                    f"integer, 'auto', or null."
                )
        return self


class CudaConfig(_Flex):
    """The ``cuda`` block. ``empty_cache_steps`` frees the allocator cache every N
    steps, trading a little speed for headroom.
    """

    empty_cache_steps: int | None = None
