from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

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
        return list(self.model_fields) + list(self.__pydantic_extra__ or {})

    def __getitem__(self, key: str):
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
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        config.file_path = str(path)
        config.apply_env_vars()
        return config

    # -- Env-var side-effect ------------------------------------------------

    def apply_env_vars(self) -> None:
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


class TaskConfig(_Flex):
    id: str | None = None
    category: str | None = None
    name: str | None = None
    metric_groups: list[_Flex] | None = None
    preproc_rules: PostprocConfig | None = None


# ---------------------------------------------------------------------------
# model.*
# ---------------------------------------------------------------------------


class AdapterConfig(_Flex):
    name: str | None = None
    source: str | None = None
    checkpoint: str | None = None
    uuid4: str | None = None


class PretrainedConfig(_Flex):
    cls: str | None = None
    args: _Flex | None = None
    name: str | None = None
    source: str | None = None
    checkpoint: str | None = None
    time_id: str | None = None
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
    train: bool | str = False
    test: bool | str = False
    validation: bool | str = False


class InputConfig(_Flex):
    key: str
    key_2: str | None = None
    key_3: str | None = None
    standardize_key: bool = False


class LabelConfig(_Flex):
    key: str
    number: int | None = None
    names: list[str] | None = None
    standardize_key: bool = False
    padded: int | None = None


class TaskIdConfig(_Flex):
    key: str
    standardize_key: bool = False


class DatasetConfig(_Flex):
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
    train_force_sequential: bool = False
    eval_force_sequential: bool = False
    test_force_sequential: bool = False
    early_stopping_after: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_threshold: float | None = None
    save_final_model: bool = True
    keep_only_final_model: bool = False
    random_task_exclusion: bool = False
    usable_columns: list[str] | None = None
    optimizer_grouped_parameters: list[_Flex] | None = None
    generation_whitelist: list[str] | None = None


class CudaConfig(_Flex):
    empty_cache_steps: int | None = None
