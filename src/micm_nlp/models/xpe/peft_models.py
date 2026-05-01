"""XPE-scoped PeftModel subclasses, one per supported ``TaskType``.

Replaces the legacy global monkey-patching. All overrides live on a shared
:class:`_XPEPeftMixin` so every task-type variant inherits the same XPE
behaviour while non-XPE ``peft_type`` values still reach the stock upstream
path via ``super()``.

Supported task types are registered in :data:`TASK_TYPE_TO_XPE_MODEL` and
resolved by :func:`xpe_model_for`. The factory (:mod:`xpe.factory`) uses this
registry to pick the right subclass for ``get_xpe_model`` /
``load_xpe_pretrained`` — no hardcoded SEQ_CLS assumption.
"""

import contextlib
import os
import warnings
from typing import Any

import torch
from peft import (
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    PeftType,
    TaskType,
)
from transformers import PreTrainedModel

from micm_nlp.models.xpe.config import CrossPromptEncoderConfig
from micm_nlp.models.xpe.encoder import CrossPromptEncoder
from micm_nlp.models.xpe.save_load import (
    xpe_get_peft_model_state_dict,
    xpe_set_peft_model_state_dict,
)


@contextlib.contextmanager
def _xpe_state_dict_helpers():
    """Patch ``peft.peft_model``'s bound names for the duration of the block.

    ``PeftModel.save_pretrained`` / ``load_adapter`` look ``get_peft_model_state_dict``
    / ``set_peft_model_state_dict`` up in their module namespace (they were
    imported with ``from .utils import ...``). Patching the module attribute is
    sufficient; nothing else in the process sees the override.
    """
    import peft.peft_model as _pm

    orig_get = _pm.get_peft_model_state_dict
    orig_set = _pm.set_peft_model_state_dict
    _pm.get_peft_model_state_dict = xpe_get_peft_model_state_dict
    _pm.set_peft_model_state_dict = xpe_set_peft_model_state_dict
    try:
        yield
    finally:
        _pm.get_peft_model_state_dict = orig_get
        _pm.set_peft_model_state_dict = orig_set


class _XPEPeftMixin:
    """XPE overrides shared across task-type-specific PeftModel subclasses.

    Each concrete subclass pairs this mixin (left) with a stock PEFT task-type
    subclass (right) — e.g. ``(_XPEPeftMixin, PeftModelForSequenceClassification)``.
    MRO lookup hits the mixin first; every override delegates to ``super()`` on
    the non-XPE branch so stock behaviour is preserved for non-XPE ``peft_type``.
    """

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if config.peft_type != PeftType.XPE:
            return super()._setup_prompt_encoder(adapter_name)

        if not hasattr(self, 'prompt_encoder'):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
        # XPE is encoder-only — force 1 submodule even for SEQ_2_SEQ_LM.
        config.num_transformer_submodules = 1

        # determine the word embeddings
        word_embeddings = None
        try:
            word_embeddings = self.base_model.get_submodule('embeddings.word_embeddings')
        except AttributeError:
            pass

        if word_embeddings is None:
            for named_param, value in list(transformer_backbone.named_parameters()):
                deepspeed_distributed_tensor_shape = getattr(value, 'ds_shape', None)
                if value.shape[0] == self.base_model.config.vocab_size or (
                    deepspeed_distributed_tensor_shape is not None
                    and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
                ):
                    word_embeddings = transformer_backbone.get_submodule(named_param.replace('.weight', ''))
                    break

        self.word_embeddings = word_embeddings

        prompt_encoder = CrossPromptEncoder(config)
        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def get_prompt(self, batch_size: int, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter].unsqueeze(0).expand(batch_size, -1).to(prompt_encoder.get_device())
        )
        return prompt_encoder(prompt_tokens, task_ids)

    def save_pretrained(self, save_directory, **kwargs):
        with _xpe_state_dict_helpers():
            return super().save_pretrained(save_directory, **kwargs)

    def load_adapter(self, model_id, adapter_name, **kwargs):
        with _xpe_state_dict_helpers():
            return super().load_adapter(model_id, adapter_name, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: str | os.PathLike,
        adapter_name: str = 'default',
        is_trainable: bool = False,
        config: PeftConfig | None = None,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ) -> 'PeftModel':
        """Load an XPE adapter, bypassing ``MODEL_TYPE_TO_PEFT_MODEL_MAPPING``.

        Two reasons this override exists:
        1. Upstream ``PeftModel.from_pretrained`` dispatches final model
           construction via ``MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]``,
           which resolves to a stock upstream subclass — not our XPE subclass.
           Our overrides would never run.

        2. Paper-era adapter_config.json files carry
           ``peft_type='P_TUNING' + encoder_ratio`` (pre-XPE PeftType). Forcing
           :class:`CrossPromptEncoderConfig` here handles both the canonical
           ``peft_type='XPE'`` form and the legacy spelling — whose
           ``__post_init__`` flips ``peft_type`` to ``PeftType.XPE`` anyway.
        """
        if config is None:
            config = CrossPromptEncoderConfig.from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f'The input config must be a PeftConfig, got {config.__class__}')

        if config.is_prompt_learning and is_trainable:
            raise ValueError('Cannot set a prompt learning adapter to trainable when loading pretrained adapter.')
        config.inference_mode = not is_trainable

        peft_model = cls(
            model,
            config,
            adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        load_result = peft_model.load_adapter(
            model_id,
            adapter_name,
            is_trainable=is_trainable,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )
        missing_keys = [
            k for k in load_result.missing_keys if 'vblora_vector_bank' not in k and 'prompt_encoder' not in k
        ]
        if missing_keys:
            warnings.warn(f'Found missing adapter keys while loading the checkpoint: {missing_keys}', stacklevel=2)
        return peft_model


class XPEPeftModelForSequenceClassification(_XPEPeftMixin, PeftModelForSequenceClassification):
    pass


class XPEPeftModelForCausalLM(_XPEPeftMixin, PeftModelForCausalLM):
    pass


TASK_TYPE_TO_XPE_MODEL: dict[TaskType, type[PeftModel]] = {
    TaskType.SEQ_CLS: XPEPeftModelForSequenceClassification,
    TaskType.CAUSAL_LM: XPEPeftModelForCausalLM,
}


def xpe_model_for(task_type) -> type[PeftModel]:
    """Resolve the XPE PeftModel subclass for a given ``task_type``.

    Accepts a :class:`TaskType` enum value or its string spelling (as read from
    ``adapter_config.json``). Raises ``ValueError`` for unsupported task types.
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    if task_type not in TASK_TYPE_TO_XPE_MODEL:
        supported = sorted(t.value for t in TASK_TYPE_TO_XPE_MODEL)
        raise ValueError(f'XPE does not support task_type={task_type!r}. Supported: {supported}')
    return TASK_TYPE_TO_XPE_MODEL[task_type]
