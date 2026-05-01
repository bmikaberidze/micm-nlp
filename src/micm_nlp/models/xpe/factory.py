"""XPE factory — single entry point for building an XPE-wrapped model.

Replaces ``get_cross_prompt_encoder`` from the legacy path. Picks the
task-type-specific XPE subclass via :func:`xpe_model_for`, constructs it
directly (no dispatch via ``MODEL_TYPE_TO_PEFT_MODEL_MAPPING``), primes grad
requirements, logs the layer summary, and optionally restores a classifier
head from a prior checkpoint (SEQ_CLS-only — a no-op for other task types).
"""

import json
import os

from peft.tuners.tuners_utils import BaseTuner
from peft.utils import _prepare_prompt_learning_config
from peft.utils.save_and_load import load_peft_weights

import micm_nlp.utils as utils
from micm_nlp.models.xpe.config import CrossPromptEncoderConfig
from micm_nlp.models.xpe.peft_models import xpe_model_for


def _filtered_kwargs(peft_config_vars):
    """Turn a mapping-like config into a kwargs dict, dropping None values.

    :class:`CrossPromptEncoderConfig.__init__` does not accept every field
    declared on the calling-site ``PeftConfig`` (e.g. ``num_tasks``). Upstream
    PEFT's constructor used to raise TypeError when a None-valued kwarg for
    an unknown field was splatted in. Stripping None values at the factory
    boundary is the single place this is handled — per bug 6.4 in
    ``refactor.xpe.md``.
    """
    data = dict(peft_config_vars)
    return {k: v for k, v in data.items() if v is not None}


def get_xpe_model(base_model, peft_config_vars):
    model_config = BaseTuner.get_model_config(base_model)
    peft_config = CrossPromptEncoderConfig(**_filtered_kwargs(peft_config_vars))
    peft_config = _prepare_prompt_learning_config(peft_config, model_config)

    adapter_name = 'default'
    xpe_cls = xpe_model_for(peft_config.task_type)
    peft_model = xpe_cls(
        base_model,
        peft_config,
        adapter_name=adapter_name,
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=False,
    )

    prompt_encoder = peft_model.prompt_encoder[adapter_name]
    prompt_encoder.set_grad_requirements()
    prompt_encoder.print_all_layers()

    if peft_config.encoder_init_state_dict_path:
        peft_model = maybe_load_pretrained_classifier_state(peft_model, peft_config.encoder_init_state_dict_path)
    return peft_model


def maybe_load_pretrained_classifier_state(base_model, init_state_dict_path):
    """Copy ``base_model.classifier.*`` weights from a saved PEFT checkpoint into
    the SEQ_CLS PeftModel's ``modules_to_save.default`` slot. No-op if the
    checkpoint has no classifier keys (e.g. CAUSAL_LM has none).
    """
    pretrained_state_dict = load_peft_weights(init_state_dict_path)
    classifier_layer_key = 'base_model.classifier'
    classifier_params_dict = {
        k.replace(classifier_layer_key, f'{classifier_layer_key}.modules_to_save.default'): v
        for k, v in pretrained_state_dict.items()
        if k.startswith(classifier_layer_key)
    }
    if classifier_params_dict:
        utils.p(f"\nInitialize Classifier from pretrained: '{init_state_dict_path}'")
        utils.p('Pretrained base model classifier state dict keys: ', classifier_params_dict.keys())
        utils.p(f'Base model state dict keys: {base_model.state_dict().keys()}')
        for name, param in classifier_params_dict.items():
            if name in base_model.state_dict():
                base_model.state_dict()[name].copy_(param)
                utils.p(f'[yellow]Initialized with weight - {name}[/yellow]   {param.shape}')
            else:
                utils.p(f'⚠️ Pretrained weight {name} not found in model. Skipping.')
    return base_model


def is_xpe_config(config):
    """Dispatch helper for callers that don't want to import PeftType directly.

    Accepts the canonical ``peft_type='XPE'`` as well as the legacy
    ``peft_type='P_TUNING' + encoder_ratio`` spelling — paper-era checkpoints
    still ship with the legacy form.
    """
    from peft import PeftType  # local import so this module is safe to import early

    peft_type = getattr(config, 'peft_type', None)
    if peft_type == PeftType.XPE:
        return True
    return peft_type == PeftType.P_TUNING and hasattr(config, 'encoder_ratio')


def _read_adapter_config(path):
    """Return parsed ``adapter_config.json`` at ``path`` or ``None``.

    ``None`` means ``path`` isn't a directory, or the file is missing.
    """
    if not os.path.isdir(path):
        return None
    config_file = os.path.join(path, 'adapter_config.json')
    if not os.path.isfile(config_file):
        return None
    with open(config_file) as f:
        return json.load(f)


def is_xpe_adapter_dir(path):
    """Peek at ``adapter_config.json`` at ``path`` and report whether it is an
    XPE adapter — canonical ``peft_type=XPE`` or legacy
    ``peft_type=P_TUNING + encoder_ratio``. Returns False for non-directories
    and directories without an adapter_config.json.
    """
    data = _read_adapter_config(path)
    if data is None:
        return False
    peft_type = data.get('peft_type')
    if peft_type == 'XPE':
        return True
    return peft_type == 'P_TUNING' and 'encoder_ratio' in data


def load_xpe_pretrained(base_model, path, **kwargs):
    """Load an XPE adapter via the task-type-specific XPE subclass.

    Reads ``task_type`` from ``adapter_config.json`` at ``path``, resolves the
    matching XPE subclass, and delegates to its ``from_pretrained`` — which
    handles both canonical and legacy config spellings and bypasses the
    upstream task-type dispatch so our overrides actually run.
    """
    data = _read_adapter_config(path) or {}
    task_type = data.get('task_type')
    xpe_cls = xpe_model_for(task_type)
    return xpe_cls.from_pretrained(base_model, path, **kwargs)
