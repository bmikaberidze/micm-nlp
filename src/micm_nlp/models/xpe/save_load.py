"""XPE-aware save/load helpers.

These are drop-in replacements for ``peft.utils.save_and_load.get_peft_model_state_dict``
and ``peft.utils.save_and_load.set_peft_model_state_dict``. They:

- Skip the ``is_prompt_learning`` fallback that collapses the prompt encoder to
  a single ``prompt_embeddings`` tensor (destructive for XPE, which stores
  multi-component state via ``modules_to_save``).
- Preserve the full upstream ``save_embedding_layers`` handling, the
  ``PEFT_TYPE_TO_PREFIX_MAPPING`` branch (safe no-op for XPE), and the MPT
  post-load branch.

See ``refactor.xpe.md`` TP5/TP6 for the delta vs upstream peft 0.14.0.
"""

import os
import warnings

import torch
from peft import PeftType
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
from peft.utils.other import EMBEDDING_LAYER_NAMES, check_file_exists_on_hf_hub
from peft.utils.save_and_load import (
    _find_mismatched_keys,
    _insert_adapter_name_into_state_dict,
    get_embedding_layer_name,
    has_valid_embedding_base_layer,
)


def _encoder_key_to_saved_key(key, adapter_name):
    """Map a prompt-encoder state-dict key to the key adapters store it under, or
    ``None`` for PEFT's frozen ``original_module`` copy, which is never saved.

    ``embedding.modules_to_save.default.weight`` and ``embedding.weight`` both map to
    ``prompt_encoder.embedding.weight`` -- the on-disk layout is the same whether or
    not PEFT wrapped the sub-module.
    """
    if 'original_module.' in key:
        return None
    return 'prompt_encoder.' + key.replace(f'modules_to_save.{adapter_name}.', '')


def _prompt_encoder_state_dict(model, adapter_name):
    """The prompt encoder's own weights, read from the encoder itself.

    Not through ``modules_to_save``: since peft 0.17 a prompt-learning adapter wraps
    ``modules_to_save`` inside the base model only, so on CAUSAL_LM the encoder's
    sub-modules are never wrapped and a wrapper-based save writes none of them.
    SEQ_CLS still wraps them (its ``__init__`` wraps the whole PeftModel), which is
    why the mapping above strips the wrapper segment.
    """
    to_return = {}
    for key, value in model.prompt_encoder[adapter_name].state_dict().items():
        saved_key = _encoder_key_to_saved_key(key, adapter_name)
        if saved_key is not None:
            to_return[saved_key] = value
    return to_return


def xpe_get_peft_model_state_dict(
    model, state_dict=None, adapter_name='default', unwrap_compiled=False, save_embedding_layers='auto'
):
    """See module docstring. Mirrors upstream except the prompt-learning collapse."""
    if unwrap_compiled:
        model = getattr(model, '_orig_mod', model)

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    to_return = {}

    # MODULES TO SAVE -- the prompt encoder's own are taken from the encoder below
    if getattr(model, 'modules_to_save', None) is not None:
        for key, value in state_dict.items():
            if key.startswith('prompt_encoder.'):
                continue
            if any(f'{module_name}.modules_to_save.{adapter_name}' in key for module_name in model.modules_to_save):
                to_return[key.replace('modules_to_save.', '')] = value

    # DEAL WITH EMBEDDINGS
    is_embedding_in_target_modules = False
    if (
        save_embedding_layers == 'auto'
        and hasattr(config, 'target_modules')
        and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES)
    ):
        warnings.warn(
            'Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.', stacklevel=2
        )
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == 'auto':
        vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', None)
        model_id = getattr(config, 'base_model_name_or_path', None)

        has_base_config = False
        if model_id is not None:
            local_config_exists = os.path.exists(os.path.join(model_id, 'config.json'))
            exists = local_config_exists or check_file_exists_on_hf_hub(model_id, 'config.json')
            if exists is None:
                warnings.warn(
                    f'Could not find a config file in {model_id} - will assume that the vocabulary was not modified.',
                    stacklevel=2,
                )
                has_base_config = False
            else:
                has_base_config = exists

        if (
            vocab_size
            and model_id
            and has_base_config
            and (vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size)
        ):
            warnings.warn(
                'Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.',
                stacklevel=2,
            )
            save_embedding_layers = True
        else:
            save_embedding_layers = False

    if save_embedding_layers and hasattr(model, 'get_input_embeddings'):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                embedding_module_name = get_embedding_layer_name(model, layer, is_embedding_in_target_modules)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn(
            'Could not identify embedding layer(s) because the model is not a 🤗 transformers model.', stacklevel=2
        )

    # REMOVE ADAPTER NAME
    to_return = {k.replace(f'.{adapter_name}', ''): v for k, v in to_return.items()}
    to_return.update(_prompt_encoder_state_dict(model, adapter_name))
    return to_return


def xpe_set_peft_model_state_dict(
    model,
    peft_model_state_dict,
    adapter_name='default',
    ignore_mismatched_sizes: bool = False,
    low_cpu_mem_usage: bool = False,
):
    """See module docstring. Mirrors upstream except it skips the destructive
    ``prompt_embeddings`` single-tensor load and instead does a non-strict
    ``prompt_encoder.load_state_dict`` so XPE's multi-component state loads.
    """
    config = model.peft_config[adapter_name]
    saved_state_dict = peft_model_state_dict
    state_dict = {}
    if getattr(model, 'modules_to_save', None) is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f'{module_name}.modules_to_save.{adapter_name}')
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    # Since peft 0.15 prompt-learning methods are in the prefix mapping too (XPE as
    # 'xpe_'), and inserting the adapter name after 'xpe_' would corrupt the
    # xpe_embedding / xpe_head keys -- so prompt learning must skip this branch.
    if config.peft_type in PEFT_TYPE_TO_PREFIX_MAPPING and not config.is_prompt_learning:
        peft_model_state_dict = {}
        parameter_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        if config.peft_type == PeftType.VBLORA and config.save_only_topk_weights:
            num_vectors, _ = model.vblora_vector_bank[adapter_name].shape
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if '_topk_indices' in k:
                    v = state_dict[k].to(torch.long)
                    original_key = k.replace('_topk_indices', '')
                    topk_weights = state_dict[k.replace('_topk_indices', '_topk_weights')]
                    topk_weights = torch.cat([topk_weights, 1 - topk_weights.sum(-1, keepdim=True)], dim=-1)
                    topk_logits = torch.log(topk_weights)
                    matrix = (
                        torch.zeros([*(topk_logits.shape[:-1]), num_vectors])
                        .fill_(float('-inf'))
                        .to(topk_logits.device)
                        .scatter(-1, v, topk_logits)
                    )
                    state_dict[original_key] = matrix
                    del state_dict[k]
                    del state_dict[k.replace('_topk_indices', '_topk_weights')]

        peft_model_state_dict = _insert_adapter_name_into_state_dict(
            state_dict, adapter_name=adapter_name, parameter_prefix=parameter_prefix
        )

        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
        elif config.peft_type == PeftType.VERA:
            if config.save_projection and 'base_model.vera_A' not in peft_model_state_dict:
                raise ValueError(
                    'Specified to load vera_A and vera_B from state dictionary however they were not present!'
                )
            elif not config.save_projection and 'base_model.vera_A' in peft_model_state_dict:
                warnings.warn(
                    'Specified to not load vera_A and vera_B from state dictionary however they are present in state'
                    ' dictionary! Consider using them to ensure checkpoint loading is correct on all platforms using'
                    ' `peft_config.save_projection = True`',
                    stacklevel=2,
                )
            elif not config.save_projection:
                warnings.warn(
                    'Specified to not load vera_A and vera_B from state dictionary. This means we will be relying on'
                    ' PRNG initialisation to restore these projections using `config.projection_prng_key`, which may'
                    ' not be accurate on all system configurations.',
                    stacklevel=2,
                )
        elif config.peft_type == PeftType.LORA:
            old_dora_suffix = f'lora_magnitude_vector.{adapter_name}'

            def renamed_dora_weights(k):
                if k.endswith(old_dora_suffix):
                    k = k + '.weight'
                return k

            peft_model_state_dict = {renamed_dora_weights(k): v for k, v in peft_model_state_dict.items()}

    elif config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict

    elif config.peft_type == PeftType.XLORA:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    peft_model_state_dict, mismatched_keys = _find_mismatched_keys(
        model, peft_model_state_dict, ignore_mismatched_sizes=ignore_mismatched_sizes
    )
    if low_cpu_mem_usage:
        load_result = model.load_state_dict(peft_model_state_dict, strict=False, assign=True)
        for module in model.modules():
            if hasattr(module, '_move_adapter_to_device_of_base_layer'):
                module._move_adapter_to_device_of_base_layer(adapter_name)
    else:
        load_result = model.load_state_dict(peft_model_state_dict, strict=False)

    # XPE delta: the upstream `is_prompt_learning` branch collapses the prompt
    # encoder to a single `prompt_embeddings` tensor, which crashes for XPE
    # (which has embedding / xpe_embedding / xpe_head.*). Instead each of the
    # encoder's own keys is looked up under its saved name, so the load works
    # whether or not PEFT wrapped the sub-modules (see _prompt_encoder_state_dict).
    prompt_encoder = model.prompt_encoder[adapter_name]
    encoder_state_dict, missing = {}, []
    for key in prompt_encoder.state_dict():
        saved_key = _encoder_key_to_saved_key(key, adapter_name)
        if saved_key is None:
            continue
        if saved_key in saved_state_dict:
            encoder_state_dict[key] = saved_state_dict[saved_key]
        else:
            missing.append(saved_key)
    if missing:
        raise KeyError(f'XPE adapter is missing prompt-encoder weights: {missing}')
    prompt_encoder.load_state_dict(encoder_state_dict, strict=False)

    if mismatched_keys:
        mismatched_warning = '\n'.join(
            [
                f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated'
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        msg = (
            f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint '
            f'and are being ignored because you passed `ignore_mismatched_sizes=True`: {mismatched_warning}.'
        )
        warnings.warn(msg, stacklevel=2)
    return load_result
