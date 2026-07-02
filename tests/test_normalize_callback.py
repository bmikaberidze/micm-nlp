"""Regression tests for NormalizePromptEncoderEmbeddings.

The clip/unit normalization was silently a no-op because the callback hooked
`on_optimizer_step` (which under transformers 4.48 did not pass `model` -> the
body early-returned). It must use `on_step_end` + `kwargs['model']`, like the
working ParamNormLogger. These tests guard both the hook wiring and the math.
"""
from types import SimpleNamespace

import torch

from micm_nlp.models.xpe import CrossPromptEncoder
from micm_nlp.training.callbacks import NormalizePromptEncoderEmbeddings


def _min_config(**over):
    cfg = dict(
        token_dim=8, encoder_input_size=None, encoder_num_heads=1,
        encoder_num_layers=1, encoder_dropout=0.0, encoder_hidden_size=8,
        encoder_reparameterization_type='MLP', encoder_embedding_init_type='hf_default',
        encoder_ratio=0,  # SPT-only -> just self.embedding, no MLP head
        num_transformer_submodules=1, num_virtual_tokens=4,
        encoder_init_state_dict_path=None, encoder_freeze=False,
        encoder_embedding_freeze=False,
        encoder_embedding_normalize='clip', encoder_embedding_normalize_max_norm=2.0,
    )
    cfg.update(over)
    return SimpleNamespace(**cfg)


def test_hook_is_on_step_end_not_on_optimizer_step():
    """The bug was the wrong hook -> guard it explicitly."""
    d = NormalizePromptEncoderEmbeddings.__dict__
    assert 'on_step_end' in d, 'must hook on_step_end (fires + receives model)'
    assert 'on_optimizer_step' not in d, 'on_optimizer_step did not pass model in tf 4.48'


def test_clip_caps_row_norms():
    enc = CrossPromptEncoder(_min_config(encoder_embedding_normalize='clip',
                                         encoder_embedding_normalize_max_norm=2.0))
    with torch.no_grad():
        enc.embedding.weight.mul_(0).add_(10.0)  # every row norm >> 2.0
    enc.normalize_embeddings()
    row_norms = enc.embedding.weight.norm(dim=-1)
    assert torch.all(row_norms <= 2.0 + 1e-4), f'rows not clipped: {row_norms}'


def test_callback_normalizes_via_on_step_end(monkeypatch):
    """End-to-end: on_step_end pulls model from kwargs, clips, logs xpe key."""
    import micm_nlp.training.callbacks as cb
    logged = {}
    monkeypatch.setattr(cb.wandb, 'log', lambda d: logged.update(d))

    enc = CrossPromptEncoder(_min_config())
    with torch.no_grad():
        enc.embedding.weight.mul_(0).add_(10.0)
    model = SimpleNamespace(active_adapter='default',
                            prompt_encoder=torch.nn.ModuleDict({'default': enc}))

    cb.NormalizePromptEncoderEmbeddings().on_step_end(None, None, None, model=model)

    assert 'train/xpe_embedd_norm' in logged, 'callback did not run / log renamed key'
    assert torch.all(enc.embedding.weight.norm(dim=-1) <= 2.0 + 1e-4)
