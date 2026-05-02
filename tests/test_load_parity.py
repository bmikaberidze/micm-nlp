"""Parity between the two XPE loading routes on the same saved adapter.

Route #1 (fresh + warm-start via ``encoder_init_state_dict_path``):
    base = <fresh HF model>
    cfg.encoder_init_state_dict_path = <saved_dir>
    model = get_xpe_model(base, cfg)

Route #2 (``load_xpe_pretrained`` / adapter from_pretrained):
    base = <fresh HF model>
    model = load_xpe_pretrained(base, <saved_dir>)

The source adapter is built fresh, has its XPE parameters perturbed so the
saved weights are non-trivial, then round-tripped through both routes.
Both routes must produce:
  - identical PEFT state dicts (key-by-key tensor equality),
  - identical classifier weights for SEQ_CLS,
  - identical forward-pass logits.
"""

import io
import os
from contextlib import redirect_stdout

import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
)

from micm_nlp.config import PeftConfig

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')


TINY_GPT2 = dict(
    vocab_size=512,
    n_positions=64,
    n_embd=32,
    n_layer=2,
    n_head=4,
    n_inner=64,
    pad_token_id=0,
    bos_token_id=0,
    eos_token_id=0,
)

TINY_BERT = dict(
    vocab_size=1024,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=128,
    num_labels=4,
    pad_token_id=0,
)

CAUSAL_CASES = [
    (0.0, 'MLP'),
    (0.5, 'MLP'),
    (1.0, 'MLP'),
    (1.0, 'ATTN'),
]

SEQCLS_CASES = [
    (0.0, 'MLP'),
    (0.5, 'MLP'),
    (1.0, 'MLP'),
    (1.0, 'ATTN'),
]


def _make_gpt2_base():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(**TINY_GPT2))


def _make_bert_base():
    torch.manual_seed(0)
    return BertForSequenceClassification(BertConfig(**TINY_BERT))


def _make_peft_cfg(task_type, ratio, head, init_path=None):
    return PeftConfig(
        peft_type='XPE',
        task_type=task_type,
        num_virtual_tokens=8,
        encoder_reparameterization_type=head,
        encoder_hidden_size=32,
        encoder_num_layers=2,
        encoder_dropout=0.0,
        encoder_input_size=None,
        encoder_init_state_dict_path=init_path,
        encoder_freeze=False,
        encoder_embedding_freeze=False,
        encoder_embedding_init_type='hf_default',
        encoder_embedding_normalize=None,
        encoder_embedding_normalize_max_norm=1.0,
        encoder_num_heads=4,
        encoder_ratio=ratio,
        modules_to_save=None,
    )


def _quiet_build(base, cfg):
    from micm_nlp.models.xpe import get_xpe_model

    buf = io.StringIO()
    with redirect_stdout(buf):
        return get_xpe_model(base, cfg)


def _quiet_load(base, path):
    from micm_nlp.models.xpe import load_xpe_pretrained

    buf = io.StringIO()
    with redirect_stdout(buf):
        return load_xpe_pretrained(base, path)


def _quiet_save(model, path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.save_pretrained(str(path))


def _perturb_trainable(model, seed):
    """Scramble every trainable parameter to a deterministic non-init value.
    This guards against the trap where both loaders silently fail and still
    match because they both stayed at init weights.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.1)


def _peft_state_dict(model):
    from micm_nlp.models.xpe.save_load import xpe_get_peft_model_state_dict

    return xpe_get_peft_model_state_dict(model, adapter_name='default')


def _assert_state_dicts_equal(a, b, *, label):
    assert set(a.keys()) == set(b.keys()), (
        f'{label}: key mismatch.\n  only in A: {sorted(set(a) - set(b))}\n  only in B: {sorted(set(b) - set(a))}'
    )
    for k in a:
        ta, tb = a[k], b[k]
        assert ta.shape == tb.shape, f'{label}[{k}]: shape {ta.shape} vs {tb.shape}'
        assert ta.dtype == tb.dtype, f'{label}[{k}]: dtype {ta.dtype} vs {tb.dtype}'
        assert torch.equal(ta.detach().cpu(), tb.detach().cpu()), (
            f'{label}[{k}]: tensor values differ (max abs diff = '
            f'{(ta.detach().cpu() - tb.detach().cpu()).abs().max().item()})'
        )


def _forward_causal(model):
    torch.manual_seed(7)
    input_ids = torch.randint(0, TINY_GPT2['vocab_size'], (2, 12))
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits.detach().cpu()


def _forward_seqcls(model):
    torch.manual_seed(7)
    input_ids = torch.randint(0, TINY_BERT['vocab_size'], (2, 12))
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits.detach().cpu()


# ---------------------------------------------------------------------------
# CAUSAL_LM parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('ratio,head', CAUSAL_CASES)
def test_causal_lm_load_parity(ratio, head, tmp_path):
    # Build + perturb the source adapter, save to disk.
    torch.manual_seed(42)
    source = _quiet_build(_make_gpt2_base(), _make_peft_cfg('CAUSAL_LM', ratio, head))
    _perturb_trainable(source, seed=1234)
    save_dir = tmp_path / 'adapter'
    _quiet_save(source, save_dir)
    source_sd = _peft_state_dict(source)

    # Route #1: fresh PEFT + encoder_init_state_dict_path.
    m1 = _quiet_build(_make_gpt2_base(), _make_peft_cfg('CAUSAL_LM', ratio, head, init_path=str(save_dir)))
    sd1 = _peft_state_dict(m1)

    # Route #2: load_xpe_pretrained.
    m2 = _quiet_load(_make_gpt2_base(), str(save_dir))
    sd2 = _peft_state_dict(m2)

    _assert_state_dicts_equal(sd1, source_sd, label=f'route1-vs-source[ratio={ratio},head={head}]')
    _assert_state_dicts_equal(sd2, source_sd, label=f'route2-vs-source[ratio={ratio},head={head}]')
    _assert_state_dicts_equal(sd1, sd2, label=f'route1-vs-route2[ratio={ratio},head={head}]')

    # Forward-pass equivalence.
    l1 = _forward_causal(m1)
    l2 = _forward_causal(m2)
    l_src = _forward_causal(source)
    assert torch.equal(l1, l2), f'CAUSAL_LM logits differ: max abs diff = {(l1 - l2).abs().max().item()}'
    assert torch.equal(l1, l_src), f'Route #1 diverges from source: max abs diff = {(l1 - l_src).abs().max().item()}'


# ---------------------------------------------------------------------------
# SEQ_CLS parity (exercises the classifier path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('ratio,head', SEQCLS_CASES)
def test_seq_cls_load_parity(ratio, head, tmp_path):
    torch.manual_seed(42)
    source = _quiet_build(_make_bert_base(), _make_peft_cfg('SEQ_CLS', ratio, head))
    _perturb_trainable(source, seed=4321)
    save_dir = tmp_path / 'adapter'
    _quiet_save(source, save_dir)
    source_sd = _peft_state_dict(source)

    # Sanity: the saved PEFT state dict must contain classifier keys, otherwise
    # we'd be silently testing a CAUSAL_LM-like path here.
    classifier_keys = [k for k in source_sd if 'classifier' in k]
    assert classifier_keys, f'Expected classifier keys in saved state dict, got: {sorted(source_sd.keys())}'

    m1 = _quiet_build(_make_bert_base(), _make_peft_cfg('SEQ_CLS', ratio, head, init_path=str(save_dir)))
    sd1 = _peft_state_dict(m1)

    m2 = _quiet_load(_make_bert_base(), str(save_dir))
    sd2 = _peft_state_dict(m2)

    _assert_state_dicts_equal(sd1, source_sd, label=f'route1-vs-source[ratio={ratio},head={head}]')
    _assert_state_dicts_equal(sd2, source_sd, label=f'route2-vs-source[ratio={ratio},head={head}]')
    _assert_state_dicts_equal(sd1, sd2, label=f'route1-vs-route2[ratio={ratio},head={head}]')

    # Tensor-level check on the live classifier slot (post-PEFT wrapping path).
    c1 = dict(m1.named_parameters())
    c2 = dict(m2.named_parameters())
    classifier_param_keys = [k for k in c1 if 'classifier' in k and 'modules_to_save.default' in k]
    assert classifier_param_keys, f'No live classifier.modules_to_save.default.* params found on m1: {list(c1)[:20]}'
    for k in classifier_param_keys:
        assert torch.equal(c1[k].detach().cpu(), c2[k].detach().cpu()), (
            f'Classifier weight {k} differs between routes (max abs diff = '
            f'{(c1[k].detach().cpu() - c2[k].detach().cpu()).abs().max().item()})'
        )

    l1 = _forward_seqcls(m1)
    l2 = _forward_seqcls(m2)
    l_src = _forward_seqcls(source)
    assert torch.equal(l1, l2), f'SEQ_CLS logits differ: max abs diff = {(l1 - l2).abs().max().item()}'
    assert torch.equal(l1, l_src), f'Route #1 diverges from source: max abs diff = {(l1 - l_src).abs().max().item()}'
