"""XPE CausalLM smoke tests.

Verifies the :class:`XPEPeftModelForCausalLM` subclass wires up, forwards,
and round-trips through save/load. Uses a tiny randomly-initialized GPT-2
so the test is network-free and fast.

These are *functional* tests (no golden snapshot): there is no pre-refactor
baseline for CAUSAL_LM, so we assert invariants (round-trip equality, subclass
identity, output shape) rather than compare to a frozen fingerprint.
"""

import io
import os
from contextlib import redirect_stdout

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

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

RATIOS = [0.0, 0.5, 1.0]


def _make_base():
    torch.manual_seed(0)
    cfg = GPT2Config(**TINY_GPT2)
    return GPT2LMHeadModel(cfg)


def _make_peft_cfg(ratio):
    return PeftConfig(
        peft_type='XPE',
        task_type='CAUSAL_LM',
        num_virtual_tokens=8,
        encoder_reparameterization_type='MLP',
        encoder_hidden_size=32,
        encoder_num_layers=2,
        encoder_dropout=0.0,
        encoder_input_size=None,
        encoder_init_state_dict_path=None,
        encoder_freeze=False,
        encoder_embedding_freeze=False,
        encoder_embedding_init_type='hf_default',
        encoder_embedding_normalize=None,
        encoder_embedding_normalize_max_norm=1.0,
        encoder_num_heads=4,
        encoder_ratio=ratio,
        modules_to_save=None,
    )


def _build(ratio):
    from micm_nlp.models.xpe import get_xpe_model

    base = _make_base()
    cfg = _make_peft_cfg(ratio)
    buf = io.StringIO()
    with redirect_stdout(buf):
        model = get_xpe_model(base, cfg)
    return model


def _peft_state_dict_fingerprint(model):
    from micm_nlp.models.xpe.save_load import xpe_get_peft_model_state_dict

    peft_sd = xpe_get_peft_model_state_dict(model, adapter_name='default')
    out = {}
    for k, v in peft_sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = {
                'shape': list(v.shape),
                'dtype': str(v.dtype),
                'norm': round(float(v.detach().float().norm().item()), 6),
            }
    return out


@pytest.mark.parametrize('ratio', RATIOS)
def test_build_dispatches_to_causal_lm_subclass(ratio):
    from micm_nlp.models.xpe import XPEPeftModelForCausalLM

    torch.manual_seed(42)
    model = _build(ratio)
    assert isinstance(model, XPEPeftModelForCausalLM)
    assert 'default' in model.prompt_encoder
    assert model.prompt_encoder['default'].total_virtual_tokens == 8


@pytest.mark.parametrize('ratio', RATIOS)
def test_forward_produces_vocab_logits(ratio):
    torch.manual_seed(42)
    model = _build(ratio)
    model.eval()
    input_ids = torch.randint(0, TINY_GPT2['vocab_size'], (2, 16))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    # PEFT prepends virtual tokens to the sequence for CausalLM; caller is
    # responsible for slicing them off when computing loss / decoding.
    num_virtual = 8
    assert out.logits.shape == (2, 16 + num_virtual, TINY_GPT2['vocab_size'])


@pytest.mark.parametrize('ratio', RATIOS)
def test_save_load_roundtrip(ratio, tmp_path):
    from micm_nlp.models.xpe import XPEPeftModelForCausalLM, load_xpe_pretrained

    torch.manual_seed(42)
    model = _build(ratio)
    before = _peft_state_dict_fingerprint(model)

    save_dir = tmp_path / 'adapter'
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.save_pretrained(str(save_dir))

    base = _make_base()
    buf = io.StringIO()
    with redirect_stdout(buf):
        loaded = load_xpe_pretrained(base, str(save_dir))
    after = _peft_state_dict_fingerprint(loaded)

    assert isinstance(loaded, XPEPeftModelForCausalLM)
    assert before == after, (
        f'CausalLM save/load mismatch at ratio={ratio}.\n'
        f'Before keys: {sorted(before.keys())}\n'
        f'After keys:  {sorted(after.keys())}'
    )
