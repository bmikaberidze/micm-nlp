"""XPE parity harness.

Captures deterministic fingerprints of an XPE-wrapped tiny BERT model across
(encoder_ratio, encoder_reparameterization_type) combinations, and compares
against golden snapshots authored by the pre-refactor monkey-patching code.
First run writes snapshots; later runs diff.

Goal: lock the post-refactor subclass path to exact parity with pre-refactor
behavior. The harness does NOT require network, GPU, or real weights — it
uses a small randomly-initialized BERT with fixed seed.

Usage:
    pytest tests/test_parity.py -v                 # compare to golden
    REWRITE_GOLDEN=1 pytest tests/test_parity.py   # regenerate golden
"""

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from micm_nlp.config import PeftConfig

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

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

GOLDEN_DIR = Path(__file__).parent / 'golden'
GOLDEN_DIR.mkdir(exist_ok=True, parents=True)

PARAMS = [
    (0.0, 'MLP'),
    (0.3, 'MLP'),
    (0.5, 'MLP'),
    (0.7, 'MLP'),
    (1.0, 'MLP'),
    (0.5, 'ATTN'),
    (1.0, 'ATTN'),
    (1.0, 'NONE'),
    (0.5, 'LSTM'),
    (1.0, 'LSTM'),
]


def _make_base():
    torch.manual_seed(0)
    cfg = BertConfig(**TINY_BERT)
    return BertForSequenceClassification(cfg)


def _make_peft_cfg(ratio, head):
    return PeftConfig(
        peft_type='P_TUNING',
        task_type='SEQ_CLS',
        num_virtual_tokens=8,
        encoder_reparameterization_type=head,
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


def _state_dict_fingerprint(state_dict):
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = {
                'shape': list(v.shape),
                'dtype': str(v.dtype),
                'norm': round(float(v.detach().float().norm().item()), 6),
            }
    return out


def _peft_state_dict_fingerprint(model):
    """Fingerprint the PEFT-facing state dict (what gets saved to adapter_model.*).

    Must use the XPE-aware getter: upstream ``get_peft_model_state_dict`` calls
    ``prompt_encoder.embedding.weight`` unconditionally, which fails for
    pure-XPE (ratio=1.0) where there is no ``.embedding``. This is the same
    codepath that ``XPEPeftModelForSequenceClassification.save_pretrained``
    installs via its context manager.
    """
    from micm_nlp.models.xpe.save_load import xpe_get_peft_model_state_dict

    peft_sd = xpe_get_peft_model_state_dict(model, adapter_name='default')
    return _state_dict_fingerprint(peft_sd)


def _forward_fingerprint(model, seed=1):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, TINY_BERT['vocab_size'], (2, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([0, 1])
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return {
        'logits_shape': list(out.logits.shape),
        'logits_norm': round(float(out.logits.norm().item()), 6),
        'logits_row0': [round(x, 6) for x in out.logits[0].tolist()],
        'loss': round(float(out.loss.item()), 6),
    }


def _trainable_param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def _build(ratio, head):
    """Build model via the refactored entry point."""
    from micm_nlp.models.xpe import get_xpe_model

    base = _make_base()
    cfg = _make_peft_cfg(ratio, head)
    buf = io.StringIO()
    with redirect_stdout(buf):
        model = get_xpe_model(base, cfg)
    return model


def _snapshot(model):
    return {
        'peft_state_dict': _peft_state_dict_fingerprint(model),
        'full_state_dict_keys': sorted(model.state_dict().keys()),
        'forward': _forward_fingerprint(model),
        'params': _trainable_param_counts(model),
        'prompt_encoder_repr': repr(model.prompt_encoder['default']),
    }


# Fields whose value is a PyTorch implementation detail rather than a fact about
# this package. ``prompt_encoder_repr`` is ``repr()`` of a module tree: torch 2.14
# added ``bias=True`` to ``LayerNorm.__repr__``, which changed the string without
# changing anything the string was standing in for. ``full_state_dict_keys`` and
# ``params`` already pin the structure, so the repr is recorded for a human reading
# the golden file and not compared.
_IGNORED_FIELDS = frozenset({'prompt_encoder_repr'})

# Floats in a snapshot are already rounded to 6 decimal places, so they agree to
# well under this unless something real changed. An exact comparison instead fails
# on the last digit across torch versions -- a weight norm of 3.372186 vs 3.372187.
_FLOAT_TOL = 1e-5


def _compare(golden, actual, path=''):
    """Walk two snapshots, returning [(path, golden, actual)] for real differences."""
    if isinstance(golden, dict) and isinstance(actual, dict):
        diffs = []
        for key in sorted(set(golden) | set(actual)):
            if key in _IGNORED_FIELDS:
                continue
            where = f'{path}.{key}' if path else key
            if key not in golden or key not in actual:
                diffs.append((where, golden.get(key, '<missing>'), actual.get(key, '<missing>')))
            else:
                diffs += _compare(golden[key], actual[key], where)
        return diffs

    if isinstance(golden, list) and isinstance(actual, list):
        if len(golden) != len(actual):
            return [(f'{path}.len', len(golden), len(actual))]
        return [d for i, (g, a) in enumerate(zip(golden, actual, strict=True))
                for d in _compare(g, a, f'{path}[{i}]')]

    if isinstance(golden, float) and isinstance(actual, (int, float)):
        return [] if abs(golden - actual) <= _FLOAT_TOL else [(path, golden, actual)]

    return [] if golden == actual else [(path, golden, actual)]


def _key(ratio, head):
    return f'ratio{ratio}_head{head}'


@pytest.mark.parametrize('ratio,head', PARAMS)
def test_parity(ratio, head):
    """Post-refactor snapshot must match the golden authored by pre-refactor."""
    torch.manual_seed(42)
    model = _build(ratio, head)
    snap = _snapshot(model)

    golden_path = GOLDEN_DIR / f'{_key(ratio, head)}.json'
    rewrite = os.environ.get('REWRITE_GOLDEN') == '1'

    if rewrite or not golden_path.exists():
        golden_path.write_text(json.dumps(snap, indent=2, sort_keys=True))
        pytest.skip(f'Wrote golden snapshot: {golden_path.name}')

    golden = json.loads(golden_path.read_text())
    diffs = _compare(golden, snap)
    assert not diffs, (
        f'Parity mismatch for {_key(ratio, head)}.\n'
        + '\n'.join(f'  {path}: golden={g!r} actual={a!r}' for path, g, a in diffs[:10])
    )


@pytest.mark.parametrize('ratio,head', PARAMS)
def test_save_load_roundtrip(ratio, head, tmp_path):
    """Saving then loading via the XPE subclass must preserve the PEFT state dict."""
    torch.manual_seed(42)
    model = _build(ratio, head)
    before = _peft_state_dict_fingerprint(model)

    save_dir = tmp_path / 'adapter'
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.save_pretrained(str(save_dir))

    from micm_nlp.models.xpe import load_xpe_pretrained

    base = _make_base()
    buf = io.StringIO()
    with redirect_stdout(buf):
        loaded = load_xpe_pretrained(base, str(save_dir))
    after = _peft_state_dict_fingerprint(loaded)

    assert before == after, (
        f'Save/load mismatch for {_key(ratio, head)}.\n'
        f'Before keys: {sorted(before.keys())}\n'
        f'After keys:  {sorted(after.keys())}'
    )
