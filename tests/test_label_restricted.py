"""Tests for label-restricted likelihood (mcqa_ftp opt-in).

The argmax over candidate label tokens lives in
``get_preprocess_logits_for_metrics``; downstream sees a regular ``(B, S)``
prediction tensor that the standard filter_padded/accuracy path handles.
These tests cover the candidate-id helper and the callback's shape contract.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from micm_nlp.evals.eval import _label_candidate_token_ids, get_preprocess_logits_for_metrics

# Fake bare-letter token ids (mirrors bloom: 'A'->36 etc.; values here are arbitrary).
IDS = {'A': 10, 'B': 11, 'C': 12, 'D': 13}


class FakeTokenizer:
    """Single-token encoder for known label strings; 'MULTI' is multi-token."""

    def encode(self, text, add_special_tokens=False):
        if text == 'MULTI':
            return [10, 11]
        if text not in IDS:
            raise KeyError(text)
        return [IDS[text]]


def make_config(names):
    return SimpleNamespace(ds=SimpleNamespace(label=SimpleNamespace(names=names)))


def test_label_candidate_token_ids_returns_names_and_ids():
    names, ids = _label_candidate_token_ids(make_config(['A', 'B', 'C', 'D']), FakeTokenizer())
    assert names == ['A', 'B', 'C', 'D']
    assert ids == [10, 11, 12, 13]


def test_missing_label_names_raises():
    with pytest.raises(ValueError, match='names'):
        _label_candidate_token_ids(make_config(None), FakeTokenizer())


def test_multi_token_label_name_raises():
    with pytest.raises(ValueError, match='single token'):
        _label_candidate_token_ids(make_config(['A', 'MULTI']), FakeTokenizer())


def make_preprocess_config():
    return SimpleNamespace(
        task=SimpleNamespace(
            preproc_rules=SimpleNamespace(prediction_axis=-1, label_restricted_likelihood=True),
            metric_groups=[SimpleNamespace(metrics=['accuracy'])],
            category='text_generation',
        ),
        ds=SimpleNamespace(label=SimpleNamespace(names=['A', 'B', 'C', 'D'])),
    )


def test_callback_places_candidate_argmax_at_answer_slot():
    # The label_restricted callback returns (B, S) aligned with labels: every
    # position is 0 except the answer slot, which holds the argmax candidate
    # token id. Downstream filter_padded (labels != -100) reduces to (B,).
    cfg = make_preprocess_config()
    fn = get_preprocess_logits_for_metrics(cfg, tokenizer=FakeTokenizer())

    B, S, V = 2, 4, 20
    logits = torch.full((B, S, V), -5.0)
    # Causal shift: answer token at original position 3 is predicted by logits
    # at original position 2 (shift_logits position 2).
    logits[0, 2, IDS['A']] = 10.0  # row 0 -> A should win
    logits[1, 2, IDS['C']] = 10.0  # row 1 -> C should win
    # Distractor at a NON-answer slot must be ignored.
    logits[0, 0, IDS['D']] = 100.0

    labels = torch.full((B, S), -100)
    labels[0, 3] = IDS['A']
    labels[1, 3] = IDS['B']  # gold id only matters to the downstream metric, not the callback

    out = fn(logits, labels)

    assert tuple(out.shape) == (B, S)
    # Answer-slot predictions are the candidate-restricted argmax token ids.
    assert out[0, 3].item() == IDS['A']
    assert out[1, 3].item() == IDS['C']
    # Non-answer slots stay at 0 (will be masked out by labels == -100).
    mask = labels != -100
    assert (out[~mask] == 0).all()


def test_callback_argmax_is_restricted_to_candidates():
    # A non-candidate token id with the highest raw logit must NOT be picked.
    cfg = make_preprocess_config()
    fn = get_preprocess_logits_for_metrics(cfg, tokenizer=FakeTokenizer())

    B, S, V = 1, 3, 20
    logits = torch.full((B, S, V), 0.0)
    NON_CAND = 19
    logits[0, 1, NON_CAND] = 100.0  # raw vocab argmax
    logits[0, 1, IDS['B']] = 1.0    # best among candidates

    labels = torch.full((B, S), -100)
    labels[0, 2] = IDS['A']

    out = fn(logits, labels)

    assert out[0, 2].item() == IDS['B']
