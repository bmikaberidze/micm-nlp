"""Tests for build_inference_dataloader_kwargs — the helper that decides
between fixed-batch and token-budget paths for eval/test dataloaders."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from micm_nlp.training.trainers import build_inference_dataloader_kwargs
from micm_nlp.training.batching import TokenBudgetBatchSampler


class _StubCollator:
    """Stand-in for DataCollatorForSeq2Seq exposing only what the helper reads."""
    def __init__(self, *, pad_to_multiple_of=None):
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        return batch


def _make_collator(*, pad_to_multiple_of=None):
    return _StubCollator(pad_to_multiple_of=pad_to_multiple_of)


class _TinyDataset(Dataset):
    """Minimal stand-in supporting both integer indexing (DataLoader protocol)
    and string indexing (HF datasets-style column access)."""

    def __init__(self, lengths):
        self._lengths = lengths

    def __len__(self):
        return len(self._lengths)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == 'length':
                return self._lengths
            raise KeyError(idx)
        return {'length': self._lengths[idx], 'input_ids': [0] * self._lengths[idx]}

    @property
    def column_names(self):
        return ['length', 'input_ids']


def _args(**overrides):
    """Build a minimal stand-in for HF TrainingArguments."""
    defaults = dict(
        eval_batch_size=4,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_drop_last=False,
        dataloader_prefetch_factor=None,
        length_column_name='length',
        group_by_length=False,
        pad_to_multiple_of=8,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_legacy_path_when_token_budget_is_none():
    ds = _TinyDataset([100, 200, 300])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        args=_args(),
        data_collator=_make_collator(),
        token_budget=None,
    )
    assert kwargs['batch_size'] == 4
    assert 'batch_sampler' not in kwargs


def test_token_budget_path_replaces_batch_size_with_batch_sampler():
    ds = _TinyDataset([100, 200, 300, 100])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        args=_args(),
        data_collator=_make_collator(pad_to_multiple_of=8),
        token_budget=1024,
    )
    assert 'batch_size' not in kwargs
    assert 'sampler' not in kwargs
    assert isinstance(kwargs['batch_sampler'], TokenBudgetBatchSampler)


def test_token_budget_reads_lengths_from_length_column():
    ds = _TinyDataset([10, 20, 30])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        args=_args(length_column_name='length'),
        data_collator=_make_collator(pad_to_multiple_of=8),
        token_budget=64,
    )
    sampler = kwargs['batch_sampler']
    assert sampler._lengths == [10, 20, 30]


def test_token_budget_path_accepts_arbitrary_positive_int():
    # No longer about "stage" — the helper doesn't see stage. This test now
    # just confirms a budget=int is accepted (renamed for clarity).
    ds = _TinyDataset([10])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        args=_args(),
        data_collator=_make_collator(pad_to_multiple_of=8),
        token_budget=128,
    )
    assert isinstance(kwargs['batch_sampler'], TokenBudgetBatchSampler)


def test_pad_multiple_read_from_data_collator():
    """pad_to_multiple_of is sourced from data_collator (not args).
    Regression for an earlier bug where the helper tried to read it from args."""
    ds = _TinyDataset([100, 200])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        args=_args(),
        data_collator=_make_collator(pad_to_multiple_of=16),
        token_budget=512,
    )
    sampler = kwargs['batch_sampler']
    assert sampler._pad_multiple == 16


def test_pad_multiple_defaults_to_one_when_collator_lacks_attribute():
    """If the collator doesn't expose pad_to_multiple_of, fall back to 1."""
    ds = _TinyDataset([100, 200])
    plain_callable = lambda batch: batch   # no pad_to_multiple_of attribute
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        args=_args(),
        data_collator=plain_callable,
        token_budget=512,
    )
    sampler = kwargs['batch_sampler']
    assert sampler._pad_multiple == 1
