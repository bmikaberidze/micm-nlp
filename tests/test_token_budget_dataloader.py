"""Tests for build_inference_dataloader_kwargs — the helper that decides
between fixed-batch and token-budget paths for eval/test dataloaders."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from micm_nlp.training.trainers import build_inference_dataloader_kwargs
from micm_nlp.training.batching import TokenBudgetBatchSampler


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


def _custom(**overrides):
    defaults = dict(
        eval_force_sequential=False,
        test_force_sequential=False,
        eval_max_tokens_per_batch=None,
        test_max_tokens_per_batch=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_legacy_path_when_token_budget_is_none():
    ds = _TinyDataset([100, 200, 300])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        stage='test',
        args=_args(),
        custom_args=_custom(),
        data_collator=lambda x: x,
        token_budget=None,
    )
    assert kwargs['batch_size'] == 4
    assert 'batch_sampler' not in kwargs


def test_token_budget_path_replaces_batch_size_with_batch_sampler():
    ds = _TinyDataset([100, 200, 300, 100])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        stage='test',
        args=_args(),
        custom_args=_custom(test_max_tokens_per_batch=1024),
        data_collator=lambda x: x,
        token_budget=1024,
    )
    assert 'batch_size' not in kwargs
    assert 'sampler' not in kwargs
    assert isinstance(kwargs['batch_sampler'], TokenBudgetBatchSampler)


def test_token_budget_reads_lengths_from_length_column():
    ds = _TinyDataset([10, 20, 30])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        stage='test',
        args=_args(length_column_name='length'),
        custom_args=_custom(test_max_tokens_per_batch=64),
        data_collator=lambda x: x,
        token_budget=64,
    )
    sampler = kwargs['batch_sampler']
    # Should have used the 'length' column values
    assert sampler._lengths == [10, 20, 30]


def test_eval_stage_reads_eval_field():
    ds = _TinyDataset([10])
    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        stage='eval',
        args=_args(),
        custom_args=_custom(eval_max_tokens_per_batch=128),
        data_collator=lambda x: x,
        token_budget=128,
    )
    assert isinstance(kwargs['batch_sampler'], TokenBudgetBatchSampler)
