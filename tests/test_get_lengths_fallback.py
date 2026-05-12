"""Regression test for _get_lengths: works whether or not the length
column is present in the dataset's columns. HF's _remove_unused_columns
strips columns not in _signature_columns or usable_columns; this helper
must gracefully fall back to computing lengths from input_ids."""

from types import SimpleNamespace

import pytest

from micm_nlp.training.trainers import _get_lengths


class _StubDataset:
    """Stand-in supporting both `len()` iteration (DataLoader protocol)
    and string column access (HF datasets-style)."""

    def __init__(self, *, lengths_column: list[int] | None, input_ids: list[list[int]]):
        self._has_length = lengths_column is not None
        self._lengths = lengths_column or []
        self._input_ids = input_ids

    @property
    def column_names(self):
        cols = ['input_ids']
        if self._has_length:
            cols.append('length')
        return cols

    def __len__(self):
        return len(self._input_ids)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == 'length':
                if not self._has_length:
                    raise KeyError('length')
                return self._lengths
            if idx == 'input_ids':
                return self._input_ids
            raise KeyError(idx)
        return {'input_ids': self._input_ids[idx], 'length': self._lengths[idx] if self._has_length else None}

    def __iter__(self):
        for ids in self._input_ids:
            yield {'input_ids': ids}


def test_uses_length_column_when_present():
    ds = _StubDataset(
        lengths_column=[10, 20, 30],
        input_ids=[[1]*10, [1]*20, [1]*30],
    )
    assert _get_lengths(ds, 'length') == [10, 20, 30]


def test_falls_back_to_input_ids_when_column_missing():
    ds = _StubDataset(
        lengths_column=None,   # column stripped
        input_ids=[[1]*7, [1]*42, [1]*3],
    )
    assert _get_lengths(ds, 'length') == [7, 42, 3]


def test_respects_custom_length_column_name():
    # If the user configured a different name, prefer that column.
    class _DS(_StubDataset):
        @property
        def column_names(self):
            return ['input_ids', 'token_count']
        def __getitem__(self, idx):
            if idx == 'token_count':
                return self._lengths
            return super().__getitem__(idx)
    ds = _DS(lengths_column=[10, 20], input_ids=[[1]*10, [1]*20])
    assert _get_lengths(ds, 'token_count') == [10, 20]


def test_empty_dataset_returns_empty_list():
    ds = _StubDataset(lengths_column=None, input_ids=[])
    assert _get_lengths(ds, 'length') == []
