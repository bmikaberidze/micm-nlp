"""Unit tests for TokenBudgetBatchSampler.

Verifies length-sorted batching with a max-tokens-per-batch cap, where
"tokens per batch" = (batch_size) * (padded_max_length_in_batch). No PyTorch
dependency beyond the Sampler protocol; pure-Python tests."""

import random

import pytest

from micm_nlp.training.batching import TokenBudgetBatchSampler


def test_empty_dataset_yields_nothing():
    sampler = TokenBudgetBatchSampler(lengths=[], token_budget=1024, pad_multiple=8)
    assert list(sampler) == []
    assert len(sampler) == 0


def test_single_sample_one_batch():
    sampler = TokenBudgetBatchSampler(lengths=[100], token_budget=1024, pad_multiple=8)
    batches = list(sampler)
    assert batches == [[0]]
    assert len(sampler) == 1


def test_uniform_length_fills_to_budget():
    # 10 samples of length 100 → padded to 104 (pad_multiple=8).
    # Budget 1024 → max 9 per batch (9*104=936 ≤ 1024, 10*104=1040 > 1024).
    sampler = TokenBudgetBatchSampler(
        lengths=[100] * 10, token_budget=1024, pad_multiple=8,
    )
    batches = list(sampler)
    assert all(len(b) <= 9 for b in batches)
    assert sum(len(b) for b in batches) == 10
    # No batch exceeds budget:
    for batch in batches:
        max_len = max(((100 + 7) // 8) * 8 for _ in batch)
        assert len(batch) * max_len <= 1024


def test_variable_lengths_sorted_ascending():
    # Lengths in random order; sampler should sort ascending before batching.
    lengths = [200, 50, 500, 100, 300]
    sampler = TokenBudgetBatchSampler(
        lengths=lengths, token_budget=1024, pad_multiple=8,
    )
    batches = list(sampler)
    # Each batch's indices must point to samples of similar length (sorted order):
    for batch in batches:
        batch_lens = [lengths[i] for i in batch]
        assert batch_lens == sorted(batch_lens), f'batch {batch} not length-sorted: {batch_lens}'


def test_no_batch_exceeds_budget():
    random.seed(0)
    lengths = [random.randint(50, 800) for _ in range(50)]
    sampler = TokenBudgetBatchSampler(
        lengths=lengths, token_budget=2048, pad_multiple=8,
    )
    for batch in sampler:
        padded_lens = [((lengths[i] + 7) // 8) * 8 for i in batch]
        max_len = max(padded_lens)
        assert len(batch) * max_len <= 2048


def test_all_samples_covered_exactly_once():
    lengths = [50, 100, 200, 50, 800, 300, 100, 50]
    sampler = TokenBudgetBatchSampler(
        lengths=lengths, token_budget=512, pad_multiple=8,
    )
    seen = []
    for batch in sampler:
        seen.extend(batch)
    assert sorted(seen) == list(range(len(lengths)))


def test_len_matches_actual_iteration():
    random.seed(1)
    lengths = [random.randint(50, 1000) for _ in range(100)]
    sampler = TokenBudgetBatchSampler(
        lengths=lengths, token_budget=2048, pad_multiple=8,
    )
    declared = len(sampler)
    actual = sum(1 for _ in sampler)
    assert declared == actual


def test_oversized_sample_yielded_as_singleton():
    # Sample exceeds budget — has to be its own batch (cannot drop test samples).
    lengths = [100, 5000, 100]
    sampler = TokenBudgetBatchSampler(
        lengths=lengths, token_budget=1024, pad_multiple=8,
    )
    batches = list(sampler)
    # Sorted ascending: [100, 100, 5000]. First two fit together; the 5000 sits alone.
    batches_by_lengths = [[lengths[i] for i in b] for b in batches]
    # Oversized sample is alone:
    assert [5000] in batches_by_lengths
    # Two same-length 100s share a batch:
    assert [100, 100] in batches_by_lengths


def test_pad_multiple_one_no_rounding():
    sampler = TokenBudgetBatchSampler(
        lengths=[100, 100, 100], token_budget=300, pad_multiple=1,
    )
    batches = list(sampler)
    # Exactly 3 fit (3*100 = 300), so one batch.
    assert batches == [[0, 1, 2]]


def test_rejects_non_positive_token_budget():
    with pytest.raises(ValueError, match='token_budget'):
        TokenBudgetBatchSampler(lengths=[100], token_budget=0, pad_multiple=8)
    with pytest.raises(ValueError, match='token_budget'):
        TokenBudgetBatchSampler(lengths=[100], token_budget=-1, pad_multiple=8)


def test_rejects_pad_multiple_below_one():
    with pytest.raises(ValueError, match='pad_multiple'):
        TokenBudgetBatchSampler(lengths=[100], token_budget=1024, pad_multiple=0)
    with pytest.raises(ValueError, match='pad_multiple'):
        TokenBudgetBatchSampler(lengths=[100], token_budget=1024, pad_multiple=-2)


def test_extra_tokens_per_sample_counted_per_row():
    # 10 rows of length 100 (pad 8 -> 104) + 20 virtual tokens = 124 per row.
    # Budget 1024 -> 8 rows (8*124=992 <= 1024, 9*124=1116 > 1024); without the
    # extra it would be 9 (9*104=936), i.e. 1116 real tokens -- over the budget.
    sampler = TokenBudgetBatchSampler(lengths=[100] * 10, token_budget=1024, pad_multiple=8,
                                      extra_tokens_per_sample=20)
    batches = list(sampler)
    assert [len(b) for b in batches] == [8, 2]


def test_no_batch_exceeds_budget_with_extra_tokens():
    rng = random.Random(0)
    lengths = [rng.randint(20, 400) for _ in range(500)]
    budget, extra, pm = 4096, 20, 4
    sampler = TokenBudgetBatchSampler(lengths=lengths, token_budget=budget, pad_multiple=pm,
                                      extra_tokens_per_sample=extra)
    for batch in sampler:
        widest = max(((lengths[i] + pm - 1) // pm) * pm for i in batch)
        assert len(batch) == 1 or len(batch) * (widest + extra) <= budget
