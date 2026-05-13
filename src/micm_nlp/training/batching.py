"""Token-budget batching for eval/test: variable-size batches whose total
token count is bounded by an auto-detected per-GPU budget.

Two pieces:

  * ``calibrate_token_budget`` — binary search on sorted lengths. Finds
    the largest ``k`` such that ``(k, padded(L_k))`` (the widest shape
    the sampler can yield at budget = k × L_k) fits in VRAM. Real-shape
    probe, deterministic.

  * ``TokenBudgetBatchSampler`` — length-sorted ``BatchSampler`` that
    yields variable-size index batches bounded by that budget.

Designed for eval/test only: variable batch size affects optimizer
dynamics in training but is semantically invisible for forward-only
inference. ``training_args.group_by_length`` is ignored on this path —
the sampler always length-sorts internally.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


_HEADROOM = 0.85   # 15% safety margin for batch-shape / activation jitter


def calibrate_token_budget(
    *,
    model,
    lengths: Sequence[int],
    pad_multiple: int = 1,
    floor: int = 256,
    tolerance: int = 64,
) -> int:
    """Find the largest token budget such that the sampler's widest batch
    fits in VRAM.

    The sampler yields length-sorted batches packed to ``budget`` total
    tokens. For budget B and sorted lengths, the first (widest) batch has
    shape ``(k, padded(L_k))`` where k samples fit before the budget is
    exceeded. We binary-search k to find the largest fitting shape.

    Because the sampler's batches all have total_tokens ≤ B, and Aya's
    sdpa→flash attention is linear in seq_len (not quadratic), and the
    LayerNorm-fp32-cast is linear in total_tokens, every batch at this
    budget will fit if the calibration shape fits. Guaranteed.

    Args:
        model: callable model on its target device.
        lengths: per-sample sequence lengths (the dataset's length column).
        pad_multiple: collator's pad_to_multiple_of (1 = no rounding).
        floor: raise if no fitting shape produces a budget at or above this.
        tolerance: binary search stops when (hi - lo) <= this many samples.

    Returns:
        int: token budget = (largest fitting k) × padded(L_k) × _HEADROOM.

    Raises:
        ValueError: empty lengths.
        RuntimeError: not even shape (1, padded(L_1)) fits, or returned
            budget < floor.
    """
    if not lengths:
        raise ValueError('lengths must be non-empty')
    if pad_multiple < 1:
        raise ValueError(f'pad_multiple must be >= 1, got {pad_multiple}')

    def _padded(L: int) -> int:
        return ((L + pad_multiple - 1) // pad_multiple) * pad_multiple

    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    device = next(model.parameters()).device

    def _probe(k: int) -> bool:
        """Test shape (k, padded(sorted_lens[k-1])) — what the sampler's
        first batch would look like if budget = k × that length."""
        L = _padded(sorted_lens[k - 1])
        try:
            ids = torch.zeros(k, L, dtype=torch.long, device=device)
            attn = torch.ones_like(ids)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=attn)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return False

    # Verify the smallest possible batch fits at all.
    if not _probe(1):
        raise RuntimeError(
            f'shape (1, {_padded(sorted_lens[0])}) does not fit on {device}; '
            f'GPU too small for this model+dataset'
        )

    # Binary search: largest k in [1, n] where (k, padded(sorted_lens[k-1])) fits.
    lo, hi = 1, n
    while hi - lo > tolerance:
        mid = (lo + hi + 1) // 2
        if _probe(mid):
            lo = mid
        else:
            hi = mid - 1

    budget_raw = lo * _padded(sorted_lens[lo - 1])
    if budget_raw < floor:
        raise RuntimeError(
            f'largest fitting budget {budget_raw} is below floor {floor}'
        )
    return int(budget_raw * _HEADROOM)


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Length-sorted batch sampler with a per-batch token cap.

    Args:
        lengths: per-sample sequence lengths (ints), same length as the dataset.
        token_budget: max (batch_size * padded_max_length) per batch.
        pad_multiple: alignment for padded length (matches data_collator's
            ``pad_to_multiple_of``); 1 disables rounding.

    Notes:
        * A sample longer than ``token_budget`` is still yielded — as a
          singleton batch. We don't silently drop test samples; the caller
          must either raise the budget or accept the OOM risk for that batch.
        * ``__len__`` is exact (computed once by simulating one iteration).
    """

    def __init__(
        self,
        lengths: Sequence[int],
        token_budget: int,
        pad_multiple: int = 1,
    ) -> None:
        if token_budget <= 0:
            raise ValueError(f'token_budget must be positive, got {token_budget}')
        if pad_multiple < 1:
            raise ValueError(f'pad_multiple must be >= 1, got {pad_multiple}')
        self._lengths = list(lengths)
        self._token_budget = token_budget
        self._pad_multiple = pad_multiple
        self._order = sorted(range(len(self._lengths)), key=lambda i: self._lengths[i])
        self._cached_len: int | None = None

    def _padded(self, length: int) -> int:
        pm = self._pad_multiple
        return ((length + pm - 1) // pm) * pm

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        batch_max = 0
        for idx in self._order:
            L = self._padded(self._lengths[idx])
            new_max = max(batch_max, L)
            new_size = len(batch) + 1
            if batch and new_size * new_max > self._token_budget:
                yield batch
                batch = [idx]
                batch_max = L
            else:
                batch.append(idx)
                batch_max = new_max
        if batch:
            yield batch

    def __len__(self) -> int:
        if self._cached_len is None:
            self._cached_len = sum(1 for _ in self.__iter__())
        return self._cached_len
