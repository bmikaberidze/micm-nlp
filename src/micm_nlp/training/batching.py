"""Token-budget batch sampler for eval/test.

Yields variable-size batches whose total token count
(samples * padded_max_length_in_batch) does not exceed a fixed budget.
Samples are sorted ascending by length so adjacent indices have similar
padded sizes — this both minimizes padding waste and lets the budget be
nearly saturated.

Designed for eval/test only: variable batch size affects optimizer
dynamics in training but is semantically invisible for forward-only
inference.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


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
