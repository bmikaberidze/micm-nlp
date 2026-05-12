"""Token-budget batching for eval/test: variable-size batches whose total
token count is bounded by an auto-detected per-GPU budget.

Two pieces:

  * ``calibrate_token_budget`` — empirical probe that finds the largest
    token budget (samples × padded_max_length) that fits in VRAM. Two-phase
    (halve-down + ramp-up + binary refine) with 15% headroom.

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
    max_sample_len: int,
    start: int = 65536,
    floor: int = 256,
    hard_cap: int = 524288,
    tolerance: int = 256,
) -> int:
    """Probe ``model`` to find the largest token budget that fits in VRAM.

    Args:
        model: callable with ``model(input_ids=..., attention_mask=...)``
            signature. Must already be on the target device.
        max_sample_len: longest sequence the dataset will produce. Probe
            samples are capped at this length (probing longer is pointless
            — real batches won't exceed it).
        start: initial token budget to attempt. Halved on OOM (Phase 1) or
            doubled on fit (Phase 1b).
        floor: minimum acceptable budget; raise if no budget at or above
            this fits.
        hard_cap: ceiling on Phase 1b ramp-up. If the model never OOMs
            even at ``hard_cap``, the function returns ``hard_cap × _HEADROOM``.
            (Phase 1b ramps by doubling; the cap itself is then probed once
            so non-power-of-two hard_caps still get refined to the true
            ceiling.)
        tolerance: Phase 2 stops when (failed - fitted) ≤ this many tokens.
            Smaller → more probes, tighter result.

    Returns:
        Token budget with ``_HEADROOM`` already applied (85% of the largest
        budget the probe confirmed fits).

    Raises:
        RuntimeError: if no budget >= ``floor`` fits.
        ValueError: if ``start < floor`` or ``hard_cap < start``.
    """
    if start < floor:
        raise ValueError(
            f'start={start} is below floor={floor}; the probe loop would '
            f'never execute. Configuration error.'
        )
    if hard_cap < start:
        raise ValueError(
            f'hard_cap={hard_cap} is below start={start}; Phase 1b ramp-up '
            f'would never execute. Configuration error.'
        )

    device = next(model.parameters()).device

    def _probe(budget: int) -> bool:
        try:
            n_tokens = max(1, min(int(max_sample_len) or 1, budget))
            input_ids = torch.zeros(1, n_tokens, dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return False

    # Phase 1: halve from ``start`` until something fits.
    fitted: int | None = None
    failed: int | None = None
    budget = start
    while budget >= floor:
        if _probe(budget):
            fitted = budget
            break
        failed = budget
        budget //= 2

    if fitted is None:
        first_probe_len = max(1, min(int(max_sample_len) or 1, start))
        raise RuntimeError(
            f'no token budget >= floor={floor} fits on {device} '
            f'(start={start}, first_probe_len={first_probe_len}). GPU may be '
            f'heavily contended or {first_probe_len}-token forward is impossible.'
        )

    # Phase 1b: if ``start`` fit immediately, double until OOM or hard_cap.
    if failed is None:
        budget = fitted * 2
        while budget <= hard_cap:
            if _probe(budget):
                fitted = budget
                budget *= 2
            else:
                failed = budget
                break
        if failed is None:
            # Ramp exited because budget*2 would exceed hard_cap, but we may
            # not have actually probed hard_cap itself (only powers of two of
            # start). Probe it directly so Phase 2 has a real bracket if it
            # doesn't fit, or so we return the true ceiling if it does.
            if fitted >= hard_cap:
                return int(fitted * _HEADROOM)
            if _probe(hard_cap):
                return int(hard_cap * _HEADROOM)
            failed = hard_cap
            # Fall through to Phase 2 binary refine in [fitted, hard_cap].

    # Phase 2: binary refine within [fitted, failed].
    lo, hi = fitted, failed
    while hi - lo > tolerance:
        mid = (lo + hi) // 2
        if _probe(mid):
            lo = mid
        else:
            hi = mid

    return int(lo * _HEADROOM)


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
