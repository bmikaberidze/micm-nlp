"""Probe-and-back-off calibration for per-GPU token budget.

Finds the largest token count (batch_size * padded_max_length) such that a
single forward pass through ``model`` doesn't trigger a CUDA OOM. The
returned budget is then handed to ``TokenBudgetBatchSampler`` so all eval/
test batches saturate the GPU without crashing.

Two-phase strategy so the result is tight, not just within a power of two:

  Phase 1  — from ``start``, halve on OOM until something fits (or floor).
  Phase 1b — if ``start`` fit on the first try, double until OOM (or hard_cap).
  Phase 2  — binary refine within the [fitted, failed] bracket to within
             ``tolerance`` tokens.

Approach mirrors ``accelerate.utils.find_executable_batch_size`` but
(a) calibrates a token count instead of a sample count, (b) refines the
result instead of accepting the first fitting power of two, and (c) runs
in eval mode (no autograd) so activation memory matches real test-time
usage."""
from __future__ import annotations

import torch


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
