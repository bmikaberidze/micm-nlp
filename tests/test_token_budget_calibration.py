"""Tests for calibrate_token_budget (two-phase: halve-down + ramp-up + binary refine).

Most tests use a mock model that raises a controllable OOM-like exception
so the calibration loop can be exercised on CPU. One integration test runs
the real loop against a tiny GPT-2 — guarded by torch.cuda.is_available()."""

import os
import pytest
import torch

from micm_nlp.training.batching import calibrate_token_budget


class _OOMAbove:
    """Mock model: raises torch.cuda.OutOfMemoryError when input_ids.numel()
    >= threshold; returns a dummy output otherwise. The probe's input length
    is min(max_sample_len, budget), so numel scales with budget when
    max_sample_len is large enough."""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.calls = []   # records numel of each forward attempt

    def __call__(self, input_ids, attention_mask):
        self.calls.append(input_ids.numel())
        if input_ids.numel() >= self.threshold:
            raise torch.cuda.OutOfMemoryError(f'mock OOM at {input_ids.numel()}')
        return type('Out', (), {'logits': torch.zeros(input_ids.shape + (10,))})()

    def parameters(self):
        # calibrate_token_budget calls next(model.parameters()).device for device.
        yield torch.nn.Parameter(torch.zeros(1))


def test_ramps_to_hard_cap_when_model_never_ooms():
    """No OOM ever → Phase 1b doubles up until hard_cap; return cap × headroom."""
    model = _OOMAbove(threshold=10**9)
    budget = calibrate_token_budget(
        model=model, max_sample_len=16384,
        start=1024, floor=64, hard_cap=8192, tolerance=128,
    )
    # 1024 fits → ramp 2048, 4096, 8192 (all fit). Next would be 16384 > hard_cap → stop.
    assert budget == int(8192 * 0.85)


def test_phase1_halves_when_start_ooms_then_phase2_refines():
    """Start OOMs → halve until fit → binary refine within [fitted, failed]."""
    model = _OOMAbove(threshold=5000)   # OOM at numel >= 5000
    budget = calibrate_token_budget(
        model=model, max_sample_len=16384,
        start=65536, floor=64, hard_cap=131072, tolerance=64,
    )
    # Phase 1 halves 65536 → 32768 → 16384 → 8192 → 4096 (fits).
    # Phase 2 refines within (4096, 8192) to within 64 tokens of 5000.
    # Result is just below 5000, then × 0.85.
    expected_upper = int(5000 * 0.85)
    expected_lower = int((5000 - 256) * 0.85)
    assert expected_lower <= budget <= expected_upper


def test_phase1b_ramps_up_when_start_fits_then_phase2_refines():
    """Start fits → ramp up until OOM → binary refine."""
    model = _OOMAbove(threshold=10000)
    budget = calibrate_token_budget(
        model=model, max_sample_len=16384,
        start=1024, floor=64, hard_cap=65536, tolerance=64,
    )
    # 1024 fits → ramp 2048, 4096, 8192 (all fit) → 16384 OOMs.
    # Refine within (8192, 16384) to within 64 of 10000.
    expected_upper = int(10000 * 0.85)
    expected_lower = int((10000 - 256) * 0.85)
    assert expected_lower <= budget <= expected_upper


def test_refined_budget_beats_naive_halving():
    """The whole point of the refinement: closer to true optimum than halving alone.

    Naive halving on threshold=15000 with start=65536 would converge at 8192
    × 0.85 = 6963. The refined algorithm should get close to 15000 × 0.85 = 12750.
    """
    model = _OOMAbove(threshold=15000)
    budget = calibrate_token_budget(
        model=model, max_sample_len=16384,
        start=65536, floor=64, hard_cap=131072, tolerance=64,
    )
    naive_halving_result = int(8192 * 0.85)
    assert budget > naive_halving_result * 1.5, (
        f'refined budget {budget} should beat naive halving {naive_halving_result} by >1.5×'
    )
    assert budget <= int(15000 * 0.85)


def test_raises_when_floor_reached():
    """Everything OOMs → no fitting budget at or above floor → raise."""
    model = _OOMAbove(threshold=1)
    with pytest.raises(RuntimeError, match='no token budget'):
        calibrate_token_budget(
            model=model, max_sample_len=128,
            start=1024, floor=64, hard_cap=2048, tolerance=64,
        )


def test_probe_uses_min_of_max_sample_len_and_budget():
    """When max_sample_len < budget, probe sample length is clamped to
    max_sample_len. Longer probes are pointless: real batches won't exceed it."""
    model = _OOMAbove(threshold=10**9)
    calibrate_token_budget(
        model=model, max_sample_len=64,
        start=1024, floor=32, hard_cap=2048, tolerance=64,
    )
    # All probes should use numel = 64 (not budget).
    assert all(n == 64 for n in model.calls)


def test_max_sample_len_zero_treated_as_one():
    """Defensive: empty/degenerate dataset edge case doesn't crash."""
    model = _OOMAbove(threshold=10**9)
    budget = calibrate_token_budget(
        model=model, max_sample_len=0,
        start=128, floor=32, hard_cap=128, tolerance=32,
    )
    # 128 fits; can't ramp past hard_cap=128. Return 128 × 0.85.
    assert budget == int(128 * 0.85)


def test_tolerance_controls_refinement_precision():
    """Larger tolerance → fewer refine probes → looser final budget."""
    model_loose = _OOMAbove(threshold=10000)
    model_tight = _OOMAbove(threshold=10000)
    loose = calibrate_token_budget(
        model=model_loose, max_sample_len=16384,
        start=1024, floor=64, hard_cap=65536, tolerance=2048,
    )
    tight = calibrate_token_budget(
        model=model_tight, max_sample_len=16384,
        start=1024, floor=64, hard_cap=65536, tolerance=64,
    )
    # Tight refinement must use strictly MORE probes than loose. (>=) would
    # silently pass if a future change made both paths take the same count.
    assert len(model_tight.calls) > len(model_loose.calls)
    # Both must be ≤ true ceiling × headroom.
    assert loose <= int(10000 * 0.85)
    assert tight <= int(10000 * 0.85)


def test_hard_cap_probed_when_not_power_of_two_multiple_of_start():
    """When hard_cap isn't a clean power-of-two multiple of start, Phase 1b
    must probe hard_cap directly so the result reflects the true ceiling.

    With start=1024, hard_cap=5000, model fits everything (threshold huge):
    ramp probes 1024, 2048, 4096 (all fit). 8192 > hard_cap so ramp exits.
    Without the hard_cap probe fix, fitted=4096 and budget would be int(4096*0.85)=3481.
    With the fix, hard_cap=5000 is probed, fits, and budget = int(5000*0.85)=4250."""
    model = _OOMAbove(threshold=10**9)
    budget = calibrate_token_budget(
        model=model, max_sample_len=16384,
        start=1024, floor=64, hard_cap=5000, tolerance=64,
    )
    assert budget == int(5000 * 0.85), (
        f'expected {int(5000 * 0.85)}, got {budget} — hard_cap probably not probed'
    )


def test_hard_cap_probed_when_it_ooms_falls_through_to_refine():
    """If hard_cap itself OOMs, Phase 2 refines in [fitted_so_far, hard_cap]."""
    model = _OOMAbove(threshold=4500)
    budget = calibrate_token_budget(
        model=model, max_sample_len=16384,
        start=1024, floor=64, hard_cap=5000, tolerance=64,
    )
    # Ramp: 1024(fit), 2048(fit), 4096(fit), 8192>cap → probe hard_cap=5000.
    # 5000 OOMs (>= 4500). Phase 2 refines [4096, 5000] toward 4500.
    # Final result is just below 4500, × 0.85.
    expected_upper = int(4500 * 0.85)
    expected_lower = int((4500 - 256) * 0.85)
    assert expected_lower <= budget <= expected_upper


def test_rejects_start_below_floor():
    """start < floor is a configuration error, not an OOM."""
    model = _OOMAbove(threshold=10**9)
    with pytest.raises(ValueError, match='start.*below.*floor'):
        calibrate_token_budget(
            model=model, max_sample_len=128,
            start=64, floor=256, hard_cap=1024, tolerance=64,
        )


def test_rejects_hard_cap_below_start():
    """hard_cap < start is a configuration error."""
    model = _OOMAbove(threshold=10**9)
    with pytest.raises(ValueError, match='hard_cap.*below.*start'):
        calibrate_token_budget(
            model=model, max_sample_len=128,
            start=1024, floor=64, hard_cap=512, tolerance=64,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_real_calibration_on_tiny_gpt2():
    """Integration: tiny GPT-2 on CUDA, real calibration loop. Sanity check
    only — asserts the function returns a positive budget without raising."""
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=512, n_positions=256, n_embd=32,
        n_layer=2, n_head=4, n_inner=64,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg).to('cuda').eval()
    budget = calibrate_token_budget(model=model, max_sample_len=128)
    assert budget >= 256   # ≥ floor
