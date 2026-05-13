"""Tests for calibrate_token_budget (binary search on sorted lengths).

Most tests use a mock model that raises a controllable OOM-like exception
so the calibration loop can be exercised on CPU. One integration test runs
the real loop against a tiny GPT-2 — guarded by torch.cuda.is_available()."""

import os
import pytest
import torch

from micm_nlp.training.batching import calibrate_token_budget

_HEADROOM = 0.85


class _OOMAbove:
    """Mock model: raises torch.cuda.OutOfMemoryError when
    input_ids.numel() (= k * L) >= threshold. Models the LayerNorm-fp32
    cost which is linear in total tokens."""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.calls = []   # list of (k, L) shape tuples

    def __call__(self, input_ids, attention_mask, labels=None):
        k, L = input_ids.shape
        self.calls.append((k, L))
        if input_ids.numel() >= self.threshold:
            raise torch.cuda.OutOfMemoryError(f'mock OOM at numel={input_ids.numel()}')
        return type('Out', (), {'logits': torch.zeros(input_ids.shape + (10,))})()

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))


def test_empty_lengths_raises_value_error():
    model = _OOMAbove(threshold=10**9)
    with pytest.raises(ValueError, match='lengths must be non-empty'):
        calibrate_token_budget(model=model, lengths=[])


def test_single_sample_returns_one_sample_times_padded_length():
    """lengths=[100], pad_multiple=8: padded(100)=104, budget=int(1*104*0.85)=88."""
    model = _OOMAbove(threshold=10**9)
    budget = calibrate_token_budget(model=model, lengths=[100], pad_multiple=8, floor=1)
    assert budget == int(1 * 104 * _HEADROOM)


def test_smallest_shape_doesnt_fit_raises_runtime_error():
    """threshold=1: even (1, padded(L_0)) OOMs -> RuntimeError."""
    model = _OOMAbove(threshold=1)
    with pytest.raises(RuntimeError, match='does not fit'):
        calibrate_token_budget(model=model, lengths=[100])


def test_binary_search_finds_largest_fitting():
    """Hand-traced: 200 samples of length 50, then 1 of length 100.
    Sorted: [50]*200 + [100]. padded(50)=56, padded(100)=104, pad_multiple=8.

    At threshold=5000 (mock OOMs when k * L >= threshold):
      k=89: 89 * 56 = 4984 (fits)
      k=90: 90 * 56 = 5040 (OOM)
    True max_k = 89. With tolerance=64, returned lo is in [25, 89].
    Expected budget range: [int((89-64) * 56 * 0.85), int(89 * 56 * 0.85)]
                         = [1190, 4236].
    """
    threshold = 5000
    lengths = [50] * 200 + [100]
    model = _OOMAbove(threshold=threshold)
    budget = calibrate_token_budget(
        model=model, lengths=lengths, pad_multiple=8, floor=1, tolerance=64,
    )
    # Hand-traced bounds:
    assert 1190 <= budget <= 4236, f'got {budget}'

    # Also verify by tightening tolerance: smaller tolerance must give a
    # tighter (higher) result.
    model2 = _OOMAbove(threshold=threshold)
    tight = calibrate_token_budget(
        model=model2, lengths=lengths, pad_multiple=8, floor=1, tolerance=4,
    )
    assert tight >= budget, f'tighter tolerance should not give a smaller budget; tight={tight}, loose={budget}'
    assert tight >= int(85 * 56 * 0.85)  # within tolerance=4 of optimum 89


def test_handles_small_dataset_below_tolerance():
    """Critical regression: when n <= tolerance, the binary search loop
    doesn't execute. Without the post-loop hi-probe, lo stays at 1 and
    the function under-reports budget by orders of magnitude."""
    model = _OOMAbove(threshold=10**9)  # never OOM
    n = 10  # well below default tolerance=64
    lengths = [50] * n
    budget = calibrate_token_budget(
        model=model, lengths=lengths, pad_multiple=8, floor=1, tolerance=64,
    )
    # All shapes fit, so largest k = n = 10. padded(50)=56.
    # Expected: int(10 * 56 * 0.85) = 476.
    assert budget == int(10 * 56 * _HEADROOM)


def test_pad_multiple_respected():
    """Probe shapes must use padded lengths. With pad_multiple=16, length=10
    should be padded to 16, so the probe shape is (1, 16) not (1, 10)."""
    model = _OOMAbove(threshold=10**9)
    calibrate_token_budget(model=model, lengths=[10], pad_multiple=16, floor=1)
    # With n=1, only _probe(1) is called (lo=hi=1 from the start, no loop).
    # The initial _probe(1) and the final result use padded length.
    assert all(L == 16 for (k, L) in model.calls), (
        f'expected all probes to use padded length 16, got {model.calls}'
    )


def test_below_floor_raises():
    """threshold=200, lengths=[10], floor=1000: budget_raw=int(1*16*0.85)=13 < 1000."""
    model = _OOMAbove(threshold=200)
    with pytest.raises(RuntimeError, match='below floor'):
        calibrate_token_budget(model=model, lengths=[10], pad_multiple=16, floor=1000)


def test_real_calibration_on_tiny_gpt2():
    """Integration: tiny GPT-2 on CUDA, real calibration loop. Sanity check
    only — asserts the function returns a positive budget without raising."""
    pytest.importorskip('transformers')
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA')

    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=512, n_positions=256, n_embd=32,
        n_layer=2, n_head=4, n_inner=64,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg).to('cuda').eval()
    lengths = [32, 64, 128, 64, 32, 96, 128, 48]
    budget = calibrate_token_budget(model=model, lengths=lengths, pad_multiple=8)
    assert budget > 0
