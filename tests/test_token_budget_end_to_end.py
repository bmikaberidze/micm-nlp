"""End-to-end: tiny GPT-2 + the project's custom trainer + token-budget mode.

Asserts batches actually vary in size when token budget mode is on. Skipped
without CUDA — the calibration probe uses ``torch.cuda.OutOfMemoryError``
detection which is meaningful only on a real device."""

import os
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='needs CUDA for real calibration'
)

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')


def test_token_budget_yields_variable_batch_sizes():
    from datasets import Dataset
    from transformers import GPT2Config, GPT2LMHeadModel
    from torch.utils.data import DataLoader
    from types import SimpleNamespace

    from micm_nlp.training.trainers import build_inference_dataloader_kwargs
    from micm_nlp.training.batching import calibrate_token_budget

    # Tiny model on GPU
    cfg = GPT2Config(
        vocab_size=512, n_positions=256, n_embd=32,
        n_layer=2, n_head=4, n_inner=64,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg).to('cuda').eval()

    # Synthetic dataset with widely varying lengths.
    lengths = [16, 32, 64, 128, 16, 32, 64, 128, 16, 200]
    rows = [
        {
            'input_ids': [1] * L,
            'attention_mask': [1] * L,
            'labels': [-100] * L,
            'length': L,
        }
        for L in lengths
    ]
    ds = Dataset.from_list(rows)

    # Calibrate against the real GPU.
    budget = calibrate_token_budget(
        model=model, max_sample_len=max(lengths), start=2048, floor=64,
    )
    assert budget >= 64

    args = SimpleNamespace(
        eval_batch_size=2, dataloader_num_workers=0,
        dataloader_pin_memory=False, dataloader_persistent_workers=False,
        dataloader_drop_last=False, dataloader_prefetch_factor=None,
        length_column_name='length', group_by_length=False,
    )

    # Collator that exposes pad_to_multiple_of, since the helper reads it
    # from data_collator (not args).
    class _Collator:
        def __init__(self, pad_to_multiple_of):
            self.pad_to_multiple_of = pad_to_multiple_of
        def __call__(self, batch):
            max_len = max(len(b['input_ids']) for b in batch)
            pm = self.pad_to_multiple_of
            pad_to = ((max_len + pm - 1) // pm) * pm
            return {
                'input_ids': torch.tensor([b['input_ids'] + [0]*(pad_to-len(b['input_ids'])) for b in batch]),
                'attention_mask': torch.tensor([b['attention_mask'] + [0]*(pad_to-len(b['attention_mask'])) for b in batch]),
            }

    collator = _Collator(pad_to_multiple_of=8)

    kwargs = build_inference_dataloader_kwargs(
        dataset=ds,
        args=args,
        data_collator=collator,
        token_budget=budget,
    )
    loader = DataLoader(ds, **kwargs)

    batch_sizes = []
    seen_samples = 0
    for batch in loader:
        bs = batch['input_ids'].shape[0]
        batch_sizes.append(bs)
        seen_samples += bs
        # Each batch must fit the calibrated budget.
        assert bs * batch['input_ids'].shape[1] <= budget

    # All 10 samples covered exactly once.
    assert seen_samples == len(lengths)
    # At least two different batch sizes — proves variable-size batching.
    assert len(set(batch_sizes)) >= 2, f'expected variable batch sizes, got {batch_sizes}'
