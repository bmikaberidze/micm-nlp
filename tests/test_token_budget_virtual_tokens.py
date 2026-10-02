"""_virtual_tokens_per_row: the per-row token count a prompt learner adds, which the
eval/test token budget must count (they are absent from the length column)."""
from types import SimpleNamespace

import torch

from micm_nlp.training.trainers import _virtual_tokens_per_row


class _Encoder(torch.nn.Module):
    def __init__(self, total):
        super().__init__()
        self.total_virtual_tokens = total


def test_plain_model_has_none():
    assert _virtual_tokens_per_row(torch.nn.Linear(2, 2)) == 0


def test_reads_encoder_total_virtual_tokens():
    model = SimpleNamespace(prompt_encoder=_Encoder(20))
    assert _virtual_tokens_per_row(model) == 20


def test_unwraps_peft_module_dict():
    model = SimpleNamespace(prompt_encoder=torch.nn.ModuleDict({'default': _Encoder(30)}))
    assert _virtual_tokens_per_row(model) == 30


def test_falls_back_to_prompt_learning_config():
    config = SimpleNamespace(is_prompt_learning=True, num_virtual_tokens=10, num_transformer_submodules=1)
    model = SimpleNamespace(prompt_encoder=torch.nn.ModuleDict({'default': torch.nn.Embedding(10, 4)}),
                            active_peft_config=config)
    assert _virtual_tokens_per_row(model) == 10
