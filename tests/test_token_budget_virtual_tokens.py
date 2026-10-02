"""The per-row virtual-token count reaches the trainer from one source.

The runner computes it with PEFT.get_total_virtual_tokens and passes it to the
trainer, which adds it per row to the eval/test token budget (the length column
does not contain the virtual tokens a prompt learner prepends).
"""
from types import SimpleNamespace

import torch

from micm_nlp.models.peft import PEFT
from micm_nlp.training.trainers import custom_trainer_class_factory


class _Base:
    def __init__(self, *args, **kwargs):
        pass


class _Encoder(torch.nn.Module):
    def __init__(self, total):
        super().__init__()
        self.total_virtual_tokens = total


def test_trainer_stores_the_count():
    Trainer = custom_trainer_class_factory(_Base)
    assert Trainer(custom_args=None, virtual_tokens_per_row=20).virtual_tokens_per_row == 20


def test_trainer_defaults_to_zero():
    assert custom_trainer_class_factory(_Base)(custom_args=None).virtual_tokens_per_row == 0


def test_peft_count_reads_the_encoder():
    base = SimpleNamespace(_model=SimpleNamespace(prompt_encoder=_Encoder(20)))
    assert PEFT.get_total_virtual_tokens(base) == 20


def test_peft_count_unwraps_the_module_dict():
    encoders = torch.nn.ModuleDict({PEFT.prompt_encoder_key: _Encoder(30)})
    base = SimpleNamespace(_model=SimpleNamespace(prompt_encoder=encoders))
    assert PEFT.get_total_virtual_tokens(base) == 30


def test_peft_count_is_none_without_a_prompt_encoder():
    assert PEFT.get_total_virtual_tokens(SimpleNamespace(_model=torch.nn.Linear(2, 2))) is None
