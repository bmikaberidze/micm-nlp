"""CustomTrainerMixin.prediction_step always ignores past_key_values.

Composite (multimodal) configs leave keys_to_ignore_at_inference empty, so without
this the DynamicCache reaches accelerate's _pad_across_processes and raises.
"""
from types import SimpleNamespace

import pytest

from micm_nlp.training.trainers import CustomTrainerMixin


class _Base:
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        return ignore_keys


class _Trainer(CustomTrainerMixin, _Base):
    def __init__(self):
        self.custom_args = SimpleNamespace(generation_whitelist=None)


def _model(keys):
    return SimpleNamespace(config=SimpleNamespace(keys_to_ignore_at_inference=keys))


@pytest.mark.parametrize('config_keys,passed,expected', [
    ([], None, ['past_key_values']),                                   # composite config
    (None, None, ['past_key_values']),                                 # attribute unset
    (['past_key_values'], None, ['past_key_values']),                  # text config: no duplicate
    (['hidden_states'], None, ['hidden_states', 'past_key_values']),   # config keys kept
    (['hidden_states'], ['foo'], ['foo', 'past_key_values']),          # explicit keys win, cache added
])
def test_past_key_values_always_ignored(config_keys, passed, expected):
    got = _Trainer().prediction_step(_model(config_keys), {}, False, ignore_keys=passed)
    assert got == expected
