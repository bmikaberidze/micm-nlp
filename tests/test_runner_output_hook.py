"""The trainer's output hooks on a bare instance: ``_setup_output`` builds the
``RunOutput`` (dir, snapshot, info.json, link) and ``_emit_order`` reads the
last dataloader's order. No model, no HF Trainer."""

import json
import os
from types import SimpleNamespace

from micm_nlp.config import CONFIG
from micm_nlp.training.run_output import CONFIG_FILE, RUN_INFO_FILE, RunOutput
from micm_nlp.training.runner import TRAINER


def _bare(tmp_path):
    t = object.__new__(TRAINER)
    t._config = CONFIG(mode='preprocess', model={'architecture': 'toy'},
                       output={'dir': str(tmp_path / 'run'), 'columns': {'group': 'g'}})
    t._model = SimpleNamespace(name='m', uuid4='u-1', path=str(tmp_path / 'm'))
    return t


def test_setup_output(tmp_path):
    t = _bare(tmp_path)
    t._setup_output()
    assert isinstance(t._output, RunOutput) and t._output.dir == tmp_path / 'run'
    assert (tmp_path / 'run' / CONFIG_FILE).exists()
    assert json.loads((tmp_path / 'run' / RUN_INFO_FILE).read_text())['paths']['model'] == str(tmp_path / 'm')
    assert os.readlink(tmp_path / 'run' / 'model') == str(tmp_path / 'm')
    assert t._output.columns['group'] == 'g' and t._output.columns['uuid4'] == 'u-1'


def test_run_returns_the_run_output(tmp_path, monkeypatch):
    """``run()`` hands back the same ``RunOutput`` it wrote through -- not the old
    ``full_shot`` / ``zero_shot`` pair."""
    from micm_nlp.training import runner

    t = _bare(tmp_path)
    t._config = CONFIG(mode='test', model={'architecture': 'toy', 'pretrained': {'source': 'local', 'name': 'm'}},
                       output={'dir': str(tmp_path / 'run')}, test={'run': True})
    t._model.hf = SimpleNamespace(wandb_run=None)
    t._setup_output()
    monkeypatch.setattr(runner.wandb, 'run', None)
    monkeypatch.setattr(t, '_init_wandb', lambda: None, raising=False)
    tested = []
    monkeypatch.setattr(t, '_test', lambda prefix, stage=None: tested.append((prefix, stage)), raising=False)

    assert t.run() is t._output
    assert tested == [('test', None)]
    assert 'finished' in json.loads((tmp_path / 'run' / RUN_INFO_FILE).read_text())


def test_emit_order(tmp_path):
    t = _bare(tmp_path)
    t.trainer = SimpleNamespace(_last_test_batch_sampler=SimpleNamespace(order=[2, 0, 1]))
    assert t._emit_order('_last_test_batch_sampler') == [2, 0, 1]
    t.trainer = SimpleNamespace(_last_test_batch_sampler=None)
    assert t._emit_order('_last_test_batch_sampler') is None
    assert t._emit_order('_no_such_sampler') is None
