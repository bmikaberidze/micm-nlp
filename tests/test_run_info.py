"""run.json and the artefact links: the environment glob, wandb capture, merge
semantics of write_run_info, idempotent symlinks, and the two TRAINER hooks on a
bare instance. No model, no wandb process."""

import json
import os
from types import SimpleNamespace

from micm_nlp.config import CONFIG
from micm_nlp.evals.results import RUN_INFO_FILE, ResultsWriter, environment_info, wandb_info
from micm_nlp.training.runner import TRAINER


def test_environment_info_globs_slurm(monkeypatch):
    for k in list(os.environ):
        if k.startswith('SLURM'):
            monkeypatch.delenv(k)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '7')
    monkeypatch.setenv('SLURMD_NODENAME', 'serv-1')      # no underscore after SLURM: still caught
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    info = environment_info()
    assert info['slurm'] == {'SLURMD_NODENAME': 'serv-1', 'SLURM_ARRAY_TASK_ID': '7', 'SLURM_JOB_ID': '1'}
    assert info['cuda_visible_devices'] == '0,1'
    assert info['host'] and info['python'].startswith('3.')
    assert set(info['versions']) == {'micm_nlp', 'torch', 'transformers', 'peft', 'datasets'}
    assert info['versions']['torch']


def test_environment_info_without_slurm(monkeypatch):
    for k in list(os.environ):
        if k.startswith('SLURM'):
            monkeypatch.delenv(k)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    info = environment_info()
    assert info['slurm'] == {} and info['cuda_visible_devices'] is None


def test_wandb_info():
    run = SimpleNamespace(id='ab12', url='https://wandb.ai/x/y/runs/ab12', dir='/w/run-1/files', path='x/y/ab12')
    assert wandb_info(run) == {'id': 'ab12', 'url': 'https://wandb.ai/x/y/runs/ab12', 'dir': '/w/run-1', 'path': 'x/y/ab12'}
    assert wandb_info(None) is None


def test_wandb_info_survives_url_error():
    class _Run:
        id, dir, path = 'ab12', '/w/run-1/files', 'x/y/ab12'

        @property
        def url(self):
            raise RuntimeError('offline')

    assert wandb_info(_Run())['url'] is None


def test_write_run_info_merges(tmp_path):
    w = ResultsWriter(tmp_path / 'run')
    w.write_run_info(started='t0', slurm={'SLURM_JOB_ID': '1'})
    w.write_run_info(finished='t1')
    saved = json.loads((tmp_path / 'run' / RUN_INFO_FILE).read_text())
    assert saved == {'started': 't0', 'slurm': {'SLURM_JOB_ID': '1'}, 'finished': 't1'}


def test_link_is_absolute_and_idempotent(tmp_path):
    w = ResultsWriter(tmp_path / 'run')
    target = tmp_path / 'models' / 'ckpt'            # does not exist yet: a dangling link is fine
    assert w.link('model', target) == tmp_path / 'run' / 'model'
    assert os.readlink(tmp_path / 'run' / 'model') == str(target)
    assert w.link('model', target) == tmp_path / 'run' / 'model'   # second call: no error, unchanged
    assert w.link('wandb', None) is None
    assert not (tmp_path / 'run' / 'wandb').is_symlink()


def _bare(tmp_path):
    t = object.__new__(TRAINER)
    t._config = CONFIG(mode='preprocess', model={'architecture': 'toy'})
    t._model = SimpleNamespace(eval_path=str(tmp_path / 'run'), uuid4='u-1', path=str(tmp_path / 'm'),
                               hf=SimpleNamespace(wandb_run=None))
    return t


def test_setup_results_writes_run_info_and_model_link(tmp_path):
    t = _bare(tmp_path)
    t._setup_results()
    saved = json.loads((tmp_path / 'run' / RUN_INFO_FILE).read_text())
    assert saved['started']
    assert saved['paths'] == {'run_dir': str(tmp_path / 'run'), 'model': str(tmp_path / 'm')}
    assert 'slurm' in saved and 'versions' in saved and 'finished' not in saved
    assert os.readlink(tmp_path / 'run' / 'model') == str(tmp_path / 'm')


def test_note_wandb(tmp_path):
    t = _bare(tmp_path)
    t._results = t._setup_results()
    files = tmp_path / 'w' / 'run-1' / 'files'
    files.mkdir(parents=True)
    t._model.hf.wandb_run = SimpleNamespace(id='ab12', url=None, dir=str(files), path='x/y/ab12')
    t._note_wandb()
    saved = json.loads((tmp_path / 'run' / RUN_INFO_FILE).read_text())
    assert saved['wandb'] == {'id': 'ab12', 'url': None, 'dir': str(files.parent), 'path': 'x/y/ab12'}
    assert os.readlink(tmp_path / 'run' / 'wandb') == str(files.parent)


def test_note_wandb_without_a_run(tmp_path):
    t = _bare(tmp_path)
    t._results = t._setup_results()
    t._note_wandb()                                   # hf.wandb_run is None and wandb.run is None
    assert 'wandb' not in json.loads((tmp_path / 'run' / RUN_INFO_FILE).read_text())


def test_write_run_info_deep_merges_and_keeps_started(tmp_path):
    w = ResultsWriter(tmp_path / 'run')
    w.write_run_info(started='t0', paths={'run_dir': '/r', 'model': '/m'}, wandb={'id': 'a'})
    w.write_run_info(started='t1', paths={'run_dir': '/r'}, wandb={'id': 'b'})
    saved = json.loads((tmp_path / 'run' / RUN_INFO_FILE).read_text())
    assert saved['started'] == 't0'
    assert saved['paths'] == {'run_dir': '/r', 'model': '/m'}
    assert saved['wandb'] == {'id': 'b'}


def test_link_replaces_a_link_to_a_different_target(tmp_path):
    w = ResultsWriter(tmp_path / 'run')
    w.link('wandb', tmp_path / 'w1')
    w.link('wandb', tmp_path / 'w2')
    assert os.readlink(tmp_path / 'run' / 'wandb') == str(tmp_path / 'w2')


def test_two_trainers_one_run_dir(tmp_path):
    first = _bare(tmp_path)                       # tune phase: has a checkpoint dir
    first._setup_results()
    second = _bare(tmp_path)
    second._model = SimpleNamespace(eval_path=str(tmp_path / 'run'), uuid4='u-2', path=None,
                                    hf=SimpleNamespace(wandb_run=None))
    second._setup_results()
    saved = json.loads((tmp_path / 'run' / RUN_INFO_FILE).read_text())
    assert saved['paths']['model'] == str(tmp_path / 'm')     # first phase's checkpoint kept
    assert os.readlink(tmp_path / 'run' / 'model') == str(tmp_path / 'm')
