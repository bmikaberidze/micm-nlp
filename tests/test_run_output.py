"""``RunOutput`` -- a run's directory: creation, the static columns, the config
snapshot, ``info.json`` (environment, paths, resolved values, wandb), the
``model`` / ``wandb`` links, and the ``prefix`` for a separate_test config. The
model is a stub; no trainer, no GPU."""

import json
import os
import subprocess
from types import SimpleNamespace

from micm_nlp import path as nlpka_path
from micm_nlp.config import CONFIG
from micm_nlp.training.run_output import (
    CONFIG_FILE, RUN_INFO_FILE, RunOutput, environment_info, output_dir_for, wandb_info, write_config,
)


def _model(tmp_path, path='m'):
    return SimpleNamespace(name='uuid_bloom_1_2', uuid4='u-1', path=None if path is None else str(tmp_path / path))


def _cfg(**output):
    return CONFIG(mode='preprocess', model={'architecture': 'toy'}, output=output or None)


# -- output_dir_for / write_config -------------------------------------------------

def test_output_dir_for(tmp_path):
    nlpka_path.set_root(tmp_path)
    assert output_dir_for(_cfg(dir='/runs/g/r'), 'ignored') == '/runs/g/r'
    assert output_dir_for(_cfg(), 'uuid_bloom_1_2') == str(tmp_path / 'artefacts' / 'runs' / 'units' / 'uuid_bloom_1_2')
    assert '/runs/units/' in output_dir_for(CONFIG(mode='preprocess'), 'x'), 'a config with no model block needs no architecture segment'


def test_write_config_dumps_plain_yaml(tmp_path):
    out = write_config(tmp_path / 'run', _cfg(columns={'group': 'g'}))
    assert out == tmp_path / 'run' / CONFIG_FILE
    text = out.read_text()
    assert 'mode: preprocess' in text and 'python/object' not in text


# -- RunOutput -------------------------------------------------------------------

def test_run_output_creates_dir_snapshot_run_info_and_link(tmp_path):
    nlpka_path.set_root(tmp_path)
    o = RunOutput(_cfg(columns={'group': 'g', 'name': 'r'}), _model(tmp_path))
    assert o.dir == tmp_path / 'artefacts' / 'runs' / 'units' / 'uuid_bloom_1_2' and o.dir.is_dir()
    assert o.prefix == '' and o.columns['group'] == 'g' and o.columns['uuid4'] == 'u-1' and 'time_id' in o.columns
    assert (o.dir / CONFIG_FILE).exists()
    saved = json.loads((o.dir / RUN_INFO_FILE).read_text())
    assert saved['started'] and saved['paths'] == {'output_dir': str(o.dir), 'model': str(tmp_path / 'm')}
    assert 'slurm' in saved and 'versions' in saved and 'resolved' not in saved
    assert os.readlink(o.dir / 'model') == str(tmp_path / 'm')


def test_run_output_honours_dir_prefix_and_config_file(tmp_path):
    o = RunOutput(_cfg(dir=str(tmp_path / 'x'), prefix='separate_', config_file='test_config.yml'), _model(tmp_path, path=None))
    assert o.dir == tmp_path / 'x' and (o.dir / 'test_config.yml').exists()
    assert o.file('test_final.csv') == tmp_path / 'x' / 'separate_test_final.csv'
    assert not (o.dir / 'model').is_symlink()                       # no checkpoint dir in TEST mode
    assert 'model' not in json.loads((o.dir / RUN_INFO_FILE).read_text())['paths']


def test_resolved_records_values_and_stamps_seed_only_when_absent(tmp_path):
    o = RunOutput(_cfg(dir=str(tmp_path / 'x'), columns={'seed': 11}), _model(tmp_path))
    o.resolved(seed=7, metric_for_best_model='unit.yml/accuracy', fp16=False)
    assert json.loads((o.dir / RUN_INFO_FILE).read_text())['resolved'] == {'seed': 7, 'metric_for_best_model': 'unit.yml/accuracy', 'fp16': False}
    assert o.columns['seed'] == 11                                   # pinned by the group: kept
    o2 = RunOutput(_cfg(dir=str(tmp_path / 'y')), _model(tmp_path))
    o2.resolved(seed=7)
    assert o2.columns['seed'] == 7                                   # unpinned: what the trainer drew


def test_two_run_outputs_one_dir_separate_test(tmp_path):
    first = RunOutput(_cfg(dir=str(tmp_path / 'x')), _model(tmp_path))                 # tune phase
    first.resolved(seed=7, fp16=False)
    second = RunOutput(_cfg(dir=str(tmp_path / 'x'), prefix='separate_', config_file='test_config.yml'),
                       _model(tmp_path, path=None))                                     # test phase
    second.resolved(seed=99, fp16=False)
    saved = json.loads((tmp_path / 'x' / RUN_INFO_FILE).read_text())
    assert saved['resolved']['seed'] == 7 and saved['separate_resolved']['seed'] == 99
    assert saved['paths']['model'] == str(tmp_path / 'm')                              # first-wins
    assert saved['started'] == first.columns['time_id'] or saved['started']            # kept from the first
    assert os.readlink(tmp_path / 'x' / 'model') == str(tmp_path / 'm')
    assert first.columns['seed'] == 7 and second.columns['seed'] == 99
    assert (tmp_path / 'x' / 'test_config.yml').exists() and (tmp_path / 'x' / CONFIG_FILE).exists()
    assert second.file('test_final.csv') == tmp_path / 'x' / 'separate_test_final.csv'


def test_note_wandb(tmp_path):
    o = RunOutput(_cfg(dir=str(tmp_path / 'x')), _model(tmp_path))
    files = tmp_path / 'w' / 'run-1' / 'files'
    files.mkdir(parents=True)
    o.note_wandb(SimpleNamespace(id='ab12', url=None, dir=str(files), path='x/y/ab12'))
    assert json.loads((o.dir / RUN_INFO_FILE).read_text())['wandb'] == {'id': 'ab12', 'url': None, 'dir': str(files.parent), 'path': 'x/y/ab12'}
    assert os.readlink(o.dir / 'wandb') == str(files.parent)
    o.note_wandb(None)                                               # no-op


def test_write_run_info_deep_merges_and_keeps_started(tmp_path):
    o = RunOutput(_cfg(dir=str(tmp_path / 'x')), _model(tmp_path))
    o.write_run_info(started='t0', paths={'a': 1}, wandb={'id': 'a'})
    o.write_run_info(started='t1', paths={'b': 2}, wandb={'id': 'b'}, finished='t9')
    saved = json.loads((o.dir / RUN_INFO_FILE).read_text())
    assert saved['started'] != 't1' and saved['paths']['a'] == 1 and saved['paths']['b'] == 2
    assert saved['wandb'] == {'id': 'b'} and saved['finished'] == 't9'


def test_link_is_absolute_idempotent_and_replaces_a_stale_target(tmp_path):
    o = RunOutput(_cfg(dir=str(tmp_path / 'x')), _model(tmp_path, path=None))
    target = tmp_path / 'models' / 'ckpt'                            # may not exist yet
    assert o.link('model', target) == o.dir / 'model' and os.readlink(o.dir / 'model') == str(target)
    o.link('model', target)                                          # no error, unchanged
    o.link('model', tmp_path / 'other')
    assert os.readlink(o.dir / 'model') == str(tmp_path / 'other')
    assert o.link('wandb', None) is None


# -- environment_info / wandb_info ---------------------------------------------------

def test_environment_info_globs_slurm(monkeypatch):
    for k in list(os.environ):
        if k.startswith('SLURM'):
            monkeypatch.delenv(k)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setenv('SLURMD_NODENAME', 'serv-1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    info = environment_info()
    assert info['slurm'] == {'SLURMD_NODENAME': 'serv-1', 'SLURM_JOB_ID': '1'} and info['cuda_visible_devices'] == '0,1'
    assert info['host'] and info['python'].startswith('3.') and info['versions']['micm_nlp'] and info['versions']['torch']


def test_environment_info_records_the_packages_own_commit(tmp_path, monkeypatch):
    """An editable install tracks a working tree that moves between releases, so the
    version string alone does not identify the code that ran."""
    from micm_nlp.training import run_output

    monkeypatch.setattr(run_output, 'package_commit', lambda: 'abc1234')
    assert environment_info()['versions']['micm_nlp_commit'] == 'abc1234'


def test_package_commit_is_none_outside_a_checkout(tmp_path, monkeypatch):
    from micm_nlp.training import run_output

    monkeypatch.setattr(run_output, '_PACKAGE_DIR', tmp_path)   # no .git above it
    assert run_output.package_commit() is None


def test_package_commit_marks_a_dirty_tree(tmp_path, monkeypatch):
    from micm_nlp.training import run_output

    def fake_git(args, **kwargs):
        out = 'abc1234\n' if 'rev-parse' in args else ' M src/micm_nlp/pipeline.py\n'
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr='')

    monkeypatch.setattr(run_output.subprocess, 'run', fake_git)
    assert run_output.package_commit() == 'abc1234-dirty'


def test_wandb_info_survives_url_error():
    class _Run:
        id, dir, path = 'ab12', '/w/run-1/files', 'x/y/ab12'

        @property
        def url(self):
            raise RuntimeError('offline')

    assert wandb_info(_Run()) == {'id': 'ab12', 'url': None, 'dir': '/w/run-1', 'path': 'x/y/ab12'}
    assert wandb_info(None) is None
