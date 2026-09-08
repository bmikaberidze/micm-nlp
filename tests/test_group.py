# tests/test_group.py
"""The group runner: loading/validation, entry selection, overrides, seed,
resolution into a config + context, run-dir layout with its config snapshot,
runner loading and dispatch. A stub runner records what it was called with; no
model is built."""

from pathlib import Path

import pytest
import yaml

from micm_nlp import group as grp
from micm_nlp import path as nlpka_path
from micm_nlp.config import CONFIG
from micm_nlp.group import (
    RunContext, apply_override, apply_seed, config_seed, load_group, load_runner, resolve_entry,
    run_group, run_solo, scalar_columns, select_indices,
)

CALLS = []


def stub_runner(config, ctx):
    CALLS.append((config, ctx))


def _unit(tmp_path, name='unit.yml', **extra):
    body = {'mode': 'preprocess', 'model': {'architecture': 'toy'},
            'peft': {'encoder_hidden_size': 64}, 'training_args': {'args': {'seed': None}}, **extra}
    p = tmp_path / name
    p.write_text(yaml.safe_dump(body))
    return p


def _group(tmp_path, runs, configs=None, stem='g1'):
    # Writes unit.yml and eval.yml fresh every time; call _unit() AFTER this
    # if a test needs a customised unit config.
    _unit(tmp_path)
    _unit(tmp_path, 'eval.yml')
    body = {'configs': configs or {'u': './unit.yml', 'e': './eval.yml'}, 'runs': runs}
    p = tmp_path / f'{stem}.yml'
    p.write_text(yaml.safe_dump(body))
    return p


# -- load_group ------------------------------------------------------------------

def test_load_group_resolves_paths_relative_to_file(tmp_path):
    g = load_group(_group(tmp_path, [{'config': 'u', 'name': 'a'}]))
    assert g['group'] == 'g1'
    assert g['configs']['u'] == str(tmp_path / 'unit.yml')
    assert g['runs'][0]['name'] == 'a'


@pytest.mark.parametrize('runs, msg', [
    ([], 'runs'),
    ([{'config': 'nope', 'name': 'a'}], "config 'nope'"),
    ([{'config': 'u'}], 'needs a name'),
    ([{'config': 'u', 'name': 'a'}, {'config': 'u', 'name': 'a'}], 'unique'),
    ([{'config': 'u', 'name': 'a', 'seed': '11'}], 'seed must be an int'),
    ([{'config': 'u', 'name': 'a', 'seed': True}], 'seed must be an int'),
    ([{'config': 'u', 'name': 'a', 'overrides': ['x']}], 'overrides must be a mapping'),
    ([{'config': 'u', 'name': 'a', 'separate_test': {'config': 'nope'}}], "config 'nope'"),
    ([{'config': 'u', 'name': 'a', 'separate_test': {'config': 'e', 'overrides': 3}}], 'overrides must be a mapping'),
    ([{'config': 'u', 'name': 'a', 'index': 7}], 'reserved column'),
    ([{'config': 'u', 'name': 'a', 'time_id': 'x'}], 'reserved column'),
    ([{'config': 'u', 'name': 'a/b'}], 'single path segment'),
])
def test_load_group_validation(tmp_path, runs, msg):
    with pytest.raises(ValueError, match=msg):
        load_group(_group(tmp_path, runs))


def test_load_group_rejects_reserved_stem_and_bad_config_paths(tmp_path):
    with pytest.raises(ValueError, match='must not start with'):
        load_group(_group(tmp_path, [{'config': 'u', 'name': 'a'}], stem='_bad'))
    with pytest.raises(ValueError, match='missing.yml'):
        load_group(_group(tmp_path, [{'config': 'u', 'name': 'a'}], configs={'u': './missing.yml'}))
    with pytest.raises(ValueError, match='must be a path'):
        load_group(_group(tmp_path, [{'config': 'u', 'name': 'a'}], configs={'u': 3}))


# -- select_indices --------------------------------------------------------------

def test_select_env_wins(monkeypatch):
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
    assert select_indices(4, task_id=0) == [2]


def test_select_task_id(monkeypatch):
    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    assert select_indices(4, task_id=3) == [3]


def test_select_all(monkeypatch):
    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    assert select_indices(3, task_id=None) == [0, 1, 2]


def test_select_out_of_range(monkeypatch):
    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    with pytest.raises(ValueError, match='range'):
        select_indices(3, task_id=3)


# -- overrides / seed ------------------------------------------------------------

def _cfg():
    return CONFIG(mode='preprocess', model={'architecture': 'toy'},
                  peft={'encoder_hidden_size': 64},
                  custom_training_args={'optimizer_grouped_parameters': [{'lr': 1e-5}]},
                  training_args={'args': {}})


def test_override_nested_and_list_index():
    c = _cfg()
    apply_override(c, 'peft.encoder_hidden_size', 192)
    apply_override(c, 'custom_training_args.optimizer_grouped_parameters.0.lr', 5e-3)
    assert c.peft.encoder_hidden_size == 192
    assert c.custom_training_args.optimizer_grouped_parameters[0].lr == 5e-3


def test_override_unknown_key_raises():
    with pytest.raises(AttributeError):
        apply_override(_cfg(), 'peft.no_such_key', 1)


def test_seed_sets_training_args_and_creates_args_block():
    c = CONFIG(mode='preprocess', model={'architecture': 'toy'}, training_args={})
    assert config_seed(c) is None
    apply_seed(c, 11)
    assert c.training_args.args.seed == 11 and config_seed(c) == 11
    apply_seed(c, None)
    assert c.training_args.args.seed == 11
    assert config_seed(CONFIG(mode='preprocess', model={'architecture': 'toy'})) is None


def test_scalar_columns_skips_reserved_and_nested():
    entry = {'config': 'u', 'name': 'a', 'seed': 1, 'overrides': {}, 'separate_test': {},
             'source_group': 'joshi5', 'fold': 0, 'flag': True, 'nested': {'x': 1}, 'lst': [1]}
    assert scalar_columns(entry) == {'source_group': 'joshi5', 'fold': 0, 'flag': True}


# -- resolve_entry ---------------------------------------------------------------

def test_resolve_entry_full(tmp_path):
    nlpka_path.set_root(tmp_path)
    g = load_group(_group(tmp_path, [{
        'config': 'u', 'name': 'a', 'seed': 11,
        'overrides': {'peft.encoder_hidden_size': 192},
        'separate_test': {'config': 'e', 'overrides': {'peft.encoder_hidden_size': 8}},
        'source_group': 'joshi5',
    }]))
    config, ctx = resolve_entry(g, 0, cli_seed=99, extras={'fold': '0'})
    assert config.peft.encoder_hidden_size == 192
    assert config.training_args.args.seed == 11          # entry beats CLI
    assert config.results.config_file == 'config.yml'
    assert ctx.separate_test.peft.encoder_hidden_size == 8
    assert config_seed(ctx.separate_test) is None                # no seed on the test config
    assert ctx.separate_test.results.config_file == 'test_config.yml'
    assert ctx.separate_test.results.dir == config.results.dir
    assert ctx.separate_test.results.columns == config.results.columns
    cols = config.results.columns
    assert cols['group'] == 'g1' and cols['name'] == 'a' and cols['index'] == 0
    assert cols['config'] == 'u' and cols['seed'] == 11 and cols['source_group'] == 'joshi5'
    run = Path(config.results.dir)
    assert cols['time_id'] and run.name == f"{cols['time_id']}_a"
    assert run.parent == tmp_path / 'artefacts' / 'evals' / 'runs' / 'toy' / 'g1'
    assert (run / 'config.yml').exists() and (run / 'test_config.yml').exists()   # the snapshot
    assert ctx == RunContext(separate_test=ctx.separate_test, entry={'source_group': 'joshi5'},
                             group='g1', name='a', index=0, run_dir=str(run), extras={'fold': '0'})


def test_resolve_entry_seed_column_follows_overrides(tmp_path):
    nlpka_path.set_root(tmp_path)
    g = load_group(_group(tmp_path, [{'config': 'u', 'name': 'a', 'overrides': {'training_args.args.seed': 5}}]))
    config, _ = resolve_entry(g, 0, cli_seed=99)
    assert config.training_args.args.seed == 5           # override beats seed
    assert config.results.columns['seed'] == 5           # and the column says what the run uses
    g = load_group(_group(tmp_path, [{'config': 'u', 'name': 'b'}]))
    config, _ = resolve_entry(g, 0, cli_seed=99)
    assert config.training_args.args.seed == 99 and config.results.columns['seed'] == 99
    g = load_group(_group(tmp_path, [{'config': 'u', 'name': 'c'}]))
    config, _ = resolve_entry(g, 0)
    assert 'seed' not in config.results.columns          # unpinned: the trainer stamps it later


def test_resolve_entry_keeps_user_columns_framework_wins(tmp_path):
    nlpka_path.set_root(tmp_path)
    gp = _group(tmp_path, [{'config': 'u', 'name': 'a'}])
    _unit(tmp_path, results={'columns': {'note': 'x', 'group': 'user'}, 'dir': '/user/dir'})   # after _group
    config, _ = resolve_entry(load_group(gp), 0)
    assert config.results.columns['note'] == 'x' and config.results.columns['group'] == 'g1'
    assert config.results.dir != '/user/dir'


def test_resolve_entry_no_model_block(tmp_path):
    nlpka_path.set_root(tmp_path)
    gp = _group(tmp_path, [{'config': 'u', 'name': 'a'}])
    (tmp_path / 'unit.yml').write_text(yaml.safe_dump({'mode': 'preprocess'}))
    config, ctx = resolve_entry(load_group(gp), 0)
    assert '/runs/_nomodel/g1/' in ctx.run_dir


def test_two_dispatches_two_dirs_same_second_raises(tmp_path, monkeypatch):
    nlpka_path.set_root(tmp_path)
    g = load_group(_group(tmp_path, [{'config': 'u', 'name': 'a'}]))
    stamps = iter(['20260907_000001', '20260907_000002', '20260907_000002'])
    monkeypatch.setattr(grp.utils, 'get_time_id', lambda: next(stamps))
    _, c1 = resolve_entry(g, 0)
    _, c2 = resolve_entry(g, 0)
    assert c1.run_dir != c2.run_dir and Path(c1.run_dir).is_dir() and Path(c2.run_dir).is_dir()
    with pytest.raises(FileExistsError):
        resolve_entry(g, 0)


# -- load_runner / dispatch ------------------------------------------------------

def test_load_runner_default_and_custom():
    from micm_nlp import pipeline
    assert load_runner(None) is pipeline.run
    assert load_runner('tests.test_group:stub_runner') is stub_runner
    with pytest.raises(ValueError, match='module:attr'):
        load_runner('no_colon')


def test_run_group_all_then_one(tmp_path, monkeypatch):
    nlpka_path.set_root(tmp_path)
    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    stamps = (f'20260907_{i:06d}' for i in range(10))
    monkeypatch.setattr(grp.utils, 'get_time_id', lambda: next(stamps))
    CALLS.clear()
    gp = _group(tmp_path, [{'config': 'u', 'name': 'a'}, {'config': 'u', 'name': 'b'}])
    assert run_group(gp, runner='tests.test_group:stub_runner', extras={'k': 'v'}) == [0, 1]
    assert [c.name for _, c in CALLS] == ['a', 'b'] and CALLS[0][1].extras == {'k': 'v'}
    CALLS.clear()
    assert run_group(gp, runner='tests.test_group:stub_runner', task_id=1) == [1]
    assert CALLS[0][1].index == 1


def test_run_solo(tmp_path):
    nlpka_path.set_root(tmp_path)
    CALLS.clear()
    run_solo(_unit(tmp_path), runner='tests.test_group:stub_runner', extras={'x': True})
    config, ctx = CALLS[0]
    assert isinstance(config, CONFIG) and config.results is None
    assert ctx == RunContext(separate_test=None, entry={}, group='_solo', name=None, index=None,
                             run_dir=None, extras={'x': True})


def test_pipeline_run_accepts_ctx():
    import inspect
    from micm_nlp import pipeline
    assert 'ctx' in inspect.signature(pipeline.run).parameters
