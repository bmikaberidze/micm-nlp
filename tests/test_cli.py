"""The CLI: argument parsing for ``run`` / ``run-group``, unknown flags becoming
runner extras, no abbreviation swallowing, and dispatch into ``group``.
``micm_nlp.init`` and the group functions are stubbed so nothing touches a
workspace."""

import pytest

from micm_nlp import cli


def test_parse_extras_forms():
    assert cli.parse_extras(['--source-group', 'joshi5', '--skip-test', '--fold', '0']) == {
        'source_group': 'joshi5', 'skip_test': True, 'fold': '0',
    }
    assert cli.parse_extras([]) == {}
    assert cli.parse_extras(['--fold=1', '--source-group=joshi5']) == {'fold': '1', 'source_group': 'joshi5'}


def test_parse_extras_rejects_positional():
    with pytest.raises(ValueError, match='stray'):
        cli.parse_extras(['stray'])


def test_run_group_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, '_init_workspace', lambda: seen.setdefault('init', True))
    monkeypatch.setattr(cli.group, 'run_group',
                        lambda path, runner, task_id, seed, extras: seen.update(locals()) or [0])
    rc = cli.main(['run-group', '--group-config', 'g.yml', '--runner', 'm:f', '--task-id', '2',
                   '--seed', '7', '--source-group', 'joshi5', '--se', '3'])
    assert rc == 0 and seen['init']
    assert seen['path'] == 'g.yml' and seen['runner'] == 'm:f' and seen['task_id'] == 2
    assert seen['seed'] == 7
    assert seen['extras'] == {'source_group': 'joshi5', 'se': '3'}   # --se is not an abbreviation of --seed


def test_run_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, '_init_workspace', lambda: None)
    monkeypatch.setattr(cli.group, 'run_solo', lambda path, runner, extras: seen.update(locals()))
    assert cli.main(['run', '--config', 'c.yml', '--fold', '1']) == 0
    assert seen['path'] == 'c.yml' and seen['runner'] is None and seen['extras'] == {'fold': '1'}


def test_run_with_stray_positional_exits(monkeypatch):
    monkeypatch.setattr(cli, '_init_workspace', lambda: None)
    with pytest.raises(SystemExit):
        cli.main(['run', '--config', 'c.yml', 'stray'])


def test_init_examples_rejects_unknown_flags(tmp_path):
    # Regression guard for existing behaviour; passes before and after.
    with pytest.raises(SystemExit):
        cli.main(['init-examples', str(tmp_path), '--bogus'])


def test_init_workspace_without_root_exits_with_a_message(monkeypatch):
    monkeypatch.delenv('PROJECT_ROOT_PATH', raising=False)
    with pytest.raises(SystemExit, match='PROJECT_ROOT_PATH'):
        cli._init_workspace()


def test_init_examples_copies_the_groups_subdir(tmp_path):
    assert cli.main(['init-examples', str(tmp_path / 'ex')]) == 0
    assert (tmp_path / 'ex' / 'xsc_finetune.yml').exists()
    assert (tmp_path / 'ex' / 'groups' / 'xsc_group.yml').exists()
