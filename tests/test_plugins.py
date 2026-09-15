"""@micm_plugin: registry, discovery, and resolution through resolve_cls."""

import sys
from pathlib import Path

import pytest

from micm_nlp import cli, plugins, utils
from micm_nlp import path as nlpka_path


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(plugins, '_PLUGINS', {})
    monkeypatch.setattr(plugins, '_discovered', False)


def test_registers_a_class_and_returns_it_unchanged():
    class Thing:
        pass

    assert plugins.micm_plugin(Thing) is Thing
    assert plugins._PLUGINS['Thing'] is Thing


def test_registers_a_function():
    def my_f1(preds, labels):
        return 1.0

    assert plugins.micm_plugin(my_f1) is my_f1
    assert plugins._PLUGINS['my_f1'] is my_f1


def test_same_class_twice_is_a_no_op():
    class Thing:
        pass

    plugins.micm_plugin(Thing)
    plugins.micm_plugin(Thing)
    assert plugins._PLUGINS == {'Thing': Thing}


def test_different_class_under_a_taken_name_raises():
    def make():
        class Thing:
            pass
        return Thing

    first, second = make(), make()
    second.__module__ = 'somewhere.else'
    plugins.micm_plugin(first)
    with pytest.raises(ValueError, match="'Thing' is taken"):
        plugins.micm_plugin(second)


def test_exported_from_the_package():
    import micm_nlp

    assert micm_nlp.micm_plugin is plugins.micm_plugin


PLUGIN = 'from micm_nlp import micm_plugin\n\n@micm_plugin\nclass {name}:\n    pass\n'
BOMB = 'raise RuntimeError("this file must never be imported")\n'


def _from(module, root: Path) -> bool:
    """Whether a module was loaded from under ``root`` (a file, or a namespace package)."""
    locations = [getattr(module, '__file__', None) or '', *list(getattr(module, '__path__', []) or [])]
    return any(str(loc).startswith(str(root)) for loc in locations)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace root. Afterwards, forget only the modules imported from it —
    never micm_nlp's own, which later tests rely on being the same objects."""
    monkeypatch.setattr(sys, 'path', list(sys.path))
    yield tmp_path
    for name, module in list(sys.modules.items()):
        if module is not None and _from(module, tmp_path):
            del sys.modules[name]


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_only_files_that_declare_a_plugin_are_imported(ws):
    _write(ws, 'pkg_a/trainers.py', PLUGIN.format(name='WsTrainer'))
    _write(ws, 'pkg_a/script.py', BOMB)                        # no @micm_plugin
    _write(ws, 'artefacts/x.py', '@micm_plugin\n' + BOMB)      # skipped dir
    _write(ws, 'tests/y.py', '@micm_plugin\n' + BOMB)
    _write(ws, '.hidden/z.py', '@micm_plugin\n' + BOMB)
    _write(ws, 'pkg_a/test_w.py', '@micm_plugin\n' + BOMB)     # skipped file
    _write(ws, 'venv/pyvenv.cfg', '')
    _write(ws, 'venv/lib/v.py', '@micm_plugin\n' + BOMB)

    assert plugins.plugin_files(ws) == [ws / 'pkg_a' / 'trainers.py']
    plugins.discover(ws)
    assert plugins._PLUGINS['WsTrainer'].__module__ == 'pkg_a.trainers'


def test_qualified_decorator_counts(ws):
    _write(ws, 'pkg_b/m.py', 'import micm_nlp\n\n@micm_nlp.micm_plugin\ndef ws_metric(p, l):\n    return 0\n')
    assert plugins.plugin_files(ws) == [ws / 'pkg_b' / 'm.py']


def test_invalid_folder_name_loads_by_path_with_a_warning(ws, capsys):
    _write(ws, 'my-proj/trainers.py', PLUGIN.format(name='DashTrainer'))
    plugins.discover(ws)
    assert 'DashTrainer' in plugins._PLUGINS
    assert 'not a valid Python module path' in capsys.readouterr().out


def test_a_name_already_taken_by_another_module_loads_by_path(ws, capsys):
    # A workspace `json.py`: `import json` would return the standard library's.
    _write(ws, 'json.py', PLUGIN.format(name='JsonTrainer'))
    plugins.discover(ws)
    assert 'JsonTrainer' in plugins._PLUGINS
    assert 'already names another module' in capsys.readouterr().out


def test_a_plugin_file_that_fails_to_import_raises_naming_it(ws):
    _write(ws, 'pkg_c/broken.py', PLUGIN.format(name='BrokenTrainer') + 'import no_such_module_xyz\n')
    with pytest.raises(ImportError, match='pkg_c/broken.py'):
        plugins.discover(ws)


def test_a_shadowed_stdlib_name_loads_by_path_without_hard_failing(ws, capsys):
    # A workspace `email/` package: `import email.trainers` must not import the
    # stdlib `email` package's own top-level code, nor hard-fail discovery when
    # `email.trainers` does not exist there.
    _write(ws, 'email/trainers.py', PLUGIN.format(name='EmailTrainer'))
    plugins.discover(ws)
    assert 'EmailTrainer' in plugins._PLUGINS
    assert 'already names another module' in capsys.readouterr().out


def test_discovery_runs_once(ws):
    _write(ws, 'pkg_d/a.py', PLUGIN.format(name='OnceTrainer'))
    plugins.discover(ws)
    _write(ws, 'pkg_d/b.py', PLUGIN.format(name='LateTrainer'))
    plugins.discover(ws)
    assert 'LateTrainer' not in plugins._PLUGINS


def test_find_without_a_workspace_root_does_not_scan(monkeypatch):
    from micm_nlp import path

    monkeypatch.setattr(path, '_workspace', None)
    assert plugins.find('Anything') is None
    assert plugins._discovered is False


def test_resolve_cls_finds_a_plugin_in_the_workspace(ws, monkeypatch):
    _write(ws, 'pkg_e/trainers.py', PLUGIN.format(name='ETrainer'))
    monkeypatch.setattr(nlpka_path, '_workspace', ws)
    assert utils.resolve_cls('ETrainer', ['collections']).__module__ == 'pkg_e.trainers'


def test_a_plugin_shadowing_a_built_in_wins_and_says_so(ws, monkeypatch, capsys):
    _write(ws, 'pkg_f/od.py', PLUGIN.format(name='OrderedDict'))
    monkeypatch.setattr(nlpka_path, '_workspace', ws)
    assert utils.resolve_cls('OrderedDict', ['collections'], 'trainer.cls').__module__ == 'pkg_f.od'
    assert "using plugin pkg_f.od.OrderedDict, not collections.OrderedDict" in capsys.readouterr().out


def test_built_in_names_resolve_as_before_without_init(monkeypatch):
    monkeypatch.setattr(nlpka_path, '_workspace', None)
    import collections

    assert utils.resolve_cls('OrderedDict', ['collections']) is collections.OrderedDict
    with pytest.raises(ValueError, match='Not a @micm_plugin'):
        utils.resolve_cls('NoSuchThing', ['collections'])


SEEN = []


def plugin_runner(config, ctx):
    SEEN.append(utils.resolve_cls('GroupTrainer', ['collections']).__name__)


def test_a_group_run_resolves_a_plugin(ws, monkeypatch):
    import yaml

    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    monkeypatch.setattr(cli, '_init_workspace', lambda root_path=None: nlpka_path.set_root(ws))
    _write(ws, 'pkg_g/trainers.py', PLUGIN.format(name='GroupTrainer'))
    _write(ws, 'configs/unit.yml', yaml.safe_dump({'mode': 'preprocess'}))
    group = _write(ws, 'configs/plug.yml', yaml.safe_dump(
        {'configs': {'u': './unit.yml'}, 'runs': [{'config': 'u', 'name': 'one'}]}))
    SEEN.clear()
    assert cli.main(['run-group', '--group-config', str(group), '--runner', 'tests.test_plugins:plugin_runner']) == 0
    assert SEEN == ['GroupTrainer']


def test_trainer_args_are_merged_into_the_constructor_kwargs():
    from micm_nlp.config import _Flex
    from micm_nlp.training.runner import trainer_kwargs

    merged = trainer_kwargs({'model': 'm', 'args': 'a'}, _Flex(alpha=0.5))
    assert merged == {'model': 'm', 'args': 'a', 'alpha': 0.5}
    assert trainer_kwargs({'model': 'm'}, None) == {'model': 'm'}


def test_trainer_args_may_not_override_framework_kwargs():
    from micm_nlp.config import _Flex
    from micm_nlp.training.runner import trainer_kwargs

    with pytest.raises(ValueError, match=r"\['custom_args', 'model'\]"):
        trainer_kwargs({'model': 'm'}, _Flex(model='x', custom_args=1))
