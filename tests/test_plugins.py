"""@micm_plugin: registry, discovery, and resolution through resolve_cls."""

import re
import sys
from pathlib import Path
from types import ModuleType

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


def _class_in_fake_module(monkeypatch, module_name, file):
    """A class ``Thing`` defined by a module ``module_name`` whose ``__file__`` is ``file``."""
    module = ModuleType(module_name)
    module.__file__ = str(file)
    monkeypatch.setitem(sys.modules, module_name, module)
    return type('Thing', (), {'__module__': module_name})


def test_the_same_file_under_two_module_names_is_one_plugin(tmp_path, monkeypatch):
    file = tmp_path / 'trainers.py'
    plugins.micm_plugin(_class_in_fake_module(monkeypatch, 'fake_src.fake_proj.trainers', file))
    plugins.micm_plugin(_class_in_fake_module(monkeypatch, 'fake_proj.trainers', file))
    other = _class_in_fake_module(monkeypatch, 'fake_other.trainers', tmp_path / 'other.py')
    with pytest.raises(ValueError, match="'Thing' is taken"):
        plugins.micm_plugin(other)


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


def test_a_decorator_only_shown_in_a_docstring_is_not_imported(ws):
    # An example snippet in a docstring declares no plugin; importing the file would
    # run its top-level code for nothing.
    _write(ws, 'pkg_j/doc.py', 'def helper():\n    """Example:\n\n    @micm_plugin\n    class MyTrainer: ...\n    """\n'
           + BOMB)
    assert plugins.plugin_files(ws) == []
    plugins.discover(ws)   # BOMB would raise if the file were imported


def test_unparsable_source_is_left_to_the_import_to_report(ws):
    _write(ws, 'pkg_k/broken_syntax.py', PLUGIN.format(name='SyntaxTrainer') + 'def (:\n')
    assert plugins.plugin_files(ws) == [ws / 'pkg_k' / 'broken_syntax.py']
    with pytest.raises(ImportError, match=re.escape('pkg_k/broken_syntax.py')):
        plugins.discover(ws)


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
    with pytest.raises(ImportError, match=re.escape('pkg_c/broken.py')):
        plugins.discover(ws)


def test_a_plugin_in_a_package_init_registers_once_as_the_package(ws):
    _write(ws, 'pkg_h/__init__.py', PLUGIN.format(name='InitTrainer'))
    plugins.discover(ws)
    assert plugins._PLUGINS['InitTrainer'].__module__ == 'pkg_h'
    assert 'pkg_h.__init__' not in sys.modules


def test_a_file_already_loaded_as_the_running_script_is_not_imported_again(ws, monkeypatch):
    # `python train.py`: the script is `__main__`, its decorators already ran.
    file = _write(ws, 'pkg_i/train.py', PLUGIN.format(name='MainTrainer')
                  + "if __name__ != 'fake_main':\n    raise RuntimeError('script re-executed')\n")
    main = ModuleType('fake_main')
    main.__file__ = str(file)
    monkeypatch.setitem(sys.modules, 'fake_main', main)
    exec(compile(file.read_text(), str(file), 'exec'), main.__dict__)
    plugins.discover(ws)
    assert plugins._PLUGINS['MainTrainer'].__module__ == 'fake_main'
    assert 'pkg_i.train' not in sys.modules


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


def test_resolve_cls_rejects_a_plugin_function(monkeypatch):
    def ws_fn():
        return None

    monkeypatch.setattr(plugins, '_discovered', True)   # no scan: the plugin is registered here
    plugins.micm_plugin(ws_fn)
    with pytest.raises(TypeError, match=re.escape("trainer.cls='ws_fn' is a @micm_plugin function, not a class")):
        utils.resolve_cls('ws_fn', ['collections'], 'trainer.cls')


SEEN = []


def plugin_runner(config, ctx):
    SEEN.append(utils.resolve_cls('GroupTrainer', ['collections']).__name__)


def test_a_group_run_resolves_a_plugin(ws, monkeypatch):
    import yaml

    monkeypatch.setattr(nlpka_path, '_workspace', None)   # restored after, undoing set_root(ws)
    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    monkeypatch.setattr(cli, '_init_workspace', lambda root_path=None: nlpka_path.set_root(ws))
    _write(ws, 'pkg_g/trainers.py', PLUGIN.format(name='GroupTrainer'))
    _write(ws, 'configs/unit.yml', yaml.safe_dump({'mode': 'preprocess'}))
    group = _write(ws, 'configs/plug.yml', yaml.safe_dump(
        {'configs': {'u': './unit.yml'}, 'runs': [{'config': 'u', 'name': 'one'}]}))
    SEEN.clear()
    assert cli.main(['run-group', '--group-config', str(group), '--runner', 'tests.test_plugins:plugin_runner']) == 0
    assert SEEN == ['GroupTrainer']


def test_a_plugin_training_arguments_subclass_resolves(ws, monkeypatch):
    # A setting of your own belongs in a TrainingArguments subclass: it reaches the
    # trainer as `self.args.alpha`, and is saved and logged with the run.
    _write(ws, 'pkg_l/targs.py',
           'from dataclasses import dataclass\n\nfrom micm_nlp import micm_plugin\n\n'
           '@micm_plugin\n@dataclass\nclass MyTrainingArguments:\n    alpha: float = 0.5\n')
    monkeypatch.setattr(nlpka_path, '_workspace', ws)
    TArgs = utils.resolve_cls('MyTrainingArguments', ['transformers'], 'training_args.cls')
    assert TArgs(alpha=0.7).alpha == 0.7
