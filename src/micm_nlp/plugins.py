"""User classes and functions a config can name in ``cls``.

Decorate a class or function anywhere in the workspace::

    from micm_nlp import micm_plugin

    @micm_plugin
    class MyTrainer(Trainer): ...

and name it in a config — ``trainer: {cls: MyTrainer}``. Nothing to install and
nothing to import: on the first ``cls`` lookup, :func:`discover` scans the workspace
for files that declare a plugin and imports only those, so the decorators run.
:func:`micm_nlp.utils.resolve_cls` asks :func:`find` before the built-in modules.

Imports only the standard library and :mod:`micm_nlp.path`, so ``import micm_nlp``
stays clear of torch.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType

from micm_nlp import path

_PLUGINS: dict[str, object] = {}
_discovered = False


def _qualified(obj) -> str:
    return f'{obj.__module__}.{obj.__qualname__}'


def micm_plugin(obj):
    """Register a class or function under its ``__name__``; return it unchanged.

    The same object, or a re-import of the same module, registers again silently.
    A different object under a name already taken raises.
    """
    name = obj.__name__
    taken = _PLUGINS.get(name)
    if taken is not None and _qualified(taken) != _qualified(obj):
        raise ValueError(f'@micm_plugin name {name!r} is taken: {_qualified(taken)} and {_qualified(obj)}')
    _PLUGINS[name] = obj
    return obj


SKIP_DIRS = frozenset({'artefacts', 'tests', 'test', '__pycache__', 'node_modules', 'build', 'dist'})
_DECLARES = re.compile(r'^\s*@(?:[A-Za-z_]\w*\.)*micm_plugin\b', re.MULTILINE)


def _skip_dir(directory: Path) -> bool:
    name = directory.name
    return (name in SKIP_DIRS or name.startswith('.') or (directory / 'pyvenv.cfg').is_file()
            or directory.resolve() == path.PACKAGE_DIR.resolve())


def _skip_file(name: str) -> bool:
    return name == 'conftest.py' or name.startswith('test_') or name.endswith('_test.py')


def plugin_files(root: Path) -> list[Path]:
    """Every ``.py`` under ``root`` that declares a ``@micm_plugin``, in path order."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(here / d))
        for filename in sorted(filenames):
            if not filename.endswith('.py') or _skip_file(filename):
                continue
            file = here / filename
            try:
                text = file.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            if _DECLARES.search(text):
                found.append(file)
    return found


def import_plugin_file(root: Path, file: Path) -> ModuleType:
    """Import one plugin file so its decorators run.

    By dotted path from ``root`` (on ``sys.path``), so the file's own absolute imports
    work and a later normal import is the same module. Loaded from the file instead,
    with a warning, when the path is not a valid dotted name, or when that name
    already belongs to another module (a workspace ``utils.py`` would otherwise
    silently resolve to someone else's ``utils``).
    """
    rel = file.relative_to(root)
    parts = rel.with_suffix('').parts
    try:
        if all(part.isidentifier() for part in parts):
            if str(root) not in sys.path:
                sys.path.append(str(root))   # last: never shadow an installed package
            dotted = '.'.join(parts)
            module = importlib.import_module(dotted)
            if Path(getattr(module, '__file__', '') or '').resolve() == file.resolve():
                return module
            reason = f'{dotted!r} already names another module'
        else:
            reason = 'not a valid Python module path'
        print(f'[micm_nlp] {rel}: {reason}, loaded by file path — relative imports in it will not '
              'work; consider renaming (letters, digits, _; not starting with a digit; not an '
              'existing module name)')
        name = 'micm_nlp_plugin__' + '__'.join(re.sub(r'\W', '_', part) for part in parts)
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module
    except Exception as exc:
        raise ImportError(f'@micm_plugin file {rel} failed to import: {exc}') from exc


def discover(root: Path | None = None) -> None:
    """Import every plugin file in the workspace, once per process.

    Without ``root`` it uses the workspace ``init()`` set; before ``init()`` it does
    nothing and stays undone, so a later call after ``init()`` still scans.
    """
    global _discovered
    if _discovered:
        return
    if root is None:
        try:
            root = path.workspace()
        except RuntimeError:
            return
    _discovered = True
    root = Path(root)
    for file in plugin_files(root):
        import_plugin_file(root, file)


def find(name: str):
    """The plugin registered under ``name``, discovering first; ``None`` if there is none."""
    discover()
    return _PLUGINS.get(name)
