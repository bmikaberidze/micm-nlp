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


def _defining_file(obj) -> Path | None:
    """The resolved ``__file__`` of the module that defines ``obj``, if it has one."""
    file = getattr(sys.modules.get(obj.__module__), '__file__', None)
    return Path(file).resolve() if isinstance(file, str) else None


def _same_plugin(a, b) -> bool:
    """One definition, even when its file was imported under two module names
    (``src.my_proj.trainers`` and ``my_proj.trainers``)."""
    if _qualified(a) == _qualified(b):
        return True
    file = _defining_file(a)
    return a.__qualname__ == b.__qualname__ and file is not None and file == _defining_file(b)


def micm_plugin(obj):
    """Register a class or function under its ``__name__``; return it unchanged.

    The same object, or a re-import of the same file (under any module name),
    registers again silently. A different object under a name already taken raises.
    """
    name = obj.__name__
    taken = _PLUGINS.get(name)
    if taken is not None and not _same_plugin(taken, obj):
        raise ValueError(f'@micm_plugin name {name!r} is taken: {_qualified(taken)} and {_qualified(obj)}')
    _PLUGINS[name] = obj
    return obj


SKIP_DIRS = frozenset({'artefacts', 'tests', 'test', '__pycache__', 'node_modules', 'build', 'dist'})
_DECLARES = re.compile(r'^\s*@(?:[A-Za-z_]\w*\.)*micm_plugin\b', re.MULTILINE)
_PACKAGE_DIR = path.PACKAGE_DIR.resolve()   # once, not per scanned directory


def _skip_dir(directory: Path) -> bool:
    name = directory.name
    return (name in SKIP_DIRS or name.startswith('.') or (directory / 'pyvenv.cfg').is_file()
            or directory.resolve() == _PACKAGE_DIR)


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


def _loaded_module(file: Path) -> ModuleType | None:
    """An already-imported module whose ``__file__`` is ``file`` — e.g. ``__main__`` when
    the running script declares a plugin. Its decorators already ran; importing the
    file again would re-run the script."""
    target = file.resolve()
    for module in list(sys.modules.values()):
        try:
            loc = getattr(module, '__file__', None)
        except Exception:   # a lazy module can raise on attribute access
            continue
        if isinstance(loc, str) and os.path.basename(loc) == file.name and Path(loc).resolve() == target:
            return module
    return None


def import_plugin_file(root: Path, file: Path) -> ModuleType:
    """Import one plugin file so its decorators run.

    A file some loaded module (``__main__`` included) already came from is not
    imported again. A package's ``__init__.py`` is imported as the package itself.
    Otherwise by dotted path from ``root`` (on ``sys.path``), so the file's own absolute
    imports work and a later normal import is the same module. Loaded from the file instead,
    with a warning, when the path is not a valid dotted name, or when that top-level
    name already belongs to another module — checked with ``find_spec`` *before*
    importing, so a workspace ``scripts/trainers.py`` shadowed by an installed
    ``scripts`` package (or a workspace ``email/`` shadowed by the stdlib) neither
    hard-fails discovery nor runs a stranger's top-level package first. The
    ``__file__`` check after the dotted import is a second guard, kept for the case
    ``find_spec`` says the name is ours but the import still resolves elsewhere.
    """
    loaded = _loaded_module(file)
    if loaded is not None:
        return loaded
    rel = file.relative_to(root)
    parts = rel.with_suffix('').parts
    if len(parts) > 1 and parts[-1] == '__init__':
        parts = parts[:-1]   # `pkg.__init__` would run the file a second time, after `pkg`
    try:
        if all(part.isidentifier() for part in parts):
            if str(root) not in sys.path:
                sys.path.append(str(root))   # last: never shadow an installed package
            dotted = '.'.join(parts)
            resolved_root = root.resolve()
            try:
                owner_spec = importlib.util.find_spec(parts[0])
            except (ImportError, ValueError):
                owner_spec = None
            locations = list(owner_spec.submodule_search_locations or []) if owner_spec else []
            if owner_spec is not None and owner_spec.origin:
                locations.append(owner_spec.origin)
            owns_it = any(Path(loc).resolve().is_relative_to(resolved_root) for loc in locations)
            if owns_it:
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
    nothing and stays undone, so a later call after ``init()`` still scans. If a
    plugin file fails to import, ``_discovered`` is reset to ``False`` before the
    exception propagates, so a later call retries (and re-raises the real error)
    instead of silently reporting no plugins.
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
    try:
        for file in plugin_files(root):
            import_plugin_file(root, file)
    except Exception:
        _discovered = False
        raise


def find(name: str):
    """The plugin registered under ``name``, discovering first; ``None`` if there is none."""
    discover()
    return _PLUGINS.get(name)
