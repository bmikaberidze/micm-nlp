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
