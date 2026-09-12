"""Filesystem layout: the workspace root and the ``artefacts/`` tree beneath it.

Two kinds of path live here. ``PACKAGE_DIR`` points inside the installed package and
is read-only. Everything else hangs off the *workspace* — the user's project
directory — which must be set once via ``set_root()`` before any accessor is called;
they raise otherwise. ``micm_nlp.init()`` does that for you::

    workspace()/artefacts/{models,datasets,tokenizers,runs,wandb}
"""

import os
from importlib import resources
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Package directory (inner paths — read-only, shipped with the package)
PACKAGE_DIR = Path(__file__).parent

EXAMPLE_CONFIG_PACKAGE = 'micm_nlp.configs'
"""Where the example configs ship. Inside the package rather than downloaded, so
the copy you get always matches the version you installed."""


def available_examples() -> list[tuple[Path, Any]]:
    """Every config shipped in the example package.

    The package root and one level of subdirectories (``groups/``), sorted by path.

    :returns: ``(relative path, traversable)`` pairs. The first line above is kept
        short and self-contained because the API reference renders it as the
        summary, and a summary that stops mid-``literal`` is a docs build warning.
    """
    root = resources.files(EXAMPLE_CONFIG_PACKAGE)
    found: list[tuple[Path, Any]] = []
    for entry in root.iterdir():
        if entry.name.endswith('.yml'):
            found.append((Path(entry.name), entry))
        elif entry.is_dir() and not entry.name.startswith('_'):
            found.extend((Path(entry.name) / sub.name, sub) for sub in entry.iterdir() if sub.name.endswith('.yml'))
    return sorted(found, key=lambda pair: str(pair[0]))


def example(name: str) -> Path:
    """The path of one config shipped inside the package.

    ::

        run(example('xsc_finetune.yml'))

    Reading a config straight out of the installed package, rather than making
    ``run`` fall back to it for a name that is not on disk -- one visible call is
    easier to explain than a second lookup rule hidden inside the pipeline.
    ``micm-nlp init-examples`` is still the way to get an editable copy.

    :param name: the file name, with or without ``.yml``; ``groups/`` names work
        too (``example('groups/xsc_tune_across_seeds.yml')``).
    :raises FileNotFoundError: naming what does ship, since the set is small and
        fixed -- a typo should not send anyone to the documentation.
    """
    wanted = name if name.endswith('.yml') else f'{name}.yml'
    for relative, traversable in available_examples():
        if str(relative) == wanted:
            return Path(str(traversable))
    shipped = ', '.join(str(relative) for relative, _ in available_examples())
    raise FileNotFoundError(f'No example config {name!r}. The package ships: {shipped}')

# Workspace (outer paths — user's project, read-write)
_workspace = None


def set_root(workspace: str | Path):
    """Set the workspace root. Call once at startup.

    :raises ValueError: if no root was given. ``micm_nlp.init()`` passes its
        ``root_path`` straight through, and that is ``None`` when neither the call
        nor ``PROJECT_ROOT_PATH`` supplied one -- which used to surface here as a
        ``TypeError`` from ``Path(None)``, naming neither the setting nor the fix.
    """
    global _workspace
    if workspace is None:
        raise ValueError(
            "No workspace root. Pass one to init(), as micm_nlp.init('/path/to/your/workspace'), "
            'or set PROJECT_ROOT_PATH in the environment or in .env.'
        )
    _workspace = Path(workspace)


def workspace() -> Path:
    """Return the workspace root.

    :raises RuntimeError: if :func:`set_root` has not been called. Every accessor
        below goes through here, so an unset root fails loudly at the first path
        request rather than silently writing into the current directory.
    """
    if _workspace is None:
        raise RuntimeError("Call micm_nlp.path.set_root('/path/to/your/workspace') first")
    return _workspace


def artefacts_dir() -> Path:
    """``<workspace>/artefacts`` — the root of everything this package writes."""
    return workspace() / 'artefacts'


def models_dir() -> Path:
    """``artefacts/models`` — saved checkpoints and PEFT adapters."""
    return artefacts_dir() / 'models'


def datasets_dir() -> Path:
    """``artefacts/datasets`` — raw, preprocessed and tokenized datasets."""
    return artefacts_dir() / 'datasets'


def tokenizers_dir() -> Path:
    """``artefacts/tokenizers`` — tokenizers trained by this package."""
    return artefacts_dir() / 'tokenizers'


def evals_dir() -> Path:
    """``artefacts/evals`` — legacy location, no longer written to.

    Runs live under :func:`runs_dir` since 0.4.0; this function has no callers and
    is kept only so an existing tree stays addressable.
    """
    return artefacts_dir() / 'evals'


UNIT_RUNS = 'units'
"""Where a run started outside any group config lands: ``runs/units/{name}``."""

GROUP_RUNS = 'groups'
"""Where a group's runs land: ``runs/groups/{group}/{name}``. One directory per
group keeps an experiment whole -- which is why no segment above it varies with
the run. An earlier layout led with the model architecture, and split any group
that varied the backbone across two trees."""


def runs_dir() -> Path:
    """``artefacts/runs`` -- one directory per run, under ``units/`` or
    ``groups/{group}/``. A run is neither an eval nor a training; it is the unit
    the trainer executes."""
    return artefacts_dir() / 'runs'


def output_dir(group: str | None, name: str) -> Path:
    """The directory one run writes into: config snapshot, ``info.json``,
    metrics, predictions, links, logs.

    :param group: the group's name, or ``None`` for a run started outside any
        group config -- which lands under :data:`UNIT_RUNS` instead.
    """
    return runs_dir() / GROUP_RUNS / group / name if group else runs_dir() / UNIT_RUNS / name


def wandb_dir() -> Path:
    """Parent directory for wandb's own ``wandb/`` folder.

    Returns ``artefacts/`` itself, not a subdirectory: wandb appends ``wandb/`` to
    whatever it is given, so the run tree lands at ``artefacts/wandb/``.
    """
    return artefacts_dir()


# Directory utilities ------------------------------------------------------------------------------------------------------------------
def find_dirs_by_prefix(root_dir, dir_prefix):
    """Recursively find every directory under ``root_dir`` whose name starts with
    ``dir_prefix``.

    Walks the whole tree, so cost grows with the size of ``root_dir`` — the progress
    bar is there because an ``artefacts/`` tree can be large.

    :param root_dir: directory to search under.
    :param dir_prefix: prefix a directory's *name* must start with.
    :returns: absolute paths as strings, in walk order.
    """
    # return [str(p) for p in Path(root_dir).rglob(f'{dir_prefix}*/') if p.is_dir()]
    matching_dirs = []
    print(root_dir, dir_prefix)
    for dirpath, dirnames, _filenames in tqdm(os.walk(root_dir), desc='Walking through directories'):
        for dirname in dirnames:
            if dirname.startswith(dir_prefix):
                full_path = os.path.join(dirpath, dirname)
                matching_dirs.append(full_path)
    return matching_dirs


def get_dir_items(dir_path, only_dirs=False, only_files=False):
    """List the immediate contents of a directory, non-recursively.

    :param dir_path: directory to list.
    :param only_dirs: return only subdirectories.
    :param only_files: return only files.
    :returns: names, with a trailing ``/`` on directories. A missing ``dir_path``
        yields ``[]`` rather than raising, so callers can probe a path that may not
        exist yet.
    """
    p = Path(dir_path)
    if not p.exists():
        return []
    items = []
    for item in p.iterdir():
        if only_dirs and not item.is_dir():
            continue
        if only_files and not item.is_file():
            continue
        items.append(f'{item.name}/' if item.is_dir() else item.name)
    return items
