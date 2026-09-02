"""Filesystem layout: the workspace root and the ``artefacts/`` tree beneath it.

Two kinds of path live here. ``PACKAGE_DIR`` points inside the installed package and
is read-only. Everything else hangs off the *workspace* — the user's project
directory — which must be set once via ``set_root()`` before any accessor is called;
they raise otherwise. ``micm_nlp.init()`` does that for you::

    workspace()/artefacts/{models,datasets,tokenizers,evals}
"""

import os
from pathlib import Path

from tqdm import tqdm

# Package directory (inner paths — read-only, shipped with the package)
PACKAGE_DIR = Path(__file__).parent

# Workspace (outer paths — user's project, read-write)
_workspace = None


def set_root(workspace: str | Path):
    """Set the workspace root. Call once at startup."""
    global _workspace
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
    """``artefacts/evals`` — evaluation runs: metrics, predictions, plots."""
    return artefacts_dir() / 'evals'


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
