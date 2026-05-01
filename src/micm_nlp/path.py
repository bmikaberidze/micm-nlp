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
    if _workspace is None:
        raise RuntimeError("Call micm_nlp.path.set_root('/path/to/your/workspace') first")
    return _workspace


def artefacts_dir() -> Path:
    return workspace() / 'artefacts'


def models_dir() -> Path:
    return artefacts_dir() / 'models'


def datasets_dir() -> Path:
    return artefacts_dir() / 'datasets'


def tokenizers_dir() -> Path:
    return artefacts_dir() / 'tokenizers'


def evals_dir() -> Path:
    return artefacts_dir() / 'evals'


def wandb_dir() -> Path:
    return artefacts_dir()


# Directory utilities ------------------------------------------------------------------------------------------------------------------
def find_dirs_by_prefix(root_dir, dir_prefix):
    """
    Find directory's path in the specified root directory
    that starts with the specified prefix
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
    """List all files and directories in the specified directory"""
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
