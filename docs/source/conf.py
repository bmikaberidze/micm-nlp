"""Sphinx configuration for the micm-nlp documentation.

The API reference is produced by ``sphinx-autoapi``, which parses ``src/``
statically. Nothing here imports ``micm_nlp``, so the docs build does not need
torch, transformers, spacy or lightning installed — see ``docs/requirements.txt``.
"""

from __future__ import annotations

import re
from pathlib import Path

# -- Project metadata --------------------------------------------------------
# Read straight out of pyproject.toml rather than importing the package, so the
# version can never drift from the packaging metadata. tomllib is 3.11+, and this
# repo's own container is 3.10, so fall back to a targeted regex there.

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = (_REPO_ROOT / 'pyproject.toml').read_text()

try:
    import tomllib

    _release = tomllib.loads(_PYPROJECT)['project']['version']
except ImportError:  # pragma: no cover - Python 3.10
    _match = re.search(r'^version\s*=\s*[\'"]([^\'"]+)[\'"]', _PYPROJECT, re.M)
    if _match is None:
        raise RuntimeError('could not read version from pyproject.toml') from None
    _release = _match.group(1)

project = 'micm-nlp'
author = 'Beso Mikaberidze'
copyright = '2026, Muskhelishvili Institute of Computational Mathematics'
release = _release
version = '.'.join(release.split('.')[:2])

# -- General -----------------------------------------------------------------

extensions = [
    'myst_parser',
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build']

# Pages are authored in Markdown.
myst_enable_extensions = ['colon_fence', 'deflist']
myst_heading_anchors = 3

# -- autoapi -----------------------------------------------------------------

autoapi_dirs = [str(_REPO_ROOT / 'src')]
autoapi_type = 'python'
autoapi_root = 'autoapi'
autoapi_member_order = 'groupwise'
autoapi_python_class_content = 'both'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]
# The generated tree is reached through the curated api.md page instead of
# being injected at the top level of the sidebar.
autoapi_add_toctree_entry = False
autoapi_keep_files = False

# -- HTML --------------------------------------------------------------------

html_theme = 'furo'
html_title = f'micm-nlp {version}'
html_static_path = []
html_theme_options = {
    'source_repository': 'https://github.com/bmikaberidze/micm-nlp/',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
}
