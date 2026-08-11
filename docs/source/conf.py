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

# api.md links every module page explicitly, with short titles, so the sidebar shows
# one flat level instead of "API reference > micm_nlp > module". autoapi still emits a
# top-level stub page (autoapi/micm_nlp/index) that nothing now links to, which costs
# one "isn't included in any toctree" warning per build. That is left visible on
# purpose: suppressing it would need a blanket `toc` suppression, which would also
# hide a module page genuinely missing from api.md's hand-written toctrees.

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
# The full release (0.2.1), not the short version (0.2) — a patch-level fix is
# exactly the thing a reader needs to know they are looking at.
html_title = f'micm-nlp {release}'
html_static_path = ['_static']

# Branding is picked up from docs/source/_static if present, so adding artwork is a
# drop-in with no edit here. Furo takes two logos and swaps them with the theme;
# a single logo.* is used for both when no variants exist.
_STATIC = Path(__file__).parent / '_static'


def _first(*names: str) -> str | None:
    for name in names:
        for suffix in ('.svg', '.png'):
            if (_STATIC / f'{name}{suffix}').is_file():
                return f'{name}{suffix}'
    return None


html_css_files = ['custom.css']

_logo = _first('logo')
_logo_light = _first('logo-light') or _logo
_logo_dark = _first('logo-dark') or _logo
_favicon = _first('favicon') or _logo

if _favicon:
    html_favicon = f'_static/{_favicon}'
html_theme_options = {
    'source_repository': 'https://github.com/bmikaberidze/micm-nlp/',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
    **({'light_logo': _logo_light} if _logo_light else {}),
    **({'dark_logo': _logo_dark} if _logo_dark else {}),
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/bmikaberidze/micm-nlp',
            'html': (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                '8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49 '
                '-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01'
                '-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07'
                '-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12'
                '0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 '
                '2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 '
                '3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55'
                '.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            'class': '',
        },
    ],
}
