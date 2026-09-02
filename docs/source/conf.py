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

# The API reference section is autoapi's tree, with two deliberate interventions and
# no others.
#
# 1. Order. index.md lists the five subpackages explicitly, because a toctree is the
#    only lever on sidebar order (autoapi_member_order sorts members *within* a page,
#    not pages) and the useful order is the pipeline's -- tokenizers, datasets,
#    models, training, evals -- not the alphabet's. A :glob: entry would be
#    maintenance-free but alphabetical.
# 2. Core. The six top-level modules are grouped behind api/core.md, which holds a
#    toctree and nothing else. "Core" is not a package in src/, so this is the one
#    place the sidebar shows a grouping the source does not have. The principled fix
#    is a src/micm_nlp/core/ package; that renames micm_nlp.config and friends, which
#    every consumer imports, so it is a major-version decision -- see
#    docs/internal/roadmap.md.
#
# Everything below the top level is autoapi's own toctrees, so the tree there mirrors
# src/micm_nlp/ exactly. Curation belongs in the package docstrings, which autoapi
# renders -- see models/xpe/__init__.py for the shape.
#
# autoapi/micm_nlp/index is not linked, so that stub is orphaned and costs one
# "isn't included in any toctree" warning per build -- see the note below.

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
html_js_files = ['copy-for-llm.js']

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


# Sidebar labels are the leaf module name -- "runner", not
# "micm_nlp.training.runner". The parent package is the entry directly above in the
# tree, so the dotted prefix repeats it at every level.
#
# Sphinx renders a toctree from ``env.tocs`` -- the per-document TOC built by
# TocTreeCollector at priority 500 -- not from ``env.titles``. Rewriting the label in
# ``env.tocs`` after that collector has run shortens every toctree entry while leaving
# the page's own H1, the breadcrumb and ``env.titles`` fully qualified, which is where
# a reader actually needs the dotted path.


_SUBMODULE_ORDER = {
    'tokenizers': ['tokenizer', 'decoding', 'architectures', 'ka_sen_tok'],
    'models': ['model', 'architectures', 'peft', 'xpe'],
    'training': [
        'runner',
        'trainers',
        'callbacks',
        'batching',
        'data_collators',
        'logits_processors',
    ],
    'evals': ['eval', 'plot', 'metrics'],
}
"""Sidebar order for each package's submodules, by leaf name.

autoapi's package template sorts submodules alphabetically. That puts ``batching``
before ``runner`` and ``architectures`` before ``tokenizer``, which reads as a list
rather than as the order someone meets these modules in. Anything not named here
keeps its alphabetical position at the end, so a new module still appears -- it is
never silently dropped from the sidebar.

The six top-level modules are not here: they have no package page to reorder, and
their order lives in the toctree in ``api/core.md``.
"""


def _order_autoapi_submodules(app, doctree):
    """Reorder a package page's submodule toctree to _SUBMODULE_ORDER."""
    from sphinx import addnodes

    docname = app.env.docname
    parts = docname.split('/')
    # autoapi/micm_nlp/<package>/index
    if len(parts) != 4 or parts[0] != autoapi_root or parts[3] != 'index':
        return

    order = _SUBMODULE_ORDER.get(parts[2])
    if not order:
        return

    def rank(entry):
        # Entries are (title, ref); ref is the generated page, ".../<leaf>/index".
        ref = entry[1]
        suffix = '/index'
        if ref.endswith(suffix):
            ref = ref[: -len(suffix)]
        leaf = ref.rsplit('/', 1)[-1]
        # Unlisted modules keep their alphabetical position, after the listed ones.
        return (order.index(leaf), '') if leaf in order else (len(order), leaf)

    for node in doctree.findall(addnodes.toctree):
        node['entries'] = sorted(node['entries'], key=rank)


_SIDEBAR_REDIRECTS = {
    'autoapi/micm_nlp/evals/metrics/string_f1/index': (
        'autoapi/micm_nlp/evals/metrics/string_f1/string_f1/index'
    ),
}
"""Sidebar entries to point at a module page instead of its package page.

``evals/metrics/string_f1`` must be a directory holding a module of the same name --
that is what ``evaluate.load('<dir>')`` requires, see the docstring in
``evals/metrics/__init__.py``. Linking the package page therefore nests "string_f1"
inside "string_f1" in the sidebar. Here, and only here, the source shape is fixed by
an external tool rather than by us, so the fix is presentational: link the module
page, which carries the actual class documentation. The package page is still built
and still reachable from the generated tree.
"""


def _redirect_sidebar_entries(app, doctree):
    """Repoint toctree entries listed in _SIDEBAR_REDIRECTS."""
    from sphinx import addnodes

    for node in doctree.findall(addnodes.toctree):
        node['entries'] = [
            (title, _SIDEBAR_REDIRECTS.get(ref, ref)) for title, ref in node['entries']
        ]


def _shorten_autoapi_sidebar_labels(app, doctree):
    """Label autoapi pages by their leaf name in every toctree that lists them."""
    from docutils import nodes as _nodes

    docname = app.env.docname
    if not docname.startswith(f'{autoapi_root}/'):
        return

    toc = app.env.tocs.get(docname)
    if toc is None:
        return

    # The document's own entry is the only reference with an empty anchor.
    for ref in toc.findall(_nodes.reference):
        if ref.get('anchorname'):
            continue
        full = ref.astext()
        leaf = full.rsplit('.', 1)[-1]
        if leaf and leaf != full:
            ref.children = [_nodes.Text(leaf)]
        break


def setup(app):
    # Ordering must run before Sphinx's TocTreeCollector (priority 500) reads the
    # toctree nodes; relabelling must run after it has built env.tocs.
    app.connect('doctree-read', _order_autoapi_submodules, priority=400)
    app.connect('doctree-read', _redirect_sidebar_entries, priority=400)
    app.connect('doctree-read', _shorten_autoapi_sidebar_labels, priority=900)
