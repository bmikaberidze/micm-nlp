# Content map — where each part of the site is written

The rule: **documentation is written where the thing it describes lives.** The site
at [micm-nlp.readthedocs.io](https://micm-nlp.readthedocs.io/) is assembled at build
time from docstrings, `README.md` and `CHANGELOG.md`. Almost nothing is written in
`docs/source/` itself.

So if you want to change what the site says, this file tells you which file to edit.
Every path below is relative to the repository root.

---

## 1. "I want to change X" → edit this

| Page on the site | What it holds | Edit |
|---|---|---|
| **Home** | tagline, `pip install`, About | `README.md` (see §2) |
| **Home** | *What it is*, *Scope*, *Links* | `docs/source/index.md` ← **hand-written** |
| **Home** | *Provenance*, *Citation* | `README.md` (see §2) |
| **Install** | everything | `README.md` (see §2) |
| **Install** | *Hardware* section, Python-version line | `docs/source/install.md` ← **hand-written** |
| **Quickstart** | everything | `README.md` (see §2) |
| **YAML configuration** | everything | `docs/source/config.md` ← **hand-written, 243 lines** |
| **API reference** → any module page | the whole page | the **docstrings** in that module (see §3) |
| **API reference** → *Core* heading | just the grouping + blurb | `docs/source/api/core.md` |
| **Changelog** | everything | `CHANGELOG.md` |
| Sidebar order and labels | — | `docs/source/conf.py` (see §4) |

---

## 2. README blocks pulled into the site

`README.md` is the canonical copy of every passage the site shares with it. Each block
is delimited by HTML comments, and a docs page pulls it in with MyST `{include}`.
**Renaming or deleting a marker breaks the page that uses it** — Sphinx logs
`CRITICAL: text not found` but still exits 0, so the section silently renders empty.

| Marker in `README.md` | Used by |
|---|---|
| `tagline` | `index.md` |
| `about` | `index.md` |
| `install-pypi` | `index.md` **and** `install.md` |
| `install-requires` | `install.md` |
| `install-source` | `install.md` |
| `install-docker` | `install.md` |
| `install-env` | `install.md` |
| `quickstart` | `quickstart.md` |
| `stages` | `quickstart.md` |
| `examples` | `quickstart.md` |
| `architectures` | `quickstart.md` |
| `acknowledgements` | `index.md` |
| `citation` | `index.md` |

`docs/source/changelog.md` includes the whole of `CHANGELOG.md`, with no markers.

---

## 3. The API reference is `src/micm_nlp/`

The sidebar under **API reference** is autoapi's tree, so it mirrors the package
directory exactly. To find the file behind a page, read the path off the sidebar:

```
API reference > models > xpe > encoder     ->   src/micm_nlp/models/xpe/encoder.py
API reference > training > runner          ->   src/micm_nlp/training/runner.py
API reference > tokenizers > ka_sen_tok    ->   src/micm_nlp/tokenizers/ka_sen_tok.py
```

Within a page: the intro paragraph is the **module docstring**; each class and
function is its own docstring. A package page (`models`, `training`, `evals`,
`tokenizers`, `datasets`) is that package's `__init__.py` docstring — that is where a
group overview belongs. `src/micm_nlp/models/xpe/__init__.py` is the model to copy.

**Two consequences worth remembering:**

- A grouping that looks wrong in the sidebar means the *package layout* is wrong.
  Move the source; do not hand-list pages in a toctree.
- Docstrings are parsed as **reStructuredText**, not Markdown. A Markdown code fence
  (```` ```py ````) produces a build warning and renders wrong. Use
  `.. code-block:: python`.

**The one exception:** *Core* is not a package. `docs/source/api/core.md` groups the
six top-level modules (`bootstrap`, `pipeline`, `config`, `path`, `enums`, `utils`)
behind one sidebar entry. It holds a toctree and a short blurb, nothing more.

---

## 4. Presentation levers in `docs/source/conf.py`

Not content — but this is where the sidebar's *shape* is decided.

| What | Where |
|---|---|
| Order of the five subpackages | the `API reference` toctree in `index.md` |
| Order of modules inside a package | `_SUBMODULE_ORDER` in `conf.py` |
| Leaf-only sidebar labels (`runner`, not `micm_nlp.training.runner`) | `_shorten_autoapi_sidebar_labels` in `conf.py` |
| `string_f1` pointing at the module page, not its package page | `_SIDEBAR_REDIRECTS` in `conf.py` |
| Logo size, table alignment, sidebar bottom spacing | `docs/source/_static/custom.css` |
| The "copy page source" button | `docs/source/_static/copy-for-llm.js` |

---

## 5. The complete list of hand-written docs

Everything else on the site comes from somewhere else. Only these are written in
`docs/source/` and nowhere else:

| File | Lines | Why it is not a docstring or README section |
|---|---|---|
| `docs/source/config.md` | 243 | The YAML schema: which keys exist and what they mean. No single module owns it, and it is far too long for the README. |
| `docs/source/index.md` | ~45 | *What it is* (the pipeline chain and symbol table), *Scope*, *Links*. Site orientation, not package documentation. |
| `docs/source/install.md` | ~8 | The *Hardware* note and the Python-version line. |
| `docs/source/api/core.md` | ~6 | Describes a grouping that exists only in the sidebar. |

Roughly 300 lines in total, and 243 of them are `config.md`.

---

## 6. Checking your change

The docs build on every push to `main`; Read the Docs rebuilds the site itself.
To see it locally first:

```bash
pip install -r docs/requirements.txt
python -m sphinx -b html docs/source docs/_build/html
```

The build needs only those four packages — `sphinx-autoapi` parses `src/` statically
and never imports it, so torch and transformers are not required.

A healthy build reports **1 warning** — `autoapi/micm_nlp/index.rst isn't included in
any toctree`, which is deliberate. Anything above that is new and worth reading:
a broken `{include}` marker, a dead cross-reference, or malformed RST in a docstring
all show up there.

(`.readthedocs.yaml` sets `fail_on_warning: false`, so none of them fail the build
today. Making them fail is planned.)
