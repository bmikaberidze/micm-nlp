"""The package describes itself in four places; they must not drift apart.

The tagline lives in the README (pulled into the docs landing page), in
``pyproject.toml``'s ``description`` (PyPI's one-line summary, which cannot be
the README because search listings do not render markdown), in the package
docstring (``help(micm_nlp)`` and the API reference), and in the citation block.
Four hand-maintained copies of one sentence drift: the citation's ``version``
field silently fell behind the real version twice before this test existed.

Rather than demand four identical strings -- they cannot be, since a docstring
summary line has a length limit and a BibTeX title is a title -- these tests pin
the phrase they share and the claim they all make.
"""

import re
from pathlib import Path

import micm_nlp

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / 'README.md'
PYPROJECT = REPO_ROOT / 'pyproject.toml'

# The words every self-description carries, verbatim. Change it here and the
# failures tell you every place that has to follow.
CORE_PHRASE = 'a research framework for NLP'

# The full sentence, shared by the README tagline and the PyPI summary.
TAGLINE = (
    'A research framework for NLP: the whole pipeline in a single YAML, run alone '
    'or in groups. Builds on the HuggingFace stack and adds a layer of features of its own.'
)


def _readme() -> str:
    return README.read_text(encoding='utf-8')


def _marker_block(text: str, name: str) -> str:
    """The text between ``<!-- start:name -->`` and ``<!-- end:name -->``."""
    match = re.search(rf'<!-- start:{name} -->\n(.*?)\n<!-- end:{name} -->', text, re.S)
    assert match, f'README has no {name!r} marker block'
    return match.group(1).strip()


def _project_field(key: str) -> str:
    """One simple string field from ``pyproject.toml``'s ``[project]`` table.

    Read with a regex rather than ``tomllib``, which is 3.11+ while the package
    supports 3.10. Both fields this test needs are plain one-line strings.
    """
    text = PYPROJECT.read_text(encoding='utf-8')
    section = re.search(r'^\[project\]\n(.*?)(?=^\[)', text, re.S | re.M)
    assert section, 'pyproject.toml has no [project] table'
    match = re.search(rf'^{key} = "(.*)"$', section.group(1), re.M)
    assert match, f'[project] has no {key!r} field on a single line'
    return match.group(1)


def test_readme_tagline_is_the_canonical_sentence():
    assert _marker_block(_readme(), 'tagline') == TAGLINE


def test_pypi_summary_matches_the_readme_tagline():
    """``description`` is PyPI's summary; ``readme`` supplies the page body."""
    assert _project_field('description') == TAGLINE


def test_package_docstring_carries_the_core_phrase():
    """Split across summary and body -- a docstring's first line is capped."""
    doc = micm_nlp.__doc__ or ''
    assert CORE_PHRASE in doc.splitlines()[0], f'first docstring line: {doc.splitlines()[0]!r}'
    assert 'the whole pipeline in a single yaml' in doc.lower()


def test_citation_title_carries_the_core_phrase():
    citation = _marker_block(_readme(), 'citation')
    title = re.search(r'^\s*title\s*=\s*\{(.+?)\},?\s*$', citation, re.M)
    assert title, 'citation block has no title field'
    assert CORE_PHRASE in title.group(1)


def test_citation_version_matches_the_real_version():
    """This is the drift that already happened twice."""
    citation = _marker_block(_readme(), 'citation')
    version = re.search(r'^\s*version\s*=\s*\{(.+?)\},?\s*$', citation, re.M)
    assert version, 'citation block has no version field'
    assert version.group(1) == _project_field('version')


def test_no_stale_self_descriptions_remain():
    """'toolkit' undersells and was the previous word; it should be gone from
    the places where the package names its own category."""
    for path in (README, PYPROJECT, REPO_ROOT / 'src' / 'micm_nlp' / '__init__.py'):
        text = path.read_text(encoding='utf-8')
        # pyproject's dependency-pinning comment says "This is a library",
        # which is the correct term in that context -- only 'toolkit' is banned.
        assert 'toolkit' not in text.lower(), f'{path.name} still says "toolkit"'
