"""The documentation URLs written by hand must all name the same host.

``README.md`` and ``FEATURES.md`` are rendered by GitHub, which substitutes
nothing -- no MyST role, no Sphinx substitution, no include. So their links into
the documentation have to be absolute, and there are twenty-five of them. Markdown
offers no variable to hold the host in one place, which is what these tests are
for: the host lives once, in ``conf.py``'s ``html_baseurl``, and a mismatch fails
here instead of shipping a dead link.

That matters at exactly one moment -- moving the docs to a custom domain. Change
``html_baseurl``, run the suite, and the failures enumerate every line to follow.
"""

import re
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_PY = REPO_ROOT / 'docs' / 'source' / 'conf.py'

# The files whose links are hand-written. Pages under docs/source/ are excluded on
# purpose: they cross-reference with {doc} roles, which Sphinx resolves itself.
HAND_WRITTEN = ('README.md', 'FEATURES.md')

# Any http(s) URL, stopping at the markdown delimiters that can close one.
_URL = re.compile(r'https?://[^\s)\]<>"]+')

# A bare hostname in prose -- README's docs link shows its host as the link text.
# Applied only after the URLs have been cut out of the text, so it cannot match
# the host half of one; the guards keep it from starting mid-label ('-nlp.…').
_BARE_RTD_HOST = re.compile(r'(?<![\w.-])[\w-]+\.readthedocs\.io(?![\w.-])')

# The fallback literal in ``html_baseurl = os.environ.get(..., '<here>')``. Read
# with a regex rather than by importing conf.py, which would need Sphinx present.
_BASEURL = re.compile(r"^html_baseurl\s*=.*?'(https?://[^']+)'", re.MULTILINE)


def canonical_base() -> str:
    """The documentation base URL, with exactly one trailing slash."""
    match = _BASEURL.search(CONF_PY.read_text())
    assert match, 'conf.py no longer defines html_baseurl with a literal fallback'
    return match.group(1).rstrip('/') + '/'


def urls_in(filename: str) -> list[str]:
    return _URL.findall((REPO_ROOT / filename).read_text())


def test_conf_py_defines_a_canonical_base():
    """Without it Sphinx emits no <link rel="canonical"> and nothing is pinned."""
    base = canonical_base()
    assert urlparse(base).scheme == 'https'
    assert urlparse(base).netloc


def test_doc_page_links_use_the_canonical_base():
    """Every link to a documentation *page* starts with the canonical base.

    A documentation page is recognised by the ``/en/<version>/`` segment Read the
    Docs puts in every URL it serves -- present whatever the host, so this test
    keeps working after a move to a custom domain.
    """
    base = canonical_base()
    offenders = [
        (filename, url)
        for filename in HAND_WRITTEN
        for url in urls_in(filename)
        if re.search(r'/en/[\w.-]+/', url) and not url.startswith(base)
    ]
    assert not offenders, f'documentation links not on {base}: {offenders}'


def test_no_stale_readthedocs_host_survives():
    """No ``*.readthedocs.io`` address may appear once the canonical host is elsewhere.

    Before a custom domain this is satisfied by the canonical host itself. After
    one, it is the test that finds the links a move left behind -- including
    README's, which prints its host as the link text and so is not a URL at all.

    ``readthedocs.org`` is a different host, serving the build badge, and is left
    alone deliberately: it does not move when the documentation does.
    """
    canonical_host = urlparse(canonical_base()).netloc
    offenders = []
    for filename in HAND_WRITTEN:
        text = (REPO_ROOT / filename).read_text()
        hosts = [urlparse(url).netloc for url in _URL.findall(text)]
        hosts += _BARE_RTD_HOST.findall(_URL.sub(' ', text))
        offenders += [
            (filename, host)
            for host in hosts
            if host.endswith('.readthedocs.io') and host != canonical_host
        ]
    assert not offenders, f'stale documentation host, canonical is {canonical_host}: {offenders}'
