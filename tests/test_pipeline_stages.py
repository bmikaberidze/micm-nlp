"""The docs' stage-by-stage listing is ``pipeline.run``'s own body.

``run`` calls the core classes directly rather than the single-stage wrappers
beside it, so that the sequence a consumer copies to intervene between two stages
is the sequence the package actually executes. The listing in
``docs/source/quickstart.md`` is a literal copy of it -- Sphinx has no directive
that includes one function's body as code -- and a literal copy drifts. This test
is what stops it: change ``run`` without changing the listing and the failure names
the line.
"""

import inspect
import re
from pathlib import Path

from micm_nlp import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = REPO_ROOT / 'docs' / 'source' / 'quickstart.md'

# The python block under the 'Drive the stages yourself' heading.
_STAGES_BLOCK = re.compile(r'## Drive the stages yourself.*?```python\n(.*?)```', re.DOTALL)

# Lines that belong to the listing but not to a function body: the imports a
# reader needs, the config load ``run`` takes as an argument, and the trailing
# line naming what ``run`` returns.
_LISTING_ONLY = ('from ', 'import ', 'config = CONFIG.from_yaml', 'output = trainer.output')


def _statements(text: str) -> list[str]:
    """The executable lines of a listing: comments and blanks dropped.

    Both sides are normalised the same way, so a stage comment on one and not the
    other is not drift -- only a statement that differs is.
    """
    return [stripped for line in text.splitlines() if (stripped := line.split('#')[0].strip())]


def docs_stage_lines() -> list[str]:
    """The executable statements of the docs listing, in order."""
    match = _STAGES_BLOCK.search(QUICKSTART.read_text())
    assert match, 'the docs quickstart no longer has a python block under Drive the stages yourself'
    return [line for line in _statements(match.group(1)) if not line.startswith(_LISTING_ONLY)]


def run_body_lines() -> list[str]:
    """``pipeline.run``'s statements, docstring and comments stripped."""
    return _statements(inspect.getsource(pipeline.run).split('"""')[2])


def test_the_docs_listing_is_not_empty():
    """Guards the two parsers: a regex that silently matched nothing would make
    every assertion below vacuously true."""
    assert len(docs_stage_lines()) >= 5
    assert len(run_body_lines()) >= 5


def test_every_docs_stage_line_appears_in_run_in_order():
    """The listing is a subsequence of ``run``'s body.

    A subsequence, not equality: ``run`` additionally accepts a path in place of a
    config and returns early for ``mode: preprocess``, neither of which belongs in
    a listing about the stages.
    """
    body = run_body_lines()
    position = -1
    for line in docs_stage_lines():
        assert line in body, f'the docs listing shows a line pipeline.run does not run: {line!r}'
        index = body.index(line)
        assert index > position, f'the docs listing shows {line!r} out of the order pipeline.run uses'
        position = index


def test_run_does_not_route_through_the_single_stage_wrappers():
    """The point of the arrangement: were ``run`` to call ``preprocess_dataset`` or
    ``load_model`` again, the listing would stop describing what it does."""
    body = '\n'.join(run_body_lines())
    for wrapper in ('preprocess_dataset(', 'load_dataset(', 'load_model('):
        assert wrapper not in body, f'pipeline.run calls {wrapper} instead of the class it wraps'
