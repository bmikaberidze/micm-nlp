"""Render GitHub alert quotes (``> [!NOTE]``) as Sphinx admonitions.

The README is shown on GitHub and included into these docs, so a note has to look
right in both. MyST's ``:::{note}`` prints as raw text on GitHub, and a plain
``> **Note:**`` quote is a grey block on both. GitHub's alert syntax renders as a
coloured box there; this turns the same quotes into Furo's note / warning cards here::

    > [!NOTE]
    > The preprocessing phase can run in every unit run.

A block quote whose text starts with ``[!NOTE]``, ``[!TIP]``, ``[!IMPORTANT]``,
``[!WARNING]`` or ``[!CAUTION]`` becomes that admonition, marker removed. Anything
else stays a block quote. It runs on ``doctree-read``, after ``{include}`` has pulled
README blocks in, so included notes convert too.
"""

from __future__ import annotations

import re

from docutils import nodes

_KINDS = {
    'NOTE': nodes.note,
    'TIP': nodes.tip,
    'IMPORTANT': nodes.important,
    'WARNING': nodes.warning,
    'CAUTION': nodes.caution,
}
_MARKER = re.compile(r'^\s*\[!(' + '|'.join(_KINDS) + r')\]\s*')


def _strip_leading(paragraph: nodes.paragraph, count: int) -> None:
    """Remove the first ``count`` characters of text, then any whitespace after them."""
    for child in list(paragraph.children):
        if not isinstance(child, nodes.Text):
            break
        text = str(child)
        if count >= len(text):
            count -= len(text)
            paragraph.remove(child)
            continue
        rest = text[count:].lstrip()
        count = 0
        if rest:
            paragraph.replace(child, nodes.Text(rest))
            break
        paragraph.remove(child)


def convert(app, doctree: nodes.document) -> None:
    for quote in list(doctree.findall(nodes.block_quote)):
        first = quote.children[0] if quote.children else None
        if not isinstance(first, nodes.paragraph):
            continue
        match = _MARKER.match(first.astext())
        if not match:
            continue
        _strip_leading(first, match.end())
        if not first.astext().strip():
            quote.remove(first)
        admonition = _KINDS[match.group(1)]()
        admonition.extend(quote.children)
        quote.replace_self(admonition)


def setup(app):
    app.connect('doctree-read', convert)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
