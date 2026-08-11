"""Georgian sentence tokenizer.

Wraps NLTK's ``PunktSentenceTokenizer`` with a Georgian abbreviation list, then
post-processes the result: a sentence ending in a period whose final token is a known
abbreviation ending is re-joined with the sentence that follows it. Newlines and tabs
are treated as sentence boundaries before splitting.

Two data files ship with the package, both extracted from Georgian Wikipedia:

``data/wiki.abbrs.txt``
    885 abbreviations, one per line, each written with its trailing period. They are
    handed to Punkt as ``abbrev_types`` (which expects them *without* the period), so
    Punkt does not treat those periods as sentence ends.

``data/abbr.ends.txt``
    379 tokens that commonly end an abbreviated form. Punkt splits on some of these
    anyway; the post-processing pass in :meth:`KaSenTok.tokenize` glues those
    sentences back together.

Research module, from the Georgian tokenization comparison (ICNLSP 2024).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from string import digits

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

_DATA_PACKAGE = 'micm_nlp.tokenizers.lib.sent'
_DATA_DIR = 'data'
_WIKI_ABBRS_FILE = 'wiki.abbrs.txt'
_ABBR_ENDS_FILE = 'abbr.ends.txt'


def _ensure_punkt() -> None:
    """Make sure NLTK's sentence-splitting models are available.

    Only downloads when they are missing, so importing or constructing does not touch
    the network on a machine that already has them. ``punkt_tab`` is what NLTK 3.9+
    looks for; ``punkt`` covers older versions.
    """
    for resource in ('punkt_tab', 'punkt'):
        try:
            nltk.data.find(f'tokenizers/{resource}')
            return
        except LookupError:
            continue

    for resource in ('punkt_tab', 'punkt'):
        if nltk.download(resource, quiet=True):
            return

    raise RuntimeError(
        "NLTK's punkt models are missing and could not be downloaded. Install them "
        "with nltk.download('punkt_tab') on a machine with network access."
    )


def _read_lines(file_name: str, override: str | Path | None) -> list[str]:
    if override is not None:
        text = Path(override).read_text(encoding='utf-8')
    else:
        text = resources.files(_DATA_PACKAGE).joinpath(_DATA_DIR, file_name).read_text(encoding='utf-8')
    return [line.strip() for line in text.splitlines() if line.strip()]


class KaSenTok:
    """Sentence-split Georgian text.

    :param abbreviations_path: override for ``data/wiki.abbrs.txt``.
    :param abbr_ends_path: override for ``data/abbr.ends.txt``.

    Both default to the copies shipped with the package.
    """

    def __init__(self, abbreviations_path=None, abbr_ends_path=None):
        _ensure_punkt()

        # Punkt wants abbreviations without their trailing period.
        self.abbreviations = {line.removesuffix('.') for line in _read_lines(_WIKI_ABBRS_FILE, abbreviations_path)}
        self.abbr_ends = set(_read_lines(_ABBR_ENDS_FILE, abbr_ends_path))

        self.custom_sent_ends_tt = str.maketrans({'\n': '. ', '\t': '. '})
        self.remove_digits_tt = str.maketrans('', '', digits)

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = self.abbreviations
        self.tokenizer = PunktSentenceTokenizer(punkt_param)

    def tokenize(self, text: str) -> list[str]:
        """Split ``text`` into sentences."""
        # Treat newlines and tabs as sentence endings.
        text = text.translate(self.custom_sent_ends_tt)
        sentences = self.tokenizer.tokenize(text)

        remove_indices = []
        for i in range(1, len(sentences)):
            sentence = sentences[i - 1]
            if sentence[-1] == '.':
                last_token = sentence[sentence.rfind(' ') + 1 :]
                last_token = last_token[last_token.rfind(' ') + 1 :]
                last_words = word_tokenize(last_token)
                if len(last_words) > 1:
                    last_word = last_words[-2]
                    last_part = last_word[last_word.rfind('.') + 1 :]
                    last_part = last_part.translate(self.remove_digits_tt)
                    if last_part in self.abbr_ends:
                        remove_indices.append(i - 1)
                        sentences[i] = sentences[i - 1] + ' ' + sentences[i]

        while remove_indices:
            del sentences[remove_indices.pop()]

        return sentences
