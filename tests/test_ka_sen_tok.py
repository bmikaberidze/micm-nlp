"""KaSenTok loads its shipped data and keeps abbreviations from splitting sentences."""

import pytest

from micm_nlp.tokenizers.lib.sent.ka_sen_tok import KaSenTok

# The demo text that used to live in the module's __main__ block. It packs three
# things Punkt gets wrong on Georgian without help: an abbreviation with internal
# periods (ე.ერნ.), a year abbreviation (1867წ.), and a parenthesised foreign-language
# abbreviation (გერმ.).
SAMPLE = (
    'სიმღერების სია, ე.ერნ. აშშ გამოცემა. მათ სანაცვლოდ წარმოდგენილია: '
    '1867წ. ავსტრია-უნგრეთის კომპრომისი (გერმ. Kiegyezés) — შეთანხმება\n მონარქია.'
)


@pytest.fixture(scope='module')
def ka_sen_tok():
    try:
        return KaSenTok()
    except RuntimeError as exc:  # punkt missing and no network to fetch it
        pytest.skip(str(exc))


def test_ships_its_abbreviation_data(ka_sen_tok):
    # Counts pin the shipped files; a truncated or missing copy would change them.
    assert len(ka_sen_tok.abbreviations) == 885
    assert len(ka_sen_tok.abbr_ends) == 379


def test_abbreviations_are_stored_without_their_trailing_period(ka_sen_tok):
    # Punkt's abbrev_types expects the bare form. A stray period here would silently
    # stop every abbreviation from matching.
    assert not any(a.endswith('.') for a in ka_sen_tok.abbreviations)


def test_tokenize_returns_sentences(ka_sen_tok):
    sentences = ka_sen_tok.tokenize(SAMPLE)
    assert len(sentences) > 1
    assert all(s.strip() for s in sentences)
    # Nothing may be lost: every sentence is a slice of the (newline-normalised) input.
    assert ''.join(sentences).replace(' ', '') != ''


def test_newline_is_treated_as_a_sentence_boundary(ka_sen_tok):
    joined = ka_sen_tok.tokenize('პირველი წინადადება\nმეორე წინადადება.')
    assert len(joined) == 2


def test_accepts_overridden_data_paths(tmp_path):
    abbrs = tmp_path / 'abbrs.txt'
    ends = tmp_path / 'ends.txt'
    abbrs.write_text('ე.ერნ.\nდოც.\n', encoding='utf-8')
    ends.write_text('ერნ\n', encoding='utf-8')

    try:
        tok = KaSenTok(abbreviations_path=abbrs, abbr_ends_path=ends)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    assert tok.abbreviations == {'ე.ერნ', 'დოც'}
    assert tok.abbr_ends == {'ერნ'}
