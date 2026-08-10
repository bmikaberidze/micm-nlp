"""Georgian sentence tokenizer.

Wraps NLTK's ``PunktSentenceTokenizer`` with a Georgian abbreviation list, then
post-processes the result: a sentence ending in a period whose final token is a
known abbreviation ending is re-joined with the sentence that follows it. Newlines
and tabs are treated as sentence boundaries before splitting.

Research module, from the Georgian tokenization comparison (ICNLSP 2024).

.. warning::
   **This module does not import as shipped.** It requires
   ``micm_nlp.datasets.storage.collections.abbreviations`` and the data files
   ``wiki.abbrs.txt`` and ``abbr.ends.txt``, none of which are part of the package —
   they lived in the pre-rename ``nlpka`` tree. It also calls
   ``nltk.download('punkt')`` at import time, which touches the network. Kept for
   provenance until the data is restored or the module is dropped.
"""

from string import digits

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

import micm_nlp.datasets.storage.collections.abbreviations as abbr
import micm_nlp.utils as utils

nltk.download('punkt')

abbr_path = utils.get_module_location(abbr)


class KaSenTok:
    wikiAbbrsFilename = f'{abbr_path}/wiki.abbrs.txt'
    abbrEndsFilename = f'{abbr_path}/abbr.ends.txt'

    def __init__(self):

        self.tokenizer = None
        self.abbrEnds = {}
        self.abbreviations = set()

        self.custom_sent_ends_tt = str.maketrans({'\n': '. ', '\t': '. '})
        self.remove_digits_tt = str.maketrans('', '', digits)

        with open(self.wikiAbbrsFilename) as wikiAbbrsFile, open(self.abbrEndsFilename) as abbrEndsFile:
            for a in wikiAbbrsFile:
                self.abbreviations.add(a[:-2])

            for ae in abbrEndsFile:
                self.abbrEnds[ae[:-1]] = 1

        punktParam = PunktParameters()
        punktParam.abbrev_types = self.abbreviations
        self.tokenizer = PunktSentenceTokenizer(punktParam)

    def tokenize(self, text):
        # Consider \n and \t as sentence endings
        text = text.translate(self.custom_sent_ends_tt)
        sentences = self.tokenizer.tokenize(text)
        removeIndxStack = []
        for i in range(1, len(sentences)):
            sentence = sentences[i - 1]
            if sentence[-1] == '.':
                lastToken = sentence[sentence.rfind(' ') + 1 :]
                lastToken = lastToken[lastToken.rfind(' ') + 1 :]
                lastWords = word_tokenize(lastToken)
                if len(lastWords) > 1:
                    lastWord = lastWords[-2]
                    lastPart = lastWord[lastWord.rfind('.') + 1 :]
                    lastPart = lastPart.translate(self.remove_digits_tt)
                    if lastPart in self.abbrEnds:
                        removeIndxStack.append(i - 1)
                        sentences[i] = sentences[i - 1] + ' ' + sentences[i]
                        # print(lastPart, word_tokenize(lastToken))

        while removeIndxStack:
            i = removeIndxStack.pop()
            del sentences[i]

        return sentences


if __name__ == '__main__':
    kaSenTok = KaSenTok()
    text = 'სიმღერების სია, ე.ერნ. \
აშშ გამოცემა. \
მათ სანაცვლოდ წარმოდგენილია: \
1867წ. ავსტრია-უნგრეთის კომპრომისი (გერმ. \
Kiegyezés) — შეთანხმება\n მონარქია.'

    print(sent_tokenize(text))
    print(kaSenTok.tokenizer.tokenize(text))
    print(kaSenTok.tokenize(text))
