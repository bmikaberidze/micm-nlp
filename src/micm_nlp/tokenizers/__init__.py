"""Tokenizer loading, architecture adaptation, and sentence splitting.

- :mod:`.tokenizer` — ``load()``, the ``AutoTokenizer`` factory; special tokens and
  post-processors per architecture; ``tokenize_sentences``; tokenizer training.
- :mod:`.decoding` — label-aware ``decode`` / ``batch_decode``.
- :mod:`.architectures` — the two tokenizer classes HuggingFace does not ship,
  :class:`~micm_nlp.tokenizers.architectures.BertByT5Tokenizer` and
  :class:`~micm_nlp.tokenizers.architectures.CustomXlmRoberta`.
- :mod:`.ka_sen_tok` — :class:`~micm_nlp.tokenizers.ka_sen_tok.KaSenTok`, a Georgian
  *sentence* splitter. Not a subword tokenizer, and not reachable through
  ``AutoTokenizer``.

Adding a backbone normally needs nothing here: ``tokenizer.cls`` in the YAML resolves
a class by name at runtime.
"""
