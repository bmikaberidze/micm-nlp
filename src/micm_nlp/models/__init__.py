"""Backbone construction, PEFT dispatch, and the Cross-Prompt Encoder.

- :mod:`.model` — :class:`~micm_nlp.models.model.MODEL`: ``from_pretrained`` through
  ``model.pretrained.cls``, plus the kwargs derived from the task (``num_labels`` for
  classification, and so on).
- :mod:`.peft` — :class:`~micm_nlp.models.peft.PEFT`: routes to a stock PEFT method or
  to the Cross-Prompt Encoder path.
- :mod:`.architectures` — model classes HuggingFace does not ship, currently
  ``CustomT5ForConditionalGeneration``.
- :mod:`.xpe` — the Cross-Prompt Encoder, from *Cross-Prompt Encoder for
  Low-Performing Languages* (`Findings of IJCNLP-AACL 2025
  <https://aclanthology.org/2025.findings-ijcnlp.144/>`_).

As with tokenizers, a new backbone is selected by name from ``transformers`` through
``model.pretrained.cls`` and needs no code here.
"""
