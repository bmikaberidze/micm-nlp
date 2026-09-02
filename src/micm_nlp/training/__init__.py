"""Training: the run driver, Trainer subclasses, callbacks, collators and batching.

- :mod:`.runner` — :class:`~micm_nlp.training.runner.TRAINER`: builds the HuggingFace
  ``Trainer`` from the config — arguments, collator, callbacks, evaluation — and runs
  it.
- :mod:`.trainers` — ``CustomTrainerMixin`` and ``RandomTaskExclusionBatchSampler``;
  ``custom_trainer_class_factory`` mixes the former into whichever ``Trainer`` class
  ``trainer.cls`` names.
- :mod:`.callbacks` — ``CustomEarlyStoppingCallback``, ``ParamNormLogger``,
  ``NormalizePromptEncoderEmbeddings``, ``DownstreamFineTuningCallback``,
  ``EmptyCudaCacheCallback``.
- :mod:`.batching` — ``TokenBudgetBatchSampler`` and ``calibrate_token_budget``, for
  batching by token count rather than by row count.
- :mod:`.data_collators` — collators for PLM, seq2seq-with-shifted-labels, and
  task-id-decorated batches.
- :mod:`.logits_processors` — ``ConstrainedPrefixLogitsProcessor``, applied at
  generation time.
"""
