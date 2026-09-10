"""Evaluation: metric assembly, prediction post-processing and plots.

- :mod:`.eval` — ``get_compute_metrics`` builds the ``compute_metrics`` callable the
  ``Trainer`` expects; ``get_preprocess_logits_for_metrics`` is its companion hook,
  which runs before logits are accumulated. Also handles per-task grouping and label
  restriction.
- :mod:`.results` — one evaluation event becomes one file in the run's output:
  ``save_metrics`` writes the metrics CSV, ``save_predictions`` the per-sample one.
- :mod:`.plot` — confusion matrices.
- :mod:`.metrics` — the metric implementations ``evaluate`` does not provide
  directly. See that package's docstring: it holds two different module shapes, on
  purpose.
"""
