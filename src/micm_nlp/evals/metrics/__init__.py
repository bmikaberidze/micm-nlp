"""Metric implementations the ``evaluate`` library does not provide directly.

Two shapes live here, because there are two ways a metric reaches
:func:`micm_nlp.evals.eval.get_compute_metrics`:

**Plain modules** — :mod:`.log_likelihood` and :mod:`.multirc` expose functions that
``eval.py`` imports and calls directly, for metrics whose inputs do not fit
``evaluate``'s ``(predictions, references)`` contract.

**A loadable metric directory** — :mod:`.string_f1` is an ``evaluate.Metric``
subclass, named in ``task.metric_groups[].metrics`` and handed to
``evaluate.combine()``. ``evaluate.load('<dir>')`` looks for ``<dir>/<dirname>.py``,
so that metric *must* be a directory containing a module of the same name, with an
``__init__.py`` re-exporting the class and a ``_metric`` factory. The doubled name is
that convention, not an accident — do not flatten it to a single module.
"""
