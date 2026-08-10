"""Directory wrapper for the string-F1 metric.

Re-exports :class:`~micm_nlp.evals.metrics.string_f1.string_f1.StringF1` and a
``_metric`` factory, the shape ``evaluate`` expects when loading a metric from a
local directory rather than the Hub.
"""

from .string_f1 import StringF1


def _metric(*args, **kwargs):
    return StringF1()
