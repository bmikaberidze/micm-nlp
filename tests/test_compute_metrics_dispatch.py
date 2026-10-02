"""_compute_metrics dispatches on the first metric to exactly one scorer.

Pins the MultiRC fix: the log-likelihood check used to be a separate `if`, so for
`multirc` its `else` ran too and replaced the MultiRC scores with the generic path.
"""
from types import SimpleNamespace

import pytest

import micm_nlp.evals.eval as ev


def _config(metric):
    return SimpleNamespace(task=SimpleNamespace(metric_groups=[SimpleNamespace(metrics=[metric])]))


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(ev, 'compute_multirc', lambda *a: seen.append('multirc') or {'m': 1})
    monkeypatch.setattr(ev, 'compute_log_likelihood_accurac', lambda *a: seen.append('ll') or {'l': 1})
    monkeypatch.setattr(ev, 'compute_metrics_by_metric_groups', lambda *a: seen.append('generic') or {'g': 1})
    return seen


@pytest.mark.parametrize('metric,scorer,result', [
    ('multirc', 'multirc', {'m': 1}),
    ('log_likelihood_accuracy', 'll', {'l': 1}),
    ('accuracy', 'generic', {'g': 1}),
])
def test_exactly_one_scorer_runs(calls, metric, scorer, result):
    assert ev._compute_metrics(None, None, _config(metric), None) == result
    assert calls == [scorer]
