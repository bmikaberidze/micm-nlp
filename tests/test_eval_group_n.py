"""``compute_metrics_by_metric_groups`` must report the group size alongside
its metrics, prefixed exactly as they are. ``evaluate.combine`` is stubbed so
no metric script is downloaded."""

from types import SimpleNamespace

import pytest

from micm_nlp.evals import eval as ev


class _Stub:
    def compute(self, **kwargs):
        return {'accuracy': 1.0}


def _config(per_task):
    return SimpleNamespace(
        eval=SimpleNamespace(per_task=per_task),
        task=SimpleNamespace(metric_groups=[
            SimpleNamespace(task=SimpleNamespace(id=0, name='eng'), metrics=['accuracy']),
            SimpleNamespace(task=SimpleNamespace(id=1, name='kat'), metrics=['accuracy']),
        ]),
    )


def test_n_per_task(monkeypatch):
    monkeypatch.setattr(ev.evaluate, 'combine', lambda names: _Stub())
    preds = {0: [1, 1, 1], 1: [0, 1]}
    labels = {0: [1, 1, 0], 1: [0, 1]}
    out = ev.compute_metrics_by_metric_groups(preds, labels, _config(per_task='task_ids'))
    assert out == {'eng/accuracy': 1.0, 'eng/n': 3, 'kat/accuracy': 1.0, 'kat/n': 2}


def test_n_without_per_task(monkeypatch):
    monkeypatch.setattr(ev.evaluate, 'combine', lambda names: _Stub())
    cfg = _config(per_task=None)
    cfg.task.metric_groups = cfg.task.metric_groups[:1]
    out = ev.compute_metrics_by_metric_groups([1, 0, 1, 1], [1, 0, 1, 0], cfg)
    assert out == {'accuracy': 1.0, 'n': 4}


def test_missing_group_still_raises_when_nothing_computed(monkeypatch):
    # Regression guard for existing behaviour; passes before and after.
    monkeypatch.setattr(ev.evaluate, 'combine', lambda names: _Stub())
    with pytest.raises(ValueError, match='No metrics computed'):
        ev.compute_metrics_by_metric_groups({}, {}, _config(per_task='task_ids'))
