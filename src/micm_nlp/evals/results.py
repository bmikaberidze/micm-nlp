"""One evaluation event -> one file, in the run's output.

``save_metrics`` turns the dict ``Trainer.evaluate`` / ``Trainer.predict``
returned into ``<event>.csv`` (one row per metric group, the run's static
columns stamped on every row); ``save_predictions`` turns a ``predict()``
output into ``predictions_<stage>.csv`` (one row per sample, after the same
preprocessing the metric saw). Both write into a
:class:`~micm_nlp.training.run_output.RunOutput`, once, whole. The trainer
calls them; runners never write results themselves.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from micm_nlp.evals.eval import preproc_preds_labels

# Bookkeeping keys the HuggingFace Trainer adds to every metrics dict. Not results.
_HF_NOISE = frozenset({
    'runtime', 'samples_per_second', 'steps_per_second',
    'model_preparation_time', 'jit_compilation_time',
})


def metric_rows(metrics: dict[str, Any], hf_prefix: str, strip: str = '') -> list[dict[str, Any]]:
    """Split one HuggingFace metrics dict into one row per metric group.

    Keys look like ``{hf_prefix}_{strip}[{group}/…/]{metric}``: the Trainer
    prepends ``metric_key_prefix + '_'``; the toolkit's ``postproc_metrics``
    prepends the unit config's file name (``TRAINER._metric_prefix``, passed
    here as ``strip``); under ``eval.per_task`` the group's ``task.name`` comes
    next. The group is everything between the stripped prefix and the last
    ``/``; a key with no group (``loss``) is copied onto every row. Keys with
    another prefix, and the Trainer's timing keys, are dropped.

    :returns: rows of ``{'metric_group', <metrics…>}``; empty if no key
        carried the prefix.
    """
    lead = f'{hf_prefix}_'
    shared: dict[str, Any] = {}
    groups: dict[str, dict[str, Any]] = {}
    for key, value in metrics.items():
        if not key.startswith(lead):
            continue
        rest = key[len(lead):]
        if strip and rest.startswith(strip):
            rest = rest[len(strip):]
        group, _, metric = rest.rpartition('/')
        if metric in _HF_NOISE:
            continue
        (groups.setdefault(group, {}) if group else shared)[metric] = value
    if not groups:
        return [{'metric_group': '', **shared}] if shared else []
    return [{'metric_group': g, **shared, **m} for g, m in groups.items()]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``rows`` to ``path`` -- header = the union of keys in first-seen order."""
    path = Path(path)
    header: list[str] = []
    for r in rows:
        header.extend(k for k in r if k not in header)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, restval='')
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_metrics(output, event: str, metrics: dict[str, Any], hf_prefix: str,
                 strip: str = '', step: int | None = None) -> Path | None:
    """Write one event's metrics as ``<prefix><event>.csv`` in the run's output.

    One row per metric group, the run's static columns stamped on every row,
    ``step`` added when given. Written once, whole; a second write of the same
    event replaces the file.

    :returns: the path, or ``None`` when no key carried ``hf_prefix``.
    """
    rows = metric_rows(metrics, hf_prefix, strip)
    if not rows:
        return None
    if step is not None:
        for row in rows:
            row['step'] = step
    return write_csv(output.file(f'{event}.csv'), [{**output.columns, **row} for row in rows])


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _sample_rows(preds, labels, order, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, (p, l) in enumerate(zip(preds, labels, strict=True)):
        sample = order[i] if order is not None else i
        if isinstance(p, (list, np.ndarray)):                              # token classification
            for pos, (pp, ll) in enumerate(zip(p, l, strict=True)):
                rows.append({**base, 'sample': sample, 'position': pos, 'prediction': _scalar(pp), 'label': _scalar(ll)})
        else:
            rows.append({**base, 'sample': sample, 'prediction': _scalar(p), 'label': _scalar(l)})
    return rows


def save_predictions(output, stage: str, pred_out, config, label_pad_id, tokenizer, ds_split,
                     order=None) -> Path:
    """Write one test pass's predictions as ``<prefix>predictions_<stage>.csv``.

    One row per sample (per token position for token classification), after
    the same preprocessing the metric saw (:func:`preproc_preds_labels`), so
    every metric is recomputable from the file. ``sample`` is the index into
    the test split -- the original index when ``order`` (the dataloader's emit
    order) is given, else the emit position. Under ``task.preproc_rules.per_task``
    the rows carry ``task`` as well, and ``sample`` is still the split index: the
    per-task groups are re-joined to the split's order (and to ``order`` when
    given), and the ``'all'`` copy the metrics use is skipped. No static columns:
    the file lives in the run's directory, which identifies the run.
    """
    preds, labels = preproc_preds_labels(pred_out.predictions, pred_out.label_ids, config, label_pad_id, tokenizer, ds_split)
    if isinstance(preds, dict):
        # per_task: grouped by task id, in the split's emit order, plus an 'all'
        # copy of the ungrouped arrays that the metrics use and this file does not.
        group_by = config.task.preproc_rules.per_task
        positions = list(order) if order is not None else list(range(len(ds_split)))
        indices: dict[Any, list[int]] = {}
        for i, sample in enumerate(ds_split):
            indices.setdefault(sample[group_by], []).append(positions[i])
        rows = []
        for task_id, task_preds in preds.items():
            if task_id == 'all':
                continue
            rows.extend(_sample_rows(task_preds, labels[task_id], indices[task_id], {'task': task_id}))
    else:
        rows = _sample_rows(preds, labels, order, {})
    return write_csv(output.file(f'predictions_{stage}.csv'), rows)
