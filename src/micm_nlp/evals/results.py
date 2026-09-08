"""The one writer of a run's result files.

Every run directory ends up with the resolved config and, when the phases ran,
``valid_res.csv`` (one row per metric group at the best checkpoint) and
``test_res.csv`` (one row per metric group per test pass). The trainer calls
this; runners never write results themselves, so every file in the tree has one
schema.

The CSV contract: the header is fixed by the first rows written; a later row
carrying a column not in the header raises rather than silently drifting; a
row missing a column gets a blank. Rows are appended as they are produced, so a
run killed part-way keeps what it had.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import micm_nlp.utils as utils

CONFIG_FILE = 'config.yml'
TEST_CONFIG_FILE = 'test_config.yml'
VALID_FILE = 'valid_res.csv'
TEST_FILE = 'test_res.csv'
RUN_INFO_FILE = 'run.json'

# (distribution name, key in run.json). The packages that decide numerics.
_VERSIONED = (('micm-nlp', 'micm_nlp'), ('torch', 'torch'), ('transformers', 'transformers'),
              ('peft', 'peft'), ('datasets', 'datasets'))


def environment_info() -> dict[str, Any]:
    """What the run ran on: every ``SLURM*`` variable, host, interpreter, visible
    GPUs, and the versions of the packages that decide numerics.

    The scheduler block is one glob rather than a hand-kept list, so a variable
    the scheduler adds tomorrow is recorded without a code change.
    """
    versions = {}
    for dist, key in _VERSIONED:
        try:
            versions[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return {
        'slurm': {k: v for k, v in sorted(os.environ.items()) if k.startswith('SLURM')},
        'host': platform.node(),
        'python': platform.python_version(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'versions': versions,
    }


def wandb_info(run) -> dict[str, Any] | None:
    """Identity and location of a wandb run, or ``None`` without one.

    ``dir`` is the run directory (the parent of ``run.dir``, which is its
    ``files/``); ``url`` is ``None`` offline, where some wandb versions raise
    on the property.
    """
    if run is None:
        return None
    try:
        url = run.url
    except Exception:
        url = None
    return {'id': run.id, 'url': url, 'dir': str(Path(run.dir).parent), 'path': getattr(run, 'path', None)}


# Bookkeeping keys the HuggingFace Trainer adds to every metrics dict. Not results.
_HF_NOISE = frozenset({
    'runtime', 'samples_per_second', 'steps_per_second',
    'model_preparation_time', 'jit_compilation_time',
})


def rows_from_metrics(metrics: dict[str, Any], hf_prefix: str, strip: str = '') -> list[dict[str, Any]]:
    """Split one HuggingFace metrics dict into one row per metric group.

    Keys look like ``{hf_prefix}_{strip}[{group}/…/]{metric}``: the Trainer
    prepends ``metric_key_prefix + '_'``; the toolkit's ``postproc_metrics``
    prepends the unit config's file name (``TRAINER._metric_prefix``, passed
    here as ``strip``); under ``eval.per_task``,
    :func:`compute_metrics_by_metric_groups` prepends ``{task.name}/``. The
    group is everything between the stripped prefix and the last ``/``; a key
    with no group (``loss``) is copied onto every row. Keys with another prefix,
    and the Trainer's timing keys, are dropped.

    Hand this one pass's dict at a time -- ``'test'`` would also match a
    ``test_zero_…`` key, so the two passes must not be mixed in one call.

    :param metrics: the dict ``Trainer.evaluate`` / ``Trainer.predict`` returned.
    :param hf_prefix: ``'eval'``, ``'test'`` or ``'test_zero'``.
    :param strip: the toolkit's own prefix to remove after ``hf_prefix``.
    :returns: rows of ``{'prefix', 'metric_group', <metrics…>}``; empty if no
        key carried the prefix.
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
        return [{'prefix': hf_prefix, 'metric_group': '', **shared}] if shared else []
    return [{'prefix': hf_prefix, 'metric_group': g, **shared, **m} for g, m in groups.items()]


def best_eval_record(state, metric_for_best_model: str | None) -> tuple[int | None, dict[str, Any]]:
    """The best training step and the ``log_history`` evaluation record at it.

    Prefers ``state.best_model_checkpoint`` (``…/checkpoint-<step>``). Without
    one, finds the evaluation record whose ``metric_for_best_model`` value
    equals ``state.best_metric``, matching the key by suffix because the
    toolkit may have expanded it into a task-prefixed name.

    :returns: ``(step, record)``; ``(None, {})`` when no evaluation ran.
    """
    history = [r for r in (getattr(state, 'log_history', None) or []) if any(k.startswith('eval_') for k in r)]
    best = getattr(state, 'best_model_checkpoint', None)
    step = int(str(best).rsplit('-', 1)[-1]) if best else None
    if step is None and metric_for_best_model and getattr(state, 'best_metric', None) is not None:
        suffix = metric_for_best_model.split('/')[-1]
        for rec in history:
            if any(k.startswith('eval_') and k.endswith(suffix) and v == state.best_metric for k, v in rec.items()):
                step = rec.get('step')
                break
    if step is None:
        return None, {}
    return step, next((r for r in history if r.get('step') == step), {})


class ResultsWriter:
    """Writes into one run directory, stamping ``columns`` onto every row.

    :param run_dir: the directory this run owns; created if missing.
    :param columns: static columns copied into every row (the framework's
        identity columns plus whatever the config or runner added). It stays a
        mutable attribute so the trainer can add the effective seed once the
        HuggingFace Trainer exists. A row's own value wins over a static one.
    """

    def __init__(self, run_dir: str | Path, columns: dict[str, Any] | None = None):
        self.run_dir = Path(run_dir)
        self.columns = dict(columns or {})
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config, filename: str = CONFIG_FILE) -> Path:
        """Save the resolved config as plain YAML (enums as strings)."""
        path = self.run_dir / filename
        utils.dict_to_yaml_file(config.model_dump(mode='json'), str(path))
        return path

    def write_run_info(self, **sections) -> Path:
        """Merge ``sections`` into ``run.json``.

        Read-modify-write, so the file is built up as the run learns things:
        environment and paths at setup, wandb once it exists, ``finished`` at
        the end. Values must be JSON-serialisable.
        """
        path = self.run_dir / RUN_INFO_FILE
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(sections)
        path.write_text(json.dumps(current, indent=2) + '\n')
        return path

    def link(self, name: str, target: str | Path | None) -> Path | None:
        """A symlink ``run_dir/name -> target`` (absolute), so every artefact of
        the run is reachable from its directory.

        A target that does not exist yet still gets a link -- the checkpoint
        directory appears only when the trainer saves. An existing link is left
        alone; ``None`` is a no-op.
        """
        if target is None:
            return None
        link = self.run_dir / name
        if not link.is_symlink() and not link.exists():
            os.symlink(os.path.abspath(target), link)
        return link

    def append(self, filename: str, rows: list[dict[str, Any]]) -> Path:
        """Append rows to ``filename``, creating it with a header on first use.

        :raises ValueError: if a row carries a column the file's header lacks.
        """
        path = self.run_dir / filename
        rows = [{**self.columns, **row} for row in rows]
        if not rows:
            return path
        if path.exists():
            with open(path, newline='') as f:
                header = next(csv.reader(f))
            unknown = sorted({k for r in rows for k in r} - set(header))
            if unknown:
                raise ValueError(f'{path}: columns {unknown} are not in the header {header}')
        else:
            header = []
            for r in rows:
                header.extend(k for k in r if k not in header)
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(header)
        with open(path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=header, restval='').writerows(rows)
        return path
