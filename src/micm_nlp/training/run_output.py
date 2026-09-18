"""A run's output directory, and everything the trainer writes into it that is
not a metric or a prediction.

``RunOutput`` is constructed by the trainer before the HuggingFace arguments
exist (they need the directory) and holds: the directory, the static columns
stamped onto every metrics row, the config snapshot (``config.yml`` -- the
config as the framework resolved it, written before anything non-serialisable
can reach it), ``info.json`` (environment, paths, the values the trainer
resolved, the wandb identity, start/finish) and the ``model`` / ``wandb``
symlinks. Metrics and predictions files are written by
:mod:`micm_nlp.evals.results`, into :meth:`RunOutput.file`.

The config is read-only for the trainer: what the run *did* is recorded here,
never written back into the config.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import micm_nlp.utils as utils
from micm_nlp.path import output_dir

CONFIG_FILE = 'config.yml'
TEST_CONFIG_FILE = 'test_config.yml'
RUN_INFO_FILE = 'info.json'

# (distribution name, key in info.json). The packages that decide numerics.
_VERSIONED = (('micm-nlp', 'micm_nlp'), ('torch', 'torch'), ('transformers', 'transformers'),
              ('peft', 'peft'), ('datasets', 'datasets'))

_PACKAGE_DIR = Path(__file__).resolve().parent


def _git(*args) -> str | None:
    """``git`` in the package's own directory; ``None`` when it fails or is absent."""
    try:
        done = subprocess.run(['git', '-C', str(_PACKAGE_DIR), *args],
                              capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def package_commit() -> str | None:
    """This package's own commit, ``-dirty`` when its tree has uncommitted changes;
    ``None`` when it is an installed copy rather than a checkout.

    The version alone does not identify the code: an editable install follows a working
    tree that moves between releases, which is how a result stops being reproducible.
    """
    commit = _git('rev-parse', '--short', 'HEAD')
    if not commit:
        return None
    return f'{commit}-dirty' if _git('status', '--porcelain') else commit


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
    versions['micm_nlp_commit'] = package_commit()
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


def output_dir_for(config, model_name: str) -> str:
    """The directory a run writes into.

    ``output.dir`` wins when set -- the group runner puts the run there.
    Otherwise the run belongs to no group and lands under ``runs/units/{model_name}``,
    where ``model_name`` is the generated name (uuid first), unique and equal to the
    wandb run name.
    """
    output = getattr(config, 'output', None)
    if output is not None and output.dir:
        return str(output.dir)
    return str(output_dir(None, model_name))


def write_run_info(dir_: str | Path, **sections) -> Path:
    """Merge ``sections`` into ``<dir_>/info.json`` (read-modify-write).

    Dict-valued sections merge one level deep; ``started`` is kept from the
    first write; everything else is latest-wins. Values must be
    JSON-serialisable.
    """
    path = Path(dir_) / RUN_INFO_FILE
    current = json.loads(path.read_text()) if path.exists() else {}
    for key, value in sections.items():
        if key == 'started':
            current.setdefault(key, value)
        elif isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key].update(value)
        else:
            current[key] = value
    path.write_text(json.dumps(current, indent=2) + '\n')
    return path


def write_config(dir_: str | Path, config, filename: str = CONFIG_FILE) -> Path:
    """Save a config as plain YAML (enums as strings) into ``dir_``, creating it."""
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / filename
    utils.dict_to_yaml_file(config.model_dump(mode='json'), str(path))
    return path


class RunOutput:
    """One run's output directory.

    :param config: the run's config; its ``output`` block (if any) supplies the
        directory, the file prefix, the config file name and static columns.
    :param model: the :class:`~micm_nlp.models.model.MODEL` -- its ``name``
        names a solo run's directory, its ``uuid4`` is stamped on every row,
        its ``path`` (when the mode has a checkpoint dir) is linked as ``model``.
    """

    def __init__(self, config, model):
        output = getattr(config, 'output', None)
        self.dir = Path(output_dir_for(config, model.name))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.info: dict[str, Any] = {}
        self.results: dict[str, list[dict[str, Any]]] = {}
        self.predictions: dict[str, list[dict[str, Any]]] = {}
        self.prefix = output.prefix if output is not None else ''
        self.columns: dict[str, Any] = dict(output.columns) if output is not None and output.columns else {}
        self.columns.setdefault('time_id', utils.get_time_id())
        self.columns['uuid4'] = model.uuid4
        write_config(self.dir, config, output.config_file if output is not None else CONFIG_FILE)
        model_path = getattr(model, 'path', None)
        paths = {'output_dir': str(self.dir)}
        if model_path is not None:
            paths['model'] = model_path
        self.write_run_info(started=utils.get_time_id(), paths=paths, **environment_info())
        self.link('model', model_path)

    def file(self, name: str) -> Path:
        """The path of one of this run's files, with the output prefix applied."""
        return self.dir / f'{self.prefix}{name}'

    def record(self, kind: str, event: str, rows: list[dict[str, Any]]) -> None:
        """Keep the rows of one written file, so the run is readable in memory.

        Called by :func:`~micm_nlp.evals.results.save_metrics` and
        :func:`~micm_nlp.evals.results.save_predictions` as they write, which is
        why nothing in :class:`~micm_nlp.training.runner.TRAINER` has to collect
        results: whatever reaches disk is here, under the same event name, in the
        same row shape the CSV holds. That equality is the point -- it is what lets
        a future ``RunOutput.load(dir)`` read a finished run back into the same
        attributes rather than a second, parallel shape.

        :param kind: ``'results'`` for a metrics file, ``'predictions'`` for a
            predictions file -- the two kinds a run writes.
        :param event: the event name the file is named for (``test_after_train``).
        """
        getattr(self, kind)[event] = rows

    def write_run_info(self, **sections) -> Path:
        """Merge ``sections`` into this run's ``info.json``; see the module function.

        The merged result is read back into :attr:`info`, so the attribute always
        equals the file rather than a parallel accumulation of the same sections.
        """
        path = write_run_info(self.dir, **sections)
        self.info = json.loads(path.read_text())
        return path

    def link(self, name: str, target: str | Path | None) -> Path | None:
        """A symlink ``dir/name -> target`` (absolute), so every artefact of the
        run is reachable from its directory.

        A target that does not exist yet still gets a link -- the checkpoint
        directory appears only when the trainer saves. A link that already
        points at ``target`` is left alone; one that points elsewhere is
        replaced. ``None`` is a no-op.
        """
        if target is None:
            return None
        link = self.dir / name
        abs_target = os.path.abspath(target)
        if link.is_symlink():
            if os.readlink(link) != abs_target:
                link.unlink()
                os.symlink(abs_target, link)
        elif not link.exists():
            os.symlink(abs_target, link)
        return link

    def resolved(self, **values) -> None:
        """Record the values the trainer derived from the config (the seed it
        drew, the prefixed ``metric_for_best_model``, ``fp16`` by device) in
        ``info.json`` -- under ``resolved``, or ``<prefix>resolved`` for a
        prefixed trainer (a ``separate_test`` config, or a runner-set prefix), so two never overwrite each
        other -- and stamp ``seed`` as a column unless the config pinned one (a
        ``separate_test`` trainer stamps the seed its own process used; it
        inherits a pinned one through ``output.columns``).
        """
        self.write_run_info(**{f'{self.prefix}resolved': values})
        if values.get('seed') is not None:
            self.columns.setdefault('seed', values['seed'])

    def note_wandb(self, run) -> None:
        """Record the wandb run's id, url and local dir, and link the dir."""
        info = wandb_info(run)
        if info:
            self.write_run_info(wandb=info)
            self.link('wandb', info['dir'])
