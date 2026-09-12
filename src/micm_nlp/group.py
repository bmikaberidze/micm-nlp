# src/micm_nlp/group.py
"""Run a group of runs described by one YAML -- the framework half of a run.

A group config names unit configs and lists runs over them::

    configs:
      spt: ./tune.spt.yml
      eval: ./test.eval.yml
    runs:
      - config: spt
        name: spt_s11
        seed: 11
        overrides: {peft.encoder_hidden_size: 192}
        separate_test: {config: eval, overrides: {}}   # rare
        source_group: joshi5                            # anything else -> the runner

Every name a ``separate_test`` block references must itself be a key of
``configs:`` -- loading raises otherwise.

The file stem is the group name. For the selected entry this module loads the
unit config, applies the seed, then the overrides, fills the ``output`` block
(run dir + identity columns), writes a snapshot of the resolved config into the
run dir and calls the runner -- ``run(config, ctx)`` -- which does the science.
Result rows are written by the trainer, never here. The only scheduler
*logic* in the package is the ``SLURM_ARRAY_TASK_ID`` read in
:func:`select_indices` (``info.json`` merely records ``SLURM*`` variables).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

import micm_nlp.utils as utils
from micm_nlp.config import CONFIG, OutputConfig, _Flex
from micm_nlp.training.run_output import (
    CONFIG_FILE, TEST_CONFIG_FILE, environment_info, write_config, write_run_info,
)
from micm_nlp.path import output_dir

RESERVED_KEYS = ('config', 'overrides', 'seed', 'name', 'separate_test')
# Columns the framework or the trainer stamps; an entry may not carry them.
FRAMEWORK_COLUMNS = ('group', 'index', 'time_id', 'uuid4', 'metric_group', 'step')
DEFAULT_RUNNER = 'micm_nlp.pipeline:run'


@dataclass(frozen=True)
class RunContext:
    """Everything the framework knows about one run that is not the config.

    A frozen dataclass rather than keyword arguments so that adding a field
    never breaks a runner written against an older version.

    The runner's first argument is the run's config; ``test_config``, when set,
    is the resolved second config a separate test phase runs under -- the
    entry's ``separate_test`` block, loaded, overridden and given its own
    ``output`` block (``test_config.yml``, prefix ``separate_``).
    """

    test_config: CONFIG | None
    entry: dict[str, Any]
    group: str | None          # None for a run started outside any group config
    name: str | None
    index: int | None
    output_dir: str | None
    extras: dict[str, Any]


# -- Loading -------------------------------------------------------------------

def _check_overrides(where: str, overrides) -> None:
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError(f'{where}: overrides must be a mapping of dotted.path -> value')


def load_group(path: str | Path) -> dict[str, Any]:
    """Load and validate a group config.

    Paths in ``configs`` resolve relative to the group file's own directory, so
    a group is location-independent (absolute paths pass through).

    :returns: ``{'group': stem, 'configs': {name: absolute path}, 'runs': [...]}``.
    :raises ValueError: for any schema violation, naming the entry index.
    """
    path = Path(path)
    if path.stem.startswith('_'):
        raise ValueError(f'{path}: group stems must not start with "_" (reserved for the framework)')
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'{path}: top level must be a mapping')
    configs = data.get('configs')
    if not isinstance(configs, dict) or not configs:
        raise ValueError(f'{path}: `configs:` must be a non-empty mapping')
    runs = data.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError(f'{path}: `runs:` must be a non-empty list')

    resolved = {}
    for key, rel in configs.items():
        if not isinstance(rel, str):
            raise ValueError(f'{path}: configs[{key!r}] must be a path, got {rel!r}')
        p = Path(rel)
        p = p if p.is_absolute() else (path.parent / p).resolve()
        if not p.is_file():
            raise ValueError(f'{path}: configs[{key!r}] -> {p} does not exist')
        resolved[key] = str(p)

    names = set()
    for i, entry in enumerate(runs):
        where = f'{path}: runs[{i}]'
        if not isinstance(entry, dict):
            raise ValueError(f'{where} must be a mapping')
        if entry.get('config') not in resolved:
            raise ValueError(f"{where} names config {entry.get('config')!r}, not in configs {sorted(resolved)}")
        name = entry.get('name')
        if not isinstance(name, str) or not name:
            raise ValueError(f'{where} needs a name')
        if '/' in name or name in ('.', '..'):
            raise ValueError(f'{where} name {name!r} must be a single path segment')
        if name in names:
            raise ValueError(f'{where} name {name!r} is not unique within the group')
        names.add(name)
        clash = sorted(set(entry) & set(FRAMEWORK_COLUMNS))
        if clash:
            raise ValueError(f'{where} uses reserved column name(s) {clash}')
        seed = entry.get('seed')
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(f'{where}.seed must be an int')
        _check_overrides(where, entry.get('overrides'))
        sep = entry.get('separate_test')
        if sep is not None:
            sep_cfg = sep.get('config') if isinstance(sep, dict) else None
            if sep_cfg not in resolved:
                raise ValueError(f'{where}.separate_test names config {sep_cfg!r}, not in configs {sorted(resolved)}')
            _check_overrides(f'{where}.separate_test', sep.get('overrides'))
    return {'group': path.stem, 'configs': resolved, 'runs': runs}


# -- Entry selection -----------------------------------------------------------

def select_indices(n_runs: int, run_index: int | None) -> list[int]:
    """Which entries to run: ``SLURM_ARRAY_TASK_ID`` > ``run_index`` > all of them.

    Array dispatch is authoritative -- a ``--run-index`` under SLURM is ignored
    with a note, so an array can never be silently re-pointed. With neither,
    every entry runs in order, which is the whole group on a machine with no
    scheduler.
    """
    env = os.environ.get('SLURM_ARRAY_TASK_ID')
    if env is not None:
        if run_index is not None:
            print(f'[group] --run-index={run_index} ignored (SLURM_ARRAY_TASK_ID={env} wins)')
        chosen = [int(env)]
    elif run_index is not None:
        chosen = [run_index]
    else:
        return list(range(n_runs))
    for i in chosen:
        if not 0 <= i < n_runs:
            raise ValueError(f'entry {i} out of range [0, {n_runs})')
    return chosen


# -- Resolution ----------------------------------------------------------------

def apply_override(config, dotted: str, value) -> None:
    """Set ``config.a.b.c = value`` from ``'a.b.c'``.

    An all-digit segment indexes a list (``…optimizer_grouped_parameters.0.lr``).
    Strict: the attribute is read before it is written, so an unknown key
    raises instead of silently creating one.
    """
    parts = dotted.split('.')
    obj = config
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() and isinstance(obj, list) else getattr(obj, p)
    last = parts[-1]
    if last.isdigit() and isinstance(obj, list):
        obj[int(last)] = value
    else:
        getattr(obj, last)
        setattr(obj, last, value)


def apply_seed(config, seed: int | None) -> None:
    """Pin ``training_args.args.seed``; a ``None`` seed leaves the config alone."""
    if seed is None:
        return
    if config.training_args is None:
        raise ValueError('seed given but the config has no training_args block')
    if config.training_args.args is None:
        config.training_args.args = _Flex()
    config.training_args.args.seed = seed


def config_seed(config) -> int | None:
    """The seed the config carries, if any -- read after overrides."""
    args = getattr(config.training_args, 'args', None) if config.training_args is not None else None
    return getattr(args, 'seed', None) if args is not None else None


def scalar_columns(entry: dict[str, Any]) -> dict[str, Any]:
    """The entry's unreserved scalar keys -- the axes this group varies over."""
    return {k: v for k, v in entry.items()
            if k not in RESERVED_KEYS and isinstance(v, (str, int, float, bool))}


def _load_and_override(path: str, overrides: dict[str, Any] | None, seed: int | None) -> CONFIG:
    config = CONFIG.from_yaml(path)
    apply_seed(config, seed)
    for dotted, value in (overrides or {}).items():
        apply_override(config, dotted, value)
        print(f'[group] override: {dotted} = {value}')
    return config


def _fill_output(config: CONFIG, dir_: str, config_file: str, columns: dict[str, Any],
                  prefix: str | None = None) -> None:
    """Fill the ``output`` block in place -- never through dotted overrides.

    ``dir`` and ``config_file`` are set; the framework's columns are merged
    over any the unit config declared (framework wins on a clash). ``prefix``
    is set only when given (the ``separate_test`` config gets ``separate_``);
    otherwise the unit config's own value stays, so a runner or a config may
    choose one.
    """
    if config.output is None:
        config.output = OutputConfig()
    config.output.dir = dir_
    config.output.config_file = config_file
    config.output.columns = {**(config.output.columns or {}), **columns}
    if prefix is not None:
        config.output.prefix = prefix


def resolve_entry(group: dict[str, Any], index: int, cli_seed: int | None = None,
                  extras: dict[str, Any] | None = None) -> tuple[CONFIG, RunContext]:
    """The selected entry as a resolved config plus its context.

    Order: load, seed (entry, else CLI), overrides, ``output`` block, output dir
    created with a snapshot of the resolved config and an ``info.json`` carrying
    ``started``, the output dir and the environment, so a run that dies before
    the trainer exists still says where and when it ran; then ``separate_test``
    the same way (no seed), with its own ``output`` block. The output dir is
    ``runs/{architecture}/{group}/{time_id}_{name}``; if it already exists
    (the same entry dispatched twice within one second) this raises rather
    than merging two runs into one directory.
    """
    entry = group['runs'][index]
    seed = entry['seed'] if entry.get('seed') is not None else cli_seed
    config = _load_and_override(group['configs'][entry['config']], entry.get('overrides'), seed)

    time_id = utils.get_time_id()
    dir_ = output_dir(group['group'], f"{time_id}_{entry['name']}")
    if dir_.exists():
        raise FileExistsError(f'{dir_} already exists: the same entry was dispatched twice within one second')

    columns = {**scalar_columns(entry), 'group': group['group'], 'name': entry['name'], 'index': index,
               'config': entry['config'], 'time_id': time_id}
    effective = config_seed(config)          # after overrides, so the column is what the run uses
    if effective is not None:
        columns['seed'] = effective
    _fill_output(config, str(dir_), CONFIG_FILE, columns)
    write_config(dir_, config, CONFIG_FILE)  # creates the dir
    write_run_info(dir_, started=time_id, paths={'output_dir': str(dir_)}, **environment_info())

    separate_test = None
    sep = entry.get('separate_test')
    if sep is not None:
        separate_test = _load_and_override(group['configs'][sep['config']], sep.get('overrides'), None)
        _fill_output(separate_test, str(dir_), TEST_CONFIG_FILE, columns, prefix='separate_')
        write_config(dir_, separate_test, TEST_CONFIG_FILE)

    ctx = RunContext(
        test_config=separate_test,
        entry={k: v for k, v in entry.items() if k not in RESERVED_KEYS},
        group=group['group'], name=entry['name'], index=index,
        output_dir=str(dir_), extras=dict(extras or {}),
    )
    return config, ctx


# -- Dispatch ------------------------------------------------------------------

def load_runner(spec: str | None) -> Callable:
    """The runner callable from ``'package.module:attr'`` or ``'path/to/script.py[:attr]'``.

    A path is loaded as a module by location (so the script must not do work at
    import time -- the usual ``if __name__ == '__main__':`` guard); ``attr``
    defaults to ``run``. ``None`` means the default runner.
    """
    spec = spec or DEFAULT_RUNNER
    target, _, attr = spec.partition(':')
    if target.endswith('.py') or '/' in target:
        path = Path(target)
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        # spec_from_file_location happily builds a loader for a path that does not
        # exist -- it only opens the file in exec_module -- so check it here.
        if not path.is_file() or module_spec is None or module_spec.loader is None:
            raise ValueError(f'runner script not found: {target!r}')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return getattr(module, attr or 'run')
    if not attr:
        raise ValueError(f'runner must be given as module:attr or path/to/script.py[:attr], got {spec!r}')
    return getattr(importlib.import_module(target), attr)


def run_group(group_path: str | Path, runner: str | None = None, run_index: int | None = None,
              seed: int | None = None, extras: dict[str, Any] | None = None) -> list[int]:
    """Run the selected entries of a group config. Returns the indices run."""
    group = load_group(group_path)
    fn = load_runner(runner)
    indices = select_indices(len(group['runs']), run_index)
    for i in indices:
        config, ctx = resolve_entry(group, i, cli_seed=seed, extras=extras)
        print(f"[group] {group['group']} [{i}] {ctx.name} -> {ctx.output_dir}")
        fn(config, ctx)
    return indices


def run_unit(config_path: str | Path, runner: str | None = None,
             extras: dict[str, Any] | None = None) -> None:
    """Run one unit config outside any group.

    Nothing to resolve -- the trainer's default output dir already puts the run
    under ``runs/units/`` -- so the context is built directly, with ``group``
    left ``None``: there is no group, and the stamped column says so.
    """
    config = CONFIG.from_yaml(config_path)
    ctx = RunContext(test_config=None, entry={}, group=None, name=None, index=None,
                     output_dir=None, extras=dict(extras or {}))
    load_runner(runner)(config, ctx)
