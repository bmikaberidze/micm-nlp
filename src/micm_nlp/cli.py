"""Command line entry point: ``micm-nlp`` / ``python -m micm_nlp``.

Three subcommands. ``run`` executes one unit config; ``run-group`` executes the
entries of a group config (see :mod:`micm_nlp.group`); ``init-examples`` copies
the example configurations out of the installed package into a directory you
can edit. Flags the parser does not know are forwarded to the runner as
``ctx.extras`` (``--source-group joshi5`` -> ``{'source_group': 'joshi5'}``)::

    python -m micm_nlp run       --config configs/examples/xsc_finetune.yml
    python -m micm_nlp run-group --group-config configs/examples/groups/xsc_tune_across_seeds.yml
    python -m micm_nlp init-examples

The configs ship *inside* the package rather than being downloaded, so the copy you
get always matches the version you installed. Fetching them over the network would
introduce the one failure this is meant to avoid -- a config written for a different
release, or no config at all on a machine without internet.

No example *scripts* exist: running a shipped config is ``run`` / ``run-group``
above, so a script would be a third path to what these commands already do. What
carries the content is the YAML.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import micm_nlp
from micm_nlp import group
from micm_nlp.path import available_examples

DEFAULT_DEST = 'configs/examples'


def init_examples(dest: str | Path = DEFAULT_DEST, force: bool = False) -> int:
    """Copy the shipped example configs into ``dest``.

    :param dest: directory to write into; created if missing.
    :param force: overwrite files that are already there. Without it an existing
        file is left alone and reported, so a config you have edited is never
        silently replaced.
    :returns: 0. Skipping an existing file is a normal outcome, not an error --
        returning non-zero for it would break any script that runs this more than
        once.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for rel, src in available_examples():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            skipped.append(target)
            continue
        target.write_bytes(src.read_bytes())
        written.append(target)

    for path in written:
        print(f'wrote {path}')
    for path in skipped:
        print(f'skipped {path} (already exists; pass --force to overwrite)')

    if written:
        print(
            f'\nRun one with:\n'
            f'  python -m micm_nlp run --config {dest / "xsc_preprocess.yml"}\n'
            f'  python -m micm_nlp run-group --group-config {dest / "groups" / "xsc_tune_across_seeds.yml"}'
        )
    return 0


def parse_extras(rest: list[str]) -> dict:
    """Unknown ``--flag value`` / ``--flag=value`` / ``--flag`` arguments as a
    dict for the runner.

    Values stay strings -- the runner knows their types. A bare ``--flag``
    becomes ``True``. Anything that is not a flag is an error, so a typo in a
    known option cannot silently become a runner extra.

    :raises ValueError: on a token that does not start with ``--``.
    """
    extras, i = {}, 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith('--'):
            raise ValueError(f'unexpected argument {tok!r}')
        key, eq, value = tok[2:].partition('=')
        key = key.replace('-', '_')
        if eq:
            extras[key] = value
            i += 1
        elif i + 1 < len(rest) and not rest[i + 1].startswith('--'):
            extras[key] = rest[i + 1]
            i += 2
        else:
            extras[key] = True
            i += 1
    return extras


def _init_workspace(root_path: str | None = None) -> None:
    """``micm_nlp.init()`` once, before anything reads ``micm_nlp.path``.

    The workspace root comes from ``--root-path``, else ``PROJECT_ROOT_PATH``
    (environment or the ``.env`` that ``micm_nlp.bootstrap`` loaded on import).
    Without either the package cannot place ``artefacts/``, so say so instead of
    failing deep inside ``pathlib`` -- and never fall back to the working
    directory, which would scatter one experiment's artefacts across as many
    trees as the directories it was launched from.
    """
    if not root_path and not os.environ.get('PROJECT_ROOT_PATH'):
        sys.exit('micm-nlp: pass --root-path, or set PROJECT_ROOT_PATH (environment or .env), '
                 'to your workspace root')
    micm_nlp.init(root_path)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog='micm-nlp', description='micm-nlp command line utilities.',
                                     allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)

    init = sub.add_parser('init-examples', help='copy the example configs out of the installed package',
                          allow_abbrev=False)
    init.add_argument('dest', nargs='?', default=DEFAULT_DEST, help=f'destination directory (default: {DEFAULT_DEST})')
    init.add_argument('--force', action='store_true', help='overwrite existing files')

    root_help = 'workspace root, where artefacts/ goes (default: PROJECT_ROOT_PATH from the environment or .env)'

    run = sub.add_parser('run', help='run one unit config', allow_abbrev=False)
    run.add_argument('--config', required=True, help='path to a unit config')
    run.add_argument('--runner', default=None, help='module:attr or path/to/script.py[:fn] of the runner (default: micm_nlp.pipeline:run)')
    run.add_argument('--root-path', default=None, help=root_help)

    rg = sub.add_parser('run-group', help='run the entries of a group config', allow_abbrev=False)
    rg.add_argument('--group-config', required=True, help='path to a group config; its stem is the group name')
    rg.add_argument('--runner', default=None, help='module:attr or path/to/script.py[:fn] of the runner (default: micm_nlp.pipeline:run)')
    rg.add_argument('--root-path', default=None, help=root_help)
    rg.add_argument('--run-index', type=int, default=None,
                    help='entry index to run; ignored under SLURM_ARRAY_TASK_ID; omit to run every entry')
    rg.add_argument('--seed', type=int, default=None, help='seed for entries that do not set their own')

    args, rest = parser.parse_known_args(argv)
    if args.command == 'init-examples':
        if rest:
            parser.error(f'unrecognized arguments: {" ".join(rest)}')
        return init_examples(args.dest, force=args.force)

    try:
        extras = parse_extras(rest)
    except ValueError as e:
        parser.error(str(e))
    _init_workspace(args.root_path)
    if args.command == 'run':
        group.run_unit(args.config, runner=args.runner, extras=extras)
        return 0
    group.run_group(args.group_config, runner=args.runner, run_index=args.run_index, seed=args.seed, extras=extras)
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
