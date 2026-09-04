"""Command line entry point: ``micm-nlp``.

One subcommand so far. ``init-examples`` copies the example configurations out of
the installed package and into a directory you can edit::

    micm-nlp init-examples

The configs ship *inside* the package rather than being downloaded, so the copy you
get always matches the version you installed. Fetching them over the network would
introduce the one failure this is meant to avoid -- a config written for a different
release, or no config at all on a machine without internet.

The example *scripts* are not shipped: each is four lines, and both are printed in
the Quickstart. What carries the content is the YAML.
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

_CONFIG_PACKAGE = 'micm_nlp.example_configs'
DEFAULT_DEST = 'micm-nlp-examples'


def _available_configs():
    """Every ``.yml`` shipped in the example-config package, sorted by name."""
    return sorted(
        (p for p in resources.files(_CONFIG_PACKAGE).iterdir() if p.name.endswith('.yml')),
        key=lambda p: p.name,
    )


def init_examples(dest: str | Path = DEFAULT_DEST, force: bool = False) -> int:
    """Copy the shipped example configs into ``dest``.

    :param dest: directory to write into; created if missing.
    :param force: overwrite files that are already there. Without it an existing
        file is left alone and reported, so a config you have edited is never
        silently replaced.
    :returns: a process exit code -- non-zero only if something was skipped.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for src in _available_configs():
        target = dest / src.name
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
        example = dest / 'xsc_preprocess.yml'
        print(
            f'\nRun one with:\n  python -c "'
            f'from micm_nlp.config import CONFIG; '
            f'from micm_nlp.pipeline import preprocess_dataset; '
            f'import micm_nlp; micm_nlp.init(); '
            f"preprocess_dataset(CONFIG.from_yaml('{example}'))\""
        )
    return 1 if skipped and not written else 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog='micm-nlp', description='micm-nlp command line utilities.')
    sub = parser.add_subparsers(dest='command', required=True)

    init = sub.add_parser(
        'init-examples',
        help='copy the example configs out of the installed package',
    )
    init.add_argument('dest', nargs='?', default=DEFAULT_DEST, help=f'destination directory (default: {DEFAULT_DEST})')
    init.add_argument('--force', action='store_true', help='overwrite existing files')

    args = parser.parse_args(argv)
    if args.command == 'init-examples':
        return init_examples(args.dest, force=args.force)
    parser.error(f'unknown command: {args.command}')  # pragma: no cover - argparse rejects first
    return 2


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
