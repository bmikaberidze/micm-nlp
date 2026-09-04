"""Process-level setup: ``.env`` loading, typed settings, and ``init()``.

Importing this module reads ``.env`` into ``os.environ``, so libraries that consult
environment variables directly (huggingface_hub, transformers, wandb) see the same
values that ``Env`` exposes as typed settings.

``init()`` is the one call an application makes at startup. It sets the workspace
root, strips the distributed-training variables that would otherwise push
accelerate into MULTI_GPU mode for a single-process run, and optionally installs
Rich pretty-printing and tracebacks.

Formerly split across ``env.py`` and ``setup.py``; merged here in 0.2.0.
"""

import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from micm_nlp.path import set_root

# Populate os.environ from .env so libraries that read env vars directly
# (huggingface_hub, transformers, wandb, …) see values loaded from the file.
load_dotenv(find_dotenv(usecwd=True))


class Env(BaseSettings):
    """Typed view of the environment, read from ``.env`` and the process env.

    ``extra='allow'`` and ``case_sensitive=True``: unknown variables are kept rather
    than rejected, so a project can put its own settings in the same ``.env``. The
    module-level :data:`env` instance is built at import time; nothing re-reads the
    file afterwards.
    """

    model_config = SettingsConfigDict(
        env_file=find_dotenv(usecwd=True) or '.env',
        env_file_encoding='utf-8',
        extra='allow',
        case_sensitive=True,
    )

    APP_ENV: str = 'local'
    SHOW_LOCALS: int = 0

    HF_TOKEN: str | None = None
    WANDB_API_KEY: str | None = None

    PROJECT_ROOT_PATH: Path | None = None


env = Env()


class RichConfig(BaseModel):
    """Settings for Rich's pretty-printer and traceback handler.

    ``show_locals`` renders local variables in tracebacks — useful when debugging,
    noisy in a training log.
    """

    show_locals: bool = False
    width: int = 120
    extra_lines: int = 1


class MicmNlpConfig(BaseModel):
    """What :func:`init` accepts.

    ``root_path`` defaults to ``PROJECT_ROOT_PATH`` from the environment, so
    ``init()`` with no arguments works when ``.env`` sets it. ``pretty_output`` takes
    ``True`` for Rich defaults, or a :class:`RichConfig` to tune it.
    """

    root_path: str | None = os.getenv('PROJECT_ROOT_PATH')
    pretty_output: RichConfig | bool = False


def init(config: MicmNlpConfig | dict | None = None) -> None:
    """Set up the process: workspace root, distributed env, optional Rich output.

    Call once before any pipeline call. **Not** triggered on import — until it runs,
    every accessor in :mod:`micm_nlp.path` raises, so ``artefacts/`` can never land
    in the wrong place by accident.

    Three things happen, in order: the workspace root is set from ``root_path``
    (falling back to ``PROJECT_ROOT_PATH``); the distributed-training variables are
    stripped when this is a single-process run, see
    ``_disable_distributed_if_single_process``; and Rich is installed if
    ``pretty_output`` asks for it.

    :param config: a :class:`MicmNlpConfig` or a plain dict of its fields. Omit it
        entirely to take every default -- which is the documented form when
        ``PROJECT_ROOT_PATH`` is already set in the environment or in ``.env``.
    """
    if config is None:
        config = {}
    if isinstance(config, dict):
        config = MicmNlpConfig(**config)
    set_root(config.root_path)
    _disable_distributed_if_single_process()
    if config.pretty_output:
        init_rich(config.pretty_output)


# Env vars that, when present, push HuggingFace `accelerate` into MULTI_GPU mode
# and trigger a NCCL allgather during DistributedDataParallel construction —
# even for a single-process run. Stripping them when WORLD_SIZE=1 forces the
# single-process path and avoids NCCL entirely. This matters on hardware where
# the installed PyTorch's NCCL can't initialize (e.g. Blackwell sm_120 with
# pre-2.6 PyTorch), and is harmless otherwise.
_DISTRIBUTED_TRIGGER_VARS = (
    'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'LOCAL_RANK',
    'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
    'SLURM_PROCID', 'SLURM_NTASKS', 'SLURM_NPROCS',
    'SLURM_LOCALID', 'SLURM_NTASKS_PER_NODE',
)


def _disable_distributed_if_single_process() -> None:
    if os.environ.get('WORLD_SIZE', '1') != '1':
        return  # genuine multi-process job — leave the env alone
    stripped = [v for v in _DISTRIBUTED_TRIGGER_VARS if os.environ.pop(v, None) is not None]
    if stripped:
        print(f'[micm_nlp] single-process mode: stripped {stripped}')


def init_rich(rich_config: RichConfig | dict | bool) -> None:
    """Install Rich's pretty-printer and traceback handler.

    Usually reached through :func:`init` rather than called directly. Rich is
    imported inside the function, so a project that never asks for pretty output
    does not pay the import.

    :param rich_config: ``True`` for defaults, or a :class:`RichConfig` / dict.
    """
    if rich_config is True:
        rich_config = RichConfig()
    elif isinstance(rich_config, dict):
        rich_config = RichConfig(**rich_config)
    from rich import pretty

    pretty.install()
    from rich.traceback import install

    install(**rich_config.model_dump())
