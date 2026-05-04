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
    show_locals: bool = False
    width: int = 120
    extra_lines: int = 1


class MicmNlpConfig(BaseModel):
    root_path: str | None = os.getenv('PROJECT_ROOT_PATH')
    pretty_output: RichConfig | bool = False


def init(config: MicmNlpConfig | dict) -> None:
    if isinstance(config, dict):
        config = MicmNlpConfig(**config)
    set_root(config.root_path)
    if config.pretty_output:
        init_rich(config.pretty_output)


def init_rich(rich_config: RichConfig | dict | bool) -> None:
    if rich_config is True:
        rich_config = RichConfig()
    elif isinstance(rich_config, dict):
        rich_config = RichConfig(**rich_config)
    from rich import pretty

    pretty.install()
    from rich.traceback import install

    install(**rich_config.model_dump())
