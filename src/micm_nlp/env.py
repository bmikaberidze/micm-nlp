from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    PROJECT_ROOT_PATH: Path = None


env = Env()
