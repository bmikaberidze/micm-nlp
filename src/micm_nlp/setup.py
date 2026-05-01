from pydantic import BaseModel

from micm_nlp.path import set_root


class RichConfig(BaseModel):
    show_locals: bool = False
    width: int = 120
    extra_lines: int = 1


class MicmNlpConfig(BaseModel):
    root_path: str
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
