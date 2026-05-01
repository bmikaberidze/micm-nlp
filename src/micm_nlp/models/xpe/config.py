"""Configuration dataclass for XPE."""

from dataclasses import dataclass, field

from peft import PeftType, PromptEncoderConfig


@dataclass
class CrossPromptEncoderConfig(PromptEncoderConfig):
    """
    This is the configuration class to store the configuration of a [`CrossPromptEncoder`].

    Args:
        encoder_embedding_init_type (`str`): The type of initialization to use for the embedding.
        encoder_init_state_dict_path (`str`): The path to pretraine encoder state for initialization shared embeddings and encoder heads.
        encoder_embedding_freeze (`bool`): The indicator of frozen or trainable embedding.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from CrossPromptEncoder layers to be set as trainable and saved in the final checkpoint.
        encoder_embedding_normalize (`str`): The type of normalization to use for the embedding (None,unit, clip).
        encoder_embedding_normalize_max_norm (`float`): The maximum norm for the embedding.
        encoder_input_size (`int`): The input size for the encoder.
        encoder_num_heads (`int`): The number of attention heads in the encoder.
    """

    encoder_ratio: float = field(
        default=0.7,
        metadata={'help': 'The ratio of encoded vs standard input embeddings'},
    )
    encoder_embedding_init_type: str = field(
        default='hf_default',
        metadata={
            'help': 'The type of initialization to use for the embedding (xavier_uniform, xavier_normal, hf_default)'
        },
    )
    encoder_init_state_dict_path: str = field(
        default=None,
        metadata={'help': 'The path to pretrained encoder'},
    )
    encoder_freeze: bool = field(
        default=True,
        metadata={'help': 'The indicator of frozen or trainable encoder'},
    )
    encoder_embedding_freeze: bool = field(
        default=True,
        metadata={'help': 'The indicator of frozen or trainable embedding'},
    )
    modules_to_save: list[str] | None = field(
        default=None,
        metadata={
            'help': 'List of modules apart from CrossPromptEncoder layers to be set as trainable and saved in the final checkpoint.'
        },
    )
    encoder_embedding_normalize: str = field(
        default='unit',
        metadata={'help': 'The type of normalization to use for the embedding (None, unit, clip)'},
    )
    encoder_embedding_normalize_max_norm: float = field(
        default=1.0,
        metadata={'help': 'The maximum norm for the embedding'},
    )
    encoder_input_size: int | None = field(
        default=None,
        metadata={'help': 'The input size for the encoder'},
    )
    encoder_num_heads: int = field(
        default=8,
        metadata={'help': 'The number of attention heads in the encoder'},
    )

    def __post_init__(self):
        super().__post_init__()

        self.peft_type = PeftType.XPE
        if self.modules_to_save is None:
            self.modules_to_save = []

            # Embeddings
            if self.encoder_ratio < 1:
                self.modules_to_save.append('embedding')
            if self.encoder_ratio > 0:
                self.modules_to_save.append('xpe_embedding')

            # Encoder Head
            if self.encoder_ratio > 0:
                self.modules_to_save.append('xpe_head')

    @classmethod
    def from_peft_type(cls, **kwargs):
        r"""
        Loads the configuration from a set of kwargs. Present for API parity with
        upstream PeftConfig subclasses; defers to the dataclass __init__.
        """
        return CrossPromptEncoderConfig(**kwargs)
