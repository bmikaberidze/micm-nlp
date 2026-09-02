"""XPE enums."""

from micm_nlp.enums import StrEnum


class CrossPromptEncoderReparameterizationType(StrEnum):
    """How the Cross-Prompt Encoder reparameterizes its virtual tokens.

    ``MLP``, ``LSTM`` and ``ATTN`` each map a small trainable tensor to the prompt
    embeddings through a head of that kind. ``NONE`` skips reparameterization
    entirely, which is what plain soft prompt tuning uses -- the embeddings are the
    parameters.
    """

    MLP = 'MLP'
    LSTM = 'LSTM'
    ATTN = 'ATTN'
    NONE = 'NONE'
