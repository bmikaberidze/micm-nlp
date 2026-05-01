"""XPE enums."""

from micm_nlp.enums import StrEnum


class CrossPromptEncoderReparameterizationType(StrEnum):
    MLP = 'MLP'
    LSTM = 'LSTM'
    ATTN = 'ATTN'
    NONE = 'NONE'
