"""XPE public surface.

Post-Phase-9 the XPE integration is exposed entirely through submodules:

- :mod:`.config` — :class:`CrossPromptEncoderConfig`
- :mod:`.encoder` — :class:`CrossPromptEncoder` (the prompt encoder nn.Module)
- :mod:`.enums` — :class:`CrossPromptEncoderReparameterizationType`
- :mod:`.heads` — :class:`LSTMWrapper`, :class:`LightweightSelfAttentionHead`
- :mod:`.save_load` — XPE-aware state-dict (de)serialization helpers
- :mod:`.peft_models` — :class:`XPEPeftModelForSequenceClassification`,
  :data:`TASK_TYPE_TO_XPE_MODEL`, :func:`xpe_model_for`
- :mod:`.factory` — :func:`get_xpe_model`, :func:`load_xpe_pretrained`,
  :func:`is_xpe_config`, :func:`is_xpe_adapter_dir`

This module performs two side-effects at import time:

1. Registers ``PeftType.XPE`` on the stdlib-backed ``peft.PeftType`` enum via
   ``aenum.extend_enum`` (idempotent — safe on reload).
2. Registers ``PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.XPE] = CrossPromptEncoderConfig``
   so ``PeftConfig.from_pretrained`` can resolve XPE checkpoints generically.

Both must run before any ``CrossPromptEncoderConfig`` is instantiated, which
is why they happen at module-import time.
"""

from aenum import extend_enum as _extend_enum
from peft import PEFT_TYPE_TO_CONFIG_MAPPING as _PEFT_TYPE_TO_CONFIG_MAPPING
from peft import PeftType as _PeftType

if not hasattr(_PeftType, 'XPE'):
    _extend_enum(_PeftType, 'XPE', 'XPE')

from micm_nlp.models.xpe.config import CrossPromptEncoderConfig
from micm_nlp.models.xpe.encoder import CrossPromptEncoder  # noqa: F401
from micm_nlp.models.xpe.enums import CrossPromptEncoderReparameterizationType  # noqa: F401
from micm_nlp.models.xpe.factory import (  # noqa: F401
    get_xpe_model,
    is_xpe_adapter_dir,
    is_xpe_config,
    load_xpe_pretrained,
    maybe_load_pretrained_classifier_state,
)
from micm_nlp.models.xpe.heads import LightweightSelfAttentionHead, LSTMWrapper  # noqa: F401
from micm_nlp.models.xpe.peft_models import (  # noqa: F401
    TASK_TYPE_TO_XPE_MODEL,
    XPEPeftModelForCausalLM,
    XPEPeftModelForSequenceClassification,
    xpe_model_for,
)
from micm_nlp.models.xpe.save_load import (  # noqa: F401
    xpe_get_peft_model_state_dict,
    xpe_set_peft_model_state_dict,
)

_PEFT_TYPE_TO_CONFIG_MAPPING[_PeftType.XPE] = CrossPromptEncoderConfig
