"""micm-nlp — a config-driven NLP research toolkit built on HuggingFace Transformers.

Re-exports the two names needed at startup: ``env`` (settings loaded from ``.env``)
and ``init()`` (sets the workspace root and, optionally, Rich output). ``init()`` is
**not** triggered on import — call it once before any pipeline call so ``artefacts/``
lands in the right place.
"""

from micm_nlp.bootstrap import env as env
from micm_nlp.bootstrap import init as init
