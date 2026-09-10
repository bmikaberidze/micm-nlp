"""micm-nlp — a research framework for NLP.

The whole pipeline in a single YAML, run alone or in groups. Builds on the
HuggingFace stack and adds a layer of features of its own.

Re-exports the two names needed at startup: ``env`` (settings loaded from ``.env``)
and ``init()`` (sets the workspace root and, optionally, Rich output). ``init()`` is
**not** triggered on import — call it once before any pipeline call so ``artefacts/``
lands in the right place.
"""

from micm_nlp.bootstrap import env as env
from micm_nlp.bootstrap import init as init
