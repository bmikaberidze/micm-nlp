"""micm-nlp — a research framework for NLP.

The whole pipeline in a single YAML, run alone or in groups. Builds on the
HuggingFace stack and adds a layer of features of its own.

Re-exports the names a script needs: ``env`` (settings loaded from ``.env``),
``init()`` (sets the workspace root and, optionally, Rich output), :class:`CONFIG`,
:func:`run`, and :func:`example` (the path of a config shipped in the package).
``init()`` is **not** triggered on import — call it once before any pipeline call
so ``artefacts/`` lands in the right place.

``run`` is resolved on first access rather than imported here, because
:mod:`micm_nlp.pipeline` reaches torch, transformers and peft: exporting it eagerly
would make ``import micm_nlp`` — and therefore ``init()``, whose whole job is to run
*before* the heavy stack — pay for the whole training stack. The longer
``from micm_nlp.pipeline import run`` keeps working and is unaffected.
"""

from typing import TYPE_CHECKING

from micm_nlp.bootstrap import env as env
from micm_nlp.bootstrap import init as init
from micm_nlp.config import CONFIG as CONFIG
from micm_nlp.path import example as example

if TYPE_CHECKING:
    # Never executed. It is here so the name is *visible in the source text*: type
    # checkers read it, and so does sphinx-autoapi, which parses these files
    # statically and would otherwise not know ``run`` is exported at all.
    from micm_nlp.pipeline import run as run

__all__ = ['CONFIG', 'env', 'example', 'init', 'run']


def __getattr__(name: str):
    """Resolve :func:`micm_nlp.run` on first access (:pep:`562`)."""
    if name == 'run':
        from micm_nlp.pipeline import run

        return run
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
