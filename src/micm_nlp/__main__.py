"""``python -m micm_nlp`` — the same CLI as the ``micm-nlp`` command.

Reaching the CLI through the interpreter resolves via ``sys.path`` rather than
``PATH``, so it works wherever the import works -- including a ``--target``
install that has no ``bin/`` directory for console scripts.
"""

import sys

from micm_nlp.cli import main

sys.exit(main())
