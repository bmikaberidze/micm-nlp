# Contributing

```{include} ../../README.md
:start-after: <!-- start:contributing -->
:end-before: <!-- end:contributing -->
```

## Developement

```{include} ../../README.md
:start-after: <!-- start:development -->
:end-before: <!-- end:development -->
```

## Building the docs

`sphinx-autoapi` parses `src/` statically, so the build needs neither the package nor its dependencies — four packages are enough:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
```

One warning is expected: `autoapi/micm_nlp/index.rst` is deliberately left out of every toctree, so its stub is orphaned. Anything else is real.
