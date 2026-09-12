# Contributing

```{include} ../../README.md
:start-after: <!-- start:contributing -->
:end-before: <!-- end:contributing -->
```

## Development

```{include} ../../README.md
:start-after: <!-- start:development -->
:end-before: <!-- end:development -->
```

## Building the docs

The docs build needs neither the package nor its dependencies — `sphinx-autoapi`
parses `src/` statically, so four packages are enough:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
```

One warning is expected: `autoapi/micm_nlp/index.rst` is deliberately left out of
every toctree, so its stub is orphaned. Anything else is a real problem.
