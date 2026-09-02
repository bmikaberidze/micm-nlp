# Installation

```{include} ../../README.md
:start-after: <!-- start:install-requires -->
:end-before: <!-- end:install-requires -->
```

## From PyPI

```{include} ../../README.md
:start-after: <!-- start:install-pypi -->
:end-before: <!-- end:install-pypi -->
```

## From source

```{include} ../../README.md
:start-after: <!-- start:install-source -->
:end-before: <!-- end:install-source -->
```

## Docker

Recommended for reproducibility on GPU machines:

```{include} ../../README.md
:start-after: <!-- start:install-docker -->
:end-before: <!-- end:install-docker -->
```

## Environment

Credentials and the workspace root come from a `.env` file:

```{include} ../../README.md
:start-after: <!-- start:install-env -->
:end-before: <!-- end:install-env -->
```

## Hardware

Training targets NVIDIA GPUs. CPU works for small-scale debugging; there is no
support for non-NVIDIA accelerators.

`peft` is pinned to `0.14.0`: the Cross-Prompt Encoder subclasses stock PEFT
internals, so raising that pin is a breaking-change review rather than a routine
upgrade.
