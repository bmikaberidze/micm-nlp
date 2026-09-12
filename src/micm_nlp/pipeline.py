"""
Pipeline — high-level wiring of the core components.

    CONFIG → TOKENIZER → DATASET → MODEL → TRAINER

Usage:
    from micm_nlp.pipeline import load_dataset, load_model, run

:func:`run` is the whole chain. The single-stage functions beside it are for
consumers that drive the stages themselves -- a preprocessing-only script, say --
and ``run`` deliberately does not go through them: its body is the chain written
out, so that reading it is reading the pipeline.
"""

# Ordered by the chain this module documents, not alphabetically.
# isort: off
from micm_nlp.enums import ModeSE
from micm_nlp.config import CONFIG
from micm_nlp.models.model import MODEL
from micm_nlp.training.runner import TRAINER
from micm_nlp.datasets.dataset import DATASET
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
# isort: on


def load_dataset(config):
    """CONFIG → DATASET"""
    return DATASET(config)


def preprocess_dataset(config, tokenizer=None):
    """CONFIG → TOKENIZER → DATASET"""
    tokenizer = tokenizer or load_tokenizer(config)
    dataset = load_dataset(config)
    dataset.preprocess(tokenizer)
    return dataset


def load_model(config):
    """CONFIG → MODEL"""
    return MODEL(config)


def run(config, ctx=None):
    """CONFIG → TOKENIZER → DATASET → MODEL → TRAINER → results

    ``ctx`` is the group runner's :class:`~micm_nlp.group.RunContext`; this
    default runner is single-phase and ignores it, so a group runs with no
    custom code. A ``separate_test`` in the entry is therefore not acted on
    here -- that is a runner's decision.

    The body below is the whole chain, written out: it calls the core classes
    directly rather than the wrappers above, so that reading this function *is*
    reading the pipeline, and a consumer that needs to intervene between two
    stages can copy it and change the one line. ``tests/test_pipeline_stages.py``
    pins the README's copy of the sequence against this source.

    :param config: a :class:`~micm_nlp.config.CONFIG`, or a path to the YAML of
        one -- so a script needs no separate load step.
    :param ctx: the group runner's :class:`~micm_nlp.group.RunContext`.
    :returns: the run's :class:`~micm_nlp.training.run_output.RunOutput` -- its
        directory, and the metrics and prediction rows it wrote, under the same
        event names as the files. Returning it rather than the model and the test
        result gives one object whose shape *is* the run directory, and reaches
        the four evaluation results the old pair dropped. ``None`` for a
        ``mode: preprocess`` config, which stops after the dataset is tokenised
        and saved -- no model, no trainer, no output directory.
    """

    # Config
    config = config if isinstance(config, CONFIG) else CONFIG.from_yaml(config)

    # Tokenizer
    tokenizer = load_tokenizer(config)

    # Dataset
    dataset = DATASET(config)
    dataset.preprocess(tokenizer)
    if config.mode == ModeSE.PREPROCESS:
        return None # a preprocessing config has no model to build

    # Model
    model = MODEL(config)

    # Trainer
    trainer = TRAINER(model, dataset, tokenizer)
    trainer.run()

    # Output
    return trainer.output
