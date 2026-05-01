"""
Pipeline — high-level wiring of the core components.

    CONFIG → TOKENIZER → DATASET → MODEL → TRAINER

Usage:
    from micm_nlp.pipeline import load_dataset, load_model, run
"""

from micm_nlp.datasets.dataset import DATASET
from micm_nlp.models.model import MODEL
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
from micm_nlp.training.runner import TRAINER


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


def run(config):
    """CONFIG → TOKENIZER → DATASET → MODEL → TRAINER → results"""

    tokenizer = load_tokenizer(config)
    dataset = preprocess_dataset(config, tokenizer)
    model = load_model(config)

    trainer = TRAINER(model, dataset, tokenizer)
    test_output = trainer.run()

    return model, test_output
