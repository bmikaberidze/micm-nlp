"""Validation tests for the new test/eval token-budget fields on
CustomTrainingArgsConfig. Pydantic model-level only — no model, no GPU."""

import pytest
from pydantic import ValidationError

from micm_nlp.config import CustomTrainingArgsConfig


def test_default_is_none():
    cfg = CustomTrainingArgsConfig()
    assert cfg.test_max_tokens_per_batch is None
    assert cfg.eval_max_tokens_per_batch is None


def test_accepts_auto_string():
    cfg = CustomTrainingArgsConfig(
        test_max_tokens_per_batch='auto',
        eval_max_tokens_per_batch='auto',
    )
    assert cfg.test_max_tokens_per_batch == 'auto'
    assert cfg.eval_max_tokens_per_batch == 'auto'


def test_accepts_positive_int():
    cfg = CustomTrainingArgsConfig(test_max_tokens_per_batch=8192)
    assert cfg.test_max_tokens_per_batch == 8192


def test_rejects_non_positive_int():
    with pytest.raises(ValidationError):
        CustomTrainingArgsConfig(test_max_tokens_per_batch=0)
    with pytest.raises(ValidationError):
        CustomTrainingArgsConfig(test_max_tokens_per_batch=-1)


def test_rejects_arbitrary_string():
    with pytest.raises(ValidationError):
        CustomTrainingArgsConfig(test_max_tokens_per_batch='maximum')


def test_rejects_combination_with_test_force_sequential():
    with pytest.raises(ValidationError) as excinfo:
        CustomTrainingArgsConfig(
            test_force_sequential=True,
            test_max_tokens_per_batch='auto',
        )
    assert 'test_force_sequential' in str(excinfo.value)
    assert 'test_max_tokens_per_batch' in str(excinfo.value)


def test_rejects_combination_with_eval_force_sequential():
    with pytest.raises(ValidationError) as excinfo:
        CustomTrainingArgsConfig(
            eval_force_sequential=True,
            eval_max_tokens_per_batch='auto',
        )
    assert 'eval_force_sequential' in str(excinfo.value)
    assert 'eval_max_tokens_per_batch' in str(excinfo.value)
