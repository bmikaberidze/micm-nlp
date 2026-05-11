"""Regression test: the length column survives _remove_unused_columns so the
token-budget sampler and HF's LengthGroupedSampler can read it.

We bypass TrainingArguments.__init__ (which hangs under MPI/SLURM init in this
cluster environment) by constructing a minimal trainer instance via object.__new__
and injecting only the attributes that _set_signature_columns_if_needed and
_remove_unused_columns actually read. The real methods are exercised unchanged.
"""

import os
import types

import pytest

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')


def _make_trainer_instance(*, length_column_name='length', usable_columns=None, remove_unused_columns=True):
    """Build a bare CustomTrainer instance — skipping __init__ entirely.

    Sets only the attributes that _set_signature_columns_if_needed and
    _remove_unused_columns read, so the MPI/accelerate machinery is never
    touched. The real HF parent methods are still invoked via super().
    """
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer
    from micm_nlp.training.trainers import custom_trainer_class_factory
    from micm_nlp.config import CustomTrainingArgsConfig

    CustomTrainer = custom_trainer_class_factory(Trainer)

    cfg = GPT2Config(
        vocab_size=32, n_positions=64, n_embd=8,
        n_layer=1, n_head=2, n_inner=16,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)

    # Build a stub args namespace that only exposes what the two methods need.
    args = types.SimpleNamespace(
        length_column_name=length_column_name,
        remove_unused_columns=remove_unused_columns,
    )
    custom_args = CustomTrainingArgsConfig(usable_columns=usable_columns)

    # Bypass __init__ — no accelerator, no TrainingArguments, no MPI.
    trainer = object.__new__(CustomTrainer)
    trainer.model = model
    trainer.args = args
    trainer.custom_args = custom_args
    # HF checks self._signature_columns is None before computing it.
    trainer._signature_columns = None
    # HF reads self.label_names in _set_signature_columns_if_needed.
    trainer.label_names = ['labels']

    return trainer


def test_length_column_in_signature_columns():
    """After _set_signature_columns_if_needed, 'length' must be in _signature_columns."""
    trainer = _make_trainer_instance(length_column_name='length')
    trainer._set_signature_columns_if_needed()

    assert 'length' in trainer._signature_columns, (
        f"'length' missing from signature_columns: {trainer._signature_columns}"
    )


def test_length_column_survives_remove_unused_columns():
    """The 'length' column must survive _remove_unused_columns end-to-end."""
    from datasets import Dataset

    trainer = _make_trainer_instance(length_column_name='length', usable_columns=['task_ids'])

    ds = Dataset.from_dict({
        'input_ids': [[1, 2, 3], [1, 2, 3, 4]],
        'attention_mask': [[1, 1, 1], [1, 1, 1, 1]],
        'labels': [[1, 2, 3], [1, 2, 3, 4]],
        'length': [3, 4],
        'task_ids': [0, 1],
    })

    stripped = trainer._remove_unused_columns(ds, description='test')

    assert 'length' in stripped.column_names, (
        f"'length' missing from stripped dataset: {stripped.column_names}"
    )


def test_usable_columns_still_preserved():
    """task_ids declared in usable_columns must also survive _remove_unused_columns."""
    from datasets import Dataset

    trainer = _make_trainer_instance(usable_columns=['task_ids'])

    ds = Dataset.from_dict({
        'input_ids': [[1, 2, 3], [1, 2, 3, 4]],
        'attention_mask': [[1, 1, 1], [1, 1, 1, 1]],
        'labels': [[1, 2, 3], [1, 2, 3, 4]],
        'length': [3, 4],
        'task_ids': [0, 1],
    })

    stripped = trainer._remove_unused_columns(ds, description='test')

    assert 'task_ids' in stripped.column_names, (
        f"'task_ids' missing from stripped dataset: {stripped.column_names}"
    )


def test_unknown_column_is_stripped():
    """Columns that are neither model inputs, length, nor usable_columns are dropped."""
    from datasets import Dataset

    trainer = _make_trainer_instance(length_column_name='length', usable_columns=None)

    ds = Dataset.from_dict({
        'input_ids': [[1, 2, 3], [1, 2, 3, 4]],
        'attention_mask': [[1, 1, 1], [1, 1, 1, 1]],
        'labels': [[1, 2, 3], [1, 2, 3, 4]],
        'length': [3, 4],
        'irrelevant_extra': [99, 98],
    })

    stripped = trainer._remove_unused_columns(ds, description='test')

    assert 'irrelevant_extra' not in stripped.column_names, (
        f"'irrelevant_extra' should have been stripped: {stripped.column_names}"
    )
