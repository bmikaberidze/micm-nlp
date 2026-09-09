# tests/test_results.py
"""``evals/results.py``: one HuggingFace metrics dict -> one metrics file per
event; one ``predict()`` output -> one predictions file. Pure file I/O over a
real ``RunOutput``; no model, no GPU."""

import csv
from types import SimpleNamespace

import numpy as np

from micm_nlp.config import CONFIG
from micm_nlp.evals.results import metric_rows, save_metrics, save_predictions, write_csv
from micm_nlp.training.run_output import RunOutput


def _read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _output(tmp_path, **kw):
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'}, output={'dir': str(tmp_path / 'run'), **kw})
    return RunOutput(cfg, SimpleNamespace(name='m', uuid4='u-1', path=None))


# -- metric_rows ---------------------------------------------------------------

def test_rows_one_per_metric_group():
    metrics = {'test_eng_Latn/accuracy': 0.9, 'test_kat_Geor/accuracy': 0.7, 'test_loss': 1.2,
               'test_runtime': 3.0, 'test_samples_per_second': 9.9}
    rows = metric_rows(metrics, 'test')
    assert rows == [{'metric_group': 'eng_Latn', 'loss': 1.2, 'accuracy': 0.9},
                    {'metric_group': 'kat_Geor', 'loss': 1.2, 'accuracy': 0.7}]


def test_rows_without_groups_is_one_row():
    assert metric_rows({'eval_accuracy': 0.5, 'eval_loss': 2.0, 'epoch': 1.0}, 'eval') == [
        {'metric_group': '', 'accuracy': 0.5, 'loss': 2.0}]


def test_rows_strip_config_name_prefix():
    assert metric_rows({'test_xsc_finetune.yml/accuracy': 0.9}, 'test', strip='xsc_finetune.yml/') == [
        {'metric_group': '', 'accuracy': 0.9}]
    assert metric_rows({'test_zero_tune.yml/eng/accuracy': 0.1}, 'test_zero', strip='tune.yml/') == [
        {'metric_group': 'eng', 'accuracy': 0.1}]


def test_rows_ignore_other_prefixes():
    assert metric_rows({'eval_accuracy': 0.5}, 'test') == []


# -- save_metrics --------------------------------------------------------------

def test_save_metrics_writes_one_file_per_event_with_columns(tmp_path):
    o = _output(tmp_path, columns={'group': 'g', 'name': 'r'})
    o.columns['seed'] = 7
    path = save_metrics(o, 'eval_validation_final', {'eval_a/accuracy': 0.6, 'eval_b/accuracy': 0.5}, 'eval', step=20)
    assert path == tmp_path / 'run' / 'eval_validation_final.csv'
    rows = _read(path)
    assert [(r['metric_group'], r['accuracy'], r['step'], r['seed'], r['uuid4']) for r in rows] == [
        ('a', '0.6', '20', '7', 'u-1'), ('b', '0.5', '20', '7', 'u-1')]
    assert list(rows[0].keys())[:6] == ['group', 'name', 'time_id', 'uuid4', 'seed', 'metric_group']


def test_save_metrics_uses_the_prefix_and_overwrites(tmp_path):
    o = _output(tmp_path, prefix='separate_')
    save_metrics(o, 'test_final', {'test_accuracy': 0.1}, 'test')
    save_metrics(o, 'test_final', {'test_accuracy': 0.9}, 'test')
    assert _read(tmp_path / 'run' / 'separate_test_final.csv') == [
        {'time_id': o.columns['time_id'], 'uuid4': 'u-1', 'metric_group': '', 'accuracy': '0.9'}]


def test_save_metrics_nothing_to_write(tmp_path):
    assert save_metrics(_output(tmp_path), 'test_final', {'eval_accuracy': 0.5}, 'test') is None
    assert not (tmp_path / 'run' / 'test_final.csv').exists()


# -- save_predictions ----------------------------------------------------------

def _pred_out(preds, labels):
    return SimpleNamespace(predictions=np.array(preds), label_ids=np.array(labels), metrics={})


def test_predictions_one_row_per_sample_in_original_order(tmp_path):
    o = _output(tmp_path)
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'},
                 task={'preproc_rules': {}}, ds={'label': {'names': ['neg', 'pos']}})
    path = save_predictions(o, 'final', _pred_out([1, 0, 1], [1, 1, 1]), cfg, -100, None, None, order=[2, 0, 1])
    assert path == tmp_path / 'run' / 'predictions_final.csv'
    assert _read(path) == [{'sample': '2', 'prediction': '1', 'label': '1'},
                           {'sample': '0', 'prediction': '0', 'label': '1'},
                           {'sample': '1', 'prediction': '1', 'label': '1'}]


def test_predictions_apply_the_metrics_preprocessing(tmp_path):
    o = _output(tmp_path, prefix='separate_')
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'},
                 task={'preproc_rules': {'label_id_to_name': True}}, ds={'label': {'names': ['neg', 'pos']}})
    path = save_predictions(o, 'zero_shot', _pred_out([1, 0], [0, 0]), cfg, -100, None, None)
    assert path == tmp_path / 'run' / 'separate_predictions_zero_shot.csv'
    assert _read(path) == [{'sample': '0', 'prediction': 'pos', 'label': 'neg'},
                           {'sample': '1', 'prediction': 'neg', 'label': 'neg'}]


def test_predictions_token_classification_one_row_per_position(tmp_path):
    o = _output(tmp_path)
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'},
                 task={'category': 'token_classification', 'preproc_rules': {'filter_padded': True}})
    out = _pred_out([[3, 4, 9], [5, 9, 9]], [[3, 4, -100], [5, -100, -100]])
    rows = _read(save_predictions(o, 'final', out, cfg, -100, None, None))
    assert rows == [{'sample': '0', 'position': '0', 'prediction': '3', 'label': '3'},
                    {'sample': '0', 'position': '1', 'prediction': '4', 'label': '4'},
                    {'sample': '1', 'position': '0', 'prediction': '5', 'label': '5'}]


def test_write_csv_header_is_union_in_first_seen_order(tmp_path):
    path = write_csv(tmp_path / 'x.csv', [{'a': 1}, {'a': 2, 'b': 3}])
    assert _read(path) == [{'a': '1', 'b': ''}, {'a': '2', 'b': '3'}]
