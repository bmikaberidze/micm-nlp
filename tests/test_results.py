"""The results writer: row shaping from HF metric dicts, best-step lookup, and
the CSV contract (fixed header, unknown column raises, append preserves rows,
static columns stamped). Pure file I/O -- no model, no GPU."""

import csv
from types import SimpleNamespace

import pytest

from micm_nlp.config import CONFIG
from micm_nlp.evals.results import (
    CONFIG_FILE, TEST_CONFIG_FILE, TEST_FILE, VALID_FILE, ResultsWriter, best_eval_record, rows_from_metrics,
)


def _read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# -- rows_from_metrics ---------------------------------------------------------

def test_rows_one_per_metric_group():
    metrics = {
        'test_eng_Latn/accuracy': 0.9, 'test_eng_Latn/n': 204,
        'test_kat_Geor/accuracy': 0.7, 'test_kat_Geor/n': 204,
        'test_loss': 1.2, 'test_runtime': 3.0, 'test_samples_per_second': 9.9,
    }
    rows = rows_from_metrics(metrics, 'test')
    assert [r['metric_group'] for r in rows] == ['eng_Latn', 'kat_Geor']
    assert rows[0] == {'prefix': 'test', 'metric_group': 'eng_Latn', 'loss': 1.2, 'accuracy': 0.9, 'n': 204}
    assert 'runtime' not in rows[0] and 'samples_per_second' not in rows[0]


def test_rows_without_groups_is_one_row():
    rows = rows_from_metrics({'eval_accuracy': 0.5, 'eval_loss': 2.0, 'epoch': 1.0}, 'eval')
    assert rows == [{'prefix': 'eval', 'metric_group': '', 'accuracy': 0.5, 'loss': 2.0}]


def test_rows_strip_config_name_prefix():
    # The toolkit prefixes every metric with the unit config's file name
    # (TRAINER._setup_metrics). The trainer passes it as `strip`.
    assert rows_from_metrics({'test_xsc_finetune.yml/accuracy': 0.9}, 'test', strip='xsc_finetune.yml/') == [
        {'prefix': 'test', 'metric_group': '', 'accuracy': 0.9},
    ]
    assert rows_from_metrics({'test_zero_tune.yml/eng/accuracy': 0.1}, 'test_zero', strip='tune.yml/') == [
        {'prefix': 'test_zero', 'metric_group': 'eng', 'accuracy': 0.1},
    ]


def test_rows_keep_nested_group_path_without_strip():
    rows = rows_from_metrics({'test_tune.yml/eng/accuracy': 0.1}, 'test')
    assert rows == [{'prefix': 'test', 'metric_group': 'tune.yml/eng', 'accuracy': 0.1}]


def test_rows_ignore_other_prefixes():
    assert rows_from_metrics({'eval_accuracy': 0.5}, 'test') == []


# -- best_eval_record ----------------------------------------------------------

def _state(best_ckpt, history, best_metric=None):
    return SimpleNamespace(best_model_checkpoint=best_ckpt, log_history=history, best_metric=best_metric)


def test_best_record_from_checkpoint_path():
    hist = [{'step': 10, 'eval_accuracy': 0.4}, {'step': 20, 'eval_accuracy': 0.6}, {'step': 20, 'loss': 1.0}]
    step, rec = best_eval_record(_state('/x/checkpoint-20', hist), 'accuracy')
    assert step == 20 and rec == {'step': 20, 'eval_accuracy': 0.6}


def test_best_record_falls_back_to_best_metric_suffix_match():
    hist = [{'step': 10, 'eval_tune.cfg/accuracy': 0.4}, {'step': 20, 'eval_tune.cfg/accuracy': 0.6}]
    step, rec = best_eval_record(_state(None, hist, best_metric=0.6), 'accuracy')
    assert step == 20 and rec['eval_tune.cfg/accuracy'] == 0.6


def test_best_record_none_when_no_eval():
    assert best_eval_record(_state(None, [{'step': 5, 'loss': 1.0}]), 'accuracy') == (None, {})


# -- ResultsWriter -------------------------------------------------------------

def test_write_config_dumps_plain_yaml(tmp_path):
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'}, results={'columns': {'group': 'g'}})
    w = ResultsWriter(tmp_path / 'run')
    out = w.write_config(cfg)
    assert out == tmp_path / 'run' / CONFIG_FILE
    text = out.read_text()
    assert 'mode: preprocess' in text and 'python/object' not in text   # StrEnum dumped as str
    assert w.write_config(cfg, TEST_CONFIG_FILE).name == 'test_config.yml'


def test_append_stamps_static_columns_and_fixes_header(tmp_path):
    w = ResultsWriter(tmp_path / 'run', columns={'group': 'g', 'name': 'r1'})
    w.append(TEST_FILE, [{'prefix': 'test', 'metric_group': 'a', 'accuracy': 0.5}])
    w.append(TEST_FILE, [{'prefix': 'test', 'metric_group': 'b', 'accuracy': 0.7}])
    rows = _read(tmp_path / 'run' / TEST_FILE)
    assert [r['metric_group'] for r in rows] == ['a', 'b']
    assert rows[0]['group'] == 'g' and rows[1]['name'] == 'r1'
    assert list(rows[0].keys()) == ['group', 'name', 'prefix', 'metric_group', 'accuracy']


def test_append_unknown_column_raises(tmp_path):
    w = ResultsWriter(tmp_path / 'run')
    w.append(VALID_FILE, [{'prefix': 'eval', 'metric_group': '', 'accuracy': 0.5}])
    with pytest.raises(ValueError, match='accuracy_top5'):
        w.append(VALID_FILE, [{'prefix': 'eval', 'metric_group': '', 'accuracy_top5': 0.9}])


def test_append_missing_column_is_blank(tmp_path):
    w = ResultsWriter(tmp_path / 'run')
    w.append(TEST_FILE, [{'prefix': 'test', 'metric_group': 'a', 'accuracy': 0.5, 'n': 3}])
    w.append(TEST_FILE, [{'prefix': 'test', 'metric_group': 'b', 'accuracy': 0.6}])
    assert _read(tmp_path / 'run' / TEST_FILE)[1]['n'] == ''


def test_row_beats_static_on_clash(tmp_path):
    w = ResultsWriter(tmp_path / 'run', columns={'seed': 11})
    w.append(VALID_FILE, [{'prefix': 'eval', 'metric_group': '', 'seed': 42}])
    assert _read(tmp_path / 'run' / VALID_FILE)[0]['seed'] == '42'


def test_columns_are_mutable_after_construction(tmp_path):
    # The trainer stamps the effective seed after the HF Trainer exists.
    w = ResultsWriter(tmp_path / 'run')
    w.columns.setdefault('seed', 7)
    w.append(VALID_FILE, [{'prefix': 'eval', 'metric_group': ''}])
    assert _read(tmp_path / 'run' / VALID_FILE)[0]['seed'] == '7'


def test_append_empty_writes_nothing(tmp_path):
    w = ResultsWriter(tmp_path / 'run')
    w.append(VALID_FILE, [])
    assert not (tmp_path / 'run' / VALID_FILE).exists()
