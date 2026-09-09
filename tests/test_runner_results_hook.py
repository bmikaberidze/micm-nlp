# tests/test_runner_results_hook.py
"""The TRAINER's four results hooks, exercised on a bare instance: config.yml at
setup, the effective seed stamped once the Trainer exists, one valid row per
metric group at the best step, one test row per metric group per pass, with the
toolkit's config-name prefix stripped. No model, no Trainer."""

import csv
from types import SimpleNamespace

from micm_nlp.config import CONFIG
from micm_nlp.evals.results import CONFIG_FILE, TEST_FILE, VALID_FILE
from micm_nlp.training.runner import TRAINER


def _bare(tmp_path, output=None):
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'}, output=output)
    t = object.__new__(TRAINER)
    t._config = cfg
    t._model = SimpleNamespace(eval_path=str(tmp_path / 'run'), uuid4='u-1')
    t._metric_prefix = 'unit.yml/'
    t.trainer = SimpleNamespace(args=SimpleNamespace(seed=7), state=SimpleNamespace(global_step=30))
    t.training_args = SimpleNamespace(metric_for_best_model='unit.yml/accuracy')
    return t


def _rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def test_setup_results_writes_config_and_stamps_identity(tmp_path):
    t = _bare(tmp_path, output={'columns': {'group': 'g', 'name': 'r'}, 'config_file': 'test_config.yml'})
    w = t._setup_results()
    assert (tmp_path / 'run' / 'test_config.yml').exists()
    assert w.columns['group'] == 'g' and w.columns['uuid4'] == 'u-1' and 'time_id' in w.columns


def test_setup_results_without_block_uses_defaults(tmp_path):
    w = _bare(tmp_path)._setup_results()
    assert (tmp_path / 'run' / CONFIG_FILE).exists()
    assert set(w.columns) == {'time_id', 'uuid4'}


def test_stamp_effective_seed_only_when_absent(tmp_path):
    t = _bare(tmp_path, output={'columns': {'seed': 11}})
    t._results = t._setup_results()
    t._stamp_effective_seed()
    assert t._results.columns['seed'] == 11          # pinned by the group: kept
    t = _bare(tmp_path)
    t._results = t._setup_results()
    t._stamp_effective_seed()
    assert t._results.columns['seed'] == 7           # unpinned: what the Trainer drew


def test_write_valid_res(tmp_path):
    t = _bare(tmp_path)
    t._results = t._setup_results()
    t._stamp_effective_seed()
    t.trainer.state = SimpleNamespace(
        best_model_checkpoint='/c/checkpoint-20', best_metric=0.6, global_step=30,
        log_history=[{'step': 10, 'eval_unit.yml/a/accuracy': 0.4},
                     {'step': 20, 'eval_unit.yml/a/accuracy': 0.6, 'eval_unit.yml/b/accuracy': 0.5}],
    )
    t._write_valid_res()
    rows = _rows(tmp_path / 'run' / VALID_FILE)
    assert [(r['metric_group'], r['accuracy'], r['step'], r['seed']) for r in rows] == [
        ('a', '0.6', '20', '7'), ('b', '0.5', '20', '7'),
    ]


def test_write_test_res_two_passes_one_file(tmp_path):
    t = _bare(tmp_path)
    t._results = t._setup_results()
    t._write_test_res({'test_zero_unit.yml/a/accuracy': 0.1, 'test_zero_unit.yml/a/n': 5}, 'test_zero')
    t._write_test_res({'test_unit.yml/a/accuracy': 0.9, 'test_unit.yml/a/n': 5}, 'test')
    rows = _rows(tmp_path / 'run' / TEST_FILE)
    assert [(r['prefix'], r['metric_group'], r['accuracy'], r['step']) for r in rows] == [
        ('test_zero', 'a', '0.1', '30'), ('test', 'a', '0.9', '30'),
    ]
