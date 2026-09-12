"""A run is readable in memory in exactly the shape it is readable on disk.

``save_metrics`` and ``save_predictions`` record their rows on the ``RunOutput`` as
they write them, and ``write_run_info`` reads its merged result back, so
``output.results``, ``output.predictions`` and ``output.info`` hold what the files
hold, under the same event names. That equality is the contract these tests pin: it
is what lets ``pipeline.run()`` return the output object instead of a
model-and-result pair, and what a future ``RunOutput.load(dir)`` would have to
reproduce from the files.
"""

import csv
import json
from pathlib import Path

from micm_nlp.evals import results
from micm_nlp.training.run_output import RUN_INFO_FILE, RunOutput


class _Output:
    """The surface ``save_metrics`` / ``save_predictions`` use of a RunOutput.

    A real one writes ``info.json``, snapshots the config and builds symlinks in
    ``__init__``; none of that is under test here, so this stands in for it and
    delegates ``record`` to the real implementation.
    """

    def __init__(self, tmp_path: Path, columns: dict | None = None):
        self.dir = tmp_path
        self.prefix = ''
        self.columns = columns or {'time_id': 't1', 'uuid4': 'u1'}
        self.info: dict = {}
        self.results: dict = {}
        self.predictions: dict = {}

    file = RunOutput.file
    record = RunOutput.record
    write_run_info = RunOutput.write_run_info


def _csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _as_text(rows: list[dict]) -> list[dict]:
    """Rows with every value stringified, as a CSV round-trip returns them."""
    return [{k: str(v) for k, v in row.items()} for row in rows]


def test_record_files_rows_under_their_kind_and_event(tmp_path):
    output = _Output(tmp_path)
    output.record('results', 'test_after_train', [{'accuracy': 1.0}])
    output.record('predictions', 'after_train', [{'sample': 0}])
    assert output.results == {'test_after_train': [{'accuracy': 1.0}]}
    assert output.predictions == {'after_train': [{'sample': 0}]}


def test_saved_metrics_are_recorded_exactly_as_written(tmp_path):
    output = _Output(tmp_path)
    path = results.save_metrics(output, 'test_after_train', {'test_accuracy': 0.75, 'test_loss': 0.5}, 'test', step=100)

    assert list(output.results) == ['test_after_train'], 'recorded under the event name the file is named for'
    assert _as_text(output.results['test_after_train']) == _csv_rows(path)


def test_static_columns_are_recorded_too(tmp_path):
    """The run's static columns are stamped before recording, not after, so the
    in-memory rows are not a narrower copy of the file's."""
    output = _Output(tmp_path, columns={'time_id': 't1', 'uuid4': 'u1', 'seed': 7})
    results.save_metrics(output, 'eval_validation_after_train', {'eval_accuracy': 0.5}, 'eval')

    row = output.results['eval_validation_after_train'][0]
    assert row['seed'] == 7
    assert row['uuid4'] == 'u1'


def test_nothing_is_recorded_when_nothing_is_written(tmp_path):
    """``save_metrics`` writes no file when no key carried the prefix; it must not
    leave a phantom event behind either."""
    output = _Output(tmp_path)
    assert results.save_metrics(output, 'test_after_train', {'other_accuracy': 1.0}, 'test') is None
    assert output.results == {}


def test_info_mirrors_the_file_it_writes(tmp_path):
    """``output.info`` is read back from ``info.json`` rather than accumulated
    beside it, so the two cannot diverge -- including through the merge rules
    (``started`` kept from the first write, dicts merged one level deep)."""
    output = _Output(tmp_path)
    output.write_run_info(started='t1', paths={'output_dir': str(tmp_path)})
    output.write_run_info(started='t2', paths={'model': '/somewhere'}, finished='t3')

    on_disk = json.loads((tmp_path / RUN_INFO_FILE).read_text())
    assert output.info == on_disk
    assert output.info['started'] == 't1', 'the first start time is kept'
    assert output.info['finished'] == 't3'
    assert output.info['paths'] == {'output_dir': str(tmp_path), 'model': '/somewhere'}


def test_each_event_is_recorded_separately(tmp_path):
    """The six metrics events of a full run are six keys, not one overwritten."""
    output = _Output(tmp_path)
    for event in ('eval_validation_before_train', 'test_before_train',
                  'eval_validation_after_train', 'test_after_train'):
        results.save_metrics(output, event, {'test_accuracy': 0.1}, 'test')
    assert sorted(output.results) == [
        'eval_validation_after_train', 'eval_validation_before_train',
        'test_after_train', 'test_before_train',
    ]
