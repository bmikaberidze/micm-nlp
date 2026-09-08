# tests/test_group_e2e.py
"""End to end through the CLI with a stub runner: init-examples ships a group
that resolves beside its unit config; run-group creates one run dir per entry
and each holds the config snapshot with the identity columns and the entry's
overrides applied. No model is built."""

import yaml

from micm_nlp import cli
from micm_nlp import path as nlpka_path

SEEN = []


def stub_runner(config, ctx):
    SEEN.append(ctx.name)


def test_example_group_dispatches(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    # xsc_finetune.yml's env block is applied to os.environ on load; pre-set so
    # monkeypatch restores it after the test.
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', 'true')
    monkeypatch.setattr(cli, '_init_workspace', lambda: nlpka_path.set_root(tmp_path))
    dest = tmp_path / 'examples'
    assert cli.main(['init-examples', str(dest)]) == 0
    assert (dest / 'xsc_group.yml').exists()

    SEEN.clear()
    rc = cli.main(['run-group', '--group-config', str(dest / 'xsc_group.yml'),
                   '--runner', 'tests.test_group_e2e:stub_runner'])
    assert rc == 0 and SEEN == ['seed_1', 'seed_2']

    group_dir = tmp_path / 'artefacts' / 'evals' / 'runs' / 'bloom' / 'xsc_group'
    run_dirs = sorted(group_dir.iterdir())
    assert [d.name.split('_', 2)[-1] for d in run_dirs] == ['seed_1', 'seed_2']
    saved = yaml.safe_load((run_dirs[0] / 'config.yml').read_text())
    assert saved['results']['columns']['group'] == 'xsc_group'
    assert saved['results']['columns']['seed'] == 1
    assert saved['training_args']['args']['seed'] == 1
    assert saved['test']['run'] is True and saved['test']['zero_shot_only'] is False
