"""``eval_path_for`` decides where a run's files go: an explicit
``results.dir`` wins, otherwise the solo layout under ``runs/``. Pure path logic
-- MODEL itself is not constructed (it would load weights)."""

from micm_nlp import path as nlpka_path
from micm_nlp.config import CONFIG
from micm_nlp.models.model import eval_path_for


def test_results_dir_wins(tmp_path):
    nlpka_path.set_root(tmp_path)
    cfg = CONFIG(mode='preprocess', model={'architecture': 'xlmr'}, output={'dir': '/runs/g/r'})
    assert eval_path_for(cfg, 'ignored') == '/runs/g/r'


def test_solo_default(tmp_path):
    nlpka_path.set_root(tmp_path)
    cfg = CONFIG(mode='preprocess', model={'architecture': 'xlmr'})
    assert eval_path_for(cfg, 'uuid_bloom_1_2') == str(
        tmp_path / 'artefacts' / 'runs' / 'xlmr' / '_solo' / 'uuid_bloom_1_2'
    )


def test_no_model_block(tmp_path):
    nlpka_path.set_root(tmp_path)
    cfg = CONFIG(mode='preprocess')
    assert '/runs/_nomodel/_solo/' in eval_path_for(cfg, 'x')
