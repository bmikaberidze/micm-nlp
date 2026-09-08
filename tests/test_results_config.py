"""The ``results`` config block and the run-dir helpers. Pydantic-level and
path-level only -- no model, no GPU."""

from micm_nlp import path as nlpka_path
from micm_nlp.config import CONFIG, ModelConfig, ResultsConfig


def test_results_defaults():
    r = ResultsConfig()
    assert r.dir is None
    assert r.config_file == 'config.yml'
    assert r.columns == {}


def test_results_columns_stay_a_plain_dict():
    # Declared as dict, so _Flex's extras-wrapping must NOT turn it into a _Flex.
    r = ResultsConfig(columns={'seed': 11, 'method': 'spt'})
    assert isinstance(r.columns, dict)
    assert r.columns['method'] == 'spt'


def test_config_accepts_results_block():
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'},
                 results={'dir': '/tmp/x', 'columns': {'group': 'g'}})
    assert cfg.results.dir == '/tmp/x'
    assert cfg.results.columns == {'group': 'g'}


def test_config_without_results_block():
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'})
    assert cfg.results is None


def test_model_config_runtime_name_and_path():
    m = ModelConfig(architecture='toy')
    assert m.name is None and m.path is None
    m.name, m.path = 'abc', '/models/abc'
    assert m.model_dump()['name'] == 'abc'


def test_run_dir_helpers(tmp_path):
    nlpka_path.set_root(tmp_path)
    assert nlpka_path.runs_dir() == tmp_path / 'artefacts' / 'evals' / 'runs'
    assert nlpka_path.run_dir('xlmr', 'g24a', '20260907_1431_spt') == (
        tmp_path / 'artefacts' / 'evals' / 'runs' / 'xlmr' / 'g24a' / '20260907_1431_spt'
    )
    assert nlpka_path.SOLO_GROUP == '_solo'
    assert nlpka_path.NO_MODEL_ARCH == '_nomodel'
