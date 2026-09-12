"""The ``output`` config block and the run-directory helpers. Pydantic-level and
path-level only -- no model, no GPU."""

from micm_nlp import path as nlpka_path
from micm_nlp.config import CONFIG, ModelConfig, OutputConfig


def test_output_defaults():
    o = OutputConfig()
    assert o.dir is None and o.config_file == 'config.yml' and o.prefix == '' and o.columns == {}


def test_output_columns_stay_a_plain_dict():
    o = OutputConfig(columns={'seed': 11, 'method': 'spt'})
    assert isinstance(o.columns, dict) and o.columns['method'] == 'spt'


def test_config_accepts_output_block():
    cfg = CONFIG(mode='preprocess', model={'architecture': 'toy'},
                 output={'dir': '/tmp/x', 'prefix': 'separate_', 'columns': {'group': 'g'}})
    assert cfg.output.dir == '/tmp/x' and cfg.output.prefix == 'separate_' and cfg.output.columns == {'group': 'g'}
    assert CONFIG(mode='preprocess', model={'architecture': 'toy'}).output is None


def test_model_config_has_no_runtime_name_or_path():
    m = ModelConfig(architecture='toy')
    assert 'name' not in m.model_fields and 'path' not in m.model_fields


def test_run_dir_helpers(tmp_path):
    nlpka_path.set_root(tmp_path)
    assert nlpka_path.runs_dir() == tmp_path / 'artefacts' / 'runs'
    assert nlpka_path.output_dir('g24a', '20260907_1431_spt') == (
        tmp_path / 'artefacts' / 'runs' / 'groups' / 'g24a' / '20260907_1431_spt'
    )
    assert nlpka_path.output_dir(None, 'uuid_bloom_1_2') == (
        tmp_path / 'artefacts' / 'runs' / 'units' / 'uuid_bloom_1_2'
    ), 'a run outside any group lands under units/, with no group segment'
    assert nlpka_path.UNIT_RUNS == 'units' and nlpka_path.GROUP_RUNS == 'groups'
