"""``pipeline.run`` on a ``mode: preprocess`` config stops after preprocessing:
no model is built, no trainer runs. The stages are stubbed; only the routing
is under test."""

from micm_nlp import pipeline
from micm_nlp.config import CONFIG


def _stub_stages(monkeypatch, calls):
    monkeypatch.setattr(pipeline, 'load_tokenizer', lambda config: calls.append('tokenizer') or 'tok')
    monkeypatch.setattr(pipeline, 'preprocess_dataset', lambda config, tokenizer=None: calls.append('dataset') or 'ds')
    monkeypatch.setattr(pipeline, 'load_model', lambda config: calls.append('model') or 'model')
    monkeypatch.setattr(pipeline, 'TRAINER',
                        lambda model, dataset, tokenizer: type('T', (), {'run': lambda self: 'out'})())


def test_preprocess_mode_stops_before_the_model(monkeypatch):
    calls = []
    _stub_stages(monkeypatch, calls)
    assert pipeline.run(CONFIG(mode='preprocess')) == (None, None)
    assert calls == ['tokenizer', 'dataset']


def test_other_modes_build_the_model(monkeypatch):
    calls = []
    _stub_stages(monkeypatch, calls)
    assert pipeline.run(CONFIG(mode='train', model={'architecture': 'toy'})) == ('model', 'out')
    assert calls == ['tokenizer', 'dataset', 'model']
