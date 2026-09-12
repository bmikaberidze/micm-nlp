"""``pipeline.run`` on a ``mode: preprocess`` config stops after preprocessing:
no model is built, no trainer runs. The stages are stubbed; only the routing
is under test."""

from micm_nlp import pipeline
from micm_nlp.config import CONFIG


def _stub_stages(monkeypatch, calls):
    """Stub the core classes ``run`` calls -- not the single-stage wrappers beside
    it, which ``run`` deliberately does not go through (see
    ``tests/test_pipeline_stages.py``)."""
    def _dataset(config):
        calls.append('dataset')
        return type('D', (), {'preprocess': lambda self, tokenizer: calls.append('preprocess')})()

    monkeypatch.setattr(pipeline, 'load_tokenizer', lambda config: calls.append('tokenizer') or 'tok')
    monkeypatch.setattr(pipeline, 'DATASET', _dataset)
    monkeypatch.setattr(pipeline, 'MODEL', lambda config: calls.append('model') or 'model')
    monkeypatch.setattr(
        pipeline, 'TRAINER',
        lambda model, dataset, tokenizer: type(
            'T', (), {'run': lambda self: 'ran', 'output': 'the-output'},
        )(),
    )


def test_preprocess_mode_stops_before_the_model(monkeypatch):
    """No model, no trainer -- and a bare ``None``, not a pair of them."""
    calls = []
    _stub_stages(monkeypatch, calls)
    assert pipeline.run(CONFIG(mode='preprocess')) is None
    assert calls == ['tokenizer', 'dataset', 'preprocess']


def test_other_modes_return_the_run_output(monkeypatch):
    """The trainer's ``output`` is what comes back -- not the model, not the
    trainer, and not the value ``TRAINER.run()`` returns."""
    calls = []
    _stub_stages(monkeypatch, calls)
    assert pipeline.run(CONFIG(mode='train', model={'architecture': 'toy'})) == 'the-output'
    assert calls == ['tokenizer', 'dataset', 'preprocess', 'model']
