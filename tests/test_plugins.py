"""@micm_plugin: registry, discovery, and resolution through resolve_cls."""

import pytest

from micm_nlp import plugins


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(plugins, '_PLUGINS', {})
    monkeypatch.setattr(plugins, '_discovered', False)


def test_registers_a_class_and_returns_it_unchanged():
    class Thing:
        pass

    assert plugins.micm_plugin(Thing) is Thing
    assert plugins._PLUGINS['Thing'] is Thing


def test_registers_a_function():
    def my_f1(preds, labels):
        return 1.0

    assert plugins.micm_plugin(my_f1) is my_f1
    assert plugins._PLUGINS['my_f1'] is my_f1


def test_same_class_twice_is_a_no_op():
    class Thing:
        pass

    plugins.micm_plugin(Thing)
    plugins.micm_plugin(Thing)
    assert plugins._PLUGINS == {'Thing': Thing}


def test_different_class_under_a_taken_name_raises():
    def make():
        class Thing:
            pass
        return Thing

    first, second = make(), make()
    second.__module__ = 'somewhere.else'
    plugins.micm_plugin(first)
    with pytest.raises(ValueError, match="'Thing' is taken"):
        plugins.micm_plugin(second)


def test_exported_from_the_package():
    import micm_nlp

    assert micm_nlp.micm_plugin is plugins.micm_plugin
