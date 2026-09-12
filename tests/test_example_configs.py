"""``example()`` reaches the configs shipped inside the package.

It exists so the Quickstart runs on a fresh install with nothing to fetch and no
copy step -- and so that reaching them is one visible call rather than a second
lookup rule hidden inside ``pipeline.run``.
"""

import pytest
import yaml

import micm_nlp
from micm_nlp.path import available_examples, example


def test_example_is_exported_from_the_package():
    """The Quickstart imports it from ``micm_nlp`` directly."""
    assert micm_nlp.example is example
    assert 'example' in micm_nlp.__all__


def test_the_quickstart_config_resolves_and_parses():
    path = example('xsc_finetune.yml')
    assert path.is_file()
    assert isinstance(yaml.safe_load(path.read_text()), dict)


def test_the_suffix_is_optional():
    assert example('xsc_finetune') == example('xsc_finetune.yml')


def test_a_group_config_is_reachable_by_its_subpath():
    """The shipped set is two levels: the package root and ``groups/``."""
    assert example('groups/xsc_tune_across_seeds.yml').is_file()


def test_every_shipped_config_is_reachable_by_name():
    """``available_examples`` and ``example`` must agree -- a config listed by
    ``init-examples`` that ``example()`` cannot find would be a dead entry."""
    for relative, _ in available_examples():
        assert example(str(relative)).is_file()


def test_an_unknown_name_names_what_does_ship():
    """The set is small and fixed, so a typo should not send anyone to the docs."""
    with pytest.raises(FileNotFoundError) as excinfo:
        example('xsc_finetunee.yml')
    message = str(excinfo.value)
    assert 'xsc_finetunee.yml' in message
    assert 'xsc_finetune.yml' in message
