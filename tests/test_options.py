"""Tests for the pandas-style display options API."""

import pytest

import metagen_dsl
from metagen_dsl import _options


@pytest.fixture(autouse=True)
def _reset_options():
    """Restore option state after each test."""
    saved = dict(_options._options)
    yield
    _options._options.clear()
    _options._options.update(saved)


def test_get_option_returns_default():
    assert metagen_dsl.get_option('display.resolution') == 65
    assert metagen_dsl.get_option('display.resolution_direct') == 97
    assert metagen_dsl.get_option('display.backend') == 'auto'


def test_set_get_option_roundtrip():
    metagen_dsl.set_option('display.resolution', 33)
    assert metagen_dsl.get_option('display.resolution') == 33


def test_unknown_option_raises():
    with pytest.raises(KeyError):
        metagen_dsl.get_option('display.bogus')
    with pytest.raises(KeyError):
        metagen_dsl.set_option('display.bogus', 1)


def test_reset_option():
    metagen_dsl.set_option('display.resolution', 33)
    metagen_dsl.reset_option('display.resolution')
    assert metagen_dsl.get_option('display.resolution') == 65


def test_option_context_restores_value():
    original = metagen_dsl.get_option('display.show')
    with metagen_dsl.option_context('display.show', 'sim'):
        assert metagen_dsl.get_option('display.show') == 'sim'
    assert metagen_dsl.get_option('display.show') == original


def test_option_context_restores_on_exception():
    original = metagen_dsl.get_option('display.show')
    with pytest.raises(RuntimeError):
        with metagen_dsl.option_context('display.show', 'sim'):
            raise RuntimeError('boom')
    assert metagen_dsl.get_option('display.show') == original


def test_option_context_multiple_pairs():
    with metagen_dsl.option_context(
        'display.resolution', 33,
        'display.backend', 'cpu',
    ):
        assert metagen_dsl.get_option('display.resolution') == 33
        assert metagen_dsl.get_option('display.backend') == 'cpu'
    assert metagen_dsl.get_option('display.resolution') == 65
    assert metagen_dsl.get_option('display.backend') == 'auto'


def test_option_context_rejects_odd_args():
    with pytest.raises(ValueError):
        with metagen_dsl.option_context('display.resolution'):
            pass
