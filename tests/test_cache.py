"""Tests for the Structure-instance LRU cache."""

import pytest

import metagen_dsl
from metagen_dsl import _options

from . import fixture_data as fd


def _make_simple_structure():
    code = (fd.FIXTURES / 'beam_bcc' / 'code.py').read_text()
    program = (
        'from metagen_dsl import *\n'
        f'{code}\n'
        's = make_structure()'
    )
    env = {}
    exec(program, env)
    return env['s']


def test_graph_json_cached():
    s = _make_simple_structure()
    g1 = s.graph_json()
    g2 = s.graph_json()
    assert g1 is g2  # same string object means cache hit


def test_clear_cache_resets():
    s = _make_simple_structure()
    s.graph_json()
    assert ('graph_json',) in s._cache
    s.clear_cache()
    assert s._cache == {}


def test_lru_eviction_respects_max_entries():
    """With max_entries=4, after 5 distinct cache puts, the oldest is evicted."""
    s = _make_simple_structure()
    saved = metagen_dsl.get_option('cache.max_entries')
    try:
        metagen_dsl.set_option('cache.max_entries', 2)
        s._cache_put(('a',), 1)
        s._cache_put(('b',), 2)
        s._cache_put(('c',), 3)
        # ('a',) should have been evicted; ('b',) and ('c',) remain.
        assert ('a',) not in s._cache
        assert ('b',) in s._cache
        assert ('c',) in s._cache
    finally:
        metagen_dsl.set_option('cache.max_entries', saved)


def test_cache_get_promotes_to_most_recent():
    """A get() should move the entry to the end of the OrderedDict so
    that the next eviction takes the truly oldest entry."""
    s = _make_simple_structure()
    saved = metagen_dsl.get_option('cache.max_entries')
    try:
        metagen_dsl.set_option('cache.max_entries', 2)
        s._cache_put(('a',), 1)
        s._cache_put(('b',), 2)
        # Touch 'a' so 'b' becomes oldest
        s._cache_get(('a',))
        s._cache_put(('c',), 3)
        assert ('a',) in s._cache
        assert ('b',) not in s._cache
        assert ('c',) in s._cache
    finally:
        metagen_dsl.set_option('cache.max_entries', saved)


def test_clear_cache_returns_self_consistent_state():
    """After clear_cache, calling graph_json again should regenerate (and
    return the same value, since graph_json is deterministic)."""
    s = _make_simple_structure()
    g1 = s.graph_json()
    s.clear_cache()
    g2 = s.graph_json()
    assert g1 == g2
    assert g1 is not g2  # not the same object — was regenerated
