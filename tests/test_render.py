"""Tests for the Structure rendering paths.

Default path uses matplotlib (installed via metagen-dsl[viz]) and produces
a dict of view-name → PNG bytes. We don't pixel-compare against the
fixture renders here — those came from pyrender and matplotlib won't
match them exactly. We just check the output is structurally valid.

Pixel-perfect parity with the legacy pyrender output is checked in
test_render_pyrender.py, gated behind metagen-dsl[pyrender].
"""

import pytest

from . import fixture_data as fd

# These tests need the kernel for geometry generation.
metagen_kernel = pytest.importorskip(
    'metagen_kernel',
    reason='metagen-dsl[native] not installed (need metagen-kernel)')

# Matplotlib for the default rendering path.
pytest.importorskip(
    'matplotlib',
    reason='metagen-dsl[viz] not installed (need matplotlib)')

# Use a small representative subset so test_render is fast.
RENDER_CASES = ['beam_bcc', 'cubic_foam', 'octet_foam']
RENDER_RESOLUTION = 33  # smallest, fastest


def _make_structure(case):
    """Build a Structure from a fixture case's code.py."""
    code = fd.load_code_py(case)
    env = {}
    program = (
        'from metagen_dsl import *\n'
        f'{code}\n'
        's = make_structure()'
    )
    exec(program, env)
    return env['s']


@pytest.mark.tier1
@pytest.mark.parametrize('case', RENDER_CASES)
def test_render_returns_view_dict_of_pngs(case):
    s = _make_structure(case)
    imgs = s.render(resolution=RENDER_RESOLUTION)
    expected_views = {'top_right', 'top', 'front', 'right'}
    assert set(imgs.keys()) == expected_views, \
        f'unexpected views: {set(imgs.keys())} (expected {expected_views})'
    for view, data in imgs.items():
        assert isinstance(data, bytes), f'{view} is not bytes'
        # PNG files start with the 8-byte signature \x89PNG\r\n\x1a\n.
        assert data.startswith(b'\x89PNG\r\n\x1a\n'), \
            f'{view} is not a valid PNG'
        assert len(data) > 1000, f'{view} suspiciously small ({len(data)} B)'


@pytest.mark.tier1
def test_render_caches_per_resolution_and_views():
    s = _make_structure('beam_bcc')
    a = s.render(resolution=RENDER_RESOLUTION)
    b = s.render(resolution=RENDER_RESOLUTION)
    assert a is b  # cache hit


@pytest.mark.tier1
def test_render_changes_when_views_differ():
    s = _make_structure('beam_bcc')
    a = s.render(resolution=RENDER_RESOLUTION, views=('front',))
    b = s.render(resolution=RENDER_RESOLUTION, views=('top',))
    assert set(a.keys()) == {'front'}
    assert set(b.keys()) == {'top'}


@pytest.mark.tier1
def test_repr_mimebundle_returns_text_and_html():
    """_repr_mimebundle_ is what Jupyter calls. CLI repr() must NOT trigger
    expensive work — we test that __repr__ stays cheap below."""
    s = _make_structure('beam_bcc')
    # Force a small res so this is fast in CI.
    import metagen_dsl
    with metagen_dsl.option_context('display.resolution', RENDER_RESOLUTION,
                                     'display.show', 'render',
                                     'display.simulate_in_repr', False):
        bundle = s._repr_mimebundle_()
    assert 'text/plain' in bundle
    assert 'text/html' in bundle
    # Plain repr should be a tight one-liner.
    assert '\n' not in bundle['text/plain']
    # HTML should embed at least one PNG via base64.
    assert 'data:image/png;base64' in bundle['text/html']


@pytest.mark.tier1
def test_cli_repr_is_cheap():
    """__repr__ (used by Python CLI) must NOT compute geometry."""
    s = _make_structure('beam_bcc')
    # No geometry call expected; cache should remain empty.
    out = repr(s)
    assert s._cache == {}
    assert isinstance(out, str)
    assert 'Structure' in out
