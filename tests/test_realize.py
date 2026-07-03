"""Tests for the minimal-realization (sim-only) geometry path.

Structure.geometry(realize='sim') asks the kernel for voxels only
(outputs='voxels'); Structure.simulate() uses that path by default.
Cache semantics under test:

  - full and sim-only results live under distinct cache keys;
  - a cached FULL geometry satisfies realize='sim' (superset, no new call);
  - a cached SIM-ONLY geometry never satisfies realize='full' — so a later
    render()/interactive() still realizes real meshes;
  - simulate() consumes the sim path and never forces full realization.

These tests fake the backend so they run without native deps.
"""

import pytest

from metagen_dsl import _backend

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


class _FakeGeo:
    def __init__(self, outputs):
        self.outputs = outputs                 # what the kernel was asked for
        self.voxel_active_cells = [[[1]]]
        self.cell_resolution = 1
        self.volume_fraction = 1.0


class _FakeBackend:
    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(_backend, 'generate_voxels', self._generate)
        monkeypatch.setattr(_backend, 'simulate', self._simulate)

    def _generate(self, graph_json, resolution, tpms_multistart_k=4,
                  outputs='all'):
        self.calls.append(outputs)
        return _FakeGeo(outputs)

    def _simulate(self, geo, backend='auto', E=1.0, nu=0.45, quality=None):
        return ('sim-result', geo)


@pytest.fixture
def fake_backend(monkeypatch):
    return _FakeBackend(monkeypatch)


def test_realize_sim_requests_voxels_only(fake_backend):
    s = _make_simple_structure()
    geo = s.geometry(resolution=33, realize='sim')
    assert fake_backend.calls == ['voxels']
    assert geo.outputs == 'voxels'
    assert ('geometry_sim', 33, 4) in s._cache
    assert ('geometry', 33, 4) not in s._cache


def test_realize_full_default_requests_all(fake_backend):
    s = _make_simple_structure()
    geo = s.geometry(resolution=33)
    assert fake_backend.calls == ['all']
    assert geo.outputs == 'all'
    assert ('geometry', 33, 4) in s._cache


def test_sim_result_does_not_poison_full_path(fake_backend):
    """After a sim-only geometry, realize='full' must hit the kernel again
    and return a full result (render()/interactive() depend on this)."""
    s = _make_simple_structure()
    sim_geo = s.geometry(resolution=33, realize='sim')
    full_geo = s.geometry(resolution=33)
    assert full_geo is not sim_geo
    assert full_geo.outputs == 'all'
    assert fake_backend.calls == ['voxels', 'all']


def test_full_result_satisfies_sim_request(fake_backend):
    """A cached full geometry is a superset — realize='sim' reuses it
    without another kernel call."""
    s = _make_simple_structure()
    full_geo = s.geometry(resolution=33)
    sim_geo = s.geometry(resolution=33, realize='sim')
    assert sim_geo is full_geo
    assert fake_backend.calls == ['all']


def test_sim_request_cached(fake_backend):
    s = _make_simple_structure()
    g1 = s.geometry(resolution=33, realize='sim')
    g2 = s.geometry(resolution=33, realize='sim')
    assert g1 is g2
    assert fake_backend.calls == ['voxels']


def test_simulate_uses_sim_path_when_nothing_cached(fake_backend):
    s = _make_simple_structure()
    tag, geo = s.simulate(resolution=33, backend='cpu')
    assert tag == 'sim-result'
    assert geo.outputs == 'voxels'
    assert fake_backend.calls == ['voxels']


def test_simulate_reuses_cached_full_geometry(fake_backend):
    s = _make_simple_structure()
    full_geo = s.geometry(resolution=33)
    tag, geo = s.simulate(resolution=33, backend='cpu')
    assert geo is full_geo
    assert fake_backend.calls == ['all']   # no second kernel call


def test_invalid_realize_raises(fake_backend):
    s = _make_simple_structure()
    with pytest.raises(ValueError):
        s.geometry(resolution=33, realize='meshes')
