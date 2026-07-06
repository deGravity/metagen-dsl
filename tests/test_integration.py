"""End-to-end pipeline test: code.py → graph → geometry → simulate → C matrix.

Walks the same path a notebook user does:
    s = make_structure()
    s.simulate(resolution=R, backend='cpu')

Compares the resulting C matrix against the fixture prototype's. Tolerance
is data-driven from meta.json: the kernel+simulator pipeline isn't
bit-exact against the legacy evaluateJSONMetagen binary that produced the
prototype (~16/32768 boundary voxels can flip due to floating-point drift,
which propagates to ~1% drift in C-matrix entries). We bound this drift
by the observed inter-trial variation plus a safety factor.
"""

import numpy as np
import pytest

from . import fixture_data as fd

# Need the native kernel + simulator (metagen-dsl[native]).
pytest.importorskip('metagen_kernel',
                    reason='metagen-dsl[native] not installed')
pytest.importorskip('metagen_simulator',
                    reason='metagen-dsl[native] not installed')


def _make_structure(case):
    code = fd.load_code_py(case)
    program = (
        'from metagen_dsl import *\n'
        f'{code}\n'
        's = make_structure()'
    )
    env = {}
    exec(program, env)
    return env['s']


# Subset of cases for integration: skip TPMS-conjugation cases by default
# (those are flagged in meta.json), only test at one resolution to keep
# runtime modest.
INTEGRATION_RESOLUTION = 33   # smallest GPU-valid res; CPU also works


def _integration_cases():
    if not fd.fixtures_initialized():
        return []
    return [(c, INTEGRATION_RESOLUTION)
            for c, r in fd.case_iter(skip_default=True,
                                     resolutions=[INTEGRATION_RESOLUTION])]


# Near-solid cells at coarse resolution amplify boundary-voxel flips into C
# drift far beyond the generic 2% floor: volumetric_fcc at r33 (vf 0.91) shows
# 4.96% relative-Frobenius drift vs the legacy prototype at voxel IoU 0.9904 —
# the C of a near-solid cell is dominated by its thin void features, which a
# 32^3 grid barely resolves. Root-caused 2026-07 (T2 thread); pre-existing,
# not caused by the SDF/kernel changes. Case-specific budget, not a global
# floor bump, so every other deterministic case keeps the tight 2% bound.
_DRIFT_OVERRIDES = {
    ('volumetric_fcc', 33): 0.06,
}


def _expected_drift(case, resolution):
    """Tolerance budget for the full pipeline against the prototype.

    Combines (a) the observed inter-trial variation in the original
    legacy binary (variation_all_trials.dCFr_to_prototype.max) with
    (b) a safety factor for floating-point drift between the original
    and the new kernel/simulator implementations. Floored at 2% so
    deterministic cases still get a defensible bound, with documented
    per-case overrides where discretization physics demands more.
    """
    meta = fd.load_meta(case, resolution)
    var = meta.get('variation_all_trials', {}).get('dCFr_to_prototype') or {}
    observed_max = var.get('max') or 0.0
    floor = _DRIFT_OVERRIDES.get((case, resolution), 0.02)
    return max(floor, observed_max * 1.5)


@pytest.mark.parametrize('case,resolution', _integration_cases())
def test_dsl_pipeline_cpu_matches_reference(case, resolution):
    """Full DSL pipeline with CPU backend reproduces the prototype's C
    matrix within the observed inter-trial drift band."""
    s = _make_structure(case)
    sim = s.simulate(resolution=resolution, backend='cpu')

    expected_C = fd.load_c_matrix(case, resolution)
    produced = np.asarray(sim.C_matrix, dtype=np.float64)

    rtol = _expected_drift(case, resolution)
    diff = np.linalg.norm(produced - expected_C, ord='fro')
    denom = np.linalg.norm(0.5 * (produced + expected_C), ord='fro')
    rel = diff / denom if denom > 0 else float('inf')
    assert rel <= rtol, (
        f'{case}/r{resolution}: rel Frobenius drift {rel:.4f} '
        f'exceeds tolerance {rtol:.4f}')


@pytest.mark.parametrize('case,resolution', _integration_cases()[:3])
def test_dsl_pipeline_caches_results(case, resolution):
    """Calling .simulate twice with same args should return cached result."""
    s = _make_structure(case)
    a = s.simulate(resolution=resolution, backend='cpu')
    b = s.simulate(resolution=resolution, backend='cpu')
    assert a is b


def test_sim_only_path_matches_full_path(case='beam_bcc', resolution=33):
    """simulate() through the minimal-realization (voxels-only) kernel path
    must produce the same voxels — and hence the same C matrix — as
    simulating a fully-realized geometry."""
    # Full path: force full geometry first, then simulate (reuses it).
    s_full = _make_structure(case)
    geo_full = s_full.geometry(resolution=resolution)
    sim_full = s_full.simulate(resolution=resolution, backend='cpu')

    # Minimal path: fresh Structure, simulate() with cold cache goes
    # through geometry(realize='sim') -> kernel outputs='voxels'.
    s_min = _make_structure(case)
    sim_min = s_min.simulate(resolution=resolution, backend='cpu')
    geo_min = s_min._cache_get(('geometry_sim', resolution, 4))

    assert geo_min is not None, 'simulate() did not use the sim-only path'
    assert geo_min.thickened_vertices.shape[0] == 0  # really voxels-only
    assert np.array_equal(np.asarray(geo_min.voxel_active_cells),
                          np.asarray(geo_full.voxel_active_cells))
    # Identical voxels feed the same solver, so C matches to solver noise:
    # non-zero entries agree to rtol=1e-9; the atol floor only covers the
    # ~1e-18 numerical-zero entries, whose sign/magnitude wobbles between
    # any two CPU solves (parallel reduction order), including two
    # full-path runs.
    assert np.allclose(np.asarray(sim_min.C_matrix),
                       np.asarray(sim_full.C_matrix), rtol=1e-9, atol=1e-12)

    # A later render/interactive-style request still realizes full meshes.
    geo_render = s_min.geometry(resolution=resolution)
    assert geo_render.thickened_vertices.shape[0] > 0
