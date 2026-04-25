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


def _expected_drift(case, resolution):
    """Tolerance budget for the full pipeline against the prototype.

    Combines (a) the observed inter-trial variation in the original
    legacy binary (variation_all_trials.dCFr_to_prototype.max) with
    (b) a safety factor for floating-point drift between the original
    and the new kernel/simulator implementations. Floored at 2% so
    deterministic cases still get a defensible bound.
    """
    meta = fd.load_meta(case, resolution)
    var = meta.get('variation_all_trials', {}).get('dCFr_to_prototype') or {}
    observed_max = var.get('max') or 0.0
    return max(0.02, observed_max * 1.5)


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
