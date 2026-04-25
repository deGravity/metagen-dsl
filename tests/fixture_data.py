"""Helpers for loading test fixtures from the metagen-tests submodule.

Fixtures live at tests/fixtures/ relative to this file. Layout:

    fixtures/
        manifest.json
        <case>/
            code.py
            graph.json
            <res>/
                vox_active_cells.npz   (np.savez_compressed: 'voxels' bool)
                structure_info.json    (sim outputs from prototype trial)
                meta.json              (cluster + variation + runtime stats)
            renders/<res>/
                top_right.png ...

This module is copy-pasted into each subproject's tests/ rather than
imported from a shared package, so the three test suites stay
independently runnable without an extra installable dependency.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


FIXTURES = Path(__file__).parent / 'fixtures'


def fixtures_initialized() -> bool:
    return (FIXTURES / 'manifest.json').is_file()


@lru_cache(maxsize=1)
def manifest() -> dict:
    if not fixtures_initialized():
        raise RuntimeError(
            f'fixtures not initialized at {FIXTURES}. '
            'Run: git submodule update --init tests/fixtures'
        )
    return json.loads((FIXTURES / 'manifest.json').read_text())


def case_dir(case: str) -> Path:
    return FIXTURES / case


def res_dir(case: str, res: int) -> Path:
    return FIXTURES / case / str(res)


def renders_dir(case: str, res: int) -> Path:
    return FIXTURES / case / 'renders' / str(res)


def load_voxels(case: str, res: int) -> np.ndarray:
    """Load the prototype voxel grid as a (dim, dim, dim) bool array."""
    with np.load(res_dir(case, res) / 'vox_active_cells.npz') as z:
        return z['voxels']


def load_sim(case: str, res: int) -> dict:
    """Load the prototype simulation outputs (structure_info.json)."""
    return json.loads((res_dir(case, res) / 'structure_info.json').read_text())


def load_c_matrix(case: str, res: int) -> np.ndarray:
    """Convenience: extract the 6×6 stiffness matrix as a numpy array."""
    sim = load_sim(case, res)
    return np.asarray(sim['sim_C_matrix'], dtype=np.float64)


def load_meta(case: str, res: int) -> dict:
    return json.loads((res_dir(case, res) / 'meta.json').read_text())


def load_code_py(case: str) -> str:
    return (FIXTURES / case / 'code.py').read_text()


def load_graph_json(case: str) -> str:
    return (FIXTURES / case / 'graph.json').read_text()


def render_resolutions(case: str) -> list[int]:
    rd = FIXTURES / case / 'renders'
    if not rd.is_dir():
        return []
    return sorted(int(d.name) for d in rd.iterdir()
                  if d.is_dir() and d.name.isdigit())


def case_iter(skip_default: bool = True,
              resolutions: Optional[list] = None) -> Iterator[tuple]:
    """Iterate over (case, resolution) pairs in the manifest.

    skip_default=True   omit (case, res) flagged skip_by_default in meta.json
    resolutions=[...]   restrict to specified resolutions (default: all in manifest)
    """
    m = manifest()
    target_res = set(resolutions or m['resolutions'])
    for c in m['cases']:
        for res_str, info in c['resolutions'].items():
            r = int(res_str)
            if r not in target_res:
                continue
            if skip_default and info.get('skip_by_default'):
                continue
            yield c['name'], r


def cases_with_renders() -> list[str]:
    return [c['name'] for c in manifest()['cases']
            if c.get('render_png_count', 0) > 0]


def runtime_tier(case: str, res: int) -> str:
    """Map mean elapsed time per trial to a coarse tier label.

    Used by conftest.py to mark tests for selective running.
    """
    rt = load_meta(case, res).get('runtime') or {}
    mean = rt.get('mean')
    if mean is None:
        return 'unknown'
    if mean < 5:
        return 'tier1'
    if mean < 60:
        return 'tier2'
    return 'tier3'
