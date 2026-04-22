"""Adapter between metagen_dsl and the native kernel/simulator packages.

Imports metagen_kernel and metagen_simulator lazily so metagen-dsl stays
usable for pure graph.json generation without native deps installed.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import warnings


class MetagenBackendError(RuntimeError):
    """Raised when a native dep is required but not importable."""


_INSTALL_HINT = (
    "Install native backends: pip install metagen-dsl[native]\n"
    "(requires metagen-kernel and metagen-simulator)."
)


# ---------------------------------------------------------------------------
# Availability probes (cached)
# ---------------------------------------------------------------------------

def _try_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None


_kernel = None
_simulator = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = _try_import('metagen_kernel')
    return _kernel


def _get_simulator():
    global _simulator
    if _simulator is None:
        _simulator = _try_import('metagen_simulator')
    return _simulator


def has_kernel() -> bool:
    return _get_kernel() is not None


def has_simulator() -> bool:
    return _get_simulator() is not None


def gpu_available() -> bool:
    sim = _get_simulator()
    if sim is None:
        return False
    if hasattr(sim, 'native_gpu_available') and sim.native_gpu_available():
        return True
    if hasattr(sim, 'gpu_available'):
        try:
            avail, _, _ = sim.gpu_available()
            return bool(avail)
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def generate_voxels(graph_json: str, resolution: int):
    """Call metagen_kernel.generate(graph_json, resolution). Returns GeometryResult."""
    kernel = _get_kernel()
    if kernel is None:
        raise MetagenBackendError(f"metagen_kernel not installed.\n{_INSTALL_HINT}")
    return kernel.generate(graph_json, resolution)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    C_matrix: Any               # 6x6 numpy array
    volume_fraction: float
    solver_used: str            # 'gpu' | 'cpu'
    elapsed: float = 0.0
    properties: dict = field(default_factory=dict)
    gpu_shift: Optional[tuple] = None


def _derive_properties(C, volume_fraction):
    """Derive scalar material properties from a 6x6 stiffness matrix.

    Hill (1952) and Ranganathan & Ostoja-Starzewski (2008).
    """
    import numpy as np
    S = np.linalg.inv(C)

    K_V = (C[0,0] + C[1,1] + C[2,2] + 2*(C[0,1] + C[1,2] + C[2,0])) / 9.0
    K_R = 1.0 / (S[0,0] + S[1,1] + S[2,2] + 2*(S[0,1] + S[1,2] + S[2,0]))
    K_VRH = 0.5 * (K_V + K_R)

    G_V = ((C[0,0] + C[1,1] + C[2,2]) - (C[0,1] + C[1,2] + C[2,0])
           + 3*(C[3,3] + C[4,4] + C[5,5])) / 15.0
    G_R = 15.0 / (4*(S[0,0] + S[1,1] + S[2,2]) - 4*(S[0,1] + S[1,2] + S[2,0])
                  + 3*(S[3,3] + S[4,4] + S[5,5]))
    G_VRH = 0.5 * (G_V + G_R)

    nu_VRH = (3*K_VRH - 2*G_VRH) / (6*K_VRH + 2*G_VRH)
    E_VRH = 9*K_VRH*G_VRH / (3*K_VRH + G_VRH)

    denom_AZ = C[0,0] - C[0,1]
    A_Z = (2*C[3,3] / denom_AZ) if abs(denom_AZ) > 0 else float('inf')
    A_UAI = 5.0*(G_V/G_R) + (K_V/K_R) - 6.0 if (G_R != 0 and K_R != 0) else float('inf')

    return {
        'K_VRH': float(K_VRH), 'G_VRH': float(G_VRH),
        'E_VRH': float(E_VRH), 'nu_VRH': float(nu_VRH),
        'A_Z': float(A_Z), 'A_UAI': float(A_UAI),
        'volume_fraction': float(volume_fraction),
    }


def _simulate_cpu(geo, E: float, nu: float) -> SimulationResult:
    import numpy as np
    import time
    sim = _get_simulator()
    vox = np.ascontiguousarray(geo.voxel_active_cells, dtype=np.int8)
    t0 = time.perf_counter()
    r = sim.simulate_voxels(vox, geo.cell_resolution, E=E, nu=nu)
    elapsed = time.perf_counter() - t0
    C = np.array(r.C_matrix, dtype=np.float64)
    vf = float(r.volume_fraction)
    return SimulationResult(
        C_matrix=C, volume_fraction=vf, solver_used='cpu',
        elapsed=elapsed, properties=_derive_properties(C, vf))


def _simulate_gpu(geo, E: float, nu: float, relthres: float) -> SimulationResult:
    import numpy as np
    sim = _get_simulator()
    if not hasattr(sim, 'simulate_gpu'):
        raise MetagenBackendError("metagen_simulator lacks simulate_gpu")
    vox = np.asarray(geo.voxel_active_cells, dtype=np.float32)
    r = sim.simulate_gpu(vox, cell_dim=geo.cell_resolution,
                         E=E, nu=nu, relthres=relthres)
    if not r['success']:
        raise RuntimeError(f"GPU solver failed: {r['error']}")
    C = np.array(r['C_matrix'], dtype=np.float64)
    vf = float(r['volume_fraction'])
    return SimulationResult(
        C_matrix=C, volume_fraction=vf, solver_used='gpu',
        elapsed=float(r['elapsed']), gpu_shift=r.get('shift'),
        properties=_derive_properties(C, vf))


def simulate(geo, backend: str = 'auto', E: float = 1.0, nu: float = 0.45,
             relthres: float = 5e-3) -> SimulationResult:
    """Dispatch simulation to GPU or CPU.

    backend='auto' tries GPU then falls back to CPU on any failure.
    backend='gpu' raises if GPU unavailable or fails.
    backend='cpu' always uses CPU.
    """
    if _get_simulator() is None:
        raise MetagenBackendError(f"metagen_simulator not installed.\n{_INSTALL_HINT}")

    if backend == 'cpu':
        return _simulate_cpu(geo, E, nu)

    if backend == 'gpu':
        if not gpu_available():
            raise MetagenBackendError("GPU backend requested but not available.")
        return _simulate_gpu(geo, E, nu, relthres)

    if backend == 'auto':
        if gpu_available():
            sim = _get_simulator()
            if hasattr(sim, 'is_valid_multigrid_dim') and \
               sim.is_valid_multigrid_dim(geo.cell_resolution):
                try:
                    return _simulate_gpu(geo, E, nu, relthres)
                except Exception as e:
                    warnings.warn(f"GPU solver failed ({e}); falling back to CPU.")
        return _simulate_cpu(geo, E, nu)

    raise ValueError(f"backend must be 'auto'|'gpu'|'cpu', got {backend!r}")
