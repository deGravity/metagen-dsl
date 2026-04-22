"""Visualization helpers. All external deps are imported lazily."""

import io
from typing import Optional


# Viewpoints: (elev, azim) in matplotlib convention (degrees)
_VIEWS = {
    'front':     (0,   -90),
    'back':      (0,    90),
    'right':     (0,     0),
    'left':      (0,   180),
    'top':       (90,  -90),
    'bottom':    (-90, -90),
    'top_right': (30,  -45),
}


def _get_surface(geo):
    """Extract (vertices, triangles) for rendering.

    Prefers voxel_surface_* (cleaner faces, no internal quads) over
    thickened_* (marching-cubes smoothed).
    """
    import numpy as np
    verts = np.asarray(geo.voxel_surface_vertices, dtype=float)
    tris = np.asarray(geo.voxel_surface_triangles, dtype=int)
    if verts.size == 0 or tris.size == 0:
        verts = np.asarray(geo.thickened_vertices, dtype=float)
        tris = np.asarray(geo.thickened_triangles, dtype=int)
    return verts, tris


# ---------------------------------------------------------------------------
# Static renders via matplotlib (headless-safe)
# ---------------------------------------------------------------------------

def _render_one_matplotlib(verts, tris, view: str, size: tuple) -> bytes:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    elev, azim = _VIEWS[view]
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    tri_verts = verts[tris]
    pc = Poly3DCollection(tri_verts, linewidth=0.0, alpha=1.0)
    pc.set_facecolor((0.4, 0.5, 0.8))
    pc.set_edgecolor('none')
    ax.add_collection3d(pc)

    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def render_static(geo, views=('top_right', 'top', 'front', 'right'),
                  size=(400, 400)) -> dict:
    """Render views via matplotlib. Returns {view_name: png_bytes}."""
    verts, tris = _get_surface(geo)
    return {v: _render_one_matplotlib(verts, tris, v, size) for v in views}


# ---------------------------------------------------------------------------
# Interactive via k3d
# ---------------------------------------------------------------------------

def render_interactive(geo):
    """Return a k3d.Plot widget. Caller displays it."""
    try:
        import k3d
    except ImportError as e:
        raise ImportError(
            "interactive rendering requires k3d. Install: pip install metagen-dsl[viz]"
        ) from e
    import numpy as np
    verts, tris = _get_surface(geo)
    plot = k3d.plot()
    plot += k3d.mesh(
        verts.astype(np.float32).flatten(),
        tris.astype(np.uint32).flatten(),
        color=0x6680CC,
    )
    return plot


# ---------------------------------------------------------------------------
# Simulation summary HTML
# ---------------------------------------------------------------------------

def sim_summary_html(sim) -> str:
    """HTML table summarizing a SimulationResult."""
    props = sim.properties
    rows = []
    rows.append(('solver', sim.solver_used))
    rows.append(('volume fraction', f"{props.get('volume_fraction', 0):.4f}"))
    rows.append(('Young\'s modulus E (VRH)', f"{props.get('E_VRH', 0):.4g}"))
    rows.append(('bulk modulus K (VRH)', f"{props.get('K_VRH', 0):.4g}"))
    rows.append(('shear modulus G (VRH)', f"{props.get('G_VRH', 0):.4g}"))
    rows.append(('Poisson ν (VRH)', f"{props.get('nu_VRH', 0):.4g}"))
    rows.append(('Zener anisotropy A_Z', f"{props.get('A_Z', 0):.4g}"))
    rows.append(('universal anisotropy A_UAI', f"{props.get('A_UAI', 0):.4g}"))
    tbl = ''.join(f'<tr><td>{k}</td><td><code>{v}</code></td></tr>' for k, v in rows)

    C = sim.C_matrix
    C_rows = ''.join(
        '<tr>' + ''.join(f'<td><code>{C[i,j]:.4g}</code></td>' for j in range(6)) + '</tr>'
        for i in range(6)
    )
    return (
        f'<table style="border-collapse:collapse;margin:4px;">{tbl}</table>'
        f'<p><b>C matrix (6×6):</b></p>'
        f'<table style="border-collapse:collapse;margin:4px;font-size:0.9em;">{C_rows}</table>'
    )


# ---------------------------------------------------------------------------
# Static renders via pyrender (optional, legacy, for offline parity)
# ---------------------------------------------------------------------------

def render_pyrender(geo, views=('top_right', 'top', 'front', 'right'),
                    size=(400, 400)) -> dict:
    """Legacy pyrender path. Requires metagen-dsl[pyrender]."""
    try:
        import pyrender  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyrender is an optional dependency. Install: pip install metagen-dsl[pyrender]"
        ) from e
    raise NotImplementedError(
        "pyrender path not implemented yet; port the original renderer when needed."
    )
