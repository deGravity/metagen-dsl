import base64
from collections import OrderedDict
from enum import Enum

from .tile import Tile
from .pattern import TilingPattern
from . import _options
from . import _backend


# =======================================
#   Structures
# =======================================
class Structure:
    """A metamaterial unit cell: a Tile plus a TilingPattern.

    Methods that do expensive work (geometry generation, simulation,
    rendering) are cached per-instance in a bounded LRU.
    """

    def __init__(self, tile: Tile, pattern: TilingPattern):
        self.tile = tile
        self.pat = pattern
        self._cache = OrderedDict()

    # -----------------------------------------------------------------
    # cache helpers
    # -----------------------------------------------------------------
    def _cache_get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > _options.get_option('cache.max_entries'):
            self._cache.popitem(last=False)

    def clear_cache(self):
        self._cache.clear()

    # -----------------------------------------------------------------
    # graph serialization (cached once — cheap but unnecessary to redo)
    # -----------------------------------------------------------------
    def graph_json(self) -> str:
        """Return the procedural graph JSON string for this Structure."""
        cached = self._cache_get(('graph_json',))
        if cached is not None:
            return cached
        from .procmeta_translator import ProcMetaTranslator
        s = ProcMetaTranslator(self).to_json()
        self._cache_put(('graph_json',), s)
        return s

    # -----------------------------------------------------------------
    # geometry / simulation
    # -----------------------------------------------------------------
    def geometry(self, resolution: int = None):
        """Generate geometry via metagen_kernel. Cached by resolution."""
        if resolution is None:
            resolution = _options.get_option('display.resolution_direct')
        key = ('geometry', resolution)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        geo = _backend.generate_voxels(self.graph_json(), resolution)
        self._cache_put(key, geo)
        return geo

    def simulate(self, resolution: int = None, backend: str = None,
                 E: float = 1.0, nu: float = 0.45):
        """Run homogenization. Cached by (resolution, backend, E, nu)."""
        if resolution is None:
            resolution = _options.get_option('display.resolution_direct')
        if backend is None:
            backend = _options.get_option('display.backend')
        key = ('simulate', resolution, backend, E, nu)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        geo = self.geometry(resolution)
        sim = _backend.simulate(geo, backend=backend, E=E, nu=nu)
        self._cache_put(key, sim)
        return sim

    # -----------------------------------------------------------------
    # visualization
    # -----------------------------------------------------------------
    def render(self, resolution: int = None, views=None, size=None) -> dict:
        """Render static views via matplotlib. Returns {view: png_bytes}."""
        if resolution is None:
            resolution = _options.get_option('display.resolution_direct')
        if views is None:
            views = _options.get_option('display.render_views')
        if size is None:
            size = _options.get_option('display.render_size')
        key = ('render', resolution, tuple(views), tuple(size))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        from . import _viz
        geo = self.geometry(resolution)
        imgs = _viz.render_static(geo, views=views, size=size)
        self._cache_put(key, imgs)
        return imgs

    def interactive(self, resolution: int = None):
        """Return a k3d.Plot for notebook display."""
        if resolution is None:
            resolution = _options.get_option('display.resolution_direct')
        from . import _viz
        return _viz.render_interactive(self.geometry(resolution))

    def summary(self, resolution: int = None, backend: str = None) -> str:
        """HTML table of the simulation results."""
        from . import _viz
        return _viz.sim_summary_html(self.simulate(resolution, backend))

    # -----------------------------------------------------------------
    # repr
    # -----------------------------------------------------------------
    def __repr__(self):
        tile = type(self.tile).__name__ if hasattr(self, 'tile') else '?'
        pat = type(self.pat).__name__ if hasattr(self, 'pat') else '?'
        return f"<{type(self).__name__} tile={tile} pat={pat}>"

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Rich repr for Jupyter. CLI `repr()` uses __repr__ and never calls this."""
        show = _options.get_option('display.show')
        resolution = _options.get_option('display.resolution')
        simulate_in_repr = _options.get_option('display.simulate_in_repr')

        if show == 'auto':
            show = 'render'
        if simulate_in_repr == 'auto':
            simulate_in_repr = _backend.gpu_available()

        if show == 'none':
            return {'text/plain': repr(self)}

        if not _backend.has_kernel():
            return {
                'text/plain': repr(self),
                'text/html': (f'<p><code>{self!r}</code></p>'
                              f'<p><em>install <code>metagen-dsl[native]</code> '
                              f'for geometry and simulation.</em></p>'),
            }

        try:
            parts = [f'<p><code>{self!r}</code></p>']

            if show in ('render', 'all'):
                imgs = self.render(resolution=resolution)
                tbl_cells = ''.join(
                    f'<td><img src="data:image/png;base64,'
                    f'{base64.b64encode(b).decode()}"/></td>'
                    for b in imgs.values()
                )
                parts.append(f'<table><tr>{tbl_cells}</tr></table>')

            if show in ('sim', 'all') or (show == 'render' and simulate_in_repr):
                if _backend.has_simulator():
                    sim = self.simulate(resolution=resolution)
                    from . import _viz
                    parts.append(_viz.sim_summary_html(sim))

            return {
                'text/plain': repr(self),
                'text/html': '\n'.join(parts),
            }
        except Exception as e:
            return {
                'text/plain': repr(self),
                'text/html': (f'<p><code>{self!r}</code></p>'
                              f'<p><em>repr failed: {type(e).__name__}: {e}</em></p>'),
            }


# =======================================
#   CSG Booleans -- also Structures since they're realizable
# =======================================
class CSGBooleanTypes(Enum):
    UNION = 0
    INTERSECT = 1
    DIFFERENCE = 2


class CSGBoolean(Structure):
    def __init__(self, _A: Structure, _B: Structure, _opType: CSGBooleanTypes):
        self.A = _A
        self.B = _B
        self.op_type = _opType
        self._cache = OrderedDict()


class Union(CSGBoolean):
    def __init__(self, A: Structure, B: Structure):
        super().__init__(A, B, CSGBooleanTypes.UNION)


class Intersect(CSGBoolean):
    def __init__(self, A: Structure, B: Structure):
        super().__init__(A, B, CSGBooleanTypes.INTERSECT)


class Subtract(CSGBoolean):
    def __init__(self, A: Structure, B: Structure):
        super().__init__(A, B, CSGBooleanTypes.DIFFERENCE)
