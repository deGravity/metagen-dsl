# Import submodules that do their own relative sibling-module lookups
# (e.g. `from . import skeleton as sk`) BEFORE star-imports below can
# shadow those submodule references with same-named functions.
from . import convex_polytope as cp
from . import skeleton
from .procmeta_translator import *

from .tile import *
from .pattern import *
from .lifting import *
from .skeleton import *
from .structure import *
from .convex_polytope import *

tet = cp.CPT_Tet("tet")
cuboid = cp.CPT_Cuboid("cuboid")
triPrism = cp.CPT_TriangularPrism("triPrism")

# Display / cache options (pandas-style)
from ._options import set_option, get_option, reset_option, option_context
