from typing import Any
from enum import Enum
from .skeleton import *
import numpy as np

class ThickeningProcedure(Enum):
    SPHERICAL = 0
    NORMAL = 1

class ThicknessProfileType(Enum):
    UNIFORM = 0
    VARYING = 1

class LiftedSkeleton:
    def __init__(self, _skel:Skeleton, _thickeningProc:ThickeningProcedure, _thicknessProfileType:ThicknessProfileType, _uniformThickness:float=None, _varyingThickness:np.array=None) -> None:
        assert len(_skel.connectedComponents) > 0, "Incompatible skeleton type detected. No skeleton detected in tile. Cannot lift to conjugate TPMS."
        self.skel = _skel
        self.parentCP = _skel.parentCP
        self.thickeningProc = _thickeningProc
        self.thicknessProfileType = _thicknessProfileType
        self.uniformThicknessValue = _uniformThickness
        self.varyingThicknessProfile = _varyingThickness


# =============================
#  Beams
# =============================
class UniformBeams(LiftedSkeleton):
    def __init__(self, skel:Skeleton, thickness:float, thickenProc:ThickeningProcedure=ThickeningProcedure.SPHERICAL) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel

        assert isinstance(_skel, EdgeSkeleton), "Incompatible skeleton type detected. Only EdgeSkeletons can be lifted to beams."
        super().__init__(_skel, thickenProc, ThicknessProfileType.UNIFORM, thickness, None)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
class SpatiallyVaryingBeams(LiftedSkeleton):
    def __init__(self, skel:Skeleton, thicknessProfile:list[list[float]], thickenProc:ThickeningProcedure=ThickeningProcedure.SPHERICAL) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel
        thickness = thicknessProfile

        assert isinstance(_skel, EdgeSkeleton), "Incompatible skeleton type detected. Only EdgeSkeletons can be lifted to beams."
        super().__init__(_skel, thickenProc, ThicknessProfileType.VARYING, None, np.array(thickness))

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
    

# =============================
#  Shells
# =============================
class UniformShell(LiftedSkeleton):
    def __init__(self, _skel:Skeleton, thickness:float, thickenProc:ThickeningProcedure=ThickeningProcedure.SPHERICAL) -> None:
        assert isinstance(_skel, EdgeSkeleton), "Incompatible skeleton type detected. Shells can only be defined over an EdgeSkeleton."
        assert len(_skel.connectedComponents) == 1, "Incompatible skeleton type detected. Multiple connected components detected in skeleton. Shells can only be defined over a single closed loop."
        assert _skel.connectedComponents[0].ccType == ConnectedComponentType.SIMPLE_CLOSED_LOOP, "Incompatible skeleton type detected. Shells can only be defined over a single closed loop."
        super().__init__(_skel, thickenProc, ThicknessProfileType.UNIFORM, thickness, None)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

class UniformTPMSShellViaConjugation(UniformShell):
    def __init__(self, skel:Skeleton, thickness:float, thickenProc:ThickeningProcedure=ThickeningProcedure.SPHERICAL) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel
        
        assert len(_skel.connectedComponents) == 1, "Incompatible skeleton type detected. Multiple connected components detected in skeleton. Conjugate TPMS can only be defined on a single closed loop."
        assert _skel.connectedComponents[0].ccType == ConnectedComponentType.SIMPLE_CLOSED_LOOP, "Incompatible skeleton type detected. Conjugate TPMS can only be defined on a single closed loop."
        ## TODO: assert that every edge is contained in a CP face
        ## TODO: assert that the loop follows a suitable path through the CP
        ## TODO: assert that there are no self-intersections in the loop 
        super().__init__(_skel, thickness, thickenProc)


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
class UniformDirectShell(UniformShell):
    def __init__(self, skel:Skeleton, thickness:float, thickenProc:ThickeningProcedure=ThickeningProcedure.SPHERICAL) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel
        
        super().__init__(_skel, thickness, thickenProc)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

class UniformTPMSShellViaMixedMinimal(UniformShell):
    def __init__(self, skel:Skeleton, thickness:float, thickenProc:ThickeningProcedure=ThickeningProcedure.SPHERICAL) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel

        super().__init__(_skel, thickness, thickenProc)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)



# =============================
#  Volumes
# =============================
class Spheres(LiftedSkeleton):
    def __init__(self, skel:Skeleton, thickness:float) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel
        radius = thickness
        
        assert isinstance(_skel, PointSkeleton), "Incompatible skeleton type detected. Only PointSkeletons can be lifted to spheres."
        super().__init__(_skel, ThickeningProcedure.SPHERICAL, ThicknessProfileType.UNIFORM, radius*2, None)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
