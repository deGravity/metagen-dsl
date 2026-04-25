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
    """Lift a skeleton to a 3D structure of uniform-thickness beams.

    Procedure to lift the input skeleton to a 3D volumetric structure by
    instantiating a beam of the given thickness centered along each
    polyline/curve of the input skeleton.

    @requirements:
        The skeleton must contain only polylines and/or curves. The skeleton
        must not contain any standalone vertices.
    @params:
        skel - the skeleton to lift.
        thickness - the diameter of the beams.
    @returns:
        liftProc - the lifted skeleton.
    @example_usage:
        liftProcedure = UniformBeams(skel, 0.03)
    """
    def __init__(self, skel:Skeleton, thickness:float, thickenProc:ThickeningProcedure=ThickeningProcedure.SPHERICAL) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel

        assert isinstance(_skel, EdgeSkeleton), "Incompatible skeleton type detected. Only EdgeSkeletons can be lifted to beams."
        super().__init__(_skel, thickenProc, ThicknessProfileType.UNIFORM, thickness, None)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
class SpatiallyVaryingBeams(LiftedSkeleton):
    """Lift a skeleton to beams with a spatially-varying thickness profile.

    Procedure to lift the input skeleton to a 3D volumetric structure by
    instantiating a beam of the given spatially-varying thickness profile
    centered along each polyline/curve of the input skeleton.

    @requirements:
        The skeleton must contain only polylines and/or curves. The skeleton
        must not contain any standalone vertices.
    @params:
        skel - the skeleton to lift.
        thicknessProfile - specifications for the diameter of the beams along
                           each polyline/curve. Given as a list[list[float]],
                           where each of the n inner lists gives the
                           information for a single sample point along the
                           polyline/curve. The first element in each inner
                           list provides a position parameter t in [0,1] along
                           the polyline/curve, and the second element specifies
                           the thickness of the beam at position t.
    @returns:
        liftProc - the lifted skeleton.
    @example_usage:
        liftProcedure = SpatiallyVaryingBeams(skel, [[0.0, 0.02], [1.0, 0.06]])
    """
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
    """Lift a skeleton to a TPMS shell via the conjugate-surface construction.

    Procedure to lift the input skeleton to a 3D volumetric structure by
    inferring a triply periodic minimal surface (TPMS) that conforms to the
    boundary constraints provided by the input skeleton. The surface is
    computed via the conjugate surface construction method.

    @requirements:
        The skeleton must contain a single closed loop composed of one or
        more polylines and/or curves. The skeleton must not contain any
        standalone vertices.
        Each vertex in the polylines/curves must live on a CP edge.
        Adjacent vertices must have a shared face.
        The loop must touch every face of the CP at least once.
        If the CP has N faces, the loop must contain at least N vertices.
    @params:
        skel - the skeleton to lift.
        thickness - the thickness of the shell. The final offset is
                    thickness/2 to each side of the inferred surface.
    @returns:
        liftProc - the lifted skeleton.
    @example_usage:
        liftProcedure = UniformTPMSShellViaConjugation(skel, 0.03)
    """
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
    """Lift a skeleton to a thin shell with a directly-fitted surface.

    Procedure to lift the input skeleton to a 3D volumetric structure by
    inferring a surface that conforms to the boundary provided by the input
    skeleton. The surface is given by a simple thin shell model: the
    resulting surface is incident on the provided boundary while minimizing
    a weighted sum of bending and stretching energies. The boundary is fixed,
    though it may be constructed with a mix of polylines and curves (which
    are first interpolated into a spline, then fixed as part of the boundary).

    @requirements:
        The skeleton must contain a single closed loop composed of one or
        more polylines and/or curves. The skeleton must not contain any
        standalone vertices.
    @params:
        skel - the skeleton to lift.
        thickness - the thickness of the shell. The final offset is
                    thickness/2 to each side of the inferred surface.
    @returns:
        liftProc - the lifted skeleton.
    @example_usage:
        liftProcedure = UniformDirectShell(skel, 0.1)
    """
    def __init__(self, skel:Skeleton, thickness:float, thickenProc:ThickeningProcedure=ThickeningProcedure.SPHERICAL) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel
        
        super().__init__(_skel, thickness, thickenProc)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

class UniformTPMSShellViaMixedMinimal(UniformShell):
    """Lift a skeleton to a TPMS shell via mean curvature flow.

    Procedure to lift the input skeleton to a 3D volumetric structure by
    inferring a triply periodic minimal surface (TPMS) that conforms to the
    boundary constraints provided by the input skeleton. The surface is
    computed via mean curvature flow. All polyline boundary regions are
    considered fixed, but any curved regions may slide within their
    respective planes in order to reduce surface curvature during the solve.

    @requirements:
        The skeleton must contain a single closed loop composed of one or
        more polylines and/or curves. The skeleton must not contain any
        standalone vertices.
        Each vertex in the polylines/curves must live on a CP edge.
        Adjacent vertices must have a shared face.
    @params:
        skel - the skeleton to lift.
        thickness - the thickness of the shell. The final offset is
                    thickness/2 to each side of the inferred surface.
    @returns:
        liftProc - the lifted skeleton.
    @example_usage:
        liftProcedure = UniformTPMSShellViaMixedMinimal(skel, 0.03)
    """
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
    """Lift a point skeleton by placing a sphere at each vertex.

    Procedure to lift the input skeleton to a 3D volumetric structure by
    instantiating a sphere of the given radius centered at vertex p, for
    each vertex in the skeleton.

    @requirements:
        The skeleton must only contain standalone vertices; no polylines or
        curves can be used.
    @params:
        skel - the skeleton to lift.
        thickness - the sphere radius.
    @returns:
        liftProc - the lifted skeleton.
    @example_usage:
        s_lift = Spheres(skel, 0.25)
    """
    def __init__(self, skel:Skeleton, thickness:float) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _skel = skel
        radius = thickness
        
        assert isinstance(_skel, PointSkeleton), "Incompatible skeleton type detected. Only PointSkeletons can be lifted to spheres."
        super().__init__(_skel, ThickeningProcedure.SPHERICAL, ThicknessProfileType.UNIFORM, radius*2, None)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
