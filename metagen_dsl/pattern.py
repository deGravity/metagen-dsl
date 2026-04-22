from typing import Any
from .pattern_ops import *
from . import convex_polytope as cp
from .list_utils import *
from .math_utils import *

# dictionary for the coordinate frame  assumptions. ProcMeta assumes up is y
# TODO: add an enum instead of strings
dir2idx:dict[str, int] = {"side":0, "up": 1, "back":2}

class TilingPattern:
    def __init__(self, bv_template:cp.ConvexPolytope, patName:str) -> None:
        self.fbv = bv_template

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class Identity(TilingPattern):
    def __init__(self):
        super().__init__(None, "identity")

    def to_unit_cube() -> PatternOp:
        return NoOp()

class CuboidFullMirror(TilingPattern):
    def __init__(self):
        super().__init__(cp.CPT_Cuboid("cuboid"), "full_mirror")


    def to_unit_cube(self, fbvCorners:np.array, operands:PatternOp=None) -> PatternOp:
        assert fbvCorners.shape[1] == 3, "Must be provided as an Nx3 matrix (with each point on a row)"
        assert fbvCorners.shape[0] == self.fbv.num_corners, "The provided fbvCorners must have the same number of vertices as the template CP"
        
        numCorners = fbvCorners.shape[0]
        bbmin = np.array([1e9, 1e9, 1e9])
        bbmax = -bbmin
        for cid in range(numCorners):
            c = fbvCorners[cid, :]
            bbmin = np.minimum(bbmin, c)
            bbmax = np.maximum(bbmax, c)

        assert fp_equals(bbmin[0], 0) and fp_equals(bbmin[1], 0) and fp_equals(bbmin[2], 0), "Min corner of CP bounding box must reside at (0,0,0)"
        # TODO: check that the max bb coords are 1/2^k for some k 

        # if we already have a unit cube, no need to mirror 
        if fp_equals(bbmax[0], 1) and fp_equals(bbmax[1], 1) and fp_equals(bbmax[2], 1):
            return NoOp()

        # figure out which face pair corresponds to given component ([0,1,2]), and which of the 2 aliases in the pair should be used as the mirror plane (for cuboid, stays consistent) 
        # TODO: remove duplication of this function (also in octant mirror)
        def getSharedCoordIndexOfFace(f:cp.ConvexPolytope.AliasedCPEntityInfo) -> int:
            vids = f.parentCP.getEntity(f).vids
            isShared = [True]*3
            v0 = fbvCorners[vids[0]]
            for cid in range(1, len(vids)):
                for dir in range(3):
                    if v0[dir] != fbvCorners[vids[cid]][dir]:
                        isShared[dir] = False
            assert isShared.count(True) == 1, "A planar axis-aligned face can only have one shared direction"
            return find_first_index_of(isShared, lambda d:d==True)
        
        # TODO: remove duplication of this function (also in octant mirror)
        def getGlobalCoordOfFaceInDir(f:cp.ConvexPolytope.AliasedCPEntityInfo, dirIdx:int) -> np.array:
            vids = f.parentCP.getEntity(f).vids
            coordInDir = fbvCorners[vids[0]][dirIdx]
            for cid in range(1, len(vids)):
                assert coordInDir == fbvCorners[vids[cid]][dirIdx], "Error: the input face is not consistent along the given direction" 
            return coordInDir
        
        facePairs = [(self.fbv.faces.TOP, self.fbv.faces.BOTTOM), 
                     (self.fbv.faces.LEFT, self.fbv.faces.RIGHT), 
                     (self.fbv.faces.BACK, self.fbv.faces.FRONT)]
        
        dirIdx2AliasedFace = {}
        for f1, f2 in facePairs:
            dirIdx = getSharedCoordIndexOfFace(f1)
            assert dirIdx == getSharedCoordIndexOfFace(f2)
            assert dirIdx not in dirIdx2AliasedFace, "A different face pair already matched this direction. Investigate."
            dirCoordF1 = getGlobalCoordOfFaceInDir(f1, dirIdx)
            extFace = f1 if dirCoordF1 == bbmax[dirIdx] else f2
            dirIdx2AliasedFace[dirIdx] = extFace

        def isUnitCube(currentBB:dict[str, np.array]) -> bool:
            return  fp_equals(currentBB["min"][0], 0) and fp_equals(currentBB["min"][1], 0) and fp_equals(currentBB["min"][2], 0) and \
                    fp_equals(currentBB["max"][0], 1) and fp_equals(currentBB["max"][1], 1) and fp_equals(currentBB["max"][2], 1)

        currBB = {"min": bbmin,
                  "max": bbmax}
        mirrorOps:PatternOp = operands
        while not isUnitCube(currBB):
            extensionDirIdx = currBB["max"].argmin()     # figure out which direction to mirror in
            mirrorOps = Mirror(dirIdx2AliasedFace[extensionDirIdx], True, mirrorOps)
            currBB["max"][extensionDirIdx] *= 2

        return mirrorOps


class TriPrismFullMirror(TilingPattern):
    def __init__(self):
        prism = cp.CPT_TriangularPrism("triPrism")
        super().__init__(prism, "full_mirror")

    def to_unit_cube(self, fbvCorners:np.array) -> PatternOp:
        # find the non-axisaligned side, mirror over that, then use cuboid
        # figure out which face pair corresponds to given component ([0,1,2]), and which of the 2 aliases in the pair should be used as the mirror plane (for cuboid, stays consistent) 
        # TODO: remove duplication of this function (also in octant/cuboid mirror) - THIS IS A GENERALIZED VERSION THOUGH. 
        def getSharedCoordIndexOfFace(f:cp.ConvexPolytope.AliasedCPEntityInfo) -> int:
            vids = f.parentCP.getEntity(f).vids
            isShared = [True]*3
            v0 = fbvCorners[vids[0]]
            for cid in range(1, len(vids)):
                for dir in range(3):
                    if v0[dir] != fbvCorners[vids[cid]][dir]:
                        isShared[dir] = False
            if isShared.count(True) != 1: #A planar axis-aligned face can only have one shared direction -- this is not axis aligned
                return -1
            return find_first_index_of(isShared, lambda d:d==True)
        
        # TODO: remove duplication of this function (also in octant mirror)
        def getGlobalCoordOfFaceInDir(f:cp.ConvexPolytope.AliasedCPEntityInfo, dirIdx:int) -> np.array:
            vids = f.parentCP.getEntity(f).vids
            coordInDir = fbvCorners[vids[0]][dirIdx]
            for cid in range(1, len(vids)):
                assert coordInDir == fbvCorners[vids[cid]][dirIdx], "Error: the input face is not consistent along the given direction" 
            return coordInDir
        
        quadfaces = [self.fbv.faces.BOTTOM_QUAD, self.fbv.faces.LEFT_QUAD, self.fbv.faces.RIGHT_QUAD]
        isAxisAligned = [False]*len(quadfaces)
        for fid in range(len(quadfaces)):
            dirIdx = getSharedCoordIndexOfFace(quadfaces[fid])
            if dirIdx != -1:
                isAxisAligned[fid] = True
        assert isAxisAligned.count(False) == 1, "Only 1 face of the triangular prism can be non-axis-aligned"
        mirrorPlane = quadfaces[find_first_index_of(isAxisAligned, lambda d:d==False)]

        # TODO: generalize. this only works because we're assuming a 45deg angle
        # mirror across the not-axis-aligned face to make it axis aligned
        opToCuboid = MirrorTriPrismToCuboid(mirrorPlane, True)

        # construct the new cuboid corners
        gspecs:GlobalMirrorSpecs = opToCuboid.apply(fbvCorners)
        return CuboidFullMirror().to_unit_cube(gspecs.resultingCPCorners, opToCuboid)

class TetFullMirror(TilingPattern):
    def __init__(self):
        tet = cp.CPT_Tet("tet")
        super().__init__(tet, "full_mirror")
        self.ops = Mirror(tet.faces.TOP, True, 
                          Mirror(tet.faces.RIGHT, True,
                                 Mirror(tet.faces.TOP, True)
                                ), 
                        )

class Custom(TilingPattern):
    def __init__(self, patternOp:PatternOp):
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        pat = patternOp

        self.ops = pat

    