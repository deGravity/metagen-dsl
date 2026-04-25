from typing import Self
from . import convex_polytope as cp
from .tile import *
from queue import Queue
from scipy.spatial.transform import Rotation as spRot
from . import math_utils

class GlobalPatternOpSpecs():
    def __init__(self, _resultingCPType:cp.ConvexPolytope, _resultingCPCorners:np.array, _doCopy:bool):
        self.resultingCPType = _resultingCPType
        self.resultingCPCorners = _resultingCPCorners
        self.doCopy = _doCopy

    def __str__(self) -> str:
        return f"Resulting CP Type: {self.resultingCPType}\nResulting CP Corners: {self.resultingCPCorners}"

class GlobalMirrorSpecs(GlobalPatternOpSpecs):
    def __init__(self, _planeO:np.array, _planeN:np.array, _doCopy:bool, _resultingCPType:cp.ConvexPolytope, _resultingCPCorners:np.array):
        self.planeO = _planeO
        self.planeN = _planeN
        super().__init__(_resultingCPType, _resultingCPCorners, _doCopy)

    def __str__(self) -> str:
        return f"Plane n: {self.planeN}\nPlane o: {self.planeO}\n{super().__str__()}"

class GlobalRotateSpecs(GlobalPatternOpSpecs):
    def __init__(self, _rotAxisO:np.array, _rotAxisDir:np.array, _angleDeg:float, _doCopy:bool, _resultingCPType:cp.ConvexPolytope, _resultingCPCorners:np.array):
        self.axisO = _rotAxisO
        self.axisDir = _rotAxisDir
        self.angleDeg = _angleDeg
        super().__init__(_resultingCPType, _resultingCPCorners, _doCopy)

class GlobalTranslateSpecs(GlobalPatternOpSpecs):
    def __init__(self, _translateVec:np.array, _doCopy:bool, _resultingCPType:cp.ConvexPolytope, _resultingCPCorners:np.array):
        self.translateVec = _translateVec
        super().__init__(_resultingCPType, _resultingCPCorners, _doCopy)

class GlobalScaleSpecs(GlobalPatternOpSpecs):
    def __init__(self, _scaleVec:np.array, _doCopy:bool, _resultingCPType:cp.ConvexPolytope, _resultingCPCorners:np.array):
        self.scaleVec = _scaleVec
        super().__init__(_resultingCPType, _resultingCPCorners, _doCopy)


# =======================================
# Pattern Operations
# =======================================
class PatternOp:
    def __init__(self, _opEntities:list[cp.ConvexPolytope.AliasedCPEntityInfo], _doCopy:bool, _operand:Self=None):
        self.entities = _opEntities
        self.doCopy = _doCopy
        opQueue:Queue[Self] = _operand.opQueue if _operand != None else Queue()
        opQueue.put(self)
        self.opQueue = opQueue

    def apply(self, fbvCorners:np.array) -> GlobalPatternOpSpecs:
        pass

    def get_op_call_string(self, operandStr:str=None):
        raise NotImplementedError("This should be overriden by each subclass")
class NoOp(PatternOp):
    def __init__(self):
        super().__init__([], False, None)

    def apply(self, fbvCorners:np.array) -> GlobalPatternOpSpecs:
        return GlobalPatternOpSpecs(None, [])


class Translate(PatternOp):
    """Translation pattern op (use inside Custom).

    Pattern operation specifying a translation that effectively moves the
    fromEntity to the toEntity. Can only be used inside of a Custom
    patterning environment.

    @params:
        fromEntity - CP Entity that serves as the origin of the translation
                     vector. Currently only implemented for a CP Face.
        toEntity - CP Entity that serves as the target of the translation
                   vector. Currently only implemented for a CP Face.
        doCopy - boolean. When True, applies the operation to a copy of the
                 input, such that the original and the transformed copy
                 persist. When False, directly transforms the input.
        patternOp - [OPTIONAL] outermost pattern operation in the
                    sub-composition, if any.
    @returns:
        pat - the composed patterning procedure, which may be used as is
              (within the Custom environment), or as the input for further
              composition.
    @example_usage:
        gridPat = Custom(Translate(cuboid.faces.LEFT, cuboid.faces.RIGHT, True,
                            Translate(cuboid.faces.FRONT, cuboid.faces.BACK, True)))
    """
    def __init__(self, fromEntity:cp.ConvexPolytope.AliasedCPEntityInfo, toEntity:cp.ConvexPolytope.AliasedCPEntityInfo, doCopy:bool, patternOp:Self=None):
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _fromEntity = fromEntity
        _toEntity = toEntity
        _doCopy = doCopy
        _operand = patternOp

        assert _fromEntity.entityType == _toEntity.entityType, "Translation entities need to be of the same type"
        # TODO: assert that both entities are from the same CP
        fbv:cp.ConvexPolytope = _fromEntity.parentCP

        # create a new line between them as the translation vector; uses the centroid of higher-dimensional entities
        extraPt0:cp.ConvexPolytope.RelativeVert = fbv.make_vertex(_fromEntity)
        extraPt1:cp.ConvexPolytope.RelativeVert = fbv.make_vertex(_toEntity)
        entity = fbv.make_segment(extraPt0.entity, extraPt1.entity)
        
        self.translationOrigin:cp.ConvexPolytope.RelativeVert = extraPt0
        self.translationVec:cp.ConvexPolytope.RelativeSegment = entity
        
        super().__init__([_fromEntity, _toEntity], _doCopy, _operand)
        
    def get_op_call_string(self, operandStr:str=None):
        fromEntity = self.entities[0]
        toEntity = self.entities[1]
        if operandStr:
            opstr_lines = operandStr.split("\n")
            indentedOpLines = "\n\t" + "\n\t".join(opstr_lines)
            return f"Translate({fromEntity.getFullAliasName()}, {toEntity.getFullAliasName()}, {self.doCopy}, {indentedOpLines})"
        return f"Translate({fromEntity.getFullAliasName()}, {toEntity.getFullAliasName()}, {self.doCopy})"

    def apply(self, fbvCorners:np.array) -> GlobalTranslateSpecs:
        assert fbvCorners.shape[1] == 3, "The fbvCorner array should be Nx3, with each point on a row"
        assert fbvCorners.shape[0] == self.entities[0].parentCP.num_corners, "The fbvCorner array must have the same number of corners as the CP type"
        (translateVec, translateOrig) = self.translationVec.entity.getGlobalEdgeDir(fbvCorners)

        # TODO: this logic only works if doCopy=True. Need different logic if doCopy=False
        # TODO: The alg for inferring new bv corners also only works for mirroring Cuboid --> Cuboid
        # figure out the new bounding volume specs by manipulating the bv corners
        fbv:cp.ConvexPolytope = self.entities[0].parentCP
        if self.entities[0].entityType == cp.CP_Face:
            centralFace:cp.CP_Face = self.entities[1].getEntity()
            extFBVCorners = np.zeros([fbv.num_corners, 3])
            # for each corner on the mirror plane
            for cid in centralFace.vids:
                # find the edge that is not on the mirror plane
                adjEdges = fbv.get_CPEdgeIDs_adjacent_to_CPCornerID(cid)
                otherCID = None
                for eid in adjEdges:
                    if eid in centralFace.eids:
                        continue
                    edge = fbv.getEntityByID(cp.CP_Edge, eid)
                    otherCID = edge.vids[1] if edge.vids[0] in centralFace.vids else edge.vids[0] # get the endpoint not in the plane
                    break
                assert otherCID != None, "Unable to find otherPt. Shouldn't happen."

                otherPt:np.array = fbvCorners[otherCID, :]

                # this original point is still part of the extended cp boundary, keep in same id spot
                extFBVCorners[otherCID, :] = otherPt

                # offset the central point by translation vector -- this becomes the extreme point on the extended side of the CP.
                # Specifically, it replaces the id of the current vertex on the central plane
                transformedPt = fbvCorners[cid, :] + translateVec
                extFBVCorners[cid, :] = transformedPt
        else:
            raise NotImplementedError("Not implemented")

        return GlobalTranslateSpecs(translateVec, self.doCopy, cp.CPT_Cuboid, extFBVCorners)



class Mirror(PatternOp):
    """Mirror pattern op (use inside Custom).

    Pattern operation specifying a mirror over the provided CP entity, which
    must be a CP Face. Can only be used inside of a Custom patterning
    environment.

    @params:
        entity - CP Face that serves as the mirror plane.
        doCopy - boolean. When True, applies the operation to a copy of the
                 input, such that the original and the transformed copy
                 persist. When False, directly transforms the input.
        patternOp - [OPTIONAL] outermost pattern operation in the
                    sub-composition, if any.
    @returns:
        pat - the composed patterning procedure, which may be used as is
              (within the Custom environment), or as the input for further
              composition.
    @example_usage:
        pat = Custom(Mirror(cuboid.faces.TOP, True,
                       Mirror(cuboid.faces.LEFT, True)))
    """
    def __init__(self, entity:cp.ConvexPolytope.AliasedCPEntityInfo, doCopy:bool, patternOp:Self=None):
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _opEntity = entity
        _doCopy = doCopy
        _operand = patternOp

        assert _opEntity.entityType == cp.CP_Face
        self.mirrorPlane:cp.CP_Face = _opEntity.getEntity()
        super().__init__([_opEntity], _doCopy, _operand)

    def get_op_call_string(self, operandStr:str=None):
        mirPlane = self.entities[0]
        if operandStr:
            opstr_lines = operandStr.split("\n")
            indentedOpLines = "\n\t" + "\n\t".join(opstr_lines)
            return f"Mirror({mirPlane.getFullAliasName()}, {self.doCopy}, {indentedOpLines})"
        return f"Mirror({mirPlane.getFullAliasName()}, {self.doCopy})"

    def apply(self, fbvCorners:np.array) -> GlobalMirrorSpecs:
        assert fbvCorners.shape[1] == 3, "The fbvCorner array should be Nx3, with each point on a row"
        assert fbvCorners.shape[0] == self.entities[0].parentCP.num_corners, "The fbvCorner array must have the same number of corners as the CP type"
        n = self.mirrorPlane.getGlobalFaceNormal(fbvCorners)
        o = self.mirrorPlane.getGlobalFaceOrigin(fbvCorners)

        # TODO: this logic only works if doCopy=True. Need different logic if doCopy=False
        # TODO: The alg for inferring new bv corners also only works for mirroring Cuboid --> Cuboid
        # figure out the new bounding volume specs by manipulating the bv corners
        fbv:cp.ConvexPolytope = self.entities[0].parentCP
        extFBVCorners = np.zeros([fbv.num_corners, 3])
        # for each corner on the mirror plane
        for cid in self.mirrorPlane.vids:
            # find the edge that is not on the mirror plane
            adjEdges = fbv.get_CPEdgeIDs_adjacent_to_CPCornerID(cid)
            otherCID = None
            for eid in adjEdges:
                if eid in self.mirrorPlane.eids:
                    continue
                edge = fbv.getEntityByID(cp.CP_Edge, eid)
                otherCID = edge.vids[1] if edge.vids[0] in self.mirrorPlane.vids else edge.vids[0] # get the endpoint not in the plane
                break
            assert otherCID != None, "Unable to find otherPt. Shouldn't happen."

            otherPt:np.array = fbvCorners[otherCID, :]

            # this original point is still part of the extended cp boundary, keep in same id spot
            extFBVCorners[otherCID, :] = otherPt

            # mirror, 
            proj:float = np.dot(otherPt - o, n) 
            pplane = otherPt - n*proj
            mirroredPt = pplane - n*proj

            # this is a new point on the extended CP. Specifically, it replaces the id of the current vertex on the mirror plane
            extFBVCorners[cid, :] = mirroredPt

        return GlobalMirrorSpecs(o, n, self.doCopy, cp.CPT_Cuboid, extFBVCorners)

class MirrorTriPrismToCuboid(PatternOp):
    def __init__(self, _opEntity:cp.ConvexPolytope.AliasedCPEntityInfo, _doCopy:bool, _operand:Self=None):
        self.mirrorPlane = _opEntity.getEntity()
        super().__init__([_opEntity], _doCopy, _operand)

    def get_op_call_string(self, operandStr:str=None):
        mirPlane = self.entities[0]
        if operandStr:
            opstr_lines = operandStr.split("\n")
            indentedOpLines = "\n\t" + "\n\t".join(opstr_lines)
            return f"MirrorTriPrismToCuboid({mirPlane.getFullAliasName()}, {self.doCopy}, {indentedOpLines})"
        return f"MirrorTriPrismToCuboid({mirPlane.getFullAliasName()}, {self.doCopy})"

    def apply(self, fbvCorners:np.array) -> GlobalMirrorSpecs:
        assert fbvCorners.shape[1] == 3, "The fbvCorner array should be Nx3, with each point on a row"
        assert fbvCorners.shape[0] == self.entities[0].parentCP.num_corners, "The fbvCorner array must have the same number of corners as the CP type"
        
        o = self.mirrorPlane.getGlobalFaceOrigin(fbvCorners)
        n = self.mirrorPlane.getGlobalFaceNormal(fbvCorners)
        
        fbv:cp.ConvexPolytope = self.entities[0].parentCP
        cuboidCorners = np.zeros([8, 3])

        # HACK -- TODO: generalize. we know this is the mapping for our special case prism. This absolutely will not always work.
        numCorners = fbvCorners.shape[0]
        bbmin = np.array([1e9, 1e9, 1e9])
        bbmax = -bbmin
        for cid in range(numCorners):
            c = fbvCorners[cid, :]
            bbmin = np.minimum(bbmin, c)
            bbmax = np.maximum(bbmax, c)

        cuboidCorners[0, :] = fbvCorners[0, :]
        cuboidCorners[1, :] = fbvCorners[1, :]
        cuboidCorners[2, :] = fbvCorners[3, :]
        cuboidCorners[3, :] = fbvCorners[4, :]
        cuboidCorners[5, :] = fbvCorners[2, :]
        cuboidCorners[4, :] = np.array([bbmax[0], bbmin[1], bbmax[2]])
        cuboidCorners[7, :] = fbvCorners[5, :]
        cuboidCorners[6, :] = np.array([bbmax[0], bbmax[1], bbmax[2]])
        # --- end HACK -- to fix

        return GlobalMirrorSpecs(o, n, self.doCopy, cp.CPT_Cuboid, cuboidCorners)

class TetDoubleMirror(PatternOp):
    def __init__(self, _opEntity:cp.ConvexPolytope.AliasedCPEntityInfo, _doCopy:bool, _operand:Self=None):
        super().__init__([_opEntity], _doCopy, _operand)

    def apply(self, fbvCorners:np.array) -> GlobalMirrorSpecs:
        pass


class InPlaneMirror(PatternOp):
    def __init__(self, _opEntities:list[cp.ConvexPolytope.AliasedCPEntityInfo], _doCopy:bool, _operand:Self=None):
        super().__init__(_opEntities, _doCopy, _operand)

    def apply(self, fbvCorners:np.array) -> GlobalMirrorSpecs:
        pass



class Rotate(PatternOp):
    def __init__(self, _opEntities:list[cp.ConvexPolytope.AliasedCPEntityInfo], _angle:float, _doCopy:bool, _operand:Self=None, _isAngleInDegrees:bool=True):
        if _isAngleInDegrees:
            self.angleDeg = _angle
            self.angleRad = _angle * np.pi / 180
        else:
            self.angleDeg = _angle * 180 / np.pi
            self.angleRad = _angle
        self.inputEntities = _opEntities

        entity = None
        self.rotationType = ""
        fbv:cp.ConvexPolytope = _opEntities[0].parentCP
        if len(_opEntities) == 1:
            assert _opEntities[0].entityType == cp.CP_Edge, "Rotation about edge assumed, but provided entity is not an edge."
            entity = _opEntities[0]
            self.rotationType = "checkerboard"
        elif len(_opEntities) == 2:
            assert _opEntities[0].entityType == _opEntities[1].entityType, "Rotation entities need to be of the same type"
            # if they're points, then create a new line between them as the axis
            if _opEntities[0].entityType == cp.CP_Point:
                raise NotImplementedError("Not implemented")
            elif _opEntities[0].entityType == cp.CP_Edge:
                # take the midpoint of each edge then use the line between those
                # TODO: differentiate if inside the volume or on a face. For now we assume there's a shared face
                extraPt0:cp.ConvexPolytope.RelativeVert = fbv.make_vertex(_opEntities[0])
                extraPt1:cp.ConvexPolytope.RelativeVert = fbv.make_vertex(_opEntities[1])
                entity = fbv.make_segment(extraPt0.entity, extraPt1.entity)
                self.rotationType = ""
            elif _opEntities[0].entityType == cp.CP_Face:
                # if faces, take the centroid of each face then use the line between those
                raise NotImplementedError("Not implemented")
            else:
                raise Exception("Unsupported entity specification for Rotate")
        assert entity != None, "Unsupported entity specification for Rotate"
        
        # TODO: clean this up -- currently, self.rotAxis can be two different types based on the case. Should be the same.
        if isinstance(entity, cp.ConvexPolytope.AliasedCPEntityInfo):
            self.rotAxis:cp.CP_Edge = entity.getEntity()
        elif isinstance(entity, cp.ConvexPolytope.RelativeSegment):
            self.rotAxis:cp.ConvexPolytope.RelativeSegment = entity.entity
        else:
            raise Exception("Unsupported case for rotation")
        super().__init__([entity], _doCopy, _operand)

    def apply(self, fbvCorners:np.array) -> GlobalRotateSpecs:    
        pass

    def rotAboutAxis(self, p:np.array, r_axis:np.array, r_origin:np.array, angle:float, inDegrees:bool):
        if inDegrees:
            radAngle = angle * np.pi / 180
        else:
            radAngle = angle
        r_axis /= np.linalg.norm(r_axis)
        pCentered = p - r_origin
        r = spRot.from_rotvec(radAngle * r_axis)
        rotP = r.apply(pCentered) + r_origin
        return rotP

class Rotate90(Rotate):
    def __init__(self, _opEntities:list[cp.ConvexPolytope.AliasedCPEntityInfo], _doCopy:bool, _operand:Self=None):
        super().__init__(_opEntities, 90, _doCopy, _operand)

class Rotate180(Rotate):
    """180° rotation pattern op (use inside Custom).

    Pattern operation specifying a 180-degree rotation about the provided CP
    entity. Can only be used inside of a Custom patterning environment.

    @params:
        entities - List of CP entities, which define the axis about which to
                   rotate. If a single entity is provided, it must be a CP
                   Edge. If multiple entities, they will be used to define a
                   new entity that spans them (e.g., two corners → axis from
                   one to the other; two edges → axis from midpoint to
                   midpoint).
        doCopy - boolean. When True, applies the operation to a copy of the
                 input, such that the original and the transformed copy
                 persist. When False, directly transforms the input.
        patternOp - [OPTIONAL] outermost pattern operation in the
                    sub-composition, if any.
    @returns:
        pat - the composed patterning procedure, which may be used as is
              (within the Custom environment), or as the input for further
              composition.
    @example_usage:
        pat = Custom(Rotate180([cuboid.edges.FRONT_LEFT, cuboid.edges.FRONT_RIGHT], True))
    """
    def __init__(self, entities:list[cp.ConvexPolytope.AliasedCPEntityInfo], doCopy:bool, patternOp:Self=None):
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _opEntities = entities
        _doCopy = doCopy
        _operand = patternOp

        super().__init__(_opEntities, 180, _doCopy, _operand)

    def get_op_call_string(self, operandStr:str=None):
        entitystr = "[" + ", ".join([e.getFullAliasName() for e in self.inputEntities]) + "]"
        if operandStr:
            opstr_lines = operandStr.split("\n")
            indentedOpLines = "\n\t" + "\n\t".join(opstr_lines)
            return f"Rotate180({entitystr}, {self.doCopy}, {indentedOpLines})"
        return f"Rotate180({entitystr}, {self.doCopy})"

    def apply(self, fbvCorners:np.array) -> GlobalRotateSpecs:
        (globalRotAxis, globalRotOrigin) = self.rotAxis.getGlobalEdgeDir(fbvCorners)

        numCorners = fbvCorners.shape[0]
        jointCorners = np.zeros([numCorners*2, 3])
        # rotate all the fbv corners
        for cid in range(numCorners):
            jointCorners[cid, :] = fbvCorners[cid, :]
            jointCorners[numCorners+cid, :] = self.rotAboutAxis(fbvCorners[cid, :], globalRotAxis, globalRotOrigin, self.angleDeg, True)

        # find the new bb of the corner set
        bbMin = np.min(jointCorners, axis=0)
        bbMax = np.max(jointCorners, axis=0)

        def isBBCorner(q:np.array):
            assert q.size == 3, "Query point q must be a 3d point"
            return      (math_utils.fp_equals(q[0], bbMin[0]) or math_utils.fp_equals(q[0], bbMax[0])) \
                    and (math_utils.fp_equals(q[1], bbMin[1]) or math_utils.fp_equals(q[1], bbMax[1])) \
                    and (math_utils.fp_equals(q[2], bbMin[2]) or math_utils.fp_equals(q[2], bbMax[2]))

        # find the number of joint (orig and rotated) corners on the bb,
        numJointOnBB:int = 0
        jointCIDsOnBB:list[int] = []
        for cid in range(numCorners*2):
            if isBBCorner(jointCorners[cid, :]):
                numJointOnBB += 1
                jointCIDsOnBB.append(cid)

        fbv:cp.ConvexPolytope = self.entities[0].parentCP
        outCorners = np.zeros([numCorners, 3])
        if numJointOnBB == 4: # rotated around an original edge, to create a checkerboard pattern
            for cid in jointCIDsOnBB:
                if cid < numCorners:   # if they're original cids, put them in the corresponding outCorners
                    outCorners[cid, :] = jointCorners[cid, :]
            for axisPtId in self.rotAxis.vids:
                fid:int = -1
                for cid_cand in jointCIDsOnBB:
                    if cid_cand >= numCorners: # not an original vertex of the bv
                        continue
                    (sameFace, fid) = fbv.are_CPCorners_on_same_CPFace([axisPtId, cid_cand])
                    if sameFace:
                        break
                assert fid != -1, "No shared face was found."

                preservedCID = cid_cand
                outCorners[axisPtId, :] = jointCorners[numCorners + preservedCID, :]  # the rotated version of the preserved bb corner fills the spot of the axis point in the larger bb
                
                # set the other points
                for cid_cand in fbv.getEntityByID(cp.CP_Face, fid).vids:
                    if cid_cand == preservedCID or cid_cand == axisPtId:
                        continue
                    dir = jointCorners[cid_cand, :] - jointCorners[preservedCID, :]
                    dir *= 2
                    extpos = jointCorners[preservedCID, :] + dir
                    assert isBBCorner(extpos), "New point must be a bb corner"
                    outCorners[cid_cand, :] = extpos

        elif numJointOnBB == 8: # rotated around an edge within a face, so 1 face is glued to itself and the remaining corners form a larger cuboid
            (shareFace, fid) = fbv.are_CPCorners_on_same_CPFace([c for c in jointCIDsOnBB if c < numCorners]) # get the face shared by the original vertices 
            f:cp.CP_Face = fbv.getEntityByID(cp.CP_Face, fid)
            if not shareFace:
                raise Exception("the original vertices should be on a single face of the fbv")
            for cid in jointCIDsOnBB:
                if cid >= numCorners:   # if not original cid, skip
                    continue
                
                outCorners[cid, :] = jointCorners[cid, :]  # they're original cids, put them in the corresponding outCorners
                for eid in fbv.get_CPEdgeIDs_adjacent_to_CPCornerID(cid):
                    otherEptId = fbv.getEntityByID(cp.CP_Edge, eid).getOtherEndpointID(cid)
                    if f.containsCorner(otherEptId):
                        continue
                    dir = jointCorners[otherEptId, :] - jointCorners[cid, :]
                    dir *= 2
                    extpos = jointCorners[cid, :] + dir
                    assert isBBCorner(extpos), "New point must be a bb corner"
                    outCorners[otherEptId, :] = extpos

        else:
            raise Exception("Unsupported case for rotation")

        return GlobalRotateSpecs(globalRotOrigin, globalRotAxis, self.angleDeg, self.doCopy, cp.CPT_Cuboid, outCorners)

class Scale(PatternOp):
    def __init__(self, _opEntities:list[cp.ConvexPolytope.AliasedCPEntityInfo], _scale:np.array, _doCopy:bool, _operand:Self=None):
        pass

    def apply(self, fbvCorners:np.array) -> GlobalScaleSpecs:
        pass