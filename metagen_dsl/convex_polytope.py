import numpy as np
import math
from enum import Enum
import itertools
from . import math_utils
from typing import Self

# =======================================
# Entities that form the convex polytope
# =======================================
INVALID_ENTITY_INDEX = -1

class ConvexPolytopeEntity:
    def __init__(self, _id:int, _name:str) -> None:
        self.id = _id
        n = _name.strip()
        if len(n.split()) != 1:
            print("Warning: CP Entity name cannot contain spaces. Substituting with underscores")
            n = "_".join(n.split())
        self.name = n

        alt_names = []
        elements = n.split("_")
        if len(elements) > 1:
            for perm in itertools.permutations(elements, len(elements)):
                cand = "_".join(perm)
                if cand != self.name:
                    alt_names.append("_".join(perm))
        self.alt_names = alt_names

class CP_Point(ConvexPolytopeEntity):
    def __init__(self, _id:int, _name:str="") -> None:    
        n = _name if _name != "" else "c"+str(_id)
        super().__init__(_id, n)

class CP_Edge(ConvexPolytopeEntity):
    def __init__(self, _id:int, _vids:list[int], _name:str="") -> None:
        assert len(_vids) == 2, "Invalid CP definition: can only create an edge with 2 vertices"

        n = _name if _name != "" else "e"+str(_id)
        super().__init__(_id, n)
        self.vids = _vids

    def containsCorner(self, cornerID:int) -> bool:
        return cornerID in self.vids
    
    def sharesEndpoint(self, qEdge:Self) -> tuple[bool, int, int]:
        # TODO: should probably assert that self and qEdge are not the same edge (both endpoints shared)
        if self.vids[0] == qEdge.vids[0]:
            return (True, 0, 0)
        elif self.vids[0] == qEdge.vids[1]:
            return (True, 0, 1)
        elif self.vids[1] == qEdge.vids[0]:
            return (True, 1, 0)
        elif self.vids[1] == qEdge.vids[1]:
            return (True, 1, 1)
        return (False, -1, -1)
    
    def getOtherEndpointID(self, endptID:int) -> int:
        assert self.containsCorner(endptID)
        if self.vids[0] == endptID:
            return self.vids[1]
        return self.vids[0]
    
    def getGlobalEdgeDir(self, CPCornerCoords:np.array) -> tuple[np.array, np.array]:
        assert CPCornerCoords.shape[1] == 3 # each 3d point on separate row (Nx3)
        v0 = CPCornerCoords[self.vids[0], :]
        v1 = CPCornerCoords[self.vids[1], :]
        return (v1-v0, v0)
    
class CP_Face(ConvexPolytopeEntity):
    def __init__(self, _id:int, _eids:list[int], _vids:list[int], _name:str="") -> None:
        assert len(_vids) >= 3, "Invalid CP definition: cannot create a face with fewer than 3 vertices"
        assert len(_eids) >= 3, "Invalid CP definition: cannot create a face with fewer than 3 edges"
        assert len(_vids) == len(_eids), "Invalid CP definition: face must have the same number of vertices and edges"

        n = _name if _name != "" else "f"+str(_id)
        super().__init__(_id, n)
        self.vids = _vids
        self.eids = _eids
        self.num_corners = len(self.vids)

    def containsCorner(self, cornerID:int) -> bool:
        return cornerID in self.vids
    
    def containsEdge(self, edgeID:int) -> bool:
        return edgeID in self.eids
    
    def containsWeightedPoint(self, ptWeights:np.array) -> bool:
        numW = ptWeights.shape[0]
        for cid in range(numW):
            if math_utils.fp_equals(ptWeights[cid], 0):
                continue
            # active weight, this corner must be in our active list
            if cid not in self.vids:
                return False
        return True


    def getGlobalFaceNormal(self, CPCornerCoords:np.array) -> np.array:
        assert CPCornerCoords.shape[1] == 3 # each 3d point on separate row (Nx3)
        numCorners = CPCornerCoords.shape[0]
        v0 = CPCornerCoords[self.vids[0], :]
        v1 = CPCornerCoords[self.vids[1], :]
        v2 = CPCornerCoords[self.vids[2], :]
        e0 = v1 - v0
        e1 = v2 - v0
        n = np.cross(e0, e1)
        n = n / np.linalg.norm(n)
        # check that n points to the outside of the CP (dot product against vec to all other vertices is <=0)
        for cid in range(numCorners):
            etest = CPCornerCoords[cid, :] - v0
            if np.dot(etest, n) > 0: # we know n is pointing the wrong way, negate and break [it must be correct now, because convex half spaces]
                n = -n
                break
        return n
    
    def getGlobalFaceOrigin(self, CPCornerCoords:np.array) -> np.array:
        v0 = CPCornerCoords[self.vids[0], :]
        return v0
    
class CP_Interior(ConvexPolytopeEntity):
    def __init__(self, _id:int, _name:str="") -> None:    
        n = _name if _name != "" else "i"+str(_id)
        super().__init__(_id, n)

class CP_Extra_Entity(ConvexPolytopeEntity): # TODO: hack, clean up. Used for creating extra entities against which to define eg patterning rotations
    def __init__(self, _id:int, _subentities:list[ConvexPolytopeEntity], _name:str="") -> None:    
        n = _name if _name != "" else "ex"+str(_id)
        super().__init__(_id, n)

# =======================================
# Entities that are defined RELATIVE TO the convex polytope
# =======================================
class CPIncidenceType(Enum):
    POINT = 0
    LINE = 1
    AREA = 2
    NONE = -1

class IncidentCPEntityInfo:
    def __init__(self, _entityType:ConvexPolytopeEntity, _idx:int, _incidenceType:CPIncidenceType) -> None:
        self.entityType = _entityType
        self.idx = _idx                          # index into the CP list of entityType
        self.incidenceType = _incidenceType

    def __str__(self) -> str:
        return f"IncidentCPEntityInfo: [{self.entityType}] \tIDX: {self.idx} \tIncidence Dim: {self.incidenceType}"

class IncidentCPFaceInfo(IncidentCPEntityInfo):
    def __init__(self, _idx:int, _incidenceType:CPIncidenceType):
        super().__init__(CP_Face, _idx, _incidenceType)

class IncidentCPEdgeInfo(IncidentCPEntityInfo):
    def __init__(self, _idx:int, _incidenceType:CPIncidenceType):
        assert _incidenceType == CPIncidenceType.POINT or _incidenceType == CPIncidenceType.LINE, "POINT or LINE incidence are the only viable options for an Incident CP Edge"
        super().__init__(CP_Edge, _idx, _incidenceType)

class IncidentCPCornerInfo(IncidentCPEntityInfo):
    def __init__(self, _idx:int, _incidenceType:CPIncidenceType):
        assert _incidenceType == CPIncidenceType.POINT, "POINT incidence is the only viable option for Incident CP Corner"
        super().__init__(CP_Point, _idx, _incidenceType)


class EntityReferencedToCP:
    def __init__(self) -> None:
        self.incidentFaces:list[IncidentCPFaceInfo] = []
        self.incidentEdges:list[IncidentCPEdgeInfo] = []
        self.incidentCorners:list[IncidentCPCornerInfo] = []

    def getIncidentFaceIDs(self) -> list[int]:
        return [fInfo.idx for fInfo in self.incidentFaces]
    
    def getIncidentEdgeIDs(self) -> list[int]:
        return [eInfo.idx for eInfo in self.incidentEdges]
    
    def getIncidentCornerIDs(self) -> list[int]:
        return [cInfo.idx for cInfo in self.incidentCorners]

class PointReferencedToCP(EntityReferencedToCP):
    def __init__(self, _weights:np.array) -> None:
        assert math_utils.fp_equals(_weights.sum(), 1), "Weights must sum to 1"
        assert math_utils.fp_equals((_weights<0).sum(), 0), "Weights must be nonnegative"
        super().__init__()
        self.weights = _weights

    def getGlobalCoords(self, CPCornerCoords:np.array) -> np.array:
        assert CPCornerCoords.shape[1] == 3 # each 3d point on separate row (Nx3)
        return np.matmul(self.weights, CPCornerCoords)
    
    def sharesCPEdge(self, qPoint:Self):
        qeids = qPoint.getIncidentEdgeIDs()
        for eid in self.getIncidentEdgeIDs():
            if eid in qeids:
                return True
        return False

    def sharesCPFace(self, qPoint:Self):
        qfids = qPoint.getIncidentFaceIDs()
        for fid in self.getIncidentFaceIDs():
            if fid in qfids:
                return True
        return False

    def __str__(self) -> str:
        return f"[{type(self)}] \tWeights: {self.weights.round(decimals=2)}"

class PointOnCPInterior(PointReferencedToCP):
    def __init__(self, _weights:np.array) -> None:
        super().__init__(_weights)

class PointOnCPBoundary(PointReferencedToCP):
    def __init__(self, _weights:np.array) -> None:
        super().__init__(_weights)

class PointOnCPFace(PointOnCPBoundary):
    def __init__(self, _weights:np.array, _faceIDs:list[int]) -> None:
        super().__init__(_weights)
        self.incidentFaces = [IncidentCPFaceInfo(fid, CPIncidenceType.POINT) for fid in _faceIDs]
    
    def __str__(self) -> str:
        return super().__str__() + f" \tIncident Face IDs: {self.getIncidentFaceIDs()}"

class PointOnCPEdge(PointOnCPFace):
    def __init__(self, _weights:np.array, _edgeIDs:list[int], _adjFaceIDs:list[int]) -> None:
        super().__init__(_weights, _adjFaceIDs)
        self.incidentEdges = [IncidentCPEdgeInfo(eid, CPIncidenceType.POINT) for eid in _edgeIDs]

    def __str__(self) -> str:
        return super().__str__() + f" \tIncident Edge IDs: {self.getIncidentEdgeIDs()}"

class PointOnCPCorner(PointOnCPEdge):
    def __init__(self, _weights:np.array, _cornerIDs:list[int], _adjEdgeIDs:list[int], _adjFaceIDs:list[int]) -> None:
        super().__init__(_weights, _adjEdgeIDs, _adjFaceIDs)
        self.incidentCorners = [IncidentCPCornerInfo(cid, CPIncidenceType.POINT) for cid in _cornerIDs]

    def __str__(self) -> str:
        return super().__str__() + f" \tIncident Corner ID: {self.getIncidentCornerIDs()}"


class EdgeReferencedToCP(EntityReferencedToCP):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        assert len(_endpts) == 2, "EdgeReferencedOnCP requires exactly 2 vertices"
        super().__init__()
        self.points:list[PointReferencedToCP] = _endpts

    def sharesEndpoint(self, qEdge:Self) -> tuple[bool, int, int]:
        # TODO: should probably assert that self and qEdge are not the same edge (both endpoints shared)
        if self.points[0] == qEdge.points[0]:
            return (True, 0, 0)
        elif self.points[0] == qEdge.points[1]:
            return (True, 0, 1)
        elif self.points[1] == qEdge.points[0]:
            return (True, 1, 0)
        elif self.points[1] == qEdge.points[1]:
            return (True, 1, 1)
        return (False, -1, -1)

    def sharesCPEdge(self, qEdge:Self) -> bool:
        qeids = qEdge.getIncidentEdgeIDs()
        for eid in self.getIncidentEdgeIDs():
            if eid in qeids:
                return True
        return False
    
    def getDirection(self) -> np.array:
        # reminder: this is the weighted direction, not a globally embedded direction
        return self.points[1].weights - self.points[0].weights

    def getGlobalEdgeDir(self, CPCornerCoords:np.array) -> tuple[np.array, np.array]:
        assert CPCornerCoords.shape[1] == 3 # each 3d point on separate row (Nx3)
        v0 = np.matmul(self.points[0].weights, CPCornerCoords)
        v1 = np.matmul(self.points[1].weights, CPCornerCoords)
        return (v1-v0, v0)
        
    def contains(self, qEdge:Self) -> bool:
        # reminder: this considers only the weighted positions (relative space), not a globally embedded direction
        pDir = self.getDirection()
        pLen = np.linalg.norm(pDir)
        qDir = qEdge.getDirection()
        qLen = np.linalg.norm(qDir)
        if pLen < qLen:
            return False # self too short to contain q
        pDir = pDir / pLen
        qDir = qDir / qLen
        # check if parallel
        if not (math_utils.array_fp_equals(pDir, qDir) or math_utils.array_fp_equals(pDir, -qDir)):
            return False # direction not the same, not coincident
        # check if along same line
        a = self.points[0].weights
        b = self.points[1].weights
        c = qEdge.points[0].weights
        if abs(np.linalg.norm(b - a)) != (abs(np.linalg.norm(c - a)) + abs(np.linalg.norm(b - c))):
            return False # qEdge point is not on self line
        
        def get_t_value(pt:np.array, rayOrigin:np.array, rayDir:np.array):
            num = pt - rayOrigin
            t_vec = np.divide(num, rayDir, out=np.zeros_like(rayDir), where=rayDir!=0) # this prevents NaN when we have 0/0, but only skips if denom is 0; doesn't check that numerator is 0
            val = t_vec[0]
            valUpdated = False
            # make sure all non-zero entries of t_vec are identical (that's the scalar multiplier)
            for v in t_vec:
                if math_utils.fp_equals(val, v):
                    continue
                elif math_utils.fp_equals(v, 0):
                    continue
                elif math_utils.fp_equals(val, 0) and not valUpdated: # val is 0 but v is first non-zero; v should be our new scalar. Can only update once, since all non-zero scalars must be identical
                    val = v
                    valUpdated = True
                else:
                    raise Exception("pt not on provided ray")
            return t_vec[0]
        
        # get a tvalue for each of the endpoints
        p0 = self.points[0].weights   # we're going to use this as the origin for the direction
        t_p0 = 0
        t_p1 = get_t_value(self.points[1].weights, p0, pDir)
        t_q0 = get_t_value(qEdge.points[0].weights, p0, pDir)
        t_q1 = get_t_value(qEdge.points[1].weights, p0, pDir)
        if t_q1 < t_q0:
            t_q0, t_q1 = t_q1, t_q0 # swap the values so they're oriented the same way

        pStartsFirst = math_utils.fp_equals(t_p0, t_q0) or t_p0 < t_q0
        pEndsLast = math_utils.fp_equals(t_p1, t_q1) or t_p1 > t_q1
        return pStartsFirst and pEndsLast
        

    def isContainedBy(self, qEdge:Self) -> bool:
        return qEdge.contains(self)

    def intersects(self, qEdge:Self, ignoreSharedEndpoint=True) -> bool:
        (sharesEndpt, _, _) = self.sharesEndpoint(qEdge)
        if sharesEndpt:
            if ignoreSharedEndpoint:
                if self.contains(qEdge) or qEdge.contains(self):
                    return True # parallel and overlapping
                return False # only contain the shared endpoint which we ignore --> no intersection
        return math_utils.nd_line_segment_intersection(self.points[0].weights, self.points[1].weights, qEdge.points[0].weights, qEdge.points[1].weights)


    def sharesCPFace(self, qEdge:Self) -> bool:
        qfids = qEdge.getIncidentFaceIDs()
        for fid in self.getIncidentFaceIDs():
            if fid in qfids:
                return True
        return False
    
    def getSharedCPFaceIDs(self, qEdge:Self) -> list[int]:
        sharedFIDs = []
        qfids = qEdge.getIncidentFaceIDs()
        for fid in self.getIncidentFaceIDs():
            if fid in qfids:
                sharedFIDs.append(fid)
        return sharedFIDs
    
    def containsPointReferencedOnCP(self, qpt:PointReferencedToCP) -> bool:
        return self.points[0] == qpt or self.points[1] == qpt
    
    def getOtherEndpoint(self, endpt:PointReferencedToCP) -> PointReferencedToCP:
        assert self.containsPointReferencedOnCP(endpt)
        if self.points[0] == endpt:
            return self.points[1]
        return self.points[0]

    def __str__(self) -> str:
        return f"[{type(self)}]\nEndpoints:\n" + "\n".join([str(e) for e in self.points])

class EdgeContainedWithinCPInterior(EdgeReferencedToCP):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        allInterior = True
        for pt in _endpts:
            if not isinstance(pt, PointOnCPInterior):
                allInterior = False
                break
        assert allInterior, "Endpoints must be on the CP interior"
        super().__init__(_endpts)

class EdgeIncidentOnCPBoundary(EdgeReferencedToCP):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        somePtOnBoundary = False
        for pt in _endpts:
            if isinstance(pt, PointOnCPBoundary):
                somePtOnBoundary = True
                break
        assert somePtOnBoundary, "At least one endpoint must be on the CP boundary"
        super().__init__(_endpts)

        # Update all of the incident face info
        e0_ifids = _endpts[0].getIncidentFaceIDs()
        e1_ifids = _endpts[1].getIncidentFaceIDs()
        for ifid0 in e0_ifids:
            if ifid0 in e1_ifids: # both endpoints on same face
                self.incidentFaces.append(IncidentCPFaceInfo(ifid0, CPIncidenceType.LINE))
            else:
                self.incidentFaces.append(IncidentCPFaceInfo(ifid0, CPIncidenceType.POINT))
        for ifid1 in e1_ifids:
            if ifid1 in e0_ifids: # already added in previous loop
                continue
            else:
                self.incidentFaces.append(IncidentCPFaceInfo(ifid1, CPIncidenceType.POINT))

        # Update all of the incident edge info
        e0_ieids = _endpts[0].getIncidentEdgeIDs()
        e1_ieids = _endpts[1].getIncidentEdgeIDs()
        for ieid0 in e0_ieids:
            if ieid0 in e1_ieids: # both endpoints on same edge
                self.incidentEdges.append(IncidentCPEdgeInfo(ieid0, CPIncidenceType.LINE))
            else:
                self.incidentEdges.append(IncidentCPEdgeInfo(ieid0, CPIncidenceType.POINT))
        for ieid1 in e1_ieids:
            if ieid1 in e0_ieids: # already added in previous loop
                continue
            else:
                self.incidentEdges.append(IncidentCPEdgeInfo(ieid1, CPIncidenceType.POINT))

        # Update all of the incident corner info
        e0_icids = _endpts[0].getIncidentCornerIDs()
        e1_icids = _endpts[1].getIncidentCornerIDs()
        for icid0 in e0_icids:
            self.incidentCorners.append(IncidentCPCornerInfo(icid0, CPIncidenceType.POINT))
        for icid1 in e1_icids:
            if icid1 in e0_icids: # already added in previous loop
                continue
            else:
                self.incidentCorners.append(IncidentCPCornerInfo(icid1, CPIncidenceType.POINT))

    def __str__(self) -> str:
        return super().__str__() + \
            "\nIncident Face IDs:\n" + "\n".join([str(x) for x in self.incidentFaces]) + \
            "\nIncident Edge IDs:\n" + "\n".join([str(x) for x in self.incidentEdges]) + \
            "\nIncident Corner IDs:\n" + "\n".join([str(x) for x in self.incidentCorners])

class EdgeIncidentOnCPFace(EdgeIncidentOnCPBoundary):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        somePtOnFace = False
        for pt in _endpts:
            if isinstance(pt, PointOnCPFace):
                somePtOnFace = True
                break
        assert somePtOnFace, "At least one endpoint must be on a CP Face"
        super().__init__(_endpts)

class EdgeIncidentOnCPEdge(EdgeIncidentOnCPFace):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        somePtOnEdge = False
        for pt in _endpts:
            if isinstance(pt, PointOnCPEdge):
                somePtOnEdge = True
                break
        assert somePtOnEdge, "At least one endpoint must be on a CP Edge"
        super().__init__(_endpts)

class EdgeIncidentOnCPCorner(EdgeIncidentOnCPEdge):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        somePtOnCorner = False
        for pt in _endpts:
            if isinstance(pt, PointOnCPCorner):
                somePtOnCorner = True
                break
        assert somePtOnCorner, "At least one endpoint must be on a CP Corner"
        super().__init__(_endpts)

class EdgeContainedWithinCPBoundary(EdgeIncidentOnCPBoundary):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        somePtOnInterior = False
        for pt in _endpts:
            if isinstance(pt, PointOnCPInterior):
                somePtOnInterior = True
                break
        assert somePtOnInterior == False, "No endpoints allowed on the CP Interior"
        super().__init__(_endpts)

class EdgeContainedWithinCPFace(EdgeContainedWithinCPBoundary, EdgeIncidentOnCPFace):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        e0_faceIDs = _endpts[0].getIncidentFaceIDs()
        e1_faceIDs = _endpts[1].getIncidentFaceIDs()
        foundSharedFace = False
        for fid0 in e0_faceIDs:
            if fid0 in e1_faceIDs:
                foundSharedFace = True
                break
        assert foundSharedFace, "Both points must belong to the same CP Face"
        super().__init__(_endpts)

class EdgeContainedWithinCPEdge(EdgeContainedWithinCPFace, EdgeIncidentOnCPEdge):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        e0_edgeIDs = _endpts[0].getIncidentEdgeIDs()
        e1_edgeIDs = _endpts[1].getIncidentEdgeIDs()
        foundSharedEdge = False
        for eid0 in e0_edgeIDs:
            if eid0 in e1_edgeIDs:
                foundSharedEdge = True
                break
        assert foundSharedEdge, "Both points must belong to the same CP Edge"
        super().__init__(_endpts)

class EdgeContainedWithinCPEdgeIncludingCPCorner(EdgeContainedWithinCPEdge, EdgeIncidentOnCPCorner):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        e0_cornerIDs = _endpts[0].getIncidentCornerIDs()
        e1_cornerIDs = _endpts[1].getIncidentCornerIDs()
        assert len(e0_cornerIDs) > 0 or len(e1_cornerIDs) > 0, "At least one endpoint must be incident on a CP Corner"
        # parent class handles shared edge assertion
        super().__init__(_endpts)

class EdgeIdenticalToCPEdge(EdgeContainedWithinCPEdgeIncludingCPCorner):
    def __init__(self, _endpts=list[PointReferencedToCP]) -> None:
        e0_cornerIDs = _endpts[0].getIncidentCornerIDs()
        e1_cornerIDs = _endpts[1].getIncidentCornerIDs()
        assert len(e0_cornerIDs) > 0 and len(e1_cornerIDs) > 0, "Both points must be incident on CP Corners"
        # parent class handles shared edge assertion
        super().__init__(_endpts)


# =======================================
#   Convex Polytope
# =======================================
class ConvexPolytope:
    def __init__(self, _name:str, _numCorners:int, _cornerList:list[CP_Point], _edgeList:list[CP_Edge], _faceList:list[CP_Face]) -> None:
        assert _numCorners == len(_cornerList), "Mismatch between provided corner list and intended number of corners."
        self.name = _name
        self.num_corners = _numCorners

        # entity ids used for external references 
        self.corners = self.AliasedEntities(self, _cornerList)
        self.edges = self.AliasedEntities(self, _edgeList)
        self.faces = self.AliasedEntities(self, _faceList)
        self.INTERIOR = self.AliasedCPEntityInfo(self, CP_Interior, -1)

        self.extraEntities = self.AliasedEntities(self, [])

        # only used internally
        self.__corner_list = _cornerList
        self.__edge_list = _edgeList
        self.__face_list = _faceList
        # self.__extras_list = [] # not currently used, because the extra entities contain an entity reference. Update if needed.

    # ======================================================
    #   inner classes used for allowing aliased CP entities
    # ======================================================
    class AliasedCPEntityInfo:
        def __init__(self, _parentCP, _entityType:type, _entityIDXinCPEntityList:int):
            # amass the relevant information about the entity we're trying to reference
            assert issubclass(_entityType, ConvexPolytopeEntity)
            self.parentCP:ConvexPolytope = _parentCP
            self.entityType = _entityType
            self.entityIDinCPList:int = _entityIDXinCPEntityList
        
        def getEntity(self):
            parent:ConvexPolytope = self.parentCP
            return parent.getEntity(self)
        
        def getAliasName(self):
            parent:ConvexPolytope = self.parentCP
            return parent.getEntityAlias(self)

        def getFullAliasName(self):
            alias = self.getAliasName()
            if self.entityType == CP_Point:
                return self.parentCP.name + ".corners." + alias
            elif self.entityType == CP_Edge:
                return self.parentCP.name + ".edges." + alias
            elif self.entityType == CP_Face:
                return self.parentCP.name + ".faces." + alias
            else:
                raise Exception("Unsupported entity")
    class AliasedExtraEntityInfo(AliasedCPEntityInfo): # TODO: FIX, hack for getting new entities against which to define eg patterning rotations
        def __init__(self, _parentCP, _entityType:type, _entity:ConvexPolytopeEntity):
            super().__init__(_parentCP, _entityType, INVALID_ENTITY_INDEX)
            self.entity = _entity

    class AliasedEntities:
        def __init__(self, _parentCP, _entityList:list[ConvexPolytopeEntity]) -> None:
            unique_aliased_entities:list[ConvexPolytope.AliasedCPEntityInfo] = []
            # create a field with the given name, so we can use dot notation from the outside
            for eid in range(len(_entityList)):
                info = _parentCP.AliasedCPEntityInfo(_parentCP, type(_entityList[eid]), eid)
                self.__setattr__(_entityList[eid].name, info)
                unique_aliased_entities.append(info)    

                # add an alias for all the permutations of the name elements (pointing to same info as original)
                for altname in _entityList[eid].alt_names:
                    self.__setattr__(altname, info)
            self.unique_aliased_entities = unique_aliased_entities
        
        def addEntity(self, _parentCP, _entity:ConvexPolytopeEntity):
            info = _parentCP.AliasedExtraEntityInfo(_parentCP, type(_entity), _entity)
            self.__setattr__(_entity.name, info)

        # function to return all the 
        def getAll(self):
            return self.unique_aliased_entities

    # ======================================================
    #   inner classes used for allowing referenced CP 
    #   entities with parent knowledge
    # ======================================================
    class RelativeEntity:
        def __init__(self, _parentCP, _entity:EntityReferencedToCP):
            self.parentCP = _parentCP
            self.entity = _entity
    
    class RelativeVert(RelativeEntity):
        def __init__(self, _parentCP, _vert:PointReferencedToCP):
            super().__init__(_parentCP, _vert)

    class RelativeSegment(RelativeEntity):
        def __init__(self, _parentCP, _segment:EdgeReferencedToCP):
            super().__init__(_parentCP, _segment)

    # ======================================================
    #   ConvexPolytope -- public getters
    # ======================================================
    def getEntity(self, _entityInfo:AliasedCPEntityInfo):
        if _entityInfo.entityType == CP_Point:
            return self.__corner_list[_entityInfo.entityIDinCPList]
        elif _entityInfo.entityType == CP_Edge:
            return self.__edge_list[_entityInfo.entityIDinCPList]
        elif _entityInfo.entityType == CP_Face:
            return self.__face_list[_entityInfo.entityIDinCPList]
        assert False, "Should never get here. Invalid entity type provided."

    def getEntityAlias(self, _entityInfo:AliasedCPEntityInfo) -> str:
        return self.getEntity(_entityInfo).name

    def getEntityByID(self, _entityType:ConvexPolytopeEntity, _entityID:int):
        if _entityType == CP_Point:
            return self.__corner_list[_entityID]
        elif _entityType == CP_Edge:
            return self.__edge_list[_entityID]
        elif _entityType == CP_Face:
            return self.__face_list[_entityID]
        assert False, "Should never get here. Invalid entity type provided."

    def get_CPEdgeIDs_adjacent_to_CPCornerID(self, cornerID:int) -> list[int]: # TODO: used by pattern_ops, should find a way around this
        return self.__get_CPEdgeIDs_adjacent_to_CPCornerID(cornerID)

    # ======================================================
    #   ConvexPolytope class methods
    # ======================================================
    def __zeroWeightList(self) -> list[float]:
        return [0.0]*self.num_corners

    def __do_CPEdges_share_corner(self, edgeIDs:list[int]) -> tuple[bool, int]:
        assert len(edgeIDs) == 2
        (shareEndpt, _, _) = self.__edge_list[edgeIDs[0]].sharesEndpoint(self.__edge_list[edgeIDs[1]])
        if not shareEndpt:
            return (False, -1)

        # find the shared corner id
        corners_e0 = self.__get_CPCornerIDs_adjacent_to_CPEdgeID(edgeIDs[0])
        corners_e1 = self.__get_CPCornerIDs_adjacent_to_CPEdgeID(edgeIDs[1])
        for c0 in corners_e0:
            for c1 in corners_e1:
                if c0 == c1:
                    return (True, c0)

    def __get_CPEntities_adjacent_to_CPReferencedEntity(self, qEntity:EntityReferencedToCP) -> list[ConvexPolytopeEntity]:
        return []

    def __get_CPCornerIDs_adjacent_to_CPFaceID(self, faceID:int) -> list[int]:
        return self.__face_list[faceID].vids

    def __get_CPEdgeIDs_adjacent_to_CPFaceID(self, faceID:int) -> list[int]:
        return self.__face_list[faceID].eids

    def __get_CPFaceIDs_adjacent_to_CPFaceID(self, faceID:int) -> list[int]:
        adjFaceIDs = []
        for e in self.__face_list[faceID].eids:
            adjFaces = self.get_CPFaceIDs_adjacent_to_CPEdgeID(e)
            for fid in adjFaces:
                if fid == faceID:
                    continue
                adjFaceIDs.append(fid)
        return list(set(adjFaceIDs))

    def __get_CPCornerIDs_adjacent_to_CPEdgeID(self, edgeID:int) -> list[int]:
        return self.__edge_list[edgeID].vids

    def __get_CPEdgeIDs_adjacent_to_CPEdgeID(self, edgeID:int) -> list[int]:
        queryEdgeCornerIDs = self.__edge_list[edgeID].vids
        adjEdgeIDs = []
        for eid in range(len(self.__edge_list)):
            for cid in queryEdgeCornerIDs: 
                if self.__edge_list[eid].containsCorner[cid]:
                    adjEdgeIDs.append(eid)
                    break
        return adjEdgeIDs

    def __get_CPFaceIDs_adjacent_to_CPEdgeID(self, edgeID:int) -> list[int]:
        adjFaceIDs = []
        for fid in range(len(self.__face_list)):
            if self.__face_list[fid].containsEdge(edgeID):
                adjFaceIDs.append(fid)
        return adjFaceIDs

    def __get_CPEdgeIDs_adjacent_to_CPCornerID(self, cornerID:int) -> list[int]:
        adjEdgeIDs = []
        for eid in range(len(self.__edge_list)):
            if self.__edge_list[eid].containsCorner(cornerID):
                adjEdgeIDs.append(eid)
        return adjEdgeIDs

    def __get_CPFaceIDs_adjacent_to_CPCornerID(self, cornerID:int) -> list[int]:
        adjFaceIDs = []
        for fid in range(len(self.__face_list)):
            if self.__face_list[fid].containsCorner(cornerID):
                adjFaceIDs.append(fid)
        return adjFaceIDs

    def __are_CPCorners_on_same_CPEdge(self, cornerIDs:list[int]) -> tuple[bool, int]:
        assert len(cornerIDs) == 2
        adjEdges0 = self.__get_CPEdgeIDs_adjacent_to_CPCornerID(cornerIDs[0])
        adjEdges1 = self.__get_CPEdgeIDs_adjacent_to_CPCornerID(cornerIDs[1])
        for eid in adjEdges0:
            if eid in adjEdges1:
                return (True, eid)
        return (False, INVALID_ENTITY_INDEX)

    def are_CPCorners_on_same_CPFace(self, cornerIDs:list[int]) -> tuple[bool, int]: # should probably still be private; used for rotation
        for fid in range(len(self.__face_list)):
            face = self.__face_list[fid]
            faceHasAll = True
            for cid in cornerIDs:
                if not face.containsCorner(cid):
                    faceHasAll = False
                    break
            if faceHasAll:
                return (True, fid)
        return (False, INVALID_ENTITY_INDEX)

    def __are_CPEdges_on_same_CPFace(self, edgeIDs:list[int]) -> tuple[bool, int]:
        for fid in range(len(self.__face_list)):
            face = self.__face_list[fid]
            faceHasAll = True
            for eid in edgeIDs:
                if not face.containsEdge(eid):
                    faceHasAll = False
                    break
            if faceHasAll:
                return (True, fid)
        return (False, INVALID_ENTITY_INDEX)        

    def __create_point_from_corner_weights(self, _weights:list[float]) -> PointReferencedToCP:
        assert len(_weights) == self.num_corners, f"Weight list must be of length {self.num_corners} (num corners in reference polytope)"
        weights = np.asarray(_weights)
        assert math_utils.fp_equals(weights.sum(), 1), "Weights must sum to 1"
        assert math_utils.fp_equals((weights<0).sum(), 0), "Weights must be nonnegative"

        activeVertIDs = weights.nonzero()[0].tolist()
        if len(activeVertIDs) == 1:
            cid = activeVertIDs[0]
            return PointOnCPCorner(weights, [cid], self.__get_CPEdgeIDs_adjacent_to_CPCornerID(cid), self.__get_CPFaceIDs_adjacent_to_CPCornerID(cid))
        if len(activeVertIDs) == 2:
            (onEdge, eid) = self.__are_CPCorners_on_same_CPEdge(activeVertIDs)
            if onEdge:
                return PointOnCPEdge(weights, [eid], self.__get_CPFaceIDs_adjacent_to_CPEdgeID(eid))
        (onFace, fid) = self.are_CPCorners_on_same_CPFace(activeVertIDs)
        if onFace:
            return PointOnCPFace(weights, [fid])
        # not on boundary entities
        return PointOnCPInterior(weights)
    
    def __create_edge(self, _endpts:list[PointReferencedToCP]) -> EdgeReferencedToCP:
        assert len(_endpts) == 2, "Exactly 2 endpoints must be provided for EdgeReferencedToCP"
        # fully interior
        if isinstance(_endpts[0], PointOnCPInterior) and isinstance(_endpts[1], PointOnCPInterior):
            return EdgeContainedWithinCPInterior(_endpts)
        # at least one endpoint on interior, other on boundary
        # TODO: maybe rename all these classes to specify that it's interior to boundary, and the others are boundary to boundary through interior
        if isinstance(_endpts[0], PointOnCPInterior) or isinstance(_endpts[1], PointOnCPInterior):
            boundaryPt = _endpts[1] if isinstance(_endpts[0], PointOnCPInterior) else _endpts[0]
            if isinstance(boundaryPt, PointOnCPCorner):
                return EdgeIncidentOnCPCorner(_endpts)
            elif isinstance(boundaryPt, PointOnCPEdge):
                return EdgeIncidentOnCPEdge(_endpts)
            elif isinstance(boundaryPt, PointOnCPFace):
                return EdgeIncidentOnCPFace(_endpts)
            else:
                assert False, "This should never happen"
        # fully contained within boundary
        if _endpts[0].sharesCPEdge(_endpts[1]):
            if isinstance(_endpts[0], PointOnCPCorner) and isinstance(_endpts[1], PointOnCPCorner):
                return EdgeIdenticalToCPEdge(_endpts)
            elif isinstance(_endpts[0], PointOnCPCorner) or isinstance(_endpts[1], PointOnCPCorner):
                return EdgeContainedWithinCPEdgeIncludingCPCorner(_endpts)
            return EdgeContainedWithinCPEdge(_endpts) # edge but neither corner
        elif _endpts[0].sharesCPFace(_endpts[1]):
            return EdgeContainedWithinCPFace(_endpts)
        # both endpoints along CP boundary, but edge goes through CP interior
        # maybe: figure out the stricter point and classify based on that?
        assert isinstance(_endpts[0], PointOnCPBoundary) and isinstance(_endpts[1], PointOnCPBoundary)
        stricterEndpt = _endpts[0] if issubclass(type(_endpts[0]), type(_endpts[1])) else _endpts[1]
        if isinstance(stricterEndpt, PointOnCPCorner):
            return EdgeIncidentOnCPCorner(_endpts)
        elif isinstance(stricterEndpt, PointOnCPEdge):
            return EdgeIncidentOnCPEdge(_endpts)
        elif isinstance(stricterEndpt, PointOnCPFace):
            return EdgeIncidentOnCPFace(_endpts)
        assert False, "This should never happen"


    def __point_on_corner(self, cornerID:int) -> PointReferencedToCP:
        w = self.__zeroWeightList()
        w[cornerID] = 1.0
        return self.__create_point_from_corner_weights(w)

    def __get_parametrized_corner_weights(self, cid:int, full_cp_weights:np.array) -> np.array:
        return np.array([])

    def get_weights_of_edge_point(self, edgeID:int, t:float) -> np.array:
        tmp = self.__point_on_edge(edgeID, t)
        return tmp.weights

    def __point_on_edge(self, edgeID:int, t:float=0.5) -> PointReferencedToCP:
        assert 0 <= t <= 1, "t must be in range [0,1]"
        activeCorners = self.__get_CPCornerIDs_adjacent_to_CPEdgeID(edgeID)
        assert len(activeCorners) == 2, "Edge must have 2 adjacent corners"
        w = self.__zeroWeightList()
        w[activeCorners[0]] = t
        w[activeCorners[1]] = 1 - t
        return self.__create_point_from_corner_weights(w)
    
    def __get_parametrized_edge_weights(self, eid:int, full_cp_weights:np.array) -> np.array:
        v0id = self.__edge_list[eid].vids[0]
        v1id = self.__edge_list[eid].vids[1]
        assert math_utils.fp_equals(full_cp_weights[v0id] + full_cp_weights[v1id], 1.0), "Point given by edge weights are not on the provided edge"
        return np.array([full_cp_weights[v0id]])

    def __point_on_face(self, faceID:int, u:float, v:float) -> PointReferencedToCP:
        activeCorners = self.__get_CPCornerIDs_adjacent_to_CPFaceID(faceID)
        w = self.__zeroWeightList()
        if len(activeCorners) == 3:
            assert u + v <= 1, "u + v must be <= 1"        
            # barycentric coordinates 
            w[activeCorners[0]] = u
            w[activeCorners[1]] = v
            w[activeCorners[2]] = 1.0 - u - v
        elif len(activeCorners) == 4:
            # bilinear interpolation
            # figure out paired edges 
            activeEdges = self.__get_CPEdgeIDs_adjacent_to_CPFaceID(faceID)
            pair0_eids = [activeEdges[0], -1]

            for i in range(1, len(activeEdges)):
                cand_eid = activeEdges[i]
                (cornerIsShared, _) = self.__do_CPEdges_share_corner([pair0_eids[0], cand_eid])
                if not cornerIsShared: # this is the opposite edge
                    pair0_eids[1] = cand_eid

            # figure out the relative pairing of the interpolation corners
            e0_corners = self.__get_CPCornerIDs_adjacent_to_CPEdgeID(pair0_eids[0])
            e0_p0 = e0_corners[0]
            e0_p1 = e0_corners[1]
            e1_corners = self.__get_CPCornerIDs_adjacent_to_CPEdgeID(pair0_eids[1])
            if self.__are_CPCorners_on_same_CPEdge([e0_p0, e1_corners[0]]): # orthogonal edge connects these two points; they should both be the first interpoland of their respective edge, to ensure consistency
                e1_p0 = e1_corners[0]
                e1_p1 = e1_corners[1]
            else:
                assert self.__are_CPCorners_on_same_CPEdge(e0_p0, e1_corners[1])
                e1_p0 = e1_corners[1]
                e1_p1 = e1_corners[0]

            # assign the weights
            w[e0_p0] = u * v
            w[e0_p1] = (1.0 - u) * v
            w[e1_p0] = u * (1.0 - v)
            w[e1_p1] = (1.0 - u) * (1.0 - v)
        else:
            print("Face interpolation only implemented for triangles and quads.")
            return

        return self.__create_point_from_corner_weights(w)
    
    def __get_parametrized_face_weights(self, fid:int, full_cp_weights:np.array) -> np.array:
        # TODO: implement this correctly, rather than just returning the centroid
        numFaceVerts = len(self.__face_list[fid].vids)
        if numFaceVerts == 3:
            return np.array([1.0/3.0, 1.0/3.0])
        elif numFaceVerts == 4:
            return np.array([0.5, 0.5])
        else:
            raise Exception(f"Unsupported face polygon with {numFaceVerts} vertices")

    def __point_in_interior(self, corner_weights:list[float]) -> PointReferencedToCP:
        return self.__create_point_from_corner_weights(corner_weights)
    
    def __get_parametrized_interior_weights(self, full_cp_weights:np.array) -> np.array:
        return full_cp_weights

    # ===================================================================================
    #  Externally available methods for defining referenced points/edges
    #  Return as RelativeVert/RelativeSegment so they have a reference to the parent CP
    # ===================================================================================
    def make_vertex(self, entityInfo:AliasedCPEntityInfo, weights:list[float]=[]) -> RelativeVert:
        refdPt:PointReferencedToCP = None
        if entityInfo.entityType == CP_Point:
            if len(weights) > 0:
                print("Warning: creating a point on a CP corner does not require weights. The provided weights have been ignored.")
            refdPt = self.__point_on_corner(entityInfo.entityIDinCPList)
        elif entityInfo.entityType == CP_Edge:
            if len(weights) == 0:
                weights = [0.5] # default is the midpoint
            assert len(weights) == 1, "Creating a point on a CP edge requires exactly 1 interpolation weight."
            refdPt = self.__point_on_edge(entityInfo.entityIDinCPList, weights[0])
        elif entityInfo.entityType == CP_Face:
            if len(weights) == 0:
                if self.__face_list[entityInfo.entityIDinCPList].num_corners == 3:
                    weights = [1.0/3.0, 1.0/3.0]
                elif self.__face_list[entityInfo.entityIDinCPList].num_corners == 4:
                    weights = [0.5, 0.5]
                else:
                    print(f"Warning: Cannot assign default weight value for face with {self.__face_list[entityInfo.entityIDinCPList].num_corners} corners.")
            assert len(weights) == 2, "Creating a point on a CP edge requires exactly 2 interpolation weights."
            refdPt = self.__point_on_face(entityInfo.entityIDinCPList, weights[0], weights[1])
        elif entityInfo.entityType == CP_Interior:
            if len(weights) == 0:
                weights = [1.0 / float(self.num_corners)] * self.num_corners # default is the centroid
            assert len(weights) == self.num_corners, "Creating a point on the CP interior requires 1 weight per corner."
            refdPt = self.__point_in_interior(weights)
        else:
            assert False, "Invalid CP Entity provided."
        assert refdPt != None

        return self.RelativeVert(self, refdPt)

    def generate_all_corner_combos_of_given_order(self, numCornersToCombine:list[int], restrictToBoundary:bool=False) -> list[RelativeVert]:
        # get all combinations of n polytope corners for n in numCornersToCombine
        cornerIDs = range(self.num_corners)
        combos = []
        for n in numCornersToCombine:
            assert n <= self.num_corners, "numCornersToCombine must not exceed the number of corners"
            combos.extend(itertools.combinations(cornerIDs, n))
        
        # get CPReferenced points with weights evenly distributed between involved corners
        pts = []
        for combo in combos:
            w = 1.0 / len(combo)
            weights = np.zeros([self.num_corners])
            for cornerID in combo:
                weights[cornerID] = w
            pt:EntityReferencedToCP = self.__create_point_from_corner_weights(weights.tolist())
            if (restrictToBoundary and not isinstance(pt, PointOnCPBoundary)):
                continue
            pts.append(pt)

        return [self.RelativeVert(self, p) for p in pts]
    
    def make_segment(self, p0:PointReferencedToCP, p1:PointReferencedToCP) -> RelativeSegment:
        assert isinstance(p0, PointReferencedToCP) and isinstance(p1, PointReferencedToCP), "make_segment() requires two points"
        e = self.__create_edge([p0, p1])
        return self.RelativeSegment(self, e)

    def get_aliased_spec_from_RelVert(self, refdPt:PointReferencedToCP) -> tuple[str, list[float]]:
        alias:str = ""
        weights:np.array = None
        match refdPt:
            case PointOnCPCorner():
                assert len(refdPt.incidentCorners) == 1, "Must be incident on exactly one CP corner"
                cid:int = refdPt.incidentCorners[0].idx
                corner_alias = self.getEntityByID(CP_Point, cid).name
                alias = f"{self.name}.corners.{corner_alias}"
                weights = self.__get_parametrized_corner_weights(cid, refdPt.weights)
            case PointOnCPEdge():
                assert len(refdPt.incidentEdges) == 1, "Must be incident on exactly one CP edge"
                eid:int = refdPt.incidentEdges[0].idx
                edge = self.getEntityByID(CP_Edge, eid)
                alias = f"{self.name}.edges.{edge.name}"
                weights = self.__get_parametrized_edge_weights(eid, refdPt.weights)
            case PointOnCPFace():
                assert len(refdPt.incidentFaces) == 1, "Must be incident on exactly one CP face"
                fid:int = refdPt.incidentFaces[0].idx
                face_alias:str = self.getEntityByID(CP_Face, fid).name
                alias = f"{self.name}.faces.{face_alias}"
                weights = self.__get_parametrized_face_weights(fid, refdPt.weights) # not fully implemented
            case PointOnCPInterior():
                alias = f"{self.name}.INTERIOR"
                weights = self.__get_parametrized_interior_weights(refdPt.weights)
            case _:
                raise Exception("Unsupported point entity type")
        assert alias != "", "No matching alias found"
        if isinstance(weights, list):
            return (alias, weights)
        return (alias, weights.tolist())
    
    # ===================================================================================
    #  Externally available methods -- other
    # ===================================================================================
    def validateGlobalEmbedding(self, pos:np.array):
        assert pos.shape[0] == self.num_corners, "Invalid CP embedding: corner positions must be provided in an Nx3 array"
        assert pos.shape[1] == 3, "Invalid CP embedding: corner positions must be provided in an Nx3 array"
        
        # check that every vertex is distinct (can't collapse any portion of the CP)
        for vid_outer in range(self.num_corners):
            for vid_inner in range(vid_outer+1, self.num_corners):
                assert not math_utils.array_fp_equals(pos[vid_outer, :], pos[vid_inner, :]), f"Invalid CP embedding: vertices {vid_inner} and {vid_outer} are colocated"

        # check that all edges have non-zero length
        for eid in range(len(self.__edge_list)):
            v0id = self.__edge_list[eid].vids[0]
            v1id = self.__edge_list[eid].vids[1]
            assert not math_utils.array_fp_equals(pos[v0id, :], pos[v1id, :]), f"Invalid CP embedding: edge {eid} (with vertices {v0id} and {v1id}) has (near) zero length"

        # for each face
        for fid in range(len(self.__face_list)):
            f = self.__face_list[fid]
            # create plane from first 3 points
            p_orig = pos[f.vids[0], :]
            p_n = np.cross(pos[f.vids[1], :]-p_orig, pos[f.vids[2], :]-p_orig)
            p_n = p_n / np.linalg.norm(p_n)

            # ensure that all corners are indeed coplanar (if >3 verts, since any 3 points are coplanar)
            if len(f.vids) > 3:
                # check that all remaining points live on this plane
                for fvid in range(3, len(f.vids)):
                    vquery = pos[f.vids[fvid]] - p_orig
                    assert math_utils.fp_equals(np.dot(vquery, p_n), 0), f"Invalid CP embedding: the vertices of face {fid} (vertices {[i for i in f.vids]}) are not coplanar"

            # ensure convexity
            # get the the normal of this plane; the dot products with each other point must all be negative (normal pointing outward) or all positive (normal pointing inward)
            dotprods = []
            for cid in range(self.num_corners):
                if cid in f.vids:
                    continue        # we already know this is coplanar, no need to check
                etest = pos[cid, :] - p_orig
                dotprod = np.dot(etest, p_n)
                assert not math_utils.fp_equals(dotprod, 0), f"Invalid CP embedding: vertex {cid} is located on face {fid}, but shouldn't be"
                dotprods.append(dotprod > 0)
            assert dotprods.count(True) == len(dotprods) or dotprods.count(False) == len(dotprods), f"Invalid CP embedding: CP is not convex"
        return True



# ==============================================
#  External calls
# ============================================== 
def vertex(cpEntity:ConvexPolytope.AliasedCPEntityInfo, t:list[float]=[]) -> ConvexPolytope.RelativeVert:
    """Create a new vertex relative to a convex polytope (CP) entity.

    The vertex is defined relative to its containing convex polytope.
    It will only have an embedding in R3 once the CP has been embedded.

    @params:
        cpEntity - an entity of a convex polytope (CP), referenced by the entity names.
        t - [OPTIONAL] list of floats in range [0,1], used to interpolate to a specific
            position on the cpEntity.
              If cpEntity is a corner, t is ignored.
              If cpEntity is an edge, t must contain exactly 1 value. t is used for
                linear interpolation between the endpoints of cpEntity.
              If cpEntity is a face, t must contain exactly 2 values. If cpEntity is a
                triangular face, t is used to interpolate via barycentric coordinates.
                If cpEntity is a quad face, bilinear interpolation is used.
              If the optional interpolant t is omitted for a non-corner entity, the
              returned point will be at the midpoint (for edge) or the centroid (for
              face) of the entity. Semantically, we encourage that t be excluded
              (1) if the structure would be invalid given a different non-midpoint t,
              or (2) if the structure would remain unchanged in the presence a
              different t (e.g., in the case of a conjugate TPMS, where only the entity
              selection matters).
    @returns:
        vertex - the new vertex object.
    @example_usage:
        v0 = vertex(cuboid.edges.BACK_RIGHT, [0.5])
        v1 = vertex(cuboid.edges.TOP_LEFT)
    """
    # assign to the original parameter names (correcting mismatched signatures in code/documentation)
    entityInfo = cpEntity
    weights = t

    cp:ConvexPolytope = entityInfo.parentCP
    return cp.make_vertex(entityInfo, weights)

def edge(p0:ConvexPolytope.RelativeVert, p1:ConvexPolytope.RelativeVert) -> ConvexPolytope.RelativeSegment:
    cp:ConvexPolytope = p0.parentCP
    assert cp == p1.parentCP
    return cp.make_segment(p0.entity, p1.entity)

# ==============================================
#  Predefined Common CPs 
# ==============================================
class CPT_Tet(ConvexPolytope):
    def __init__(self, _name:str) -> None:
        corners = [CP_Point(0, "BOTTOM_RIGHT"), CP_Point(1, "BOTTOM_LEFT"), CP_Point(2, "TOP_BACK"), CP_Point(3, "BOTTOM_BACK")]
        edges = [CP_Edge(0, [0,1], "BOTTOM_FRONT"),
                 CP_Edge(1, [1,2], "TOP_LEFT"),
                 CP_Edge(2, [2,3], "BACK"),
                 CP_Edge(3, [0,3], "BOTTOM_RIGHT"),
                 CP_Edge(4, [0,2], "TOP_RIGHT"),
                 CP_Edge(5, [1,3], "BOTTOM_LEFT")]
        faces = [CP_Face(0, [0,5,3], [0,1,3], "BOTTOM"),
                 CP_Face(1, [0,1,4], [0,1,2], "TOP"),
                 CP_Face(2, [4,2,3], [0,2,3], "RIGHT"),
                 CP_Face(3, [1,2,5], [1,2,3], "LEFT")]
        super().__init__(_name, 4, corners, edges, faces)

    def embed_using_aabb_minmax(self, aabb_min_pt:list[float], aabb_max_pt:list[float]):
        if not len(aabb_min_pt) == 3 and len(aabb_max_pt) == 3:
            raise Exception("Min and max points must be in R3")
        tile_corners = [[aabb_min_pt[0], aabb_min_pt[1], aabb_min_pt[2]],
                        [aabb_max_pt[0], aabb_min_pt[1], aabb_min_pt[2]],
                        [aabb_max_pt[0], aabb_max_pt[1], aabb_max_pt[2]],
                        [aabb_max_pt[0], aabb_min_pt[1], aabb_max_pt[2]]]
        return tile_corners

    def embed(self, bounding_box_side_length:float):
        """Embed the tet CP in R^3.

        Constructs the information required to embed the tet CP in R^3.

        @params:
            bounding_box_side_length - length of axis-aligned bounding box
                                       containing the tet. Float in range
                                       [0, 1]. Must be 1/2^k for some
                                       integer k.
        @returns:
            embedding - the embedding information. Specifically, the
                        position in R^3 of all the CP corners.
        @example_usage:
            side_len = 0.5 / num_tiling_unit_repeats_per_dim
            embedding = tet.embed(side_len)
        """
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        bounding_box_side_len = bounding_box_side_length

        assert math_utils.is_power_of_2(bounding_box_side_len), "bounding_box_side_length must be 1/2^k for some k" # required for the mirror compositions to fill a unit cube precisely
        return self.embed_using_aabb_minmax([0,0,0], [bounding_box_side_len, bounding_box_side_len, bounding_box_side_len])

    def infer_embed_call_from_corners(self, tile_corners:np.array) -> str:
        maxpt = tile_corners.max(axis=0) # max aabb point
        minpt = tile_corners.min(axis=0) # min aabb point
        if not (math_utils.array_fp_equals(minpt, np.array([0.0, 0.0, 0.0]))):
            raise Exception("Check that these corners are ok --- haven't implemented non-origin min points")
        if not (math_utils.array_fp_equals(maxpt, np.array([maxpt[0], maxpt[0], maxpt[0]])) and math_utils.is_power_of_2(maxpt[0])):
            raise Exception("Check that these corners are ok --- we only account for a specific type of tet with the max corner at [1/2^k, 1/2^k, 1/2^k]  for some k")
        return f"{self.name}.embed({maxpt[0]})"

class CPT_TriangularPrism(ConvexPolytope):
    def __init__(self, _name:str) -> None:
        corners = [CP_Point(0, "FRONT_BOTTOM_LEFT"), CP_Point(1, "FRONT_TOP"), CP_Point(2, "FRONT_BOTTOM_RIGHT"),
                   CP_Point(3, "BACK_BOTTOM_LEFT"), CP_Point(4, "BACK_TOP"), CP_Point(5, "BACK_BOTTOM_RIGHT")]
        edges = [CP_Edge(0, [0,1], "FRONT_LEFT"),
                 CP_Edge(1, [1,2], "FRONT_RIGHT"),
                 CP_Edge(2, [0,2], "FRONT_BOTTOM"),
                 CP_Edge(3, [3,4], "BACK_LEFT"),
                 CP_Edge(4, [4,5], "BACK_RIGHT"),
                 CP_Edge(5, [3,5], "BACK_BOTTOM"),
                 CP_Edge(6, [0,3], "BOTTOM_LEFT"),
                 CP_Edge(7, [1,4], "TOP"),
                 CP_Edge(8, [2,5], "BOTTOM_RIGHT")]
        faces = [CP_Face(0, [0,1,2], [0,1,2], "FRONT_TRI"),
                 CP_Face(1, [3,4,5], [3,4,5], "BACK_TRI"),
                 CP_Face(2, [0,7,3,6], [0,1,4,3], "LEFT_QUAD"),
                 CP_Face(3, [1,7,4,8], [1,4,5,2], "RIGHT_QUAD"),
                 CP_Face(4, [2,6,5,8], [0,3,5,2], "BOTTOM_QUAD")]
        super().__init__(_name, 6, corners, edges, faces)

    def embed_using_aabb_minmax(self, aabb_min_pt:list[float], aabb_max_pt:list[float]):
        if not len(aabb_min_pt) == 3 and len(aabb_max_pt) == 3:
            raise Exception("Min and max points must be in R3")
        tile_corners = [[aabb_max_pt[0], aabb_min_pt[1], aabb_min_pt[2]],
                        [aabb_min_pt[0], aabb_min_pt[1], aabb_min_pt[2]],
                        [aabb_min_pt[0], aabb_min_pt[1], aabb_max_pt[2]],
                        [aabb_max_pt[0], aabb_max_pt[1], aabb_min_pt[2]],
                        [aabb_min_pt[0], aabb_max_pt[1], aabb_min_pt[2]],
                        [aabb_min_pt[0], aabb_max_pt[1], aabb_max_pt[2]]]
        return tile_corners

    def embed(self, bounding_box_side_length:float):
        """Embed the triangular prism CP in R^3.

        Constructs the information required to embed the triangular prism
        CP in R^3.

        @params:
            bounding_box_side_length - length of axis-aligned bounding box
                                       containing the triangular prism. Float
                                       in range [0, 1]. Must be 1/2^k for
                                       some integer k.
        @returns:
            embedding - the embedding information. Specifically, the
                        position in R^3 of all the CP corners.
        @example_usage:
            side_len = 0.5 / num_tiling_unit_repeats_per_dim
            embedding = triPrism.embed(side_len)
        """
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        bounding_box_side_len = bounding_box_side_length

        assert math_utils.is_power_of_2(bounding_box_side_len), "bounding_box_side_length must be 1/2^k for some k" # required for the mirror compositions to fill a unit cube precisely
        return self.embed_using_aabb_minmax([0,0,0], [bounding_box_side_len, bounding_box_side_len, bounding_box_side_len])

    def infer_embed_call_from_corners(self, tile_corners:np.array) -> str:
        maxpt = tile_corners.max(axis=0) # max aabb point
        minpt = tile_corners.min(axis=0) # min aabb point
        if not (math_utils.array_fp_equals(minpt, np.array([0.0, 0.0, 0.0]))):
            raise Exception("Check that these corners are ok --- haven't implemented non-origin min points")
        if not (math_utils.array_fp_equals(maxpt, np.array([maxpt[0], maxpt[0], maxpt[0]])) and math_utils.is_power_of_2(maxpt[0])):
            raise Exception("Check that these corners are ok --- we only account for a specific type of tet with the max corner at [1/2^k, 1/2^k, 1/2^k]  for some k")
        return f"{self.name}.embed({maxpt[0]})"

class CPT_Cuboid(ConvexPolytope):
    def __init__(self, _name=str) -> None:
        corners = [CP_Point(0, "FRONT_BOTTOM_LEFT"), CP_Point(1, "FRONT_BOTTOM_RIGHT"), CP_Point(2, "FRONT_TOP_LEFT"), CP_Point(3, "FRONT_TOP_RIGHT"),
                   CP_Point(4, "BACK_BOTTOM_LEFT"), CP_Point(5, "BACK_BOTTOM_RIGHT"), CP_Point(6, "BACK_TOP_LEFT"), CP_Point(7, "BACK_TOP_RIGHT")]
        edges = [CP_Edge(0, [0,1], "FRONT_BOTTOM"),
                 CP_Edge(1, [0,2], "FRONT_LEFT"),
                 CP_Edge(2, [2,3], "FRONT_TOP"),
                 CP_Edge(3, [1,3], "FRONT_RIGHT"),
                 CP_Edge(4, [4,5], "BACK_BOTTOM"),
                 CP_Edge(5, [4,6], "BACK_LEFT"),
                 CP_Edge(6, [6,7], "BACK_TOP"),
                 CP_Edge(7, [5,7], "BACK_RIGHT"),
                 CP_Edge(8, [0,4], "BOTTOM_LEFT"),
                 CP_Edge(9, [2,6], "TOP_LEFT"),
                 CP_Edge(10, [3,7], "TOP_RIGHT"),
                 CP_Edge(11, [1,5], "BOTTOM_RIGHT")]
        faces = [CP_Face(0, [0,1,2,3], [0,1,2,3], "FRONT"),
                 CP_Face(1, [4,5,6,7], [4,5,6,7], "BACK"),
                 CP_Face(2, [2,9,6,10], [2,3,6,7], "TOP"),
                 CP_Face(3, [0,8,4,11], [0,1,4,5], "BOTTOM"),
                 CP_Face(4, [1,8,5,9], [0,2,6,4], "LEFT"),
                 CP_Face(5, [3,10,7,11], [1,3,5,7], "RIGHT")]
        super().__init__(_name, 8, corners, edges, faces)

    def embed_using_values_of_componentwise_aliases(self, front_comp:np.array, back_comp:np.array,
                                                   bottom_comp:np.array, top_comp:np.array,
                                                   left_comp:np.array, right_comp:np.array) -> list[list[float]]:
        tile_corners = [(front_comp + bottom_comp   + left_comp ).tolist(), # FRONT_BOTTOM_LEFT
                        (front_comp + bottom_comp   + right_comp).tolist(), # FRONT_BOTTOM_RIGHT
                        (front_comp + top_comp      + left_comp ).tolist(), # FRONT_TOP_LEFT
                        (front_comp + top_comp      + right_comp).tolist(), # FRONT_TOP_RIGHT
                        (back_comp  + bottom_comp   + left_comp ).tolist(), # BACK_BOTTOM_LEFT
                        (back_comp  + bottom_comp   + right_comp).tolist(), # BACK_BOTTOM_RIGHT
                        (back_comp  + top_comp      + left_comp ).tolist(), # BACK_TOP_LEFT
                        (back_comp  + top_comp      + right_comp).tolist()] # BACK_TOP_RIGHT
        return tile_corners

    def embed_via_minmax(self, aabb_min_pt:list[float], aabb_max_pt:list[float], cornerAtMinPt:ConvexPolytope.AliasedCPEntityInfo=None, **kwargs):
        """Embed the cuboid CP in R^3 by min/max corner.

        Constructs the information required to embed the cuboid CP in R^3
        from explicit min/max corner positions (alternative to embed()).

        @params:
            aabb_min_pt - minimum point of the cuboid, in R^3. Given as a
                          list of length 3, where each component must be a
                          float in range [0, 1], with 1/2^k for some integer k.
            aabb_max_pt - maximum point of the cuboid, in R^3. Given as a
                          list of length 3, where each component must be a
                          float in range [0, 1], with 1/2^k for some integer k.
            cornerAtMinPt - CP corner entity (e.g.,
                            cuboid.corners.FRONT_BOTTOM_LEFT) that should
                            be collocated with the cuboid's minimum
                            position in R^3.
        @returns:
            embedding - the embedding information. Specifically, the
                        position in R^3 of all the CP corners.
        @example_usage:
            side_len = 0.5 / num_tiling_unit_repeats_per_dim
            embedding = cuboid.embed_via_minmax([0,0,0], [side_len, side_len, side_len], cuboid.corners.BACK_BOTTOM_RIGHT)
        """
        # check kwargs for cornerAtAABBMin, which is the argument name for a very similar function (cuboid.embed)
        # they're semantically similar and role is the same, so just accept this if passed
        # but ignore this value if the actual named parameter is provided
        if not cornerAtMinPt and "cornerAtAABBMin" in kwargs:
            assert isinstance(kwargs["cornerAtAABBMin"], ConvexPolytope.AliasedCPEntityInfo)
            cornerAtMinPt = kwargs.get("cornerAtAABBMin")
        
        if not len(aabb_min_pt) == 3 and len(aabb_max_pt) == 3:
            raise Exception("Min and max points must be in R3")
        if not cornerAtMinPt:
            cornerAtMinPt = self.corners.FRONT_BOTTOM_LEFT
        cornerAliasAtMin = self.getEntityAlias(cornerAtMinPt)
        
        front_comp  = np.array([aabb_min_pt[0], 0.0, 0.0])
        back_comp   = np.array([aabb_max_pt[0], 0.0, 0.0])
        if "FRONT" not in cornerAliasAtMin:
            front_comp, back_comp = back_comp, front_comp # switch, since back is aligned with the "min" point
        bottom_comp = np.array([0.0, aabb_min_pt[1], 0.0])
        top_comp    = np.array([0.0, aabb_max_pt[1], 0.0])
        if "BOTTOM" not in cornerAliasAtMin:
            bottom_comp, top_comp = top_comp, bottom_comp # switch, since top is aligned with the "min" point
        left_comp   = np.array([0.0, 0.0, aabb_min_pt[2]])
        right_comp  = np.array([0.0, 0.0, aabb_max_pt[2]])
        if "LEFT" not in cornerAliasAtMin:
            left_comp, right_comp = right_comp, left_comp # switch, since right is aligned with the "min" point

        return self.embed_using_values_of_componentwise_aliases(front_comp, back_comp, bottom_comp, top_comp, left_comp, right_comp)

    def embed_using_aliased_corners(self, pos_at_aliased_pt:dict[ConvexPolytope.AliasedCPEntityInfo, list[float]]=None):
        for alias, pos in pos_at_aliased_pt:
            if len(pos) != 3:
                raise Exception("Error - positions must be in R3")
        alias_names = [self.getEntityAlias(cpEntityInfo) for cpEntityInfo in pos_at_aliased_pt.keys()]
        comps = {"FRONT":   {"idx": 0, "val": -1}, 
                    "BACK":    {"idx": 0, "val": -1}, 
                    "LEFT":    {"idx": 2, "val": -1}, 
                    "RIGHT":   {"idx": 2, "val": -1},
                    "TOP":     {"idx": 1, "val": -1}, 
                    "BOTTOM":  {"idx": 1, "val": -1}}
        for comp_name, comp_info in comps:
            # get all the specified aliased entities with this component
            relevant_aliases = [a for a in alias_names if comp_name in a]
            if len(relevant_aliases) == 0:
                raise Exception(f"not enough information provided -- no specification for {comp_name}")
            # ensure that all the entities have a consistent value for this
            val = -1
            for alias in relevant_aliases:
                pos = pos_at_aliased_pt[alias]
                idx_to_check = comp_info["idx"]
                new_val = pos[idx_to_check]
                if val != -1 and new_val != val: # already found, but curr value doesn't match the previous value
                    raise Exception(f"Error - inconsistent embedding. {comp_name} is at {val} and {new_val}")
            comp_info["val"] = val
        return self.embed_using_values_of_componentwise_aliases(comps["FRONT"]["val"], comps["BACK"]["val"], comps["BOTTOM"]["val"], comps["TOP"]["val"], comps["LEFT"]["val"], comps["RIGHT"]["val"])

    def embed(self, width:float, height:float, depth:float, cornerAtAABBMin:ConvexPolytope.AliasedCPEntityInfo=None, **kwargs):
        """Embed the cuboid CP in R^3.

        Constructs the information required to embed the cuboid CP in R^3.

        @params:
            width - length of cuboid side from left to right. Float in range
                    [0, 1]. Must be 1/2^k for some integer k.
            height - length of cuboid side from top to bottom. Float in range
                     [0, 1]. Must be 1/2^k for some integer k.
            depth - length of cuboid side from front to back. Float in range
                    [0, 1]. Must be 1/2^k for some integer k.
            cornerAtAABBMin - CP corner entity (e.g.,
                              cuboid.corners.FRONT_BOTTOM_LEFT) that should
                              be collocated with the cuboid's minimum
                              position in R^3.
        @returns:
            embedding - the embedding information. Specifically, the
                        position in R^3 of all the CP corners.
        @example_usage:
            side_len = 0.5 / num_tiling_unit_repeats_per_dim
            embedding = cuboid.embed(side_len, side_len, side_len, cornerAtAABBMin=cuboid.corners.FRONT_BOTTOM_LEFT)
        """
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        width_LtoR = width
        height_TtoB = height
        depth_FtoB = depth

        # check kwargs for cornerAtMinPt, which was originally listed instead of cornerAtAABBMin
        # they're semantically similar and role is the same, so just accept this if passed, for sake of backward compatibility
        # but ignore this value if the non-deprecated parameter is provided 
        if not cornerAtAABBMin and "cornerAtMinPt" in kwargs:
            assert isinstance(kwargs["cornerAtMinPt"], ConvexPolytope.AliasedCPEntityInfo)
            cornerAtAABBMin = kwargs.get("cornerAtMinPt")
        
        assert math_utils.is_power_of_2(width_LtoR), "Width must be 1/2^k for some k" # required for the mirror compositions to fill a unit cube precisely
        assert math_utils.is_power_of_2(height_TtoB), "Height must be 1/2^k for some k" # required for the mirror compositions to fill a unit cube precisely
        assert math_utils.is_power_of_2(depth_FtoB), "Depth must be 1/2^k for some k" # required for the mirror compositions to fill a unit cube precisely

        return self.embed_via_minmax([0,0,0], [depth_FtoB, height_TtoB, width_LtoR], cornerAtAABBMin)
    
    def infer_embed_call_from_corners(self, tile_corners:np.array) -> str:
        maxpt = tile_corners.max(axis=0) # max aabb point
        minpt = tile_corners.min(axis=0) # min aabb point
        if not (math_utils.is_power_of_2(maxpt[0]) and math_utils.is_power_of_2(maxpt[1]) and math_utils.is_power_of_2(maxpt[2])):
            raise Exception("Check that these corners are ok --- we only account for a specific type of tet with the max corner at [1/2^i, 1/2^j, 1/2^k]  for some integers i,j,k")
        
        # find the cid of the min tile corner
        cid = 0
        while not math_utils.array_fp_equals(tile_corners[cid], minpt):
            cid += 1
        aliasedCornerAtMinPt = self.name + ".corners." + self.getEntityByID(CP_Point, cid).name
        
        if math_utils.array_fp_equals(minpt, np.array([0.0, 0.0, 0.0])):
            depth = maxpt[0]
            height = maxpt[1]
            width = maxpt[2]
            return f"{self.name}.embed({width}, {height}, {depth}, {aliasedCornerAtMinPt})"
        else:
            return f"{self.name}.embed_via_minmax({minpt.tolist()}, {maxpt.tolist()}, {aliasedCornerAtMinPt})"
        