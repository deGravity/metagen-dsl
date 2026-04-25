from dataclasses import dataclass
import numpy as np
from . import convex_polytope as cp
from enum import Enum
from typing import Self, Union
from . import list_utils
from queue import Queue

NO_CC_LIMIT:int = -1

# =======================================
# Components that form the skeleton
# =======================================
class SkeletonComponent:
    def __init__(self, _parentCP:cp.ConvexPolytope):
        self.parentCP = _parentCP

class Vertex(SkeletonComponent):
    def __init__(self, _point:cp.ConvexPolytope.RelativeVert) -> None:
        self.cp_point = _point.entity
        super().__init__(_point.parentCP)

    def getCPRefdPt(self) -> cp.PointReferencedToCP:
        return self.cp_point

class ECTraversalDirection(Enum):
    BACKWARD = -1
    FORWARD = 1

class EdgeChainLowLevel(SkeletonComponent):
    def __init__(self, _orderedEdgeSegments:list[cp.ConvexPolytope.RelativeSegment], _edgeTraversalOrder:list[ECTraversalDirection], _isSmooth:bool=False) -> None:
        super().__init__(_orderedEdgeSegments[0].parentCP) # TODO: should assert that they all come from the same CP

        edgeSegs:list[cp.EdgeReferencedToCP] = [e.entity for e in _orderedEdgeSegments]

        # assert that the provided edges are continuously tranversible
        for esid in range(len(edgeSegs)-1):
            # get "second" point of the current edge
            currEndPt = edgeSegs[esid].points[1]
            if _edgeTraversalOrder[esid] == ECTraversalDirection.BACKWARD:
                currEndPt = edgeSegs[esid].points[0]
            # get "first" point of the current edge
            nextStartPt = edgeSegs[esid+1].points[0]
            if _edgeTraversalOrder[esid+1] == ECTraversalDirection.BACKWARD:
                nextStartPt = edgeSegs[esid+1].points[1]
            assert currEndPt == nextStartPt, "The provided edges are not continuously traversible"

        self.edgeSegments = edgeSegs
        self.edgeTravOrder = _edgeTraversalOrder
        self.isSmooth = _isSmooth
        
        # compute the footprint of this edgechain on the CP boundary
        self.incidentCPPoints = {}
        self.incidentCPEdges = {}
        self.incidentCPFaces = {}

        #TODO: actual computation, once I figure out what information we want to keep/percolate up

    def get_ordered_edges_along_chain(self) -> list[cp.EdgeReferencedToCP]:
        return self.edgeSegments

    def get_ordered_points_along_chain(self) -> list[cp.PointReferencedToCP]:
        ordered_vert_list = []
        for esid in range(len(self.edgeSegments)):
            # add the starting vertex of the current segment (end added by next segment)
            if self.edgeTravOrder[esid] == ECTraversalDirection.FORWARD:
                ordered_vert_list.append(self.edgeSegments[esid].points[0])
            else:
                ordered_vert_list.append(self.edgeSegments[esid].points[1])
        # add the ending vertex of the last segment
        if self.edgeTravOrder[esid] == ECTraversalDirection.FORWARD:
            ordered_vert_list.append(self.edgeSegments[esid].points[1])
        else:
            ordered_vert_list.append(self.edgeSegments[esid].points[0])
        return ordered_vert_list
        
    def is_endpoint(self, qpt:cp.PointReferencedToCP) -> bool:
        pts = self.get_ordered_points_along_chain()
        return pts[0] == qpt or pts[len(pts)] == qpt
    
    def is_vert_in_chain(self, qpt:cp.PointReferencedToCP) -> bool:
        pts = self.get_ordered_points_along_chain()
        for pt in pts:
            if pt == qpt:
                return True
        return False
    
    def get_edges_incident_on_vert(self, qpt:cp.PointReferencedToCP) -> list[cp.EdgeReferencedToCP]:
        incidentEdges:list[int] = {}
        for e in self.edgeSegments:
            for i in [0,1]:
                if e.points[i] == qpt:
                    incidentEdges.append(e)
        return incidentEdges
    
    @staticmethod
    def inferEdgeTraversalDirections(_orderedEdgeSegments:list[cp.EdgeReferencedToCP]) -> list[ECTraversalDirection]:
        # assign the traversal direction for all edges / assert that they're continuously traversable
        _edgeTraversalOrder = [ECTraversalDirection.FORWARD]*len(_orderedEdgeSegments)

        if len(_orderedEdgeSegments) == 1:
            return _edgeTraversalOrder #no additional information, assume forward is fine. TODO: see if this causes any problems in the skeleton if wrong decision made 

        # figure out if the first edge is forward/backward
        (sharePt, currEPID, _) = _orderedEdgeSegments[0].sharesEndpoint(_orderedEdgeSegments[1])
        assert sharePt == True, "The provided edges are not continuously traversible."
        if currEPID == 0: # first edge traversed 1->0
            _edgeTraversalOrder[0] = ECTraversalDirection.BACKWARD
        for esid in range(1, len(_orderedEdgeSegments)):
            (sharePt, prevEPID, currEPID) = _orderedEdgeSegments[esid-1].sharesEndpoint(_orderedEdgeSegments[esid])
            assert sharePt == True, "The provided edges are not continuously traversible."

            if prevEPID == 0: # curr edge shares endpoint with prev edge's 0th point; prev edge must be traversed 1->0 for this to be valid
                assert _edgeTraversalOrder[esid-1] == ECTraversalDirection.BACKWARD, "The provided edges are not continuously traversible."
            else:
                assert _edgeTraversalOrder[esid-1] == ECTraversalDirection.FORWARD, "The provided edges are not continuously traversible."

            if currEPID == 1: # curr edge must be traversed 1->0
                _edgeTraversalOrder[esid] = ECTraversalDirection.BACKWARD
            # otherwise, we're happy with the default direction of FORWARD, leave it.
        return _edgeTraversalOrder

class EdgeChain(EdgeChainLowLevel):
    def __init__(self, _orderedEntities:list[cp.ConvexPolytope.RelativeEntity], _isSmooth:bool):
        # TODO: should assert that all the RelativeEntities are of the same type (vert, segment)
        repEl = _orderedEntities[0]

        edgeTraversalOrder = []
        orderedEdgeSegments = []
        match repEl:
            case cp.ConvexPolytope.RelativeVert():
                _orderedPoints = _orderedEntities
                fbv:cp.ConvexPolytope = _orderedPoints[0].parentCP # TODO: should assert that all points belong to the same CP
                edgeTraversalOrder = [ECTraversalDirection.FORWARD]*(len(_orderedPoints)-1)
                orderedEdgeSegments = [None]*(len(_orderedPoints)-1)
                for pid in range(len(_orderedPoints)-1):
                    p = _orderedPoints[pid]
                    p_next = _orderedPoints[pid+1]
                    eRel = fbv.make_segment(p.entity, p_next.entity)
                    orderedEdgeSegments[pid] = eRel
            case cp.ConvexPolytope.RelativeSegment():
                orderedEdgeSegments = _orderedEntities
                edgeSegs = [e.entity for e in orderedEdgeSegments]
                edgeTraversalOrder = EdgeChain.inferEdgeTraversalDirections(edgeSegs)
            case _:
                raise Exception("Unknown RelativeEntity type")
        assert len(edgeTraversalOrder) > 0 and len(orderedEdgeSegments) == len(edgeTraversalOrder)

        super().__init__(orderedEdgeSegments, edgeTraversalOrder, _isSmooth)


class Curve(EdgeChain):
    """Smooth path through ordered vertices.

    Creates a path along the ordered input vertices. This path will be smoothed
    at a later stage (e.g., to a Bezier curve), depending on the lifting
    procedures that are chosen. All input vertices must be referenced to the
    same CP (e.g., all relative to cuboid entities).

    @params:
        ordered_verts - a list of vertices, in the order you'd like them to be
                        traversed. A closed loop may be created by repeating
                        the zeroth element at the end of the list. No other
                        vertex may be repeated. Only simple paths are permitted.
    @returns:
        curve - the new curve object.
    @example_usage:
        c0 = Curve([v2, v3])
        c0 = Curve([v0, v1, v2, v3, v4, v5, v0])
    """
    def __init__(self, ordered_verts:list[cp.ConvexPolytope.RelativeEntity]):
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _orderedEntities = ordered_verts

        super().__init__(_orderedEntities, True)

class PlanarCurve(Curve):
    def __init__(self, _orderedEntities:list[cp.ConvexPolytope.RelativeEntity]):
        # TODO: assert that all points are on the same plane.
        super().__init__(_orderedEntities, True)

class Polyline(EdgeChain):
    """Piecewise-linear path through ordered vertices.

    Creates a piecewise-linear path along the ordered input vertices. All
    vertices must be referenced to the same CP (e.g., all relative to cuboid
    entities). The resulting path will remain a polyline in any structures
    that include it.

    @params:
        ordered_verts - a list of vertices, in the order you'd like them to be
                        traversed. A closed loop may be created by repeating
                        the zeroth element at the end of the list. No other
                        vertex may be repeated. Only simple paths are permitted.
    @returns:
        polyline - the new polyline object.
    @example_usage:
        p0 = Polyline([v2, v3])
        p0 = Polyline([v0, v1, v2, v3, v4, v5, v0])
    """
    def __init__(self, ordered_verts:list[cp.ConvexPolytope.RelativeEntity]):
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _orderedEntities = ordered_verts

        super().__init__(_orderedEntities, False)

    @classmethod
    def generate_polylines_from_segment_soup(cls:Self, _relSegments:list[cp.ConvexPolytope.RelativeSegment]) -> list[Self]:
        segments:list[cp.EdgeReferencedToCP] = [e.entity for e in _relSegments]

        # create dictionary of vertices to adjacent edges
        vert2IncidentEdges:dict[cp.PointReferencedToCP, list[int]] = {}
        for eid in range(len(segments)):
            e = segments[eid]
            for i in [0,1]:
                if e.points[i] in vert2IncidentEdges:
                    vert2IncidentEdges[e.points[i]].append(eid)
                else:
                    vert2IncidentEdges[e.points[i]] = [eid]
            
        ecs = []
        edgeVisited = [False for i in range(len(segments))]
        while not all(edgeVisited):
            # start with an edge (we choose lexically-first unvisited, could be random)
            rootEid:int = list_utils.find_first_index_of(edgeVisited, lambda e:e==False)
            edgeVisited[rootEid] = True
            ecSegIDs = [rootEid]
            ecSegTravOrders = [ECTraversalDirection.FORWARD] # this edge traversed forward (first index 0 then 1)

            # try extending it from each end point. 1 first for forward walking direction
            for endptid in [1,0]:
                walkingForward = True if endptid == 1 else False
                
                # walk as far as we can from this point (in a simply traversible way; branches will [necessarily] end up in separate edge chains)
                currPt = segments[rootEid].points[endptid]
                while currPt != None:
                    possNextSegIds:list[int] = vert2IncidentEdges[currPt]
                    nextSegIdxInPossList = list_utils.find_first_index_of(possNextSegIds, lambda e : edgeVisited[e] == False) # find an edge of these that's unvisited

                    # if they're all visited, we've gone as far as we can along this direction of the edge chain; break
                    if nextSegIdxInPossList == list_utils.INVALID_LIST_INDEX: 
                        break
                    
                    # otherwise, add this edge
                    nextSegID = possNextSegIds[nextSegIdxInPossList]
                    assert(segments[nextSegID].containsPointReferencedOnCP(currPt))
                    if not walkingForward: # if we're walking backwards, then a matching [1] endpoint means it's forward traversed in the full chain
                        segTravOrder = ECTraversalDirection.FORWARD if currPt == segments[nextSegID].points[1] else ECTraversalDirection.BACKWARD
                    else: # if walking forward, a matching [0] endpoint means it's forward traversed
                        segTravOrder = ECTraversalDirection.FORWARD if currPt == segments[nextSegID].points[0] else ECTraversalDirection.BACKWARD

                    list_pos_from_walk_dir = 0 if endptid == 0 else len(ecSegIDs) # insert in front of ecsegs if walking from endpoint 0 of edge, append to end of list if walking from endpoint 1
                    ecSegIDs.insert(list_pos_from_walk_dir, nextSegID)
                    ecSegTravOrders.insert(list_pos_from_walk_dir, segTravOrder)
                    edgeVisited[nextSegID] = True
                    currPt = segments[nextSegID].getOtherEndpoint(currPt)

            # connect these into a chain 
            ecs.append(cls([_relSegments[esid] for esid in ecSegIDs]))
        return ecs


class ConnectedComponentType(Enum):
    SIMPLE_CLOSED_LOOP = 0
    SIMPLE_OPEN_PATH = 1
    BRANCHED = 2
    POINT = 3
    UNSPECIFIED = 4

class ConnectedComponent(SkeletonComponent):
    def __init__(self, _id:int, _elements:list[SkeletonComponent], _ccType:ConnectedComponentType):
        # TODO: assert that all elements have the same CP
        self.parentCP = _elements[0].parentCP
        self.id = _id
        self.elements = _elements
        self.ccType = _ccType

class PointConnectedComponent(ConnectedComponent):
    def __init__(self, _id:int, _pt:Vertex):
        super().__init__(_id, [_pt], ConnectedComponentType.POINT)

    @classmethod
    def generate_connected_components_from_point_soup(cls:Self, pts:list[cp.ConvexPolytope.RelativeVert]) -> list[Self]:
        # every point is, by definition, its own singleton connected component
        ccs:list[Self] = []
        id = 0
        for p in pts:
            skelv = Vertex(p)
            ccs.append(cls(id, skelv))
            id += 1
        return ccs
    
    def getCPRefdPt(self) -> cp.PointReferencedToCP:
        assert len(self.elements) == 1
        return self.elements[0].getCPRefdPt()

@dataclass
class PointInfo:
    incidentEdgeChainIDs:list[int]
    adjPoints:list[cp.PointReferencedToCP]
    adjEdges:list[cp.EdgeReferencedToCP]
    valenceLocal:int
    isEndpoint:bool

class EdgeConnectedComponent(ConnectedComponent):
    def __init__(self, _id:int, _edgeChains:list[EdgeChain]) -> None:
        self.edgeChains = _edgeChains

        # ==== within the cc, get the adjacency info for each vertex
        vert2adjEdges = {}
        vert2adjVerts = {}
        vert2adjECIDs = {}
        for ecid in range(len(self.edgeChains)):
            edges = self.edgeChains[ecid].edgeSegments
            for edge in edges:
                for ptid in [0,1]:
                    pt = edge.points[ptid]
                    other_pt = edge.points[1-ptid]

                    # ids of adjacent edge chains
                    if pt in vert2adjECIDs:
                        if ecid not in vert2adjECIDs[pt]:
                            vert2adjECIDs[pt].append(ecid)
                    else:
                        vert2adjECIDs[pt] = [ecid]

                    # refs of adjacent edges (ref'd on CP)
                    if pt in vert2adjEdges:
                        vert2adjEdges[pt].append(edge)
                    else:
                        vert2adjEdges[pt] = [edge]

                    # refs of adjacent verts (ref'd on CP)
                    if pt in vert2adjVerts:
                        if other_pt not in vert2adjVerts[pt]:
                            vert2adjVerts[pt].append(other_pt)
                    else:
                        vert2adjVerts[pt] = [other_pt]
        # gather point info into single dictionary
        ptInfo = {}
        num_endpts = 0
        for pt, adjECIDs in vert2adjECIDs.items():
            adjVerts = vert2adjVerts[pt]
            adjEdges = vert2adjEdges[pt]
            valence = len(adjEdges)
            is_endpt = valence == 1
            if is_endpt:
                num_endpts += 1
            ptInfo[pt] = PointInfo(adjECIDs, adjVerts, adjEdges, valence, is_endpt)

        self.num_endpoints = num_endpts
        self.pointInfo = ptInfo

        # ==== determine the connected component type
        interiorPtValences = [info.valenceLocal for pt, info in self.pointInfo.items() if not info.isEndpoint]
        simpleIntPt = [True if ipv == 2 else False for ipv in interiorPtValences]
        if self.num_endpoints == 0 and all(simpleIntPt):
            _ccType = ConnectedComponentType.SIMPLE_CLOSED_LOOP
        elif self.num_endpoints == 2 and all(simpleIntPt):
            _ccType = ConnectedComponentType.SIMPLE_OPEN_PATH
        else:
            _ccType = ConnectedComponentType.BRANCHED

        # TODO: compute this        
        self.incidenceOnCPEntity = None


        super().__init__(_id, _edgeChains, _ccType)

    def get_edges(self) -> list[cp.EdgeReferencedToCP]:
        es = []
        for ec in self.edgeChains:
            es.extend(ec.get_ordered_edges_along_chain())
        return es
    
    def get_vertices(self) -> list[cp.PointReferencedToCP]:
        ps = []
        for ec in self.edgeChains:
            ps.extend(ec.get_ordered_points_along_chain())
        return ps

    def get_vertex_info(self, v:cp.PointReferencedToCP) -> PointInfo:
        return self.pointInfo[v]

    @classmethod
    def generate_connected_components_from_edgechain_soup(cls:Self, ecs:list[EdgeChain]) -> list[Self]:
        vert2IncidentEdgeChainIDs:dict[cp.PointReferencedToCP, list[int]] = {}
        for ecid in range(len(ecs)):
            pts = ecs[ecid].get_ordered_points_along_chain()
            for pt in pts:
                if pt in vert2IncidentEdgeChainIDs:
                    vert2IncidentEdgeChainIDs[pt].append(ecid)
                else:
                    vert2IncidentEdgeChainIDs[pt] = [ecid]

        ecid2incidentECIDs:dict[int, list[int]] = {}
        for ecid in range(len(ecs)):
            incidentEdgeChainsIDs = []
            pts = ecs[ecid].get_ordered_points_along_chain()
            for pt in pts:
                pt_neighbors = vert2IncidentEdgeChainIDs[pt]
                for neighborID in pt_neighbors:
                    if neighborID not in incidentEdgeChainsIDs and neighborID != ecid:
                        incidentEdgeChainsIDs.append(neighborID)
            ecid2incidentECIDs[ecid] = incidentEdgeChainsIDs

        ecVisited = [False] * len(ecs)
        ccs:list[Self] = []
        while not all(ecVisited):
            ecIDsInCC = []
            # start with an edge (we choose lexically-first unvisited, could be random)
            rootECid:int = list_utils.find_first_index_of(ecVisited, lambda e:e==False)
            assert rootECid != list_utils.INVALID_LIST_INDEX

            # use bfs to add all the edge chains visible from this ec
            q = Queue()
            q.put(rootECid)
            while not q.empty():
                ecid = q.get()
                if ecVisited[ecid]:
                    continue # it was put in here multiple times [not processed before finding additional incident cc], but we've already processed a previous entry 
                ecIDsInCC.append(ecid)
                ecVisited[ecid] = True
                neighbors = ecid2incidentECIDs[ecid]
                for n in neighbors:
                    if not ecVisited[n]:
                        q.put(n)

            ccid = len(ccs)
            ccs.append(cls(ccid, [ecs[ecid] for ecid in ecIDsInCC]))
        return ccs

    def has_dangling_vertex(self, interiorOnly:bool=False):
        for p in self.get_vertices():
            pi:PointInfo = self.pointInfo[p]
            if pi.valenceLocal == 1:
                if isinstance(p, cp.PointOnCPInterior):
                    return True
                if not interiorOnly:
                    e = pi.adjEdges[0] # we know there's exactly one bc of localValence
                    if isinstance(p, cp.PointOnCPCorner):
                        return False # symmetric units can still connect in non-overlapping ways to give higher valence
                    # from here on, we know p is not on a corner
                    if isinstance(e, cp.EdgeContainedWithinCPEdge): # any additional copies of this edge will lie on top of the original one
                        return True
                    elif isinstance(e, cp.EdgeContainedWithinCPFace) and not isinstance(p, cp.PointOnCPEdge):  # any additional copies of this face lie on top of the original, and since p is in the face interior, it can't form higher valence connections
                        return True
        return False

# =======================================
#   Skeleton definition
# =======================================

class Skeleton:
    def __init__(self, _connectedComponents:list[ConnectedComponent]): 
        # TODO: assert that all components from same fbv/cp

        assert len(_connectedComponents) > 0, "Invalid skeleton input: Skeleton must contain at least one connected component."
        self.connectedComponents = _connectedComponents
        self.parentCP = _connectedComponents[0].parentCP

        # TODO: compute the incidence of each connected component

        # TODO: compute the incidence of the full skeleton
        # self.incidence:dict[cp.ConvexPolytopeEntity, list[int]] = {}

    def is_single_connected_component(self) -> bool:
        return len(self.connectedComponents) == 1
    
    def has_connected_component_type(self, permissible_types:list[ConnectedComponentType]) -> bool:
        valid = True
        for cc in self.connectedComponents:
            if cc.ccType not in permissible_types:
                valid = False
                break
        return valid

    def is_incident_on_CP_entity(self, cpEnt:cp.ConvexPolytopeEntity) -> tuple[bool, cp.CPIncidenceType]:
        adjSkelCompIDs = self.incidence[cpEnt]
        incidenceType = cp.CPIncidenceType.NONE

        # get maximum incidence type of all the skeletal elements

        return (True, incidenceType)

    def is_some_cc_on_all_faces(self) -> bool:
        # check that at least one connected component touches all bv faces
        for cc in self.connectedComponents:
            touchesAllFaces = True
            for f in self.refBV.faces:
                if not self.is_incident_on_CP_entity(f, cc):
                    touchesAllFaces = False
                    break
            if touchesAllFaces:
                return True
        return False

    def get_valid_transformations_for_entities():
        # given 2 faces, under what transformations can they match up?
        pass



class PointSkeleton(Skeleton):
    def __init__(self, _connectedComponents:list[PointConnectedComponent]):
        super().__init__(_connectedComponents)

    @classmethod
    def generate_skeleton_from_point_soup(cls:Self, points:list[cp.PointReferencedToCP]):
        ccs = PointConnectedComponent.generate_connected_components_from_point_soup(points)
        return cls(ccs)
    
    def get_points(self) -> list[cp.PointReferencedToCP]:
        pts = []
        for cc in self.connectedComponents:
            pts.append(cc.getCPRefdPt())
        return pts

class EdgeSkeleton(Skeleton):
    def __init__(self, _connectedComponents:list[EdgeConnectedComponent]):
        super().__init__(_connectedComponents)

    @classmethod
    def generate_skeleton_from_segment_soup(cls:Self, _relSegments:list[cp.ConvexPolytope.RelativeSegment]) -> Self:
        ecs = Polyline.generate_polylines_from_segment_soup(_relSegments)
        ccs = EdgeConnectedComponent.generate_connected_components_from_edgechain_soup(ecs)
        return cls(ccs)

    @classmethod
    def generate_skeleton_from_edgechain_soup(cls:Self, edgeChains:list[EdgeChain]) -> Self:
        ccs = EdgeConnectedComponent.generate_connected_components_from_edgechain_soup(edgeChains)
        return cls(ccs)

    def get_edge_chains(self) -> list[EdgeChain]:
        ecs = []
        for cc in self.connectedComponents:
            ecs.extend(cc.edgeChains)
        return ecs

    def get_edges(self) -> list[cp.EdgeReferencedToCP]:
        es = []
        for cc in self.connectedComponents:
            es.extend(cc.get_edges())
        return es
    
    def get_vertices(self) -> list[cp.PointReferencedToCP]:
        es = []
        for cc in self.connectedComponents:
            es.extend(cc.get_vertices())
        return es

    ## Functions to validate the skeleton relative to the CP
    ## TODO: this is not fully generalized. Eg, a vertex on an edge can be dangling if the segment is aligned with the edge, or if the segment is fully in the face
    def has_dangling_vertex(self):
        for cc in self.connectedComponents:
            if cc.has_dangling_vertex():
                return True
        return False
    
    def has_coinciding_edges(self) -> bool:
        # checks whether there are topologically coincident edges in the skeleton (st any scaling/vertex offset would still result in distinct edges within one another)
        edges = self.get_edges()
        for eid0 in range(len(edges)):
            e0 = edges[eid0]
            for eid1 in range(eid0+1, len(edges)):
                e1 = edges[eid1]
                if e0.contains(e1) or e1.contains(e0):
                    return True
        return False
    
    def has_intersecting_edges(self) -> bool:
        # checks whether there are guaranteed intersections between edges (skipping any explicitly shared point)
        edges = self.get_edges()
        for eid0 in range(len(edges)):
            e0 = edges[eid0]
            for eid1 in range(eid0+1, len(edges)):
                e1 = edges[eid1]
                (share_endpt, _, _) = e0.sharesEndpoint(e1)
                if share_endpt:
                    continue
                if e0.intersects(e1):
                    return True
        return False
    
    def all_edges_in_same_face(self) -> bool:
        edges = self.get_edges()
        if len(edges) == 0:
            return False # not really well defined
        for e in edges:
            if not isinstance(e, cp.EdgeContainedWithinCPFace):
                return False
        if len(edges) == 1:
                return True #single edge in specific face
            
        possCPIDs:list[int] = edges[0].getSharedCPFaceIDs(edges[1])
        for eid in range(2, len(edges)):
            adjFaces = edges[eid].getIncidentFaceIDs()
            still_valid_FIDs = []
            for fid in adjFaces:
                if fid in possCPIDs:
                    still_valid_FIDs.append(fid)
            possCPIDs = still_valid_FIDs
        if len(possCPIDs) > 0:
            return True
        return False


# types that can be accepted by the create skeleton function
SkeletonInput = Union[cp.ConvexPolytope.RelativeEntity, SkeletonComponent]

def skeleton(entities:list[SkeletonInput]) -> Skeleton:
    """Combine vertices or polylines/curves into a skeleton.

    Combines a set of vertices OR polylines/curves into a larger structure,
    over which additional information can be inferred. For example, within a
    skeleton, multiple open polylines/curves may string together to create a
    closed loop, a branched path, or a set of disconnected components.

    @params:
        entities - a list of entities (vertices or polylines/curves) to be
                   combined. A given skeleton must only have entities with the
                   same dimension — that is, it must consist of all points or
                   all polylines/curves.
    @returns:
        skeleton - the new skeleton object.
    @example_usage:
        skel = skeleton([curve0, polyline1, curve2, polyline3])
        skel = skeleton([v0])
    """
    # assign to the original parameter names (correcting mismatched signatures in code/documentation)
    _elements = entities

    # TODO: assert all are the same type 
    elRep = _elements[0] # get representative element for input type determination. 

    _cpIn:cp.ConvexPolytope = elRep.parentCP

    match elRep:
        case EdgeChain():
            return EdgeSkeleton.generate_skeleton_from_edgechain_soup(_elements)
        case cp.ConvexPolytope.RelativeEntity():
            isCpRefdPoint = [False]*len(_elements)
            for eid in range(len(_elements)):
                if isinstance(_elements[eid], cp.ConvexPolytope.RelativeVert):
                    isCpRefdPoint[eid] = True
                else:
                    assert isinstance(_elements[eid], cp.ConvexPolytope.RelativeSegment), f"Invalid skeleton input: element {eid} is not a valid skeleton entity (or differs in type from previous entities)"
            assert all(isCpRefdPoint) or not any(isCpRefdPoint), "Invalid skeleton input: all entities in given skeleton must be of the same dimension (vertices or Curves/Polylines)" # all must be of same type -- either all points or none points
            isPtSkel = isCpRefdPoint[0]

            if isPtSkel:
                return PointSkeleton.generate_skeleton_from_point_soup(_elements)
            else: 
                return EdgeSkeleton.generate_skeleton_from_segment_soup(_elements)