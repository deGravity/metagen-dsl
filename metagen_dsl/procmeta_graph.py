import json
from pathlib import Path
import networkx as nx
from . import procmeta_nodes as pmn
import numpy as np

class ProcMetaGraph:
    def __init__(self) -> None:
        self.sr = nx.DiGraph() # structure representation
        self.__idGenerator = {} # maps node type to integer id based on how many have been instantiated so far

    # private function for node id generation
    def __generateIDForNewNode(self, newNodeType:type) -> int:
        if newNodeType in self.__idGenerator:
            self.__idGenerator[newNodeType] += 1
        else:
            self.__idGenerator[newNodeType] = 0
        return self.__idGenerator[newNodeType]
    
    def add_vertex(self, pos:np.array) -> pmn.OpNode_Vertex:
        v = pmn.OpNode_Vertex(self.__generateIDForNewNode(pmn.OpNode_Vertex), pos)
        self.sr.add_node(v)
        return v
    
    def add_edge_chain(self, orderedVerts:list[pmn.OpNode_Vertex], isSmooth:bool) -> pmn.OpNode_Vertex:
        ec = pmn.OpNode_EdgeChain(self.__generateIDForNewNode(pmn.OpNode_EdgeChain), orderedVerts, isSmooth)
        self.sr.add_node(ec)
        for v in orderedVerts:
            self.sr.add_edge(v, ec)
        return ec
    
    def add_line_uniformThickness(self, orderedEdgeChains:list[pmn.OpNode_EdgeChain], thickness:float) -> pmn.OpNode_Line:
        tProfile = np.array([[0.0, thickness], [1.0, thickness]])
        return self.add_line_variableThickness(orderedEdgeChains, tProfile)

    def add_line_variableThickness(self, orderedEdgeChains:list[pmn.OpNode_EdgeChain], thicknessProfile:np.array) -> pmn.OpNode_Line:
        tProfShape = thicknessProfile.shape
        assert tProfShape[0] >= 2
        assert tProfShape[1] == 2
        l = pmn.OpNode_Line(self.__generateIDForNewNode(pmn.OpNode_Line), orderedEdgeChains, thicknessProfile)
        self.sr.add_node(l)
        for ec in orderedEdgeChains:
            self.sr.add_edge(ec, l)
        return l
    
    def add_conjugate_surface_uniformThickness(self, orderedEdgeChains:list[pmn.OpNode_EdgeChain], bvType:pmn.ProcMetaBVTypes, bvCorners:np.array, thickness:float) -> pmn.OpNode_Surface:
        # tet given as a cusom bv often with all 4 corners; prism and aabb use min/max of bounding aabb
        # assert bvType == pmn.ProcMetaBVTypes.TET and bvCorners.shape[0] == 4 \
        #         or bvType == pmn.ProcMetaBVTypes.TRIANGULARPRISM and bvCorners.shape[0] == 2 \
        #         or bvType == pmn.ProcMetaBVTypes.AABB and bvCorners.shape[0] == 2 \
        #         or bvType == pmn.ProcMetaBVTypes.CUSTOM and bvCorners.shape[0] >= 4 # minimum number of corners in closed 3d polytope 
        s = pmn.OpNode_Surface(self.__generateIDForNewNode(pmn.OpNode_Surface), orderedEdgeChains, pmn.ProcMetaSurfaceTypes.CONJUGATE, bvType, _bvCorners=bvCorners, _thicknessProfile=np.array([[0.5, 0.5, thickness]]))
        self.sr.add_node(s)
        for ec in orderedEdgeChains:
            self.sr.add_edge(ec, s)
        return s
    
    def add_direct_surface_uniformThickness(self, orderedEdgeChains:list[pmn.OpNode_EdgeChain], bvType:pmn.ProcMetaBVTypes, thickness:float) -> pmn.OpNode_Surface:
        s = pmn.OpNode_Surface(self.__generateIDForNewNode(pmn.OpNode_Surface), orderedEdgeChains, pmn.ProcMetaSurfaceTypes.DIRECT, bvType, _thicknessProfile=np.array([[0.5, 0.5, thickness]]))
        self.sr.add_node(s)
        for ec in orderedEdgeChains:
            self.sr.add_edge(ec, s)
        return s

    def add_mixed_minimal_surface_uniformThickness(self, orderedEdgeChains:list[pmn.OpNode_EdgeChain], bvType:pmn.ProcMetaBVTypes, thickness:float) -> pmn.OpNode_Surface:
        s = pmn.OpNode_Surface(self.__generateIDForNewNode(pmn.OpNode_Surface), orderedEdgeChains, pmn.ProcMetaSurfaceTypes.MIXEDMINIMAL, bvType, _thicknessProfile=np.array([[0.5, 0.5, thickness]]))
        self.sr.add_node(s)
        for ec in orderedEdgeChains:
            self.sr.add_edge(ec, s)
        return s

    def add_mirror(self, inputSkel:pmn.ProcMetaOpNode, planeO:np.array, planeN:np.array, doCopy:bool) -> pmn.OpNode_Mirror:
        m = pmn.OpNode_Mirror(self.__generateIDForNewNode(pmn.OpNode_Mirror), inputSkel, planeO, planeN, doCopy)
        self.sr.add_node(m)
        self.sr.add_edge(inputSkel, m)
        return m
    
    # private function for transform node; broken apart into translate, rotate, scale from outside
    def __add_transform(self, inputSkel:pmn.ProcMetaOpNode, rotOrigin:np.array, rotAxis:np.array, rotAngleDegrees:float, translateVec:np.array, scaleVec:np.array, doCopy:bool):
        t = pmn.OpNode_Transform(self.__generateIDForNewNode(pmn.OpNode_Transform), inputSkel, rotOrigin, rotAxis, rotAngleDegrees, translateVec, scaleVec, doCopy)
        self.sr.add_node(t)
        self.sr.add_edge(inputSkel, t)
        return t

    def add_rotate(self, inputSkel:pmn.ProcMetaOpNode, rotOrigin:np.array, rotAxis:np.array, rotAngleDegrees:float, doCopy:bool) -> pmn.OpNode_Transform:
        translateVec = np.array([0.0, 0.0, 0.0])
        scaleVec = np.array([1.0, 1.0, 1.0])
        return self.__add_transform(inputSkel, rotOrigin, rotAxis, rotAngleDegrees, translateVec, scaleVec, doCopy)

    def add_translate(self, inputSkel:pmn.ProcMetaOpNode, translateVec:np.array, doCopy:bool):
        rotOrigin = np.array([0.0, 0.0, 0.0])
        rotAxis = np.array([0.0, 0.0, 0.0])
        rotAngleDegrees = 0.0
        scaleVec = np.array([1.0, 1.0, 1.0])
        return self.__add_transform(inputSkel, rotOrigin, rotAxis, rotAngleDegrees, translateVec, scaleVec, doCopy)

    def add_scale(self, inputSkel:pmn.ProcMetaOpNode, scaleVec:np.array, doCopy:bool) -> pmn.OpNode_Transform:
        rotOrigin = np.array([0.0, 0.0, 0.0])
        rotAxis = np.array([0.0, 0.0, 0.0])
        rotAngleDegrees = 0.0
        translateVec = np.array([0.0, 0.0, 0.0])
        return self.__add_transform(inputSkel, rotOrigin, rotAxis, rotAngleDegrees, translateVec, scaleVec, doCopy)

    def add_group(self, inputSkels:list[pmn.ProcMetaOpNode]) -> pmn.OpNode_Group:
        g = pmn.OpNode_Group(self.__generateIDForNewNode(pmn.OpNode_Group), inputSkels)
        self.sr.add_node(g)
        for s in inputSkels:
            self.sr.add_edge(s, g)
        return g

    def add_object(self, inputSkel:pmn.ProcMetaOpNode, sdfRes, extrusionMethod) -> pmn.OpNode_Object:
        o = pmn.OpNode_Object(self.__generateIDForNewNode(pmn.OpNode_Object), inputSkel, sdfRes, extrusionMethod)
        self.sr.add_node(o)
        self.sr.add_edge(inputSkel, o)
        return o
    
    def add_object_boolean(self, obj1:pmn.ProcMetaOpNode, obj2:pmn.ProcMetaOpNode, csgOp:pmn.ProcMetaCSGOps) -> pmn.OpNode_BooleanObject:
        b = pmn.OpNode_BooleanObject(self.__generateIDForNewNode(pmn.OpNode_BooleanObject), obj1, obj2, csgOp)
        self.sr.add_node(b)
        self.sr.add_edge(obj1, b)
        self.sr.add_edge(obj2, b)
        return b
    
    def add_boolean_union(self, obj1:pmn.ProcMetaOpNode, obj2:pmn.ProcMetaOpNode) -> pmn.OpNode_BooleanObject:
        return self.add_object_boolean(obj1, obj2, pmn.ProcMetaCSGOps.UNION)

    def add_boolean_intersect(self, obj1:pmn.ProcMetaOpNode, obj2:pmn.ProcMetaOpNode) -> pmn.OpNode_BooleanObject:
        return self.add_object_boolean(obj1, obj2, pmn.ProcMetaCSGOps.INTERSECT)
    
    def add_boolean_difference(self, obj1:pmn.ProcMetaOpNode, obj2:pmn.ProcMetaOpNode) -> pmn.OpNode_BooleanObject:
        return self.add_object_boolean(obj1, obj2, pmn.ProcMetaCSGOps.DIFFERENCE)

    def add_voxel(self, inputObj:pmn.ProcMetaOpNode, youngsMod:float=1.0, poissonRatio:float=0.45, density:float=1.0) -> pmn.OpNode_Voxel:
        vox = pmn.OpNode_Voxel(self.__generateIDForNewNode(pmn.OpNode_Voxel), inputObj, youngsMod, poissonRatio, density)
        self.sr.add_node(vox)
        self.sr.add_edge(inputObj, vox)

    def save_to_json(self, filename:Path) -> None:
        topo_nodes = list(nx.topological_sort(self.sr))
        j = { "operations": [] }
        for node in topo_nodes:
            j["operations"].append(node.get_proc_meta_description())
        
        fh = open(filename, 'w')
        json.dump(j, fh, indent=4)
        fh.close()        
