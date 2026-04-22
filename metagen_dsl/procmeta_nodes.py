from enum import Enum
import numpy as np
from typing import Self

class ProcMetaOpClasses(Enum):
    TOPOLOGY = 0
    SKELETON = 1
    SOLID = 2
    MATERIALPROPS = 3

class ProcMetaBVTypes(Enum):
    AABB = 0
    TRIANGULARPRISM = 1
    TET = 2
    CUSTOM = 3

class ProcMetaSurfaceTypes(Enum):
    DIRECT = 0
    MIXEDMINIMAL = 1
    CONJUGATE = 2

class ProcMetaCSGOps(Enum):
    UNION = 0
    INTERSECT = 1
    DIFFERENCE = 2

class ProcMetaExtrusionMethods(Enum):
    SPHERICAL = 0
    NORMAL = 1

class ProcMetaOpNode:
    '''
    Shared super class for all operation nodes defined in Procedural Metamaterials
    === Attributes ===
        name
        opClass
        validInputOpClasses
    === Methods === 
        isValidInputNode
    '''
    def __init__(self, _name:str, _opClass:ProcMetaOpClasses, _validInputOps:list[Self], _validInputOpClasses:list[ProcMetaOpClasses]) -> None:
        '''
        @param _name (str) -- unique identifier for node, comprising two parts: <operation type code><numeric id>. The type code indicates the type of node (e.g. "v" for vertex) while the id is an integer used to differentiate between multiple instances of the same node type. For example, two vertex nodes may have codes "v0" and "v1".
        @param _opClass (ProcMetaOpClasses) -- class of the node
        @param _validInputOps (list of ProcMetaOpNode) -- node types that can serve as input for the node. Empty if node accepts no input.
        @param _validInputOpClasses (list of ProcMetaOpClasses) -- operation node classes that can serve as input for the node. Empty if no node inputs.
        '''
        self.name = _name
        self.opClass = _opClass
        self.validInputOps = _validInputOps
        self.validInputOpClasses = _validInputOpClasses

    def isValidInputNode(self, qNode:Self) -> bool:
        return qNode.opClass in self.validInputOpClasses and type(qNode) in self.validInputOps

    def get_proc_meta_description(self) -> dict:
        return {"name": self.name}

    def __hash__(self) -> str:
        return self.name.__hash__()

    def __str__(self) -> str:
        return self.name



class OpNode_Vertex(ProcMetaOpNode):
    '''
    Instantiates a vertex at the given position.
    === Attributes ===
        pos - 
    '''
    def __init__(self, _id:int, _pos:np.array) -> None:
        super().__init__("v"+str(_id), 
                        ProcMetaOpClasses.TOPOLOGY, 
                        [],
                        [])
        self.pos = _pos

    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["position"] = self.pos.tolist()
        return info


class OpNode_EdgeChain(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _orderedVertices:list[OpNode_Vertex], _smooth:bool) -> None:
        super().__init__("e"+str(_id),
                         ProcMetaOpClasses.TOPOLOGY,
                         [OpNode_Vertex],
                         [ProcMetaOpClasses.TOPOLOGY])
        self.verts = _orderedVertices
        self.smooth = _smooth
        assert len(_orderedVertices) >= 2, f"Error instantiating EdgeChain ({self.name}): need at least 2 input vertices."

    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["smooth"] = self.smooth
        vids = []
        for v in self.verts:
            vids.append(v.name)
        info["vertices"] = vids
        return info

class OpNode_Line(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _orderedEdgeChains:list[OpNode_EdgeChain], _thicknessProfile:np.array, _periodic:bool=False) -> None:
        super().__init__("l"+str(_id), 
                         ProcMetaOpClasses.SKELETON,
                         [OpNode_EdgeChain],
                         [ProcMetaOpClasses.TOPOLOGY])
        self.edges = _orderedEdgeChains
        self.periodic = _periodic
        self.thickness = _thicknessProfile
        assert len(_orderedEdgeChains) >= 1, "Error instantiating Line ({self.name}): need at least 1 input EdgeChain"

    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        ecids = []
        for e in self.edges:
            ecids.append(e.name)
        info["edges"] = ecids
        info["periodic"] = self.periodic
        info["thickness"] = self.thickness.tolist()
        return info

class OpNode_Surface(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _orderedEdgeChains:list[OpNode_EdgeChain], _surfaceType:ProcMetaSurfaceTypes,
                 _bvType:ProcMetaBVTypes, _bvCorners:np.array=np.array([]), _conjAngle:float=0.0,
                 _thicknessProfile:np.array=np.array([]), _sampleDist:float=None) -> None:
        super().__init__("s"+str(_id), 
                         ProcMetaOpClasses.SKELETON,
                         [OpNode_EdgeChain],
                         [ProcMetaOpClasses.TOPOLOGY])
        self.edgeLoop = _orderedEdgeChains
        self.surfaceType = _surfaceType
        self.bvType = _bvType
        if _bvType == ProcMetaBVTypes.CUSTOM:
            assert _bvCorners != None, "Custom BV types must provide bvCorners"
        self.bvCorners = _bvCorners
        self.conjugateAngle = _conjAngle
        self.thickness = _thicknessProfile
        self.sampleDist = _sampleDist

    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        ecids = []
        for e in self.edgeLoop:
            ecids.append(e.name)
        info["boundaries"] = ecids

        st = "direct"
        if self.surfaceType == ProcMetaSurfaceTypes.CONJUGATE:
            st = "conjugate"
        elif self.surfaceType == ProcMetaSurfaceTypes.MIXEDMINIMAL:
            st = "minimal"
        else:
            assert self.surfaceType == ProcMetaSurfaceTypes.DIRECT, "Unsupported surface type."
        info["type"] = st

        if self.surfaceType == ProcMetaSurfaceTypes.CONJUGATE:
            info["conj-angle"] = self.conjugateAngle

            bvt = "aabb"
            if self.bvType == ProcMetaBVTypes.TRIANGULARPRISM:
                bvt = "prism"
            elif self.bvType == ProcMetaBVTypes.TET:
                bvt = "tet"
            elif self.bvType == ProcMetaBVTypes.CUSTOM:
                bvt = "custom"
            else:
                assert self.bvType == ProcMetaBVTypes.AABB, "Unsupported BV type"
            info["bv-type"] = bvt
            
            if self.bvCorners.size > 0:
                info["bv"] = self.bvCorners.tolist()
        if self.thickness.size > 0:
            info["thickness"] = self.thickness.tolist()
        if self.sampleDist != None:
            info["sample-dist"] = self.sampleDist
        return info

class OpNode_DualSurface(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _surfIn:OpNode_Surface) -> None:
        super().__init__("dual"+str(_id), 
                         ProcMetaOpClasses.SKELETON,
                         [OpNode_Surface],
                         [ProcMetaOpClasses.SKELETON])
        self.surfIn = _surfIn
    
    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["src"] = self.surfIn.name
        return info

class OpNode_AssociateFamily(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _primarySurfIn:OpNode_Surface, _dualSurfIn:OpNode_DualSurface, _interpAngle:float) -> None:
        super().__init__("af"+str(_id), 
                         ProcMetaOpClasses.SKELETON,
                         [OpNode_Surface, OpNode_DualSurface],
                         [ProcMetaOpClasses.SKELETON])
        self.primary_surf_in = _primarySurfIn
        self.dual_surf_in = _dualSurfIn
        self.angle = _interpAngle

    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["s0"] = self.primary_surf_in.name
        info["s1"] = self.primary_surf_in.name
        info["angle"] = self.angle
        return info

class OpNode_Mirror(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _skelIn:ProcMetaOpNode, _mirrorPlaneOrigin:np.array, _mirrorPlaneNormal:np.array, _doCopy:bool) -> None:
        super().__init__("mirror"+str(_id), 
                         ProcMetaOpClasses.SKELETON,
                         [OpNode_Line, OpNode_Surface, OpNode_DualSurface, OpNode_AssociateFamily, OpNode_Mirror, OpNode_Transform, OpNode_Group],
                         [ProcMetaOpClasses.SKELETON])
        self.skel_in = _skelIn
        self.planeO = _mirrorPlaneOrigin
        self.planeN = _mirrorPlaneNormal
        self.doCopy = _doCopy

    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["src"] = self.skel_in.name
        info["copy"] = self.doCopy
        info["plane"] = np.concatenate((self.planeO, self.planeN)).tolist()
        return info

class OpNode_Transform(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _skelIn:ProcMetaOpNode, _origin:np.array,
                 _rotationAxis:np.array, _rotationAngleDegrees:float,
                 _translateVec:np.array, _scaleVec:np.array,
                 _doCopy:bool) -> None:
        super().__init__("t"+str(_id), 
                         ProcMetaOpClasses.SKELETON,
                         [OpNode_Line, OpNode_Surface, OpNode_DualSurface, OpNode_AssociateFamily, OpNode_Mirror, OpNode_Transform, OpNode_Group],
                         [ProcMetaOpClasses.SKELETON])
        self.skel_in = _skelIn.name
        self.origin = _origin
        self.rotationAxis = _rotationAxis
        self.rotationAngleDegrees = _rotationAngleDegrees
        self.translateVec = _translateVec
        self.scaleVec = _scaleVec
        self.doCopy = _doCopy

    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["src"] = self.skel_in
        info["origin"] = self.origin.tolist()
        info["r-axis"] = self.rotationAxis.tolist()
        info["r-angle"] = self.rotationAngleDegrees
        info["t"] = self.translateVec.tolist()
        info["s"] = self.scaleVec.tolist()
        info["copy"] = self.doCopy
        return info

class OpNode_Group(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _skelsIn:list[ProcMetaOpNode]) -> None:
        super().__init__("g"+str(_id),
                       ProcMetaOpClasses.SKELETON,
                       [OpNode_Line, OpNode_Surface, OpNode_DualSurface, OpNode_AssociateFamily, OpNode_Mirror, OpNode_Transform, OpNode_Group],
                       [ProcMetaOpClasses.SKELETON])
        self.skels_in = _skelsIn
        
    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        skelids = []
        for skel in self.skels_in:
            skelids.append(skel.name)
        info["inputs"] = skelids
        return info
        
class OpNode_Object(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _skelIn:ProcMetaOpNode, _sdfResolution:int, _extrusionMethod:ProcMetaExtrusionMethods=ProcMetaExtrusionMethods.SPHERICAL) -> None:
        super().__init__("object"+str(_id), 
                         ProcMetaOpClasses.SOLID,
                         [OpNode_Line, OpNode_Surface, OpNode_DualSurface, OpNode_AssociateFamily, OpNode_Mirror, OpNode_Transform, OpNode_Group],
                         [ProcMetaOpClasses.SKELETON])
        self.skel_in = _skelIn
        self.res = _sdfResolution
        self.extrusion_method = _extrusionMethod
    
    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["src"] = self.skel_in.name
        info["resolution"] = self.res
        extMethodCode:int = None
        match self.extrusion_method:
            case ProcMetaExtrusionMethods.SPHERICAL:
                extMethodCode = 0
            case ProcMetaExtrusionMethods.NORMAL:
                extMethodCode = 1
            case _:
                print("Error: Unsupported extrusion method. Using spherical.")
                extMethodCode = 0
        info["extrusion-method"] = extMethodCode
        return info

class OpNode_BooleanObject(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _objIn0:ProcMetaOpNode, _objIn1:ProcMetaOpNode, _csgOperator:ProcMetaCSGOps) -> None:
        super().__init__("boolean"+str(_id), 
                         ProcMetaOpClasses.SOLID,
                         [OpNode_Object, OpNode_BooleanObject],
                         [ProcMetaOpClasses.SOLID])
        self.objIn0 = _objIn0
        self.objIn1 = _objIn1
        self.csgOp = _csgOperator
    
    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["src"] = [self.objIn0.name, self.objIn1.name]
        opCode = -1
        match self.csgOp:
            case ProcMetaCSGOps.UNION:
                opCode = 0
            case ProcMetaCSGOps.INTERSECT:
                opCode = 1
            case ProcMetaCSGOps.DIFFERENCE:
                opCode = 2
        assert opCode != -1
        info["opt"] = opCode
        return info

class OpNode_Voxel(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _objIn:ProcMetaOpNode, _E:float=1.0, _nu:float=0.45, _rho:float=1.0) -> None:
        super().__init__("vox"+str(_id), 
                         ProcMetaOpClasses.SOLID,
                         [OpNode_Object, OpNode_BooleanObject],
                         [ProcMetaOpClasses.SOLID])
        self.objIn = _objIn
        self.E = _E
        self.nu = _nu
        self.rho = _rho

    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["src"] = self.objIn.name
        info["E"] = self.E
        info["nu"] = self.nu
        info["pho"] = self.rho
        return info
        
class OpNode_MaterialMatrix(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _voxIn:OpNode_Voxel) -> None:
        super().__init__("mat"+str(_id), 
                         ProcMetaOpClasses.MATERIALPROPS,
                         [OpNode_Voxel],
                         [ProcMetaOpClasses.SOLID])
        self.voxIn = _voxIn
    
    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["src"] = self.voxIn.name
        return info

class OpNode_PhononicBandGap(ProcMetaOpNode):
    '''
    '''
    def __init__(self, _id:int, _voxIn:OpNode_Voxel, _numWaves:int=5, _numDispersionCurves:int=10, _gapTolerance:float=1e-4) -> None:
        super().__init__("pbg"+str(_id), 
                         ProcMetaOpClasses.MATERIALPROPS,
                         [OpNode_Voxel],
                         [ProcMetaOpClasses.SOLID])
        self.voxIn = _voxIn
        self.numWaves = _numWaves
        self.numDispCurves = _numDispersionCurves
        self.gapTol = _gapTolerance
        
    def get_proc_meta_description(self) -> dict:
        info = super().get_proc_meta_description()
        info["src"] = self.voxIn.name
        info["numWaves"] = self.numWaves
        info["numDispersionCurves"] = self.numDispCurves
        info["gapTol"] = self.gapTol
        return info

