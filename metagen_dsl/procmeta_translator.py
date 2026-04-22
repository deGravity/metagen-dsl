from .tile import *
from .pattern import *
from .lifting import *
from .structure import *
from . import skeleton as sk
from . import convex_polytope as cp
from pathlib import Path
from . import procmeta_graph as pmg
from . import procmeta_nodes as pmn
import math
from queue import LifoQueue, Queue

class ProcMetaTranslator:
    def __init__(self, finalStructure:Structure) -> None:
        self.sr = pmg.ProcMetaGraph()

        # figure out how many substructures there are, and keep a backtrace so we can roll it forward to build up the bools at the end
        boolOpTrace = LifoQueue()
        non_bool_substructures:list[Structure] = []
        q = Queue()
        q.put(finalStructure)
        while not q.empty():
            curr_structure = q.get()
            assert isinstance(curr_structure, Structure), "Invalid input to ProcMetaTranslator: ensure that the input is a valid Structure object"
            if not isinstance(curr_structure, CSGBoolean):
                non_bool_substructures.append(curr_structure)
                continue
            # we have a boolean
            q.put(curr_structure.A)
            q.put(curr_structure.B) 
            boolOpTrace.put(curr_structure)

        # for each substructure, create the appropriate nodes of the graph
        # TODO: for now, have a separate set of operations (vertices, edge chains etc) for each one. Potentially figure out how to reuse some things later.
        substructure2pmnOutNode = {}
        for structure in non_bool_substructures:
            # get fbv information, to be used in some of the lifting / patterning operations
            bvType:pmn.ProcMetaBVTypes = None
            match structure.tile.bv_template:
                case cp.CPT_Tet():
                    bvType = pmn.ProcMetaBVTypes.TET
                case cp.CPT_TriangularPrism():
                    bvType = pmn.ProcMetaBVTypes.TRIANGULARPRISM
                case cp.CPT_Cuboid(): # TODO: make sure it's really an AABB, might be skew
                    bvType = pmn.ProcMetaBVTypes.AABB     
                case _:
                    bvType = pmn.ProcMetaBVTypes.CUSTOM
            assert bvType != None 

            # look at every lifted skeleton 
            liftedSkelPMNs = []
            for liftedSkel in structure.tile.liftedSkeletons:
                # loop over the vertices and edge chains
                CPRefdPt2ProcMetaVert:dict[cp.PointReferencedToCP, pmn.OpNode_Vertex] = {}
                SkelEC2ProcMetaEC:dict[sk.EdgeChain, pmn.OpNode_EdgeChain] = {}
                match liftedSkel.skel:
                    case sk.PointSkeleton():
                        for pt in liftedSkel.skel.get_points():
                            if pt not in CPRefdPt2ProcMetaVert:
                                v_pm = self.sr.add_vertex(pt.getGlobalCoords(structure.tile.bv_corner_positions))
                                CPRefdPt2ProcMetaVert[pt] = v_pm
                    case sk.EdgeSkeleton():
                        for ec in liftedSkel.skel.get_edge_chains():
                            ordered_vert_list = ec.get_ordered_points_along_chain()
                            # add the points and collect the new references
                            for pt in ordered_vert_list:
                                if pt not in CPRefdPt2ProcMetaVert:
                                    v_pm = self.sr.add_vertex(pt.getGlobalCoords(structure.tile.bv_corner_positions))
                                    CPRefdPt2ProcMetaVert[pt] = v_pm 

                            # add the edge chain to the graph (using the pmg references)
                            pmg_ordered_vert_list = [CPRefdPt2ProcMetaVert[p_orig] for p_orig in ordered_vert_list]
                            ec_pm = self.sr.add_edge_chain(pmg_ordered_vert_list, ec.isSmooth)
                            SkelEC2ProcMetaEC[ec] = ec_pm
                    case _:
                        print("Unsupported skeleton type")
                        return

                # get extrusion method, to be used by the object node later
                extMethod:pmn.ProcMetaExtrusionMethods = None
                match liftedSkel.thickeningProc:
                    case ThickeningProcedure.SPHERICAL:
                        extMethod = pmn.ProcMetaExtrusionMethods.SPHERICAL
                    case ThickeningProcedure.NORMAL:
                        extMethod = pmn.ProcMetaExtrusionMethods.NORMAL
                    case _:
                        print("Error: unsupported extrution method / thickening procedure. Aborting.")
                        return
                assert extMethod != None

                # apply ProcMeta Skeleton lifting procedure based on type of input
                liftOut:pmn.ProcMetaOpNode = None
                match liftedSkel:
                    case UniformBeams() | SpatiallyVaryingBeams():
                        ProcMetaLineNodes = []
                        for ec in liftedSkel.skel.get_edge_chains():
                            if liftedSkel.thicknessProfileType == ThicknessProfileType.UNIFORM:
                                l = self.sr.add_line_uniformThickness([SkelEC2ProcMetaEC[ec]], liftedSkel.uniformThicknessValue)
                            elif liftedSkel.thicknessProfileType == ThicknessProfileType.VARYING:
                                l = self.sr.add_line_variableThickness([SkelEC2ProcMetaEC[ec]], liftedSkel.varyingThicknessProfile)
                            ProcMetaLineNodes.append(l)

                        # add group node for all the lines
                        if len(ProcMetaLineNodes) > 1:
                            g0 = self.sr.add_group(ProcMetaLineNodes)
                            liftOut = g0
                        else:
                            liftOut = ProcMetaLineNodes[0]
                    # TODO: include hierarchy level through general Shell lifting to reduce repetition
                    case UniformTPMSShellViaConjugation():
                        assert len(liftedSkel.skel.connectedComponents) == 1, "Lifting to shell requires a single connected component."
                        assert liftedSkel.skel.connectedComponents[0].ccType == sk.ConnectedComponentType.SIMPLE_CLOSED_LOOP, "Lifting to shell requires a simple closed loop."
                        pmchains = [SkelEC2ProcMetaEC[skelec] for skelec in liftedSkel.skel.get_edge_chains()]
                        s0 = self.sr.add_conjugate_surface_uniformThickness(pmchains, bvType, structure.tile.bv_corner_positions, liftedSkel.uniformThicknessValue)
                        liftOut = s0
                    case UniformDirectShell():
                        assert len(liftedSkel.skel.connectedComponents) == 1, "Lifting to shell requires a single connected component."
                        assert liftedSkel.skel.connectedComponents[0].ccType == sk.ConnectedComponentType.SIMPLE_CLOSED_LOOP, "Lifting to shell requires a simple closed loop."
                        pmchains = [SkelEC2ProcMetaEC[skelec] for skelec in liftedSkel.skel.get_edge_chains()]
                        s0 = self.sr.add_direct_surface_uniformThickness(pmchains, bvType, liftedSkel.uniformThicknessValue)
                        liftOut = s0
                    case UniformTPMSShellViaMixedMinimal():
                        assert len(liftedSkel.skel.connectedComponents) == 1, "Lifting to shell requires a single connected component."
                        assert liftedSkel.skel.connectedComponents[0].ccType == sk.ConnectedComponentType.SIMPLE_CLOSED_LOOP, "Lifting to shell requires a simple closed loop."
                        pmchains = [SkelEC2ProcMetaEC[skelec] for skelec in liftedSkel.skel.get_edge_chains()]
                        s0 = self.sr.add_mixed_minimal_surface_uniformThickness(pmchains, bvType, liftedSkel.uniformThicknessValue)
                        liftOut = s0
                    case Spheres():
                        # for each point in the skeleton
                        pmnSphereLines = []
                        for pt in liftedSkel.skel.get_points():
                            # generate the colocated vertex, edge and beam
                            twinVert = self.sr.add_vertex(pt.getGlobalCoords(structure.tile.bv_corner_positions) + np.array([0.001, 0.001, 0.001]))
                            e = self.sr.add_edge_chain([CPRefdPt2ProcMetaVert[pt], twinVert], False)
                            l = self.sr.add_line_uniformThickness([e], liftedSkel.uniformThicknessValue)
                            pmnSphereLines.append(l)
                        # group them all 
                        g0 = self.sr.add_group(pmnSphereLines)
                        liftOut = g0
                    case _:
                        print(f"Unsupported lift procedure: {type(liftedSkel)}. Aborting.")
                        return
                assert not liftOut == None
                liftedSkelPMNs.append(liftOut)

            # get a single liftOut node for moving forward
            assert len(liftedSkelPMNs) > 0, "No lifted skeletons found."
            if len(liftedSkelPMNs) == 1:
                liftOut = liftedSkelPMNs[0]
            else:
                liftOut = self.sr.add_group(liftedSkelPMNs)  # group so we can continue processing them as a unit

            # process the remainder of the tile operations
            patOut:pmn.ProcMetaOpNode = None
            sqrt2:float = math.sqrt(2)
            match structure.pat:
                case TetFullMirror():
                    m1 = self.sr.add_mirror(liftOut, np.array([0.0, 0.0, 0.0]), np.array([-sqrt2, 0.0, sqrt2]), True)
                    m2 = self.sr.add_mirror(m1, np.array([0.0, 0.0, 0.0]), np.array([0.0, -sqrt2, sqrt2]), True)
                    m3 = self.sr.add_mirror(m2, np.array([0.0, 0.0, 0.0]), np.array([-sqrt2, 0.0, sqrt2]), True)

                    m4 = self.sr.add_mirror(m3, np.array([0.5, 0.5, 0.5]), np.array([1.0, 0.0, 0.0]), True)
                    m5 = self.sr.add_mirror(m4, np.array([0.5, 0.5, 0.5]), np.array([0.0, 1.0, 0.0]), True)
                    m6 = self.sr.add_mirror(m5, np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.0]), True)
                    patOut = m6              
                case CuboidFullMirror() | TriPrismFullMirror():
                    currCPCorners:np.array = structure.tile.bv_corner_positions
                    currOut = liftOut
                    opQ = structure.pat.to_unit_cube(currCPCorners).opQueue
                    while not opQ.empty():
                        op:PatternOp = opQ.get()
                        if isinstance(op, NoOp):
                            break
                        globalSpecs:GlobalPatternOpSpecs = op.apply(currCPCorners)
                        m = self.sr.add_mirror(currOut, globalSpecs.planeO, globalSpecs.planeN, globalSpecs.doCopy)
                        # update for next round
                        currOut = m
                        currCPCorners = globalSpecs.resultingCPCorners
                    patOut = currOut
                case Identity():
                    patOut = liftOut
                case Custom():
                    currCPCorners:np.array = structure.tile.bv_corner_positions
                    currOut = liftOut
                    opQ = structure.pat.ops.opQueue
                    while not opQ.empty():
                        op:PatternOp = opQ.get()
                        if isinstance(op, NoOp):
                            break
                        globalSpecs:GlobalPatternOpSpecs = op.apply(currCPCorners)
                        match globalSpecs:
                            case GlobalMirrorSpecs():
                                m = self.sr.add_mirror(currOut, globalSpecs.planeO, globalSpecs.planeN, globalSpecs.doCopy)
                            case GlobalRotateSpecs():
                                m = self.sr.add_rotate(currOut, globalSpecs.axisO, globalSpecs.axisDir, globalSpecs.angleDeg, globalSpecs.doCopy)
                            case GlobalTranslateSpecs():
                                m = self.sr.add_translate(currOut, globalSpecs.translateVec, globalSpecs.doCopy)
                            case GlobalScaleSpecs():
                                m = self.sr.add_scale(currOut, globalSpecs.scaleVec, globalSpecs.doCopy)
                            case _:
                                raise Exception("Unsupported pattern operation output type")
                        # update for next round
                        currOut = m
                        currCPCorners = globalSpecs.resultingCPCorners
                    patOut = currOut
                case _:
                    raise Exception("Unsupported pattern type")
            assert patOut != None

            o1 = self.sr.add_object(patOut, 64, extMethod)
            substructure2pmnOutNode[structure] = o1

        # once all structures have been added independently, add the CSG operations between them using the backtrace contstructed at the beginning
        booleanOpOut = None
        if len(non_bool_substructures) == 1: # no boolean operations involved, just take the output of the final structure
            assert len(substructure2pmnOutNode) == 1, "There should be exactly one substructure in this graph"
            booleanOpOut = substructure2pmnOutNode[finalStructure]
        else:
            while not boolOpTrace.empty():
                currOp:CSGBoolean = boolOpTrace.get()
                in1:Structure = substructure2pmnOutNode[currOp.A]
                in2:Structure = substructure2pmnOutNode[currOp.B]
                currOpPmnOut = None
                match currOp.op_type:
                    case CSGBooleanTypes.UNION:
                        currOpPmnOut = self.sr.add_boolean_union(in1, in2)
                    case CSGBooleanTypes.INTERSECT:
                        currOpPmnOut = self.sr.add_boolean_intersect(in1, in2)
                    case CSGBooleanTypes.DIFFERENCE:
                        currOpPmnOut = self.sr.add_boolean_difference(in1, in2)
                    case _:
                        print(f"Error: unsupported CSG Boolean type: {currOp.op_type}. Aborting.")
                        return
                assert currOpPmnOut != None
                substructure2pmnOutNode[currOp] = currOpPmnOut
                # update opOut in case this is the last one
                booleanOpOut = currOpPmnOut

        # add the vox node onto the final boolean
        vox1 = self.sr.add_voxel(booleanOpOut)


    def save(self, filename:Path):
        self.sr.save_to_json(filename)