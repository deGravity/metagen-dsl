import numpy as np
from . import procmeta_graph as pmg
from . import convex_polytope as cp
from .lifting import *

class Tile():
    def __init__(self, lifted_skeletons:list[LiftedSkeleton], embedding:list[list[float]]) -> None:
        # assign to the original parameter names (correcting mismatched signatures in code/documentation)
        _liftedSkeletons = lifted_skeletons
        _boundingVolumeCornerPositions = embedding

        if len(_liftedSkeletons) < 1:
            raise Exception("Invalid Tile construction: must provide at least one skeleton")
        self.liftedSkeletons = _liftedSkeletons
        self.bv_template:cp.ConvexPolytope = _liftedSkeletons[0].parentCP
        # TODO: assert that all elements are from the same CP
        numCorners = self.bv_template.num_corners

        assert numCorners == len(_boundingVolumeCornerPositions), f"Invalid Tile construction: for a CP with {numCorners} corners, corner array must be a list of length {numCorners}"
        bvcpos = np.zeros([numCorners, 3])
        for cid in range(numCorners):
            pos = _boundingVolumeCornerPositions[cid]
            assert len(pos) == 3, f"Invalid Tile construction: each entry of the corner list must be a 3D position, given by a list of 3 float values. Corner {cid} has {len(pos)} floats"
            for i in range(3):
                bvcpos[cid, i] = pos[i]
        self.bv_corner_positions = bvcpos

        # ensure that the tile corners yield a valid embedding of the CP
        self.bv_template.validateGlobalEmbedding(self.bv_corner_positions)

        # now that we have real coords for the corners, we should make sure all the endpoints used are located in distinct positions
        # eg, multiple weight combinations might lead to identical points -- we should collocate those and adjust the edge endpoint references accordingly

    # maybe an integer return for eg. cc intersects itself, distinct cc's intersect
    # NOTE: this doesn't consider thickness!
    # NOTE: also, this really only makes sense for straight beams, since eg curved beams aren't interpolated, surfaces aren't yet solved.
    def has_self_intersecting_skeleton(self) -> bool:
        # check if the 
        pass