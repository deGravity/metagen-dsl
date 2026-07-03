"""VoxelVolume: import external voxel geometry as a first-class Volume.

A VoxelVolume plays the same role in a Tile as a lifted skeleton
(UniformBeams, shells, Spheres): it is the "Volume"-level geometric
entity that patterning operations (Tile embedding + TilingPattern
mirrors/transforms) and CSG booleans compose on top of. Instead of
lifting a skeleton, the volume is given directly as a binary voxel
occupancy grid — from a numpy array or an .npy/.npz file.

Conventions
-----------
- The voxel array is indexed ``voxels[z, y, x]`` — the project-wide
  voxel layout, matching ``GeometryResult.voxel_active_cells`` and the
  test-fixture ``vox_active_cells.npz`` arrays.
- The voxel box spans the axis-aligned bounding box of the Tile's
  embedded cuboid (fit policy): array x-extent maps to the box
  x-extent, etc. The embedding must be an axis-aligned cuboid.
- Serialization into graph.json is inline and self-contained: packed
  bits, zlib-deflated, base64-encoded (see procmeta_nodes.
  encode_voxel_data). The kernel converts the occupancy to a signed-
  distance grid once at load, so the import is resolution-independent
  and behaves like any other SDF primitive.

@example_usage:
    vox = np.load('scan.npz')['voxels']        # (nz, ny, nx) bool
    vol = VoxelVolume(vox)
    embedding = cuboid.embed(0.5, 0.5, 0.5,
                             cornerAtAABBMin=cuboid.corners.FRONT_BOTTOM_LEFT)
    tile = Tile([vol], embedding)
    s = Structure(tile, CuboidFullMirror())
"""

from pathlib import Path

import numpy as np

from . import convex_polytope as cp
from .lifting import ThickeningProcedure


def _load_voxel_array(voxels, npz_key=None):
    """Accept a 3D ndarray or a path to a .npy/.npz file."""
    if isinstance(voxels, (str, Path)):
        path = Path(voxels)
        if not path.exists():
            raise FileNotFoundError(f"VoxelVolume: no such file: {path}")
        loaded = np.load(path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if npz_key is not None:
                key = npz_key
            elif 'voxels' in loaded.files:
                key = 'voxels'
            elif len(loaded.files) == 1:
                key = loaded.files[0]
            else:
                raise ValueError(
                    f"VoxelVolume: {path} has multiple arrays "
                    f"({loaded.files}); pass npz_key= to pick one")
            arr = loaded[key]
        else:
            arr = loaded
    else:
        arr = np.asarray(voxels)
    return arr


class VoxelVolume:
    """Import an external voxel-occupancy grid as a Volume-level entity.

    Wraps a dense 3D occupancy array so it can be placed in a Tile and
    patterned/booleaned exactly like a lifted skeleton.

    @params:
        voxels - a 3D array indexed [z, y, x] (bool, or numeric —
                 thresholded at `threshold`), or a str/Path to a .npy
                 or .npz file containing one.
        threshold - occupancy threshold for numeric input arrays
                    (cell is material iff value > threshold).
        npz_key - which array to take from an .npz file (default:
                  'voxels', or the single array if there is only one).
    @returns:
        vol - the new VoxelVolume, ready to be placed in a Tile.
    @example_usage:
        vol = VoxelVolume(sphere_mask)
        tile = Tile([vol], embedding)
    """

    def __init__(self, voxels, threshold: float = 0.5, npz_key: str = None):
        arr = _load_voxel_array(voxels, npz_key)
        if arr.ndim != 3:
            raise ValueError(
                f"VoxelVolume: expected a 3D array indexed [z, y, x], "
                f"got shape {arr.shape}")
        if arr.dtype != bool:
            arr = arr > threshold
        if arr.size == 0:
            raise ValueError("VoxelVolume: empty voxel array")
        self.voxels = np.ascontiguousarray(arr)

        # Tile integration: Tile reads .parentCP from its entities to
        # determine the bounding-volume template. Voxel volumes live in
        # a cuboid frame (the array spans the embedded cuboid's AABB).
        self.parentCP = cp.CPT_Cuboid("cuboid")
        # Translator reads .thickeningProc to pick the Object node's
        # extrusion method; voxel volumes don't use extrusion, so any
        # valid value works. SPHERICAL == extMethod 0.
        self.thickeningProc = ThickeningProcedure.SPHERICAL

    @property
    def dims(self):
        """(nx, ny, nz) — grid size along x, y, z."""
        nz, ny, nx = self.voxels.shape
        return (nx, ny, nz)

    @property
    def volume_fraction(self) -> float:
        return float(self.voxels.mean())

    def __repr__(self):
        nx, ny, nz = self.dims
        return (f"<VoxelVolume {nx}x{ny}x{nz} "
                f"vf={self.volume_fraction:.4f}>")
