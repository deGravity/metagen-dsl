"""VoxelVolume import (T5): DSL-side tests.

Covers node serialization (encode/decode round-trip), graph structure
(Volume-level placement: patterning and boolean nodes compose on top),
Tile/embedding integration, and file loading. Pure-Python — no kernel
required.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from metagen_dsl import (
    VoxelVolume, Structure, Tile, Identity, CuboidFullMirror,
    Union, Subtract, cuboid,
)
from metagen_dsl.procmeta_nodes import (
    OpNode_VoxelVolume, encode_voxel_data, decode_voxel_data,
    VOXEL_VOLUME_ENCODING,
)
from metagen_dsl.procmeta_translator import ProcMetaTranslator


def sphere_mask(n, radius_frac=0.35):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    c = (n - 1) / 2
    return ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) \
        <= (radius_frac * n) ** 2


def unit_embedding():
    return cuboid.embed(1.0, 1.0, 1.0,
                        cornerAtAABBMin=cuboid.corners.FRONT_BOTTOM_LEFT)


def octant_embedding():
    return cuboid.embed(0.5, 0.5, 0.5,
                        cornerAtAABBMin=cuboid.corners.FRONT_BOTTOM_LEFT)


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------

@pytest.mark.tier1
def test_encode_decode_roundtrip():
    rng = np.random.default_rng(20260703)
    vox = rng.random((13, 7, 21)) > 0.6      # deliberately anisotropic
    data = encode_voxel_data(vox)
    nz, ny, nx = vox.shape
    back = decode_voxel_data(data, [nx, ny, nz])
    assert back.shape == vox.shape
    assert np.array_equal(back, vox)


@pytest.mark.tier1
def test_node_description_fields():
    vox = sphere_mask(16)
    node = OpNode_VoxelVolume(0, vox,
                              np.array([0.0, 0.0, 0.0]),
                              np.array([0.5, 0.5, 0.5]))
    info = node.get_proc_meta_description()
    assert info['name'] == 'volume0'
    assert info['dims'] == [16, 16, 16]
    assert info['encoding'] == VOXEL_VOLUME_ENCODING
    assert info['bbox'] == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert np.array_equal(decode_voxel_data(info['data'], info['dims']), vox)


@pytest.mark.tier1
def test_graph_json_is_valid_json_and_self_contained():
    s = Structure(Tile([VoxelVolume(sphere_mask(16))], unit_embedding()),
                  Identity())
    g = json.loads(s.graph_json())
    names = [op['name'] for op in g['operations']]
    assert names == ['volume0', 'object0', 'vox0']
    vol = g['operations'][0]
    # inline payload — no external file references
    assert set(vol) >= {'dims', 'encoding', 'data', 'bbox'}
    assert 'path' not in vol


# ---------------------------------------------------------------------------
# Volume-level composition in the graph
# ---------------------------------------------------------------------------

@pytest.mark.tier1
def test_pattern_composes_on_volume():
    """CuboidFullMirror on an octant volume: mirror nodes take the
    volume node as src, and the object node consumes the mirror chain
    — i.e. the volume node sits at the same graph level as lifted
    skeletons ("Volume" level)."""
    s = Structure(Tile([VoxelVolume(sphere_mask(16))], octant_embedding()),
                  CuboidFullMirror())
    g = json.loads(s.graph_json())
    ops = {op['name']: op for op in g['operations']}
    assert 'volume0' in ops
    mirrors = sorted(n for n in ops if n.startswith('mirror'))
    assert len(mirrors) == 3, 'octant -> unit cube needs 3 mirrors'
    assert ops['mirror0']['src'] == 'volume0'
    assert ops['object0']['src'] == mirrors[-1]
    # volume spans the octant AABB of the embedding
    assert ops['volume0']['bbox'] == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]


@pytest.mark.tier1
def test_boolean_composes_on_volume_structures():
    a = Structure(Tile([VoxelVolume(sphere_mask(8))], unit_embedding()),
                  Identity())
    b = Structure(Tile([VoxelVolume(sphere_mask(8, 0.2))], unit_embedding()),
                  Identity())
    for op_cls, code in ((Union, 0), (Subtract, 2)):
        g = json.loads(ProcMetaTranslator(op_cls(a, b)).to_json())
        ops = {op['name']: op for op in g['operations']}
        assert 'volume0' in ops and 'volume1' in ops
        assert ops['boolean0']['opt'] == code
        assert sorted(ops['boolean0']['src']) == ['object0', 'object1']


@pytest.mark.tier1
def test_mixed_tile_volume_plus_beams():
    from metagen_dsl import UniformBeams, Polyline, vertex, skeleton
    emb = octant_embedding()
    v0 = vertex(cuboid.corners.FRONT_BOTTOM_LEFT)
    v1 = vertex(cuboid.corners.BACK_TOP_RIGHT)
    beams = UniformBeams(skeleton([Polyline([v0, v1])]), 0.05)
    s = Structure(Tile([VoxelVolume(sphere_mask(8)), beams], emb),
                  CuboidFullMirror())
    g = json.loads(s.graph_json())
    ops = {op['name']: op for op in g['operations']}
    assert 'volume0' in ops and 'l0' in ops
    # both members are grouped, then patterned together
    assert sorted(ops['g0']['inputs']) == ['l0', 'volume0']
    assert ops['mirror0']['src'] == 'g0'


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------

@pytest.mark.tier1
def test_accepts_float_array_with_threshold():
    dens = np.zeros((8, 8, 8))
    dens[2:6, 2:6, 2:6] = 0.9
    vol = VoxelVolume(dens, threshold=0.5)
    assert vol.voxels.dtype == bool
    assert vol.voxels.sum() == 4 ** 3


@pytest.mark.tier1
def test_accepts_npy_and_npz_paths(tmp_path):
    vox = sphere_mask(12)
    npy = tmp_path / 'v.npy'
    np.save(npy, vox)
    assert np.array_equal(VoxelVolume(npy).voxels, vox)

    npz = tmp_path / 'v.npz'
    np.savez_compressed(npz, voxels=vox)
    assert np.array_equal(VoxelVolume(npz).voxels, vox)

    npz2 = tmp_path / 'v2.npz'
    np.savez_compressed(npz2, a=vox, b=~vox)
    with pytest.raises(ValueError):
        VoxelVolume(npz2)          # ambiguous without a key
    assert np.array_equal(VoxelVolume(npz2, npz_key='b').voxels, ~vox)


@pytest.mark.tier1
def test_rejects_bad_input():
    with pytest.raises(ValueError):
        VoxelVolume(np.ones((4, 4)))          # not 3D
    with pytest.raises(FileNotFoundError):
        VoxelVolume('/does/not/exist.npy')


@pytest.mark.tier1
def test_fixture_voxels_load(tmp_path):
    """The fixture repo's vox_active_cells.npz files load directly."""
    from . import fixture_data as fd
    if not fd.fixtures_initialized():
        pytest.skip('fixtures submodule not initialized')
    path = Path(fd.FIXTURES) / 'beam_bcc_tet' / '97' / 'vox_active_cells.npz'
    if not path.exists():
        pytest.skip('beam_bcc_tet/97 fixture not present')
    vol = VoxelVolume(path)
    assert vol.dims == (96, 96, 96)
    assert 0.0 < vol.volume_fraction < 1.0
