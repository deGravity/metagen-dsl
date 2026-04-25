"""Pixel-parity check for the legacy pyrender path (metagen-dsl[pyrender]).

The fixtures' renders/<res>/*.png were produced by the original metagen
package's pyrender pipeline. This test checks that the optional pyrender
path in metagen_dsl reproduces those references within tight pixel
tolerance.

Skipped unless pyrender is importable AND the legacy renderer in
metagen_dsl._viz is implemented (currently a NotImplementedError stub).
"""

import pytest

from . import fixture_data as fd

pyrender = pytest.importorskip(
    'pyrender',
    reason='metagen-dsl[pyrender] not installed')


@pytest.mark.pyrender
@pytest.mark.tier2
@pytest.mark.parametrize('case', ['beam_bcc'])
@pytest.mark.parametrize('resolution', [97])
def test_pyrender_matches_reference(case, resolution):
    """Compare pyrender output against the committed fixture PNG.

    Comparison metric: per-pixel mean absolute difference, normalized
    to [0, 1]. Threshold: <0.01 (renderer determinism is high but not
    bit-exact across machines / GL versions).
    """
    pytest.skip("pyrender path not yet implemented in metagen_dsl._viz")
    # When implemented, the comparison would look like:
    #
    # from PIL import Image
    # import numpy as np
    # from metagen_dsl._viz import render_pyrender
    #
    # s = _make_structure(case)
    # geo = s.geometry(resolution=resolution)
    # produced = render_pyrender(geo)
    # for view, png_bytes in produced.items():
    #     ref_path = fd.renders_dir(case, resolution) / f'{view}.png'
    #     if not ref_path.exists():
    #         continue
    #     a = np.asarray(Image.open(io.BytesIO(png_bytes))).astype(float) / 255
    #     b = np.asarray(Image.open(ref_path)).astype(float) / 255
    #     assert a.shape == b.shape, f'shape mismatch for {view}'
    #     mad = np.abs(a - b).mean()
    #     assert mad < 0.01, f'{view} differs from reference: MAD={mad:.4f}'
