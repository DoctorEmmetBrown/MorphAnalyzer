import numpy as np
import pytest

from morphanalyzer.core import Roi, Volume, ball_offsets, connectivity_offsets


def test_volume_basics():
    v = Volume(np.zeros((4, 5, 6), dtype=bool), voxel_size=2.0, unit="um", name="t")
    assert v.shape == (4, 5, 6)
    assert v.voxel_size == (2.0, 2.0, 2.0)
    assert v.is_isotropic
    assert v.voxel_volume == 8.0
    assert v.physical_shape == (8.0, 10.0, 12.0)
    assert v.fluid.all()


def test_volume_rejects_bad_input():
    with pytest.raises(ValueError):
        Volume(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        Volume(np.zeros((2, 2, 2)), voxel_size=(1.0, 0.0, 1.0))


def test_volume_solid_requires_binary():
    v = Volume(np.zeros((3, 3, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="binaire"):
        _ = v.solid


def test_roi_box_and_cylinder():
    shape = (10, 20, 20)
    box = Roi(z=(2, 8))
    assert box.mask(shape).sum() == 6 * 20 * 20
    cyl = Roi(cylinder=True, axis=0)
    m = cyl.mask(shape)
    # disque de rayon ~9.5 replique sur 10 coupes
    assert m.sum() < np.prod(shape)
    assert m[:, 10, 10].all()  # l'axe est dedans
    assert not m[0, 0, 0]  # le coin est dehors


def test_connectivity_and_ball():
    assert len(connectivity_offsets(6)) == 6
    assert len(connectivity_offsets(18)) == 18
    assert len(connectivity_offsets(26)) == 26
    # boule de rayon 1 avec le critere floor(d) <= r : tout le cube 3x3x3
    assert len(ball_offsets(1.0)) == 27
    assert len(ball_offsets(0.0)) == 1


def test_volume_converts_to_array_without_wrapping():
    """np.asarray(volume) doit rendre le tableau, pas un objet 0-d."""
    v = Volume(np.ones((3, 4, 5), dtype=np.float32), voxel_size=2.0)
    a = np.asarray(v)
    assert a.shape == (3, 4, 5)
    assert a.dtype == np.float32
    assert np.asarray(v, dtype=np.float64).dtype == np.float64
    # meme tampon tant qu'on ne demande pas de copie
    assert np.shares_memory(a, v.data)
