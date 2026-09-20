import numpy as np
import pytest

from morphanalyzer.distance import distance_transform, geodesic_ball, nearest_seed_propagation


def test_distance_transform_is_exact_on_sphere(sphere):
    """Au centre d'une sphere de rayon r, la distance au fluide vaut r."""
    truth = sphere.meta["truth"]
    d = np.asarray(distance_transform(sphere.solid))
    centre = tuple(int(round(c)) for c in truth["centre"])
    # la transformee exacte donne le rayon au voxel pres (discretisation du bord)
    assert d[centre] == pytest.approx(truth["radius"], abs=1.0)
    assert d.max() == pytest.approx(truth["radius"], abs=1.0)


def test_distance_transform_outside(sphere):
    d_in = np.asarray(distance_transform(sphere.solid, inside=True))
    d_out = np.asarray(distance_transform(sphere.solid, inside=False))
    assert (d_in[sphere.solid] > 0).all()
    assert (d_in[~sphere.solid] == 0).all()
    assert (d_out[~sphere.solid] > 0).all()


def test_distance_transform_honours_voxel_size(sphere):
    d1 = np.asarray(distance_transform(sphere.solid, voxel_size=1.0))
    d2 = np.asarray(distance_transform(sphere.solid, voxel_size=2.0))
    assert d2.max() == pytest.approx(2.0 * d1.max(), rel=1e-5)


def test_nearest_seed_propagation_fills_target():
    vals = np.zeros((16, 16, 16), dtype=np.float32)
    seeds = np.zeros((16, 16, 16), dtype=bool)
    seeds[4, 4, 4] = True
    vals[4, 4, 4] = 7.0
    seeds[12, 12, 12] = True
    vals[12, 12, 12] = 9.0
    target = np.ones((16, 16, 16), dtype=bool)
    out = nearest_seed_propagation(vals, seeds, target)
    assert set(np.unique(out)) == {7.0, 9.0}
    assert out[5, 5, 5] == 7.0
    assert out[11, 11, 11] == 9.0


def test_nearest_seed_propagation_rejects_empty_seeds():
    z = np.zeros((8, 8, 8), dtype=bool)
    with pytest.raises(ValueError, match="aucun germe"):
        nearest_seed_propagation(np.zeros((8, 8, 8)), z, ~z)


def test_geodesic_ball_does_not_cross_a_wall():
    """Deux plaques separees par du fluide : la boule ne doit pas sauter."""
    solid = np.zeros((40, 40, 40), dtype=bool)
    solid[:, 18:20, :] = True  # plaque A
    solid[:, 22:24, :] = True  # plaque B, separee de 2 voxels
    cloud = geodesic_ball(solid, (20, 19, 20), 10.0)
    ys = np.unique(cloud[:, 1])
    assert set(ys.tolist()) <= {18, 19}, "la boule a traverse le fluide"


def test_geodesic_ball_is_inside_the_euclidean_ball():
    solid = np.ones((30, 30, 30), dtype=bool)
    centre = (15, 15, 15)
    r = 7.0
    cloud = geodesic_ball(solid, centre, r)
    d = np.linalg.norm(cloud - np.asarray(centre), axis=1)
    assert d.max() <= r + 1e-9
    # dans un milieu plein, on retrouve la boule discrete complete
    assert len(cloud) == int((d <= r).sum())


def test_geodesic_ball_follows_a_bent_tube():
    """Dans un tube coude, la boule suit le tube au lieu de couper au court."""
    solid = np.zeros((40, 40, 40), dtype=bool)
    solid[20, 10:30, 20] = True  # branche 1
    solid[20, 29, 20:35] = True  # branche 2, a angle droit
    cloud = geodesic_ball(solid, (20, 12, 20), 10.0)
    assert (cloud[:, 2] == 20).all(), "la boule est sortie du tube"
    assert cloud[:, 1].max() <= 22
