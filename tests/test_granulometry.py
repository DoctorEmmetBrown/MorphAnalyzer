import numpy as np
import pytest

import morphanalyzer as ma


def test_aperture_of_a_slab_is_half_its_thickness():
    """Dans une plaque d'epaisseur t, la plus grande boule incluse a un rayon t/2."""
    vol = ma.phantoms.plate(shape=(40, 40, 40), thickness=10.0, axis=0)
    ap = np.asarray(ma.granulometry.aperture_map(vol.solid, n_radii=24))
    assert np.median(ap[vol.solid]) == pytest.approx(5.0, abs=0.7)


def test_aperture_of_a_sphere_pack_is_the_radius(pack):
    truth = pack.meta["truth"]
    ap = np.asarray(ma.granulometry.aperture_map(pack.solid, n_radii=24))
    # la mediane sous-estime un peu : les voxels de bord appartiennent a des
    # boules plus petites que la sphere entiere
    assert np.percentile(ap[pack.solid], 90) == pytest.approx(truth["radius"], rel=0.2)


def test_aperture_is_zero_outside_and_positive_inside(pack):
    ap = np.asarray(ma.granulometry.aperture_map(pack.solid, n_radii=12))
    assert (ap[~pack.solid] == 0).all()
    assert (ap[pack.solid] > 0).all()


def test_aperture_scales_with_voxel_size():
    vol = ma.phantoms.plate(shape=(32,) * 3, thickness=8.0)
    a1 = np.asarray(ma.granulometry.aperture_map(vol.solid, voxel_size=1.0, n_radii=16))
    a2 = np.asarray(ma.granulometry.aperture_map(vol.solid, voxel_size=2.0, n_radii=16))
    assert a2.max() == pytest.approx(2.0 * a1.max(), rel=1e-5)


def test_aperture_rejects_anisotropic_voxels():
    vol = ma.phantoms.plate(shape=(24,) * 3, thickness=6.0)
    with pytest.raises(NotImplementedError, match="anisotrope"):
        ma.granulometry.aperture_map(vol.solid, voxel_size=(2.0, 1.0, 1.0))


def test_pore_size_distribution_sums_to_one(pack):
    ap = ma.granulometry.aperture_map(pack.solid, n_radii=16)
    df = ma.granulometry.pore_size_distribution(ap, bins=20)
    assert df["fraction"].sum() == pytest.approx(1.0, abs=1e-9)
    assert df["cumulative"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
