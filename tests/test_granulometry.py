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


# --- boules maximales et marqueurs ---------------------------------------


def test_maximal_balls_on_a_single_sphere_is_one_full_ball():
    """Une cavite spherique isolee : une boule, remplie a 100 %."""
    zz, yy, xx = np.ogrid[:48, :48, :48]
    fluid = (zz - 24) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2 <= 14**2
    balls = ma.granulometry.maximal_balls(fluid, min_radius=3.0)
    assert len(balls) >= 1
    big = balls.table.iloc[0]
    assert big["radius"] == pytest.approx(14.0, abs=1.0)
    assert big["fill_ratio"] > 0.9
    assert not big["touches_border"]


def test_fill_ratio_is_low_for_a_ball_swallowed_by_a_bigger_one():
    """Deux cavites tres inegales reliees : la petite cede du terrain."""
    zz, yy, xx = np.ogrid[:32, :32, :64]
    fluid = ((zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 18) ** 2 <= 13**2) | (
        (zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 38) ** 2 <= 6**2
    )
    balls = ma.granulometry.maximal_balls(fluid, min_radius=3.0)
    t = balls.table.sort_values("radius", ascending=False)
    assert len(t) >= 2
    assert t.iloc[0]["fill_ratio"] > t.iloc[-1]["fill_ratio"]


def test_cell_markers_are_one_per_cavity():
    zz, yy, xx = np.ogrid[:32, :32, :64]
    fluid = ((zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2 <= 11**2) | (
        (zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 46) ** 2 <= 11**2
    )
    fluid |= (np.abs(zz - 16) <= 2) & (np.abs(yy - 16) <= 2)
    mk = np.asarray(ma.granulometry.cell_markers(fluid, fill_ratio=0.65, keep_border_balls=False))
    assert mk.max() == 2
    assert (mk > 0).sum() == 2


def test_markers_lie_inside_the_mask():
    vol = ma.phantoms.sphere_pack(shape=(64,) * 3, radius=8.0, n=4, seed=2)
    fluid = ~vol.solid
    mk = np.asarray(ma.granulometry.cell_markers(fluid))
    assert (mk > 0).sum() > 0
    assert fluid[mk > 0].all()


def test_candidates_all_agrees_with_maxima_on_a_small_case():
    """La force brute d'iMorph et le depart par les maxima donnent la meme boule."""
    zz, yy, xx = np.ogrid[:24, :24, :24]
    fluid = (zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 12) ** 2 <= 8**2
    a = ma.granulometry.maximal_balls(fluid, candidates="maxima", min_radius=3.0)
    b = ma.granulometry.maximal_balls(fluid, candidates="all", min_radius=3.0)
    assert a.table["radius"].max() == pytest.approx(b.table["radius"].max(), abs=1e-6)


def test_bad_candidates_value_is_rejected():
    fluid = np.ones((8, 8, 8), dtype=bool)
    with pytest.raises(ValueError, match="candidates"):
        ma.granulometry.maximal_balls(fluid, candidates="magique")
