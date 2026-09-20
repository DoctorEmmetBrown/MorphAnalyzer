"""Phase 9 : decoupage angulaire, profils, connectivite empilee, Voronoi 2D."""

import numpy as np
import pandas as pd
import pytest

import morphanalyzer as ma


@pytest.fixture(scope="module")
def bone():
    return ma.phantoms.cortical_tube(
        shape=(24, 128, 128), n_canals=20, sector_weights=[3, 1, 1, 1], seed=2
    )


@pytest.fixture(scope="module")
def bone_masks(bone):
    t = bone.meta["truth"]
    nz = bone.shape[0]
    canals = np.broadcast_to(t["canal_mask"], (nz,) + t["canal_mask"].shape).copy()
    out = np.broadcast_to(~t["ring_mask"], canals.shape).copy()
    return canals, out, t


# --------------------------------------------------------------------------
# decoupage angulaire
# --------------------------------------------------------------------------
def test_sector_map_follows_the_trigonometric_convention():
    """theta = atan2(y0 - j, i - x0) : x vers la droite, y vers le HAUT.

    Le secteur 0 couvre donc le quadrant en haut a droite de l'image, malgre
    l'axe des lignes qui descend.
    """
    sm = ma.cortical.sector_map((32, 32), center=(15.5, 15.5), n_sectors=4)
    assert sm[4, 28] == 0  # haut droite
    assert sm[4, 4] == 1  # haut gauche
    assert sm[28, 4] == 2  # bas gauche
    assert sm[28, 28] == 3  # bas droite


def test_start_angle_rotates_the_sectors():
    a = ma.cortical.sector_map((32, 32), center=(15.5, 15.5), n_sectors=4)
    b = ma.cortical.sector_map((32, 32), center=(15.5, 15.5), n_sectors=4, start_angle=90.0)
    assert b[4, 4] == 0
    assert a[4, 4] == 1


def test_four_sectors_split_a_disc_evenly():
    ny = nx = 128
    j, i = np.ogrid[:ny, :nx]
    disc = ((j - 63.5) ** 2 + (i - 63.5) ** 2) <= 50.0**2
    sm = ma.cortical.sector_map((ny, nx), center=(63.5, 63.5), n_sectors=4)
    counts = np.bincount(sm[disc], minlength=4)
    assert counts.std() / counts.mean() < 0.01


def test_circular_ellipse_is_the_isotropic_case():
    iso = ma.cortical.sector_bounds(8)
    ell = ma.cortical.sector_bounds(8, ellipse=(10.0, 10.0, 30.0))
    assert np.allclose(np.sort(np.mod(ell, 2 * np.pi)), np.sort(np.mod(iso, 2 * np.pi)))


def test_ellipse_correction_equalises_sector_areas():
    """Sur une ellipse, des parts d'angle egal n'ont pas la meme aire."""
    ny = nx = 160
    j, i = np.ogrid[:ny, :nx]
    a, b = 60.0, 25.0
    ell = ((i - 79.5) / a) ** 2 + ((j - 79.5) / b) ** 2 <= 1.0
    naive = ma.cortical.sector_map((ny, nx), center=(79.5, 79.5), n_sectors=8)
    fixed = ma.cortical.sector_map((ny, nx), center=(79.5, 79.5), n_sectors=8, ellipse=(a, b, 0.0))
    cv_naive = np.bincount(naive[ell], minlength=8)
    cv_fixed = np.bincount(fixed[ell], minlength=8)
    disp = lambda c: c.std() / c.mean()  # noqa: E731
    assert disp(cv_fixed) < 0.5 * disp(cv_naive)


def test_iso_area_angles_balance_the_volume(bone_masks):
    _canals, out, t = bone_masks
    inside = ~out
    ang = ma.cortical.iso_area_angles(inside, 6, center=t["centre"])
    sm = ma.cortical.sector_map(inside.shape, center=t["centre"], angles=ang)
    counts = np.bincount(sm[inside[0]], minlength=6)
    assert counts.std() / counts.mean() < 0.02


def test_sector_map_needs_a_count_or_angles():
    with pytest.raises(ValueError, match="n_sectors"):
        ma.cortical.sector_map((8, 8), center=(3.5, 3.5))


# --------------------------------------------------------------------------
# profils
# --------------------------------------------------------------------------
def test_angular_profile_total_matches_the_exact_porosity(bone_masks):
    canals, out, t = bone_masks
    prof = ma.cortical.angular_profile(canals, center=t["centre"], n_sectors=4, mask_out=out)
    assert prof.overall == pytest.approx(t["porosity"], rel=1e-12)


def test_angular_profile_sees_the_built_in_anisotropy(bone_masks):
    canals, out, t = bone_masks
    prof = ma.cortical.angular_profile(canals, center=t["centre"], n_sectors=4, mask_out=out)
    s = prof.by_sector()
    assert s.idxmax() == 0
    assert s[0] > 1.5 * s.drop(0).mean()


def test_profile_is_constant_along_a_z_invariant_phantom(bone_masks):
    canals, out, t = bone_masks
    prof = ma.cortical.angular_profile(canals, center=t["centre"], n_sectors=6, mask_out=out)
    per_z = prof.by_slice()
    assert per_z.std() == pytest.approx(0.0, abs=1e-12)
    piv = prof.pivot()
    assert piv.shape == (canals.shape[0], 6)


def test_angular_aperture_on_a_uniform_map(bone_masks):
    canals, out, t = bone_masks
    aper = np.where(canals, 3.5, -1.0).astype(np.float32)
    prof = ma.cortical.angular_aperture(aper, center=t["centre"], n_sectors=4, mask_out=out)
    assert prof.overall == pytest.approx(3.5)
    assert np.allclose(prof.by_sector().to_numpy(), 3.5)


def test_radial_profile_localises_a_band_of_canals():
    """Des canaux confines dans une couronne ne doivent apparaitre que la."""
    nz, ny, nx = 4, 96, 96
    j, i = np.ogrid[:ny, :nx]
    r = np.hypot(j - 47.5, i - 47.5)
    band = (r >= 24) & (r <= 30)
    phase = np.broadcast_to(band, (nz, ny, nx)).copy()
    df = ma.cortical.radial_profile(phase, center=(47.5, 47.5), n_bins=8, r_max=48.0)
    per_ring = df.groupby("ring")["porosity"].mean()
    assert per_ring.idxmax() in (4, 5)
    assert per_ring[0] == pytest.approx(0.0)
    assert per_ring[7] == pytest.approx(0.0)


def test_radial_profile_equal_area_rings_have_equal_counts():
    nz, ny, nx = 1, 128, 128
    phase = np.zeros((nz, ny, nx), dtype=bool)
    df = ma.cortical.radial_profile(
        phase, center=(63.5, 63.5), n_bins=6, r_max=60.0, equal_area=True
    )
    n = df["n_voxels"].to_numpy(dtype=float)
    assert n.std() / n.mean() < 0.02


# --------------------------------------------------------------------------
# connectivite par empilement
# --------------------------------------------------------------------------
def test_parallel_canals_never_merge(bone_masks):
    canals, _out, t = bone_masks
    prof = ma.cortical.cortical_connectivity(canals)
    assert (prof.table["n_objects"] == t["n_canals"]).all()
    assert prof.table["total_volume"].is_monotonic_increasing
    assert int(np.asarray(prof.labels).max()) == t["n_canals"]


def test_a_bridge_merges_two_canals_at_the_right_slice():
    m = np.zeros((40, 40, 40), dtype=bool)
    m[:, 18:22, 10:14] = True
    m[:, 18:22, 26:30] = True
    m[30:, 18:22, 10:30] = True
    prof = ma.cortical.cortical_connectivity(m)
    n = prof.table.set_index("z")["n_objects"]
    assert n[29] == 2
    assert n[30] == 1
    assert (np.asarray(prof.labels)[m] == 1).all()


def test_volumes_per_slice_match_the_object_count(bone_masks):
    canals, _o, t = bone_masks
    prof = ma.cortical.cortical_connectivity(canals)
    for k, v in enumerate(prof.volumes):
        assert len(v) == prof.table["n_objects"].iloc[k]
        assert v.sum() == pytest.approx(prof.table["total_volume"].iloc[k])
        assert (np.diff(v) <= 0).all()


def test_n_objects_above_is_a_fraction(bone_masks):
    canals, _o, t = bone_masks
    prof = ma.cortical.cortical_connectivity(canals)
    assert prof.n_objects_above(0.0).iloc[-1] == t["n_canals"]
    assert prof.n_objects_above(0.5).iloc[-1] == 0
    assert prof.n_objects_above(1.0, cumulative=True).iloc[-1] == t["n_canals"]
    with pytest.raises(ValueError, match="fraction"):
        prof.n_objects_above(1.5)


def test_connectivity_start_slice_is_checked(bone_masks):
    canals, _o, _t = bone_masks
    with pytest.raises(ValueError, match="start"):
        ma.cortical.cortical_connectivity(canals, start=999)


# --------------------------------------------------------------------------
# Voronoi 2D
# --------------------------------------------------------------------------
def test_voronoi_splits_the_matrix_on_the_bisector():
    """Deux canaux symetriques : la frontiere est la mediatrice."""
    ny = nx = 64
    j, i = np.ogrid[:ny, :nx]
    solid = np.ones((1, ny, nx), dtype=bool)
    hole_a = ((j - 31.5) ** 2 + (i - 15.0) ** 2) <= 4.0**2
    hole_b = ((j - 31.5) ** 2 + (i - 48.0) ** 2) <= 4.0**2
    solid[0] &= ~(hole_a | hole_b)
    lab = np.asarray(ma.cortical.voronoi_2d(solid))[0]
    left = lab[31, 20]
    right = lab[31, 43]
    assert left != right
    mid = (15.0 + 48.0) / 2.0
    assert lab[31, int(mid) - 4] == left
    assert lab[31, int(mid) + 4] == right


def test_voronoi_labels_exactly_the_matrix(bone):
    solid = np.asarray(bone)
    lab, table = ma.cortical.voronoi_2d(solid, return_table=True)
    arr = np.asarray(lab)
    assert (arr[solid] > 0).all()
    assert (arr[~solid] == 0).all()
    assert isinstance(table, pd.DataFrame)
    assert table["area"].sum() == pytest.approx(solid.sum())
