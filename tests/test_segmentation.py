"""Segmentation des cellules, cols, morphometrie.

La verite terrain vient du fantome de Voronoi : la partition exacte des cellules
est connue, ce qu'aucun tomogramme ne permet.
"""

import numpy as np
import pytest

import morphanalyzer as ma

# --- le watershed lui-meme ------------------------------------------------


def _two_cavities():
    """Deux cavites spheriques reliees par une constriction."""
    shape = (24, 24, 48)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    fluid = ((zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 14) ** 2 <= 81) | (
        (zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 34) ** 2 <= 81
    )
    fluid |= (np.abs(zz - 12) <= 2) & (np.abs(yy - 12) <= 2) & (xx >= 14) & (xx <= 34)
    return fluid


def test_watershed_splits_two_cavities_at_the_constriction():
    fluid = _two_cavities()
    dist = np.asarray(ma.distance.distance_transform(fluid))
    markers = np.zeros(fluid.shape, dtype=np.int32)
    markers[12, 12, 14] = 1
    markers[12, 12, 34] = 2
    cells = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid))
    assert set(np.unique(cells)) == {0, 1, 2}
    # la frontiere passe par la constriction, au milieu
    line = cells[12, 12, 14:35]
    transition = np.flatnonzero(np.diff(line) != 0)
    assert len(transition) == 1
    assert 6 <= transition[0] <= 14
    # les deux cellules sont de taille comparable
    n1, n2 = (cells == 1).sum(), (cells == 2).sum()
    assert abs(n1 - n2) / max(n1, n2) < 0.1


def test_watershed_labels_every_masked_voxel():
    fluid = _two_cavities()
    dist = np.asarray(ma.distance.distance_transform(fluid))
    markers = np.zeros(fluid.shape, dtype=np.int32)
    markers[12, 12, 14] = 1
    markers[12, 12, 34] = 2
    cells = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid))
    assert (cells[fluid] > 0).all(), "pas de ligne de partage materialisee attendue"
    assert (cells[~fluid] == 0).all()


def test_watershed_rejects_missing_markers():
    fluid = _two_cavities()
    dist = np.asarray(ma.distance.distance_transform(fluid))
    with pytest.raises(ValueError, match="aucun marqueur"):
        ma.segmentation.watershed_cells(dist, np.zeros(fluid.shape, dtype=np.int32), mask=fluid)


def _slab_between_two_cells():
    solid = np.zeros((20, 20, 20), dtype=bool)
    solid[:, 8:12, :] = True
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[:, 0:8, :] = 1
    labels[:, 12:, :] = 2
    return solid, labels


def test_watershed_accepts_markers_outside_the_mask():
    """Cas d'usage d'iMorph : propager des labels de fluide dans le solide.

    Les germes sont ici hors du domaine inonde. En partant de l'interface, la
    plaque se partage exactement par moitie sur son plan median.
    """
    solid, labels = _slab_between_two_cells()
    dsol = np.asarray(ma.distance.distance_transform(solid))
    out = np.asarray(ma.segmentation.watershed_cells(dsol, labels, mask=solid, invert=False))
    assert (out[solid] > 0).all()
    assert set(np.unique(out[solid])) == {1, 2}
    assert (out == 1).sum() == (out == 2).sum()


def test_flood_direction_changes_the_result():
    """Le sens d'inondation n'est pas une convention neutre.

    L'algorithme est un parcours « meilleur d'abord » : le premier front arrive
    sur la crete du relief l'emporte ensuite largement. Sur une plaque
    symetrique, inonder depuis la crete donne tout a une seule cellule.
    """
    solid, labels = _slab_between_two_cells()
    dsol = np.asarray(ma.distance.distance_transform(solid))
    from_crest = np.asarray(ma.segmentation.watershed_cells(dsol, labels, mask=solid, invert=True))
    from_face = np.asarray(ma.segmentation.watershed_cells(dsol, labels, mask=solid, invert=False))
    assert len(np.unique(from_crest[solid])) == 1
    assert len(np.unique(from_face[solid])) == 2


def test_numba_and_skimage_broadly_agree():
    fluid = _two_cavities()
    dist = np.asarray(ma.distance.distance_transform(fluid))
    markers = np.zeros(fluid.shape, dtype=np.int32)
    markers[12, 12, 14] = 1
    markers[12, 12, 34] = 2
    a = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid, method="numba"))
    b = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid, method="skimage"))
    assert (a[fluid] == b[fluid]).mean() > 0.95


# --- la chaine complete contre la verite terrain --------------------------


def test_segmentation_recovers_the_voronoi_cells(segmented):
    """Les cellules interieures doivent recouvrir les regions de Voronoi."""
    truth = segmented["vol"].meta["truth"]
    cells, fluid, ref = segmented["cells"], segmented["fluid"], truth["cell_labels"]
    ious = []
    for lab in truth["interior_cells"]:
        m = ref == lab
        cand, cnt = np.unique(cells[m], return_counts=True)
        best = cand[cnt.argmax()]
        inter = int((m & (cells == best)).sum())
        union = int((m | ((cells == best) & fluid)).sum())
        ious.append(inter / union)
    ious = np.asarray(ious)
    assert len(ious) >= 5
    assert np.median(ious) > 0.8, f"IoU median {np.median(ious):.3f}"
    assert (ious > 0.7).mean() > 0.6


def test_marker_count_is_of_the_order_of_the_cell_count(segmented):
    truth = segmented["vol"].meta["truth"]
    n_markers = int(segmented["markers"].max())
    assert 0.5 * truth["n_cells"] < n_markers < 2.0 * truth["n_cells"]


def test_high_fill_ratio_under_segments(foam_cells):
    """Au-dela de 80 % de remplissage, la these annonce une sous-segmentation."""
    fluid = ~foam_cells.solid
    dist = ma.distance.distance_transform(fluid)
    balls = ma.granulometry.maximal_balls(fluid, distance=dist)
    low = np.asarray(ma.granulometry.cell_markers(fluid, balls=balls, fill_ratio=0.55)).max()
    high = np.asarray(ma.granulometry.cell_markers(fluid, balls=balls, fill_ratio=0.95)).max()
    assert high < low


# --- morphometrie ---------------------------------------------------------


def test_cell_morphometry_columns_and_border_flag(segmented):
    df = ma.segmentation.cell_morphometry(segmented["cells"])
    assert len(df) == int(segmented["cells"].max())
    for col in (
        "label",
        "volume",
        "d_equivalent",
        "a",
        "b",
        "c",
        "a_on_b",
        "b_on_c",
        "theta",
        "phi",
        "touches_border",
    ):
        assert col in df.columns
    assert df["touches_border"].any(), "des cellules touchent forcement le bord"
    assert (~df["touches_border"]).any(), "il faut aussi des cellules interieures"
    assert (df["a"] >= df["b"]).all() and (df["b"] >= df["c"]).all()


def test_cell_volumes_account_for_almost_all_the_fluid(segmented):
    """Le watershed n'invente ni ne perd de volume, aux poches sans germe pres.

    Une composante de fluide sans aucun marqueur reste a zero : l'inondation ne
    peut pas l'atteindre. C'est correct, et negligeable ici, mais il faut le
    savoir avant de sommer des volumes.
    """
    df = ma.segmentation.cell_morphometry(segmented["cells"])
    total = int(segmented["fluid"].sum())
    assert df["volume"].sum() <= total
    assert df["volume"].sum() / total > 0.999


def test_fluid_without_marker_stays_unlabelled():
    fluid = np.zeros((20, 20, 40), dtype=bool)
    fluid[8:12, 8:12, 2:10] = True  # poche avec germe
    fluid[8:12, 8:12, 30:38] = True  # poche isolee, sans germe
    dist = np.asarray(ma.distance.distance_transform(fluid))
    markers = np.zeros(fluid.shape, dtype=np.int32)
    markers[10, 10, 6] = 1
    cells = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid))
    assert (cells[:, :, 30:38] == 0).all()
    assert (cells[8:12, 8:12, 2:10] == 1).all()


def test_equivalent_diameter_matches_a_sphere():
    """Sur une cavite spherique unique, Dpore doit valoir 2r."""
    zz, yy, xx = np.ogrid[:48, :48, :48]
    fluid = (zz - 24) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2 <= 15**2
    labels = fluid.astype(np.int32)
    df = ma.segmentation.cell_morphometry(labels)
    assert df["d_equivalent"].iloc[0] == pytest.approx(30.0, rel=0.02)


def test_cell_elongation_is_in_the_foam_range(segmented):
    """Une mousse de Voronoi a des cellules legerement allongees, comme les vraies.

    La these mesure a/b ~ 1,30 et b/c ~ 1,23 sur la Recemat 1723 (tableau 3.1).
    Le fantome n'a aucune raison de reproduire ces valeurs exactement, mais il
    doit tomber dans le meme regime — des cellules a peine anisotropes.
    """
    df = ma.segmentation.cell_morphometry(segmented["cells"])
    inner = df[~df["touches_border"]]
    assert len(inner) >= 5
    assert 1.05 < inner["a_on_b"].median() < 1.6
    assert 1.0 < inner["b_on_c"].median() < 1.6


def test_volume_scales_with_voxel_size(segmented):
    a = ma.segmentation.cell_morphometry(segmented["cells"], voxel_size=1.0)
    b = ma.segmentation.cell_morphometry(segmented["cells"], voxel_size=2.0)
    assert b["volume"].sum() == pytest.approx(8.0 * a["volume"].sum(), rel=1e-9)
    assert b["d_equivalent"].median() == pytest.approx(2.0 * a["d_equivalent"].median(), rel=1e-6)
    # les rapports sont sans dimension
    assert b["a_on_b"].median() == pytest.approx(a["a_on_b"].median(), rel=1e-6)


# --- cols et reseau -------------------------------------------------------


def test_throats_are_symmetric_and_ordered(segmented):
    th = ma.segmentation.throats(segmented["cells"])
    assert len(th) > 0
    assert (th["label_a"] < th["label_b"]).all()
    assert not th.duplicated(subset=["label_a", "label_b"]).any()
    assert (th["area"] > 0).all()


def test_throat_diameter_is_smaller_than_pore_diameter(segmented):
    """Dcol < Dpore par definition d'une constriction (these, fig. 3.12)."""
    cm = ma.segmentation.cell_morphometry(segmented["cells"])
    th = ma.segmentation.throats(segmented["cells"])
    ratio = th["d_equivalent"].median() / cm["d_equivalent"].median()
    assert 0.2 < ratio < 0.9, f"Dcol/Dpore = {ratio:.2f}"


def test_throat_area_of_a_flat_interface_is_exact():
    """Deux cubes accoles : la surface du col est celle de la face commune."""
    labels = np.zeros((10, 10, 10), dtype=np.int32)
    labels[:, :, :5] = 1
    labels[:, :, 5:] = 2
    th = ma.segmentation.throats(labels)
    assert len(th) == 1
    assert th["n_faces"].iloc[0] == 100
    assert th["area"].iloc[0] == pytest.approx(100.0)
    assert th["d_equivalent"].iloc[0] == pytest.approx(2 * np.sqrt(100 / np.pi))


def test_connectivity_counts_neighbours():
    labels = np.zeros((6, 6, 18), dtype=np.int32)
    labels[:, :, :6] = 1
    labels[:, :, 6:12] = 2
    labels[:, :, 12:] = 3
    cx = ma.segmentation.connectivity(labels)
    assert cx.to_dict() == {1: 1, 2: 2, 3: 1}


def test_pore_network_returns_tables_and_graph(segmented):
    cells, th = ma.segmentation.pore_network(segmented["cells"])
    assert "length" in th.columns
    assert (th["length"] > 0).all()
    g = ma.segmentation.pore_network(segmented["cells"], as_networkx=True)
    assert g.number_of_nodes() == len(cells)
    assert g.number_of_edges() == len(th)
