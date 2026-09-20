"""Les fantomes doivent etre coherents avec leur propre verite terrain."""

import numpy as np
import pytest

import morphanalyzer as ma


def test_sphere_matches_analytic_volume(sphere):
    truth = sphere.meta["truth"]
    measured = 1.0 - sphere.solid.sum() / sphere.solid.size
    assert measured == pytest.approx(truth["porosity"], abs=2e-3)


def test_sphere_pack_has_exactly_n_components(pack):
    """Les spheres doivent rester separees meme en 26-connexite."""
    from scipy import ndimage as ndi

    st = ndi.generate_binary_structure(3, 3)
    _lab, n = ndi.label(pack.solid, structure=st)
    assert n == pack.meta["truth"]["n_spheres"]


def test_sphere_pack_is_disjoint(pack):
    truth = pack.meta["truth"]
    c = truth["centres"]
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 2 * truth["radius"], "des spheres se recouvrent"
    measured = 1.0 - pack.solid.sum() / pack.solid.size
    assert measured == pytest.approx(truth["porosity"], rel=2e-2)


def test_plate_and_cylinders_have_expected_class():
    assert ma.phantoms.plate().meta["truth"]["expected_shape_class"] == "plate"
    assert ma.phantoms.cylinders().meta["truth"]["expected_shape_class"] == "strut"


def test_straight_tube_tortuosity_is_one():
    vol = ma.phantoms.straight_tube()
    assert vol.meta["truth"]["tortuosity"] == 1.0
    assert vol.fluid.any()


def test_sinusoidal_tube_tortuosity_above_one():
    vol = ma.phantoms.sinusoidal_tube(shape=(96, 64, 64), amplitude=8.0, n_periods=2.0)
    t = vol.meta["truth"]
    assert t["tortuosity"] > 1.0
    assert t["geodesic_over_euclidean"] == pytest.approx(np.sqrt(t["tortuosity"]), rel=1e-9)


def test_voronoi_foam_is_open_and_labelled(foam):
    truth = foam.meta["truth"]
    assert truth["plateau_law_exact"]
    assert truth["n_cells"] >= 2
    labels = truth["cell_labels"]
    assert labels.shape == foam.shape
    # le solide porte le label 0, le fluide est entierement etiquete
    assert (labels[foam.solid] == 0).all()
    assert (labels[~foam.solid] > 0).all()
    # mousse ouverte : porosite elevee
    assert truth["porosity"] > 0.5


def test_voronoi_foam_has_interior_cells(foam):
    """Il faut des cellules entierement incluses : c'est sur elles que portent
    toutes les statistiques morphometriques (convention de la these)."""
    truth = foam.meta["truth"]
    assert truth["n_interior_cells"] >= 3
    labels = truth["cell_labels"]
    border = np.zeros(foam.shape, dtype=bool)
    border[[0, -1]] = True
    border[:, [0, -1]] = True
    border[:, :, [0, -1]] = True
    for lab in truth["interior_cells"]:
        assert not (labels[border] == lab).any()


def test_voronoi_foam_porosity_is_realistic(foam):
    """Entre 80 et 95 %, la gamme des mousses de la these (tableau 2.2)."""
    assert 0.80 < foam.meta["truth"]["porosity"] < 0.96


def test_voronoi_foam_cells_are_connected(foam):
    """Chaque cellule de Voronoi doit etre d'un seul tenant (convexite)."""
    from scipy import ndimage as ndi

    labels = foam.meta["truth"]["cell_labels"]
    for lab in np.unique(labels[labels > 0]):
        _, n = ndi.label(labels == lab)
        assert n == 1, f"la cellule {lab} est fragmentee en {n} morceaux"
