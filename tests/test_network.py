"""Phase 7 : drainage morphologique, percolation d'invasion, Young-Laplace."""

import numpy as np
import pytest

import morphanalyzer as ma


# --------------------------------------------------------------------------
# fantomes
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tube():
    """Tube droit de rayon 6 le long de z : reponse analytique uniforme."""
    nz, ny, nx = 48, 32, 32
    _z, y, x = np.ogrid[:nz, :ny, :nx]
    disc = ((y - 15.5) ** 2 + (x - 15.5) ** 2) <= 6.0**2
    return np.broadcast_to(disc, (nz, ny, nx)).copy()


@pytest.fixture(scope="module")
def ink_bottle():
    """Col etroit puis chambre large : le piege a mercure de manuel.

    La chambre ne peut etre envahie qu'une fois le rayon de courbure descendu
    sous le rayon du col, quelle que soit sa propre taille. C'est l'effet
    « bouteille d'encre », et c'est la seule chose qu'un algorithme de drainage
    morphologique doit absolument reproduire.
    """
    nz, ny, nx = 64, 48, 48
    z, y, x = np.ogrid[:nz, :ny, :nx]
    c = 23.5
    neck = ((y - c) ** 2 + (x - c) ** 2 <= 4.0**2) & (z < 34)
    bulb = ((z - 44) ** 2 + (y - c) ** 2 + (x - c) ** 2) <= 11.0**2
    return neck | bulb, neck, bulb & ~neck


@pytest.fixture(scope="module")
def network_tables():
    v = ma.phantoms.voronoi_foam(
        shape=(96, 96, 96), n_cells=20, strut=3.0, min_seed_gap=20.0, seed=7
    )
    fluid = ~v.solid
    dist = ma.distance.distance_transform(fluid)
    mk = ma.granulometry.cell_markers(fluid, distance=dist, fill_ratio=0.55)
    cells_img = np.asarray(ma.segmentation.watershed_cells(dist, mk, mask=fluid))
    return {
        "labels": cells_img,
        "cells": ma.segmentation.cell_morphometry(cells_img),
        "throats": ma.segmentation.throats(cells_img),
    }


# --------------------------------------------------------------------------
# Young-Laplace
# --------------------------------------------------------------------------
def test_young_laplace_reference_value():
    """Eau/air, rayon 10 um, mouillage parfait : Pc = 2 sigma / r."""
    pc = ma.network.capillary_pressure(10.0, unit="um")
    assert pc == pytest.approx(2 * 0.0728 / 10e-6, rel=1e-12)
    assert pc == pytest.approx(14560.0, rel=1e-6)


def test_imorph_convention_is_twice_laplace():
    a = ma.network.capillary_pressure(7.5, convention="laplace")
    b = ma.network.capillary_pressure(7.5, convention="imorph")
    assert b == pytest.approx(2 * a)


def test_capillary_radius_is_the_inverse():
    for conv in ("laplace", "imorph"):
        r = 3.25
        pc = ma.network.capillary_pressure(r, convention=conv, contact_angle=30.0)
        back = ma.network.capillary_radius(pc, convention=conv, contact_angle=30.0)
        assert back == pytest.approx(r, rel=1e-9)


def test_unit_scaling():
    assert ma.network.capillary_pressure(1.0, unit="mm") == pytest.approx(
        ma.network.capillary_pressure(1000.0, unit="um")
    )


def test_zero_radius_is_infinite_pressure():
    out = ma.network.capillary_pressure(np.array([0.0, -1.0, 2.0]))
    assert np.isinf(out[0]) and np.isinf(out[1]) and np.isfinite(out[2])


def test_unknown_unit_is_refused():
    with pytest.raises(ValueError, match="unite"):
        ma.network.capillary_pressure(1.0, unit="furlong")


# --------------------------------------------------------------------------
# drainage morphologique
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["hazlett", "hilpert"])
def test_tube_fills_uniformly(tube, method):
    res = ma.network.drainage(tube, face=0, method=method, step=0.5)
    f = res.filling_radius
    assert (f[tube] == f[tube][0]).all()
    assert f[tube][0] == pytest.approx(5.39, abs=0.1)
    assert (f[~tube] == -1).all()
    assert len(res.curve) == 1
    assert res.curve["saturation"].iloc[0] == pytest.approx(1.0)
    assert res.curve["penetration"].iloc[0] == tube.shape[0] - 1


@pytest.mark.parametrize("method", ["hazlett", "hilpert"])
def test_ink_bottle_shielding(ink_bottle, method):
    """La chambre n'est jamais envahie a un rayon superieur a celui du col."""
    pore, neck, bulb = ink_bottle
    res = ma.network.drainage(
        pore, face=0, method=method, step=0.25, min_radius=0.5, restrict="inlet"
    )
    f = res.filling_radius
    r_neck = f[neck].max()
    assert f[bulb].max() <= r_neck + 1e-6
    # la chambre est bien plus large que le col : sans blindage elle serait
    # envahie a un rayon proche de 11
    assert r_neck < 5.0


def test_hilpert_is_never_more_permissive_than_hazlett(ink_bottle):
    """Hilpert impose en plus un chemin continu pour le *centre* de la boule."""
    pore, _neck, _bulb = ink_bottle
    radii = np.arange(4.0, 0.4, -0.25)
    a = ma.network.drainage(
        pore, face=0, method="hazlett", radii=radii, restrict="inlet"
    ).filling_radius
    b = ma.network.drainage(
        pore, face=0, method="hilpert", radii=radii, restrict="inlet"
    ).filling_radius
    assert (b[pore] <= a[pore] + 1e-6).all()


def test_saturation_increases_as_radius_decreases(ink_bottle):
    pore, _n, _b = ink_bottle
    res = ma.network.drainage(pore, face=0, step=0.25, min_radius=0.5, restrict="inlet")
    sat = res.curve["saturation"].to_numpy()
    assert (np.diff(sat) >= -1e-9).all()
    assert sat[-1] <= 1.0 + 1e-9


def test_spanning_restriction_drops_isolated_porosity(tube):
    """Une bulle fermee n'est pas de la porosite ouverte : iMorph l'ecartait."""
    pore = tube.copy()
    _z, y, x = np.ogrid[: pore.shape[0], : pore.shape[1], : pore.shape[2]]
    blob = ((_z - 24) ** 2 + (y - 4) ** 2 + (x - 4) ** 2) <= 3.0**2
    pore |= blob
    res = ma.network.drainage(pore, face=0, restrict="spanning", step=1.0)
    assert (res.filling_radius[blob] == -1).all()
    assert (res.filling_radius[tube] > 0).all()


def test_unknown_method_and_restrict_are_refused(tube):
    with pytest.raises(ValueError, match="method"):
        ma.network.drainage(tube, method="washburn")
    with pytest.raises(ValueError, match="restrict"):
        ma.network.drainage(tube, restrict="somewhere")


def test_face_slab_covers_the_six_faces():
    shape = (5, 6, 7)
    for face in range(6):
        sl = ma.network.face_slab(shape, face)
        sub = np.zeros(shape, dtype=bool)
        sub[sl] = True
        assert sub.sum() == np.prod(shape) // shape[face // 2]
    with pytest.raises(ValueError):
        ma.network.face_slab(shape, 6)


# --------------------------------------------------------------------------
# percolation d'invasion
# --------------------------------------------------------------------------
def test_invasion_fills_the_connected_network(network_tables):
    t = network_tables
    inlet = ma.network.face_cells(t["labels"], 0)
    res = ma.network.invasion_percolation(t["cells"], t["throats"], inlet=inlet)
    assert res.n_invaded >= 0.9 * len(t["cells"])
    assert 0.0 < res.final_saturation <= 1.0


def test_filling_radius_is_a_decreasing_staircase(network_tables):
    """La pression ne fait que monter : le rayon enregistre ne remonte jamais.

    C'est ce que garantissait la boucle `r_current -= 0.2` d'iMorph et ce que
    reproduit ici le `min` avec le niveau courant.
    """
    t = network_tables
    inlet = ma.network.face_cells(t["labels"], 0)
    res = ma.network.invasion_percolation(t["cells"], t["throats"], inlet=inlet)
    inv = res.cells[res.cells["invaded"]].sort_values("order")
    r = inv["filling_radius"].to_numpy()
    assert (np.diff(r) <= 1e-9).all()


def test_trapping_only_removes_cells(network_tables):
    t = network_tables
    inlet = ma.network.face_cells(t["labels"], 0)
    outlet = ma.network.face_cells(t["labels"], 1)
    free = ma.network.invasion_percolation(t["cells"], t["throats"], inlet=inlet)
    trap = ma.network.invasion_percolation(
        t["cells"], t["throats"], inlet=inlet, outlet=outlet, trapping=True
    )
    assert trap.n_invaded <= free.n_invaded
    assert trap.final_saturation <= free.final_saturation
    assert (trap.cells["trapped"] & trap.cells["invaded"]).sum() == 0


def test_trapping_needs_an_outlet(network_tables):
    t = network_tables
    inlet = ma.network.face_cells(t["labels"], 0)
    with pytest.raises(ValueError, match="outlet"):
        ma.network.invasion_percolation(t["cells"], t["throats"], inlet=inlet, trapping=True)


def test_deformable_throats_ease_the_invasion(network_tables):
    """Des cols deformables laissent passer a plus faible pression."""
    t = network_tables
    inlet = ma.network.face_cells(t["labels"], 0)
    rigid = ma.network.invasion_percolation(t["cells"], t["throats"], inlet=inlet)
    soft = ma.network.invasion_percolation(
        t["cells"], t["throats"], inlet=inlet, deformation_rate=2.0
    )
    assert np.nanmin(soft.cells["filling_radius"]) > np.nanmin(rigid.cells["filling_radius"])
    assert soft.n_invaded >= rigid.n_invaded


def test_deformed_radius_is_identity_at_rate_one():
    r = np.array([1.0, 3.0, 7.5])
    out = ma.network.deformed_throat_radius(r, deformation_rate=1.0, r_max=10.0)
    assert np.allclose(out, r)


def test_deformed_radius_is_monotone_and_larger():
    r = np.linspace(0.5, 10.0, 25)
    out = ma.network.deformed_throat_radius(r, deformation_rate=1.8, r_max=10.0)
    assert (out >= r - 1e-9).all()
    assert (np.diff(out) > 0).all()


def test_deformation_rate_below_one_is_refused():
    with pytest.raises(ValueError, match="deformation_rate"):
        ma.network.deformed_throat_radius(2.0, deformation_rate=0.5, r_max=5.0)


def test_invalid_inlet_is_reported(network_tables):
    t = network_tables
    with pytest.raises(ValueError, match="injection"):
        ma.network.invasion_percolation(t["cells"], t["throats"], inlet=[10**6])


def test_face_cells_are_on_the_face(network_tables):
    lab = t_lab = network_tables["labels"]
    got = ma.network.face_cells(lab, 2)
    assert set(got.tolist()) <= set(np.unique(t_lab[t_lab > 0]).tolist())
    assert set(got.tolist()) == set(np.unique(lab[:, 0, :][lab[:, 0, :] > 0]).tolist())


def test_filling_map_paints_the_cells(network_tables):
    t = network_tables
    inlet = ma.network.face_cells(t["labels"], 0)
    res = ma.network.invasion_percolation(t["cells"], t["throats"], inlet=inlet)
    m = ma.network.filling_map(t["labels"], res)
    arr = np.asarray(m)
    assert arr.shape == t["labels"].shape
    assert (arr[t["labels"] == 0] == 0).all()
    lab0 = int(res.cells.index[res.cells["invaded"]][0])
    vals = np.unique(arr[t["labels"] == lab0])
    assert len(vals) == 1


def test_network_text_export_roundtrips(network_tables, tmp_path):
    t = network_tables
    p = ma.network.save_network_text(t["cells"], t["throats"], tmp_path / "net.txt")
    text = p.read_text()
    assert "Number Of Pores :" in text
    assert f"\n{len(t['cells'])}\n" in text
    assert "Throat Linking" in text
