import pytest

import morphanalyzer as ma


@pytest.fixture(scope="session")
def sphere():
    return ma.phantoms.sphere(shape=(64, 64, 64), radius=20.0)


@pytest.fixture(scope="session")
def pack():
    return ma.phantoms.sphere_pack(shape=(96, 96, 96), radius=10.0, n=12, seed=1)


@pytest.fixture(scope="session")
def foam():
    return ma.phantoms.voronoi_foam(
        shape=(128, 128, 128), n_cells=30, strut=4.0, min_seed_gap=16.0, seed=3
    )


@pytest.fixture(scope="session")
def foam_struts():
    """Mousse a brins elances : cellules larges, brins fins.

    Les proportions comptent pour la classification de forme. Une mousse dont
    les brins sont aussi epais que longs ne distingue pas un brin d'un noeud —
    ni pour l'algorithme, ni physiquement. Celle-ci a un rapport d'aspect
    d'environ 12:1, dans la gamme des mousses Recemat de la these
    (cellule 2 367 um, brin 240 um pour la 1116).
    """
    return ma.phantoms.voronoi_foam(
        shape=(128, 128, 128), n_cells=10, strut=3.0, min_seed_gap=45.0, seed=3
    )


@pytest.fixture(scope="session")
def foam_cells():
    """Mousse a nombreuses cellules, pour valider la segmentation.

    Plus de cellules et plus petites que `foam_struts` : on veut des cellules
    entierement incluses dans la boite, seules utilisables pour la morphometrie
    (convention de la these).
    """
    return ma.phantoms.voronoi_foam(
        shape=(128, 128, 128), n_cells=40, strut=3.0, min_seed_gap=20.0, seed=7
    )


@pytest.fixture(scope="session")
def segmented(foam_cells):
    """Chaine complete : distance -> marqueurs -> cellules."""
    import numpy as np

    fluid = ~foam_cells.solid
    dist = ma.distance.distance_transform(fluid)
    markers = ma.granulometry.cell_markers(fluid, distance=dist, fill_ratio=0.55)
    cells = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid))
    return {
        "vol": foam_cells,
        "fluid": fluid,
        "dist": dist,
        "markers": np.asarray(markers),
        "cells": cells,
    }
