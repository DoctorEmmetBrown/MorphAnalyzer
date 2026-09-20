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
