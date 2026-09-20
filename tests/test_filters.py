import numpy as np
import pytest

import morphanalyzer as ma
from morphanalyzer.core import Volume


def _grey_from(vol, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.where(vol.solid, 200.0, 40.0)
    if noise:
        g = g + rng.normal(0, noise, g.shape)
    return Volume(g.astype(np.float32), voxel_size=vol.voxel_size)


def test_otsu_recovers_the_binary(pack):
    grey = _grey_from(pack, noise=8.0)
    out, t = ma.filters.threshold_otsu(grey, return_threshold=True)
    assert 40 < t < 200
    agreement = (out.solid == pack.solid).mean()
    assert agreement > 0.99


def test_otsu_records_its_threshold(pack):
    out = ma.filters.threshold_otsu(_grey_from(pack))
    assert out.meta["threshold_method"] == "otsu"
    assert isinstance(out.meta["threshold"], float)


def test_hysteresis_is_between_the_two_plain_thresholds(pack):
    grey = _grey_from(pack, noise=15.0)
    low, high = 100.0, 160.0
    n_low = ma.filters.threshold_value(grey, low).solid.sum()
    n_high = ma.filters.threshold_value(grey, high).solid.sum()
    n_hyst = ma.filters.threshold_hysteresis(grey, low, high).solid.sum()
    assert n_high <= n_hyst <= n_low


def test_median_removes_salt_and_pepper():
    rng = np.random.default_rng(0)
    clean = np.zeros((24, 24, 24), dtype=np.uint8)
    clean[8:16, 8:16, 8:16] = 255
    noisy = clean.copy()
    idx = rng.integers(0, 24, size=(3, 200))
    noisy[tuple(idx)] = 255
    out = ma.filters.median(noisy, radius=1)
    assert (
        np.abs(out.astype(int) - clean.astype(int)).sum()
        < np.abs(noisy.astype(int) - clean.astype(int)).sum()
    )


def test_keep_largest_component_drops_islands():
    solid = np.zeros((32, 32, 32), dtype=bool)
    solid[4:20, 4:20, 4:20] = True  # gros bloc
    solid[28, 28, 28] = True  # ilot de bruit
    out = ma.filters.keep_largest_component(solid)
    assert out.sum() == 16**3
    assert not out[28, 28, 28]


def test_fill_holes_closes_internal_cavity():
    solid = np.ones((24, 24, 24), dtype=bool)
    solid[8:14, 8:14, 8:14] = False
    assert ma.filters.fill_holes(solid).all()


def test_erode_dilate_are_inverse_on_large_object():
    solid = np.zeros((32, 32, 32), dtype=bool)
    solid[8:24, 8:24, 8:24] = True
    back = ma.filters.dilate(ma.filters.erode(solid, 1.0), 1.0)
    assert back.sum() == pytest.approx(solid.sum(), rel=0.02)
