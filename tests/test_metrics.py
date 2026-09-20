import numpy as np
import pytest

import morphanalyzer as ma
from morphanalyzer.core import Roi


def test_porosity_on_sphere(sphere):
    assert ma.metrics.porosity(sphere) == pytest.approx(sphere.meta["truth"]["porosity"], abs=2e-3)


def test_porosity_with_mask_changes_denominator(sphere):
    roi = Roi(cylinder=True, axis=0, margin=2)
    m = roi.mask(sphere.shape)
    p_full = ma.metrics.porosity(sphere)
    p_roi = ma.metrics.porosity(sphere, mask=m)
    # le cylindre retire des coins vides : la porosite mesuree baisse
    assert p_roi < p_full


def test_porosity_per_slice_shape(pack):
    p = ma.metrics.porosity_per_slice(pack, axis=0)
    assert p.shape == (pack.shape[0],)
    assert np.all((p >= 0) & (p <= 1))
    assert p.mean() == pytest.approx(ma.metrics.porosity(pack), abs=1e-9)


def test_specific_surface_binary_overestimates(sphere):
    """Marching cubes sur un masque binaire surestime l'aire, de ~9 % ici.

    Ce n'est pas un defaut d'implementation : l'iso-surface a 0.5 d'un masque
    est un escalier, et l'interpolation lineaire de marching cubes ne peut pas
    retrouver une interface qui n'est plus dans les donnees. Le test verrouille
    l'ordre de grandeur du biais pour qu'une regression future se voie.
    """
    truth = sphere.meta["truth"]
    ratio = ma.metrics.specific_surface(sphere) / truth["specific_surface"]
    assert 1.03 < ratio < 1.15


def test_specific_surface_on_grey_field_is_accurate(sphere):
    """Sur le champ continu, marching cubes retrouve l'aire au 1 % pres.

    Consequence directe pour le portage : mesurer la surface specifique sur les
    niveaux de gris (ou une distance signee) et non sur le masque binaire. Le
    gain est d'un ordre de grandeur, pour un cout nul.
    """
    truth = sphere.meta["truth"]
    sv = ma.metrics.specific_surface(sphere, grey=truth["signed_distance"], level=0.0)
    assert sv == pytest.approx(truth["specific_surface"], rel=0.01)


def test_specific_surface_scales_with_voxel_size():
    a = ma.phantoms.sphere(shape=(48,) * 3, radius=14.0, voxel_size=1.0)
    b = ma.phantoms.sphere(shape=(48,) * 3, radius=14.0, voxel_size=2.0)
    # Sv est une longueur^-1 : doubler la taille du voxel la divise par deux
    assert ma.metrics.specific_surface(b) == pytest.approx(
        ma.metrics.specific_surface(a) / 2.0, rel=1e-6
    )


def test_open_porosity_on_closed_cavity():
    """Une cavite fermee ne doit pas compter dans la porosite ouverte."""
    solid = np.ones((32, 32, 32), dtype=bool)
    solid[4:12, 4:12, 4:12] = False  # cavite isolee
    solid[:, 20:26, 20:26] = False  # canal traversant
    frac, biggest = ma.metrics.open_porosity(solid)
    total = ma.metrics.porosity(solid)
    assert frac < total
    assert biggest.sum() == (~solid)[:, 20:26, 20:26].size


def test_rev_dispersion_decreases_with_box_size(pack):
    df = ma.metrics.representative_volume(pack.solid, sizes=(4, 8, 16), n_boxes=60, seed=0)
    assert list(df["half_size"]) == [4, 8, 16]
    assert df["std"].iloc[-1] < df["std"].iloc[0], "la dispersion devrait decroitre"
