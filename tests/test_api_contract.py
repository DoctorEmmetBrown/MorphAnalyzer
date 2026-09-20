"""Contrat d'API et tableau de bord du portage.

Deux exigences. Les fonctions annoncees dans `__all__` doivent exister, pour
qu'un notebook ecrit aujourd'hui ne casse pas demain. Et celles qui ne sont pas
encore portees doivent echouer clairement, en nommant leur phase, plutot que de
lever une `AttributeError` obscure.
"""

import numpy as np
import pytest

import morphanalyzer as ma

#: Ce qui est livre et doit etre appelable.
IMPLEMENTED = {
    "io": ["read_raw", "write_raw", "read_stack", "write_stack"],
    "filters": ["threshold_otsu", "median", "erode", "keep_largest_component"],
    "metrics": ["porosity", "specific_surface", "representative_volume"],
    "distance": ["distance_transform", "nearest_seed_propagation", "geodesic_ball"],
    "granulometry": ["aperture_map", "pore_size_distribution", "maximal_balls", "cell_markers"],
    "skeleton": ["skeletonize", "distance_ridge", "plateau_skeleton", "skeleton_graph"],
    "shape": ["local_shape_tensor", "classify_solid", "elongation_ratios", "strut_orientation"],
    "segmentation": [
        "watershed_cells",
        "cell_morphometry",
        "throats",
        "connectivity",
        "pore_network",
    ],
    "phantoms": ["sphere", "sphere_pack", "voronoi_foam", "sinusoidal_tube"],
}

#: Ce qui reste a porter : l'API est declaree, l'appel doit dire quelle phase.
PLANNED = {
    "distance": ["geodesic_distance", "travel_time", "label_propagation"],
    "skeleton": ["prune", "medial_axis_flux"],
    "tortuosity": ["point_tortuosity", "plane_tortuosity", "graph_tortuosity"],
    "mesh": ["surface_mesh", "save_mesh", "decimate"],
    "network": ["drainage", "invasion_percolation", "saturation_curve"],
    "radiative": ["ray_trace", "transmittance", "reflectance"],
    "cortical": ["radial_profile", "angular_profile"],
}


@pytest.mark.parametrize("pkg", sorted(IMPLEMENTED))
def test_implemented_api_is_callable(pkg):
    mod = getattr(ma, pkg)
    for name in IMPLEMENTED[pkg]:
        assert name in mod.__all__, f"{pkg}.{name} manque dans __all__"
        assert callable(getattr(mod, name)), f"{pkg}.{name} n'est pas appelable"


@pytest.mark.parametrize("pkg", sorted(PLANNED))
def test_planned_api_is_declared(pkg):
    mod = getattr(ma, pkg)
    for name in PLANNED[pkg]:
        assert name in mod.__all__, f"{pkg}.{name} manque dans __all__"


@pytest.mark.parametrize("pkg", sorted(PLANNED))
def test_planned_api_fails_with_its_phase(pkg):
    mod = getattr(ma, pkg)
    for name in PLANNED[pkg]:
        fn = getattr(mod, name, None)
        if fn is None:  # module entierement en attente : verifie via _todo
            with pytest.raises(NotImplementedError, match="phase"):
                mod._todo(name)
            continue
        with pytest.raises(NotImplementedError, match="phase"):
            fn()


def test_version_is_exposed():
    assert ma.__version__


def test_end_to_end_stays_headless():
    """Le parcours complet de la phase 5, sans ecran ni dependance optionnelle."""
    vol = ma.phantoms.voronoi_foam(shape=(48,) * 3, n_cells=4, strut=3.0, min_seed_gap=18.0, seed=0)
    aper = ma.granulometry.aperture_map(vol.solid, n_radii=8)
    st = ma.shape.local_shape_tensor(vol, aperture=aper, propagate=True)
    cls = ma.shape.classify_solid(st, vol.solid)
    assert set(np.unique(cls)).issubset({0, 1, 2, 3})
    assert (cls[vol.solid] > 0).all()
    assert (cls[~vol.solid] == 0).all()
