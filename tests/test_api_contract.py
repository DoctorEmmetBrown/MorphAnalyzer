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
    "phantoms": ["sphere", "sphere_pack", "voronoi_foam", "sinusoidal_tube", "cortical_tube"],
    "project": ["Project", "Layer"],
    "tortuosity": [
        "point_tortuosity",
        "plane_tortuosity",
        "directional_tortuosity",
        "poiseuille_tortuosity",
        "graph_tortuosity",
        "shortest_path",
    ],
    "mesh": ["surface_mesh", "save_mesh", "decimate", "mesh_volume", "mesh_area"],
    "network": [
        "drainage",
        "saturation_curve",
        "invasion_percolation",
        "capillary_pressure",
        "to_openpnm",
    ],
    "cortical": [
        "angular_profile",
        "angular_aperture",
        "radial_profile",
        "cortical_connectivity",
        "voronoi_2d",
    ],
}

#: Ce qui reste a porter : l'API est declaree, l'appel doit dire quelle phase.
PLANNED = {
    "skeleton": ["prune", "medial_axis_flux"],
}

#: Remplace par un equivalent : l'appel doit renvoyer vers ce qui le remplace.
SUPERSEDED = {
    "distance": ["label_propagation"],
}

#: Ecarte du portage : l'API reste declaree, l'appel doit le dire.
OUT_OF_SCOPE = {
    "radiative": ["ray_trace", "transmittance", "reflectance", "absorbance", "exchange_factors"],
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


@pytest.mark.parametrize("pkg", sorted(SUPERSEDED))
def test_superseded_api_points_to_its_replacement(pkg):
    mod = getattr(ma, pkg)
    for name in SUPERSEDED[pkg]:
        assert name in mod.__all__, f"{pkg}.{name} manque dans __all__"
        with pytest.raises(NotImplementedError, match="utiliser"):
            getattr(mod, name)()


@pytest.mark.parametrize("pkg", sorted(OUT_OF_SCOPE))
def test_out_of_scope_api_says_so(pkg):
    mod = getattr(ma, pkg)
    for name in OUT_OF_SCOPE[pkg]:
        assert name in mod.__all__, f"{pkg}.{name} manque dans __all__"
        with pytest.raises(NotImplementedError, match="hors perimetre"):
            mod._todo(name)


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


def test_project_is_exposed_at_the_top_level():
    """`Project` est du noyau : c'est le modele de donnees partage par la CLI,
    les notebooks et l'interface."""
    assert ma.Project is not None
    assert "Project" in ma.__all__


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
