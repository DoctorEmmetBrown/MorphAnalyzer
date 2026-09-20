"""Les modules pas encore portes doivent echouer clairement, pas mysterieusement."""

import pytest

import morphanalyzer as ma

PLANNED = {
    "distance": ["distance_transform", "geodesic_distance", "travel_time", "label_propagation"],
    "granulometry": ["aperture_map", "pore_size_distribution", "maximal_balls", "cell_markers"],
    "segmentation": ["watershed_cells", "cell_morphometry", "throats", "connectivity"],
    "skeleton": ["skeletonize", "plateau_skeleton", "skeleton_graph"],
    "shape": ["local_shape_tensor", "classify_solid", "strut_orientation"],
    "tortuosity": ["point_tortuosity", "plane_tortuosity", "graph_tortuosity"],
    "mesh": ["surface_mesh", "save_mesh", "decimate"],
    "network": ["drainage", "invasion_percolation", "saturation_curve"],
    "radiative": ["ray_trace", "transmittance", "reflectance"],
    "cortical": ["radial_profile", "angular_profile"],
}


@pytest.mark.parametrize("pkg", sorted(PLANNED))
def test_planned_api_is_declared(pkg):
    mod = getattr(ma, pkg)
    for name in PLANNED[pkg]:
        assert name in mod.__all__, f"{pkg}.{name} manque dans __all__"


@pytest.mark.parametrize("pkg", sorted(PLANNED))
def test_todo_message_is_actionable(pkg):
    mod = getattr(ma, pkg)
    with pytest.raises(NotImplementedError, match="phase"):
        mod._todo("une_fonction")


def test_version_is_exposed():
    assert ma.__version__
