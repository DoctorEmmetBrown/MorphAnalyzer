import numpy as np
import pytest

import morphanalyzer as ma
from morphanalyzer.core import Volume
from morphanalyzer.pipeline import Pipeline, available_steps, run_from_config


@pytest.fixture()
def grey(pack):
    return Volume(np.where(pack.solid, 200, 40).astype(np.uint8), voxel_size=pack.voxel_size)


def test_registry_is_populated():
    steps = available_steps()
    for expected in ("threshold_otsu", "porosity", "specific_surface", "keep_largest_component"):
        assert expected in steps


def test_unknown_step_is_rejected_early():
    with pytest.raises(KeyError, match="etape inconnue"):
        Pipeline().step("granulometrie_magique")


def test_pipeline_runs_headless(grey, pack):
    ctx = (
        Pipeline(verbose=False)
        .step("threshold_otsu")
        .step("porosity", out="phi")
        .step("specific_surface", out="Sv")
        .run(grey)
    )
    assert 0.0 < ctx["phi"] < 1.0
    assert ctx["phi"] == pytest.approx(ma.metrics.porosity(pack), abs=0.02)
    assert ctx["Sv"] > 0
    assert ctx["output"].dtype == bool


def test_keep_largest_on_disjoint_pack_keeps_one_sphere(pack):
    """Garde-fou : `keep_largest_component` est destine au bruit en ilots.

    Sur un milieu dont le solide n'est pas connexe — un empilement de spheres
    disjointes, des particules, un os trabeculaire fragmente — il detruit
    l'echantillon. iMorph l'appliquait par defaut ; ici c'est un choix explicite.
    """
    out = ma.filters.keep_largest_component(pack.solid)
    n = pack.meta["truth"]["n_spheres"]
    assert out.sum() == pytest.approx(pack.solid.sum() / n, rel=0.02)


def test_pipeline_from_dict(grey):
    cfg = {
        "verbose": False,
        "steps": ["threshold_otsu", {"porosity": {"out": "phi"}}],
    }
    ctx = run_from_config(cfg, grey)
    assert "phi" in ctx


def test_pipeline_from_json_file(grey, tmp_path):
    import json

    p = tmp_path / "pipe.json"
    p.write_text(
        json.dumps({"verbose": False, "steps": ["threshold_otsu", {"porosity": {"out": "phi"}}]})
    )
    ctx = run_from_config(p, grey)
    assert 0.0 < ctx["phi"] < 1.0


def test_shape_steps_are_registered():
    steps = available_steps()
    for expected in ("distance_transform", "aperture_map", "skeletonize", "shape_classification"):
        assert expected in steps


def test_full_shape_pipeline_from_dict(grey):
    """Chaine complete binarisation -> classification, pilotee par des donnees."""
    import numpy as np

    cfg = {
        "verbose": False,
        "steps": [
            "threshold_otsu",
            {"porosity": {"out": "phi"}},
            {"shape_classification": {"out": "classes", "expand_factor": 2.0}},
        ],
    }
    ctx = run_from_config(cfg, grey)
    assert 0.0 < ctx["phi"] < 1.0
    assert set(np.unique(ctx["classes"])).issubset({0, 1, 2, 3})
