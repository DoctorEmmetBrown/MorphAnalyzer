"""L'interface web : rendu de coupes, API, execution de pipeline.

Aucun navigateur n'est necessaire : on teste le rendu comme une fonction, et
l'API avec le client de test de FastAPI.
"""

import io
import time

import numpy as np
import pandas as pd
import pytest

import morphanalyzer as ma
from morphanalyzer.project import Project
from morphanalyzer.webapp import render as R
from morphanalyzer.webapp.jobs import JobRunner, unpack

fastapi = pytest.importorskip("fastapi", reason="extra 'web' non installe")
from fastapi.testclient import TestClient  # noqa: E402

from morphanalyzer.webapp.server import create_app  # noqa: E402


# ──────────────────────────── rendu ────────────────────────────────
def test_ramps_are_single_hue_and_monotone():
    """Une rampe sequentielle doit s'assombrir sans jamais repartir en arriere."""
    for name in R.RAMPS:
        lut = R.colormap(name)
        lum = lut.astype(float) @ np.array([0.2126, 0.7152, 0.0722])
        assert lut.shape == (256, 3)
        assert np.all(np.diff(lum) <= 0.6), f"{name} n'est pas monotone"
        assert lum[0] > lum[-1], f"{name} ne va pas du clair au fonce"


def test_unknown_ramp_lists_the_others():
    with pytest.raises(ValueError, match="arc-en-ciel|rampe inconnue"):
        R.colormap("rainbow")


def test_label_colors_are_stable_and_distinct():
    a, b = R.label_colors(24), R.label_colors(24)
    assert np.array_equal(a, b)  # deterministe d'une session a l'autre
    assert tuple(a[0]) == (0, 0, 0)  # 0 = fond
    uniques = {tuple(c) for c in a[1:]}
    assert len(uniques) == 24


def test_binary_layer_is_transparent_outside():
    m = np.zeros((4, 8, 8), dtype=bool)
    m[:, 2:6, 2:6] = True
    rgba = R.render_slice([R.LayerView(m, kind="binary", color="#eb6834")], axis=0, index=0)
    assert rgba.shape == (8, 8, 4)
    assert tuple(rgba[3, 3, :3]) == (235, 104, 52)
    # hors du masque, c'est le fond qui reste visible
    assert tuple(rgba[0, 0, :3]) != (235, 104, 52)


def test_overlay_respects_alpha():
    base = np.ones((2, 4, 4), dtype=bool)
    top = np.ones((2, 4, 4), dtype=bool)
    opaque = R.render_slice(
        [
            R.LayerView(base, kind="binary", color="#000000"),
            R.LayerView(top, kind="binary", color="#ffffff", alpha=1.0),
        ],
        index=0,
    )
    half = R.render_slice(
        [
            R.LayerView(base, kind="binary", color="#000000"),
            R.LayerView(top, kind="binary", color="#ffffff", alpha=0.5),
        ],
        index=0,
    )
    assert tuple(opaque[0, 0, :3]) == (255, 255, 255)
    assert 100 < half[0, 0, 0] < 160


def test_scalar_windowing():
    a = np.linspace(0, 10, 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    full = R.render_slice([R.LayerView(a, kind="scalar", vmin=0, vmax=10)], index=0)
    tight = R.render_slice([R.LayerView(a, kind="scalar", vmin=0, vmax=1)], index=0)
    # en resserrant la fenetre, tout sature vers le fonce
    assert tight[..., 2].mean() < full[..., 2].mean()


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="formes differentes"):
        R.render_slice(
            [
                R.LayerView(np.zeros((4, 4, 4), dtype=bool), kind="binary"),
                R.LayerView(np.zeros((4, 4, 5), dtype=bool), kind="binary"),
            ],
            index=0,
        )


def test_png_encoders_agree():
    """L'encodeur de secours doit produire la meme image que Pillow."""
    rgba = R.render_slice(
        [
            R.LayerView(
                np.random.default_rng(0).random((3, 12, 9)).astype(np.float32), kind="scalar"
            )
        ],
        index=1,
    )
    from PIL import Image

    a = np.array(Image.open(io.BytesIO(R.png_bytes(rgba))))
    b = np.array(Image.open(io.BytesIO(R._png_fallback(rgba))))
    assert np.array_equal(a, b)


def test_large_slices_are_decimated():
    big = np.zeros((2, 4000, 4000), dtype=bool)
    rgba = R.render_slice([R.LayerView(big, kind="binary")], index=0, max_size=1000)
    assert max(rgba.shape[:2]) <= 1000


# ──────────────────────────── projet de test ───────────────────────
@pytest.fixture(scope="module")
def served(tmp_path_factory):
    path = tmp_path_factory.mktemp("web") / "projet"
    vol = ma.phantoms.voronoi_foam(shape=(32,) * 3, n_cells=4, strut=3.0, min_seed_gap=14.0, seed=0)
    proj = Project.create(path, volume=ma.Volume(vol.fluid, voxel_size=2.0), name="projet")
    proj.add_layer("distance", ma.distance.distance_transform(proj.layer("volume")))
    proj.add_table(
        "courbe", pd.DataFrame({"rayon": [3.0, 2.0, 1.0], "saturation": [0.1, 0.6, 1.0]})
    )
    return TestClient(create_app(path)), proj


def test_project_route_describes_everything(served):
    client, _ = served
    p = client.get("/api/project").json()
    assert p["name"] == "projet"
    assert p["shape"] == [32, 32, 32]
    assert p["voxel_size"] == [2.0, 2.0, 2.0]
    assert {layer["name"] for layer in p["layers"]} == {"volume", "distance"}
    assert p["tables"][0]["name"] == "courbe"
    assert "blue" in p["ramps"]


def test_slice_route_returns_a_png_of_the_right_size(served):
    client, _ = served
    spec = '{"axis":0,"index":16,"layers":[{"name":"distance","ramp":"blue"}]}'
    r = client.get("/api/slice.png", params={"spec": spec})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    from PIL import Image

    assert Image.open(io.BytesIO(r.content)).size == (32, 32)


def test_slice_route_rejects_nonsense(served):
    client, _ = served
    assert client.get("/api/slice.png", params={"spec": "pas du json"}).status_code == 400
    assert (
        client.get("/api/slice.png", params={"spec": '{"layers":[{"name":"absent"}]}'}).status_code
        == 404
    )


def test_value_route_reads_one_voxel(served):
    client, proj = served
    r = client.get("/api/value", params={"z": 16, "y": 16, "x": 16}).json()
    assert set(r["values"]) == {"volume", "distance"}
    expected = float(np.asarray(proj.layer("distance"))[16, 16, 16])
    assert r["values"]["distance"]["value"] == pytest.approx(expected, abs=1e-5)


def test_histogram_route(served):
    client, _ = served
    h = client.get("/api/histogram", params={"layer": "distance", "bins": 16}).json()
    assert len(h["counts"]) == 16
    assert len(h["edges"]) == 17
    assert client.get("/api/histogram", params={"layer": "absent"}).status_code == 404


def test_table_routes(served):
    client, _ = served
    t = client.get("/api/table/courbe").json()
    assert t["columns"] == ["rayon", "saturation"]
    assert t["rows"][0]["rayon"] == 3.0
    csv = client.get("/api/table/courbe.csv")
    assert csv.status_code == 200
    assert client.get("/api/table/absente").status_code == 404


def test_steps_route_exposes_parameters(served):
    client, _ = served
    steps = client.get("/api/steps").json()
    names = {s["name"] for s in steps}
    assert {"distance_transform", "aperture_map", "drainage", "watershed_cells"} <= names
    aperture = next(s for s in steps if s["name"] == "aperture_map")
    assert any(p["name"] == "n_radii" and p["default"] == 32 for p in aperture["parameters"])
    # le volume d'entree n'est pas un reglage
    assert all(p["name"] != "mask" for p in aperture["parameters"])


def test_run_route_executes_and_persists(served):
    client, proj = served
    job = client.post(
        "/api/run",
        json={
            "steps": [{"step": "porosity", "params": {"out": "porosite"}}],
            "input_layer": "volume",
        },
    ).json()
    for _ in range(100):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert state["status"] == "done", state.get("error")
    assert 0.0 < state["results"]["porosite"] < 1.0
    assert any(h["step"] == "porosity" for h in client.get("/api/project").json()["history"])


def test_run_route_refuses_an_empty_queue(served):
    client, _ = served
    assert client.post("/api/run", json={"steps": []}).status_code == 400


def test_pipeline_export_is_replayable(served):
    client, _ = served
    text = client.get("/api/pipeline.yaml").text
    assert "porosity" in text


def test_unknown_job_is_reported(served):
    client, _ = served
    assert client.get("/api/jobs/inexistant").status_code == 404


# ──────────────────────────── taches ───────────────────────────────
def test_layer_references_are_resolved(tmp_path):
    vol = ma.phantoms.voronoi_foam(shape=(32,) * 3, n_cells=4, strut=3.0, min_seed_gap=14.0, seed=0)
    proj = Project.create(tmp_path / "p", volume=ma.Volume(vol.fluid, voxel_size=1.0))
    job = JobRunner(proj).run_sync(
        [
            {"step": "distance_transform", "params": {"out": "distance"}},
            {
                "step": "cell_markers",
                "params": {
                    "out": "marqueurs",
                    "distance": "@distance",
                    "fill_ratio": 0.5,
                    "min_radius": 2.0,
                },
            },
        ]
    )
    assert job.status == "done", job.error
    assert set(proj.layers) >= {"volume", "distance", "marqueurs"}


def test_a_scalar_step_does_not_consume_the_volume(tmp_path):
    """Une porosite au milieu d'une chaine ne doit pas casser la suite."""
    vol = ma.phantoms.voronoi_foam(shape=(32,) * 3, n_cells=4, strut=3.0, min_seed_gap=14.0, seed=0)
    proj = Project.create(tmp_path / "p", volume=ma.Volume(vol.fluid, voxel_size=1.0))
    job = JobRunner(proj).run_sync(
        [
            {"step": "porosity", "params": {"out": "porosite"}},
            {"step": "distance_transform", "params": {"out": "distance"}},
        ]
    )
    assert job.status == "done", job.error
    assert "distance" in proj.layers


def test_errors_are_reported_not_raised(tmp_path):
    vol = ma.phantoms.sphere(shape=(16,) * 3, radius=5.0)
    proj = Project.create(tmp_path / "p", volume=vol)
    job = JobRunner(proj).run_sync([{"step": "aperture_map", "params": {"n_radii": "beaucoup"}}])
    assert job.status == "error"
    assert job.error
    assert any("traceback" in entry for entry in job.log)


def test_composite_results_are_unpacked():
    """Un drainage doit donner une carte ET une courbe, pas une chaine de caracteres."""
    tube = np.zeros((24, 24, 24), dtype=bool)
    tube[:, 8:16, 8:16] = True
    res = ma.network.drainage(tube, face=0, step=1.0, min_radius=1.0)
    parts = unpack(res, "drainage")
    kinds = {kind for kind, _, _ in parts}
    assert kinds == {"layer", "table"}
    assert [name for kind, name, _ in parts if kind == "table"] == ["drainage_courbe"]


def test_unpack_falls_back_to_a_value():
    assert unpack(0.42, "x") == [("value", "x", 0.42)]


# ──────────────────────────── etancheite ───────────────────────────
def test_importing_the_core_does_not_pull_the_server():
    """Le noyau ne doit connaitre ni FastAPI ni le serveur."""
    import subprocess
    import sys

    code = (
        "import sys, morphanalyzer as ma, morphanalyzer.granulometry, morphanalyzer.network;"
        "bad=[m for m in sys.modules if m.split('.')[0] in {'fastapi','uvicorn','starlette'}"
        " or m == 'morphanalyzer.webapp'];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout
