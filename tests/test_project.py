"""Le dossier de projet : lisible sans le logiciel, et rejouable."""

import json

import numpy as np
import pandas as pd
import pytest

import morphanalyzer as ma
from morphanalyzer.project import MANIFEST_NAME, Project


@pytest.fixture
def small(tmp_path):
    vol = ma.phantoms.voronoi_foam(shape=(32,) * 3, n_cells=4, strut=3.0, min_seed_gap=14.0, seed=0)
    return Project.create(
        tmp_path / "essai",
        volume=ma.Volume(vol.fluid, voxel_size=7.46, unit="um"),
        name="essai",
    )


def test_layout_is_ordinary_files(small):
    """Rien de proprietaire : du JSON, des .npy, des .csv."""
    assert (small.path / MANIFEST_NAME).exists()
    assert (small.path / "layers" / "volume.npy").exists()
    manifest = json.loads((small.path / MANIFEST_NAME).read_text())
    assert manifest["unit"] == "um"
    assert manifest["voxel_size"] == [7.46, 7.46, 7.46]
    # le calque se relit sans morphanalyzer
    assert np.load(small.path / "layers" / "volume.npy").shape == (32, 32, 32)


def test_reopen_keeps_everything(small):
    small.add_layer("distance", ma.distance.distance_transform(small.layer("volume")))
    again = Project.open(small.path)
    assert again.name == "essai"
    assert set(again.layers) == {"volume", "distance"}
    assert again.voxel_size == (7.46, 7.46, 7.46)
    assert again.shape == (32, 32, 32)


def test_layers_are_memory_mapped(small):
    """Ouvrir un projet ne doit rien charger : c'est ce qui rend les gros volumes utilisables."""
    arr = small.layer("volume")
    assert isinstance(arr, np.memmap)


def test_kind_is_inferred_and_ranges_recorded(small):
    dist = ma.distance.distance_transform(small.layer("volume"))
    layer = small.add_layer("distance", dist)
    assert layer.kind == "scalar"
    assert layer.vmin == pytest.approx(float(dist.min()))
    assert layer.vmax == pytest.approx(float(dist.max()))

    lab = np.zeros((32, 32, 32), dtype=np.int32)
    lab[4:8, 4:8, 4:8] = 5
    assert small.add_layer("etiquettes", lab).kind == "labels"
    assert small.layers["etiquettes"].n_labels == 5
    assert small.layers["volume"].kind == "binary"


def test_tables_are_csv(small):
    frame = pd.DataFrame({"rayon": [1.0, 2.0], "saturation": [0.2, 0.9]})
    small.add_table("retention", frame, description="courbe")
    assert (small.path / "tables" / "retention.csv").exists()
    back = Project.open(small.path).table("retention")
    pd.testing.assert_frame_equal(back, frame)


def test_history_replays_as_a_pipeline(small):
    """Ce qui a ete fait a la souris doit redevenir un fichier de configuration."""
    small.log_step("distance_transform", {}, outputs=["distance"], duration=0.1)
    small.log_step(
        "cell_markers",
        {"fill_ratio": 0.55, "distance": "@distance"},
        outputs=["marqueurs"],
        duration=0.2,
    )
    config = small.to_pipeline_config()
    assert config["steps"][0] == {"distance_transform": {"out": "distance"}}
    assert config["steps"][1]["cell_markers"]["distance"] == "@distance"
    assert config["steps"][1]["cell_markers"]["out"] == "marqueurs"


def test_removing_a_layer_removes_its_file(small):
    small.add_layer("jetable", np.zeros((32, 32, 32), dtype=bool))
    path = small.path / "layers" / "jetable.npy"
    assert path.exists()
    small.remove_layer("jetable")
    assert not path.exists()
    assert "jetable" not in small.layers
    with pytest.raises(KeyError):
        small.remove_layer("jetable")


def test_bad_names_are_refused(small):
    for bad in ("../evade", "sous/dossier", ".cache"):
        with pytest.raises(ValueError, match="invalide"):
            small.add_layer(bad, np.zeros((2, 2, 2), dtype=bool))


def test_unknown_layer_names_the_alternatives(small):
    with pytest.raises(KeyError, match="volume"):
        small.layer("inexistant")


def test_creating_twice_is_refused(small):
    with pytest.raises(FileExistsError):
        Project.create(small.path)
    assert Project.create(small.path, exist_ok=True).name == "essai"


def test_opening_a_plain_directory_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="morphanalyzer.json"):
        Project.open(tmp_path)


def test_volume_carries_the_voxel_size(small):
    vol = small.volume()
    assert isinstance(vol, ma.Volume)
    assert vol.voxel_size == (7.46, 7.46, 7.46)
    assert vol.unit == "um"


def test_a_project_carries_its_own_gitignore(small):
    """Un projet se versionne : la recette et les tables, pas les gigaoctets."""
    text = (small.path / ".gitignore").read_text()
    assert "layers/" in text
    assert "morphanalyzer run" in text  # la recette suffit a les reconstruire
