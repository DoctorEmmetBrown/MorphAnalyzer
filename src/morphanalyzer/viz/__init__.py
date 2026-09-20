"""Visualisation — **strictement optionnelle**.

Rien dans le noyau de morphanalyzer n'importe ce module, et ce module n'est
jamais importe automatiquement : `import morphanalyzer` ne tire ni napari, ni
matplotlib, ni Qt. Un test (`tests/test_no_gui_imports.py`) le verifie a chaque
execution de la suite.

    import morphanalyzer as ma
    from morphanalyzer import viz          # explicite, et seulement si voulu
    viz.show(vol, labels=cells)

Phase de portage : 11.
"""

from __future__ import annotations

__all__ = ["show", "plot_distribution", "plot_orientation_polar"]


def show(volume, **layers):  # pragma: no cover - necessite un affichage
    """Ouvre le volume dans napari, avec des calques nommes optionnels."""
    from morphanalyzer._deps import require

    napari = require("napari", reason="la visualisation 3D interactive")
    from morphanalyzer.core.volume import Volume, as_array

    viewer = napari.Viewer(ndisplay=3)
    scale = volume.voxel_size if isinstance(volume, Volume) else (1.0, 1.0, 1.0)
    viewer.add_image(as_array(volume).astype("float32"), name="volume", scale=scale)
    for name, layer in layers.items():
        arr = as_array(layer)
        if arr.dtype.kind in "iu":
            viewer.add_labels(arr, name=name, scale=scale)
        else:
            viewer.add_image(arr, name=name, scale=scale)
    return viewer


def plot_distribution(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("morphanalyzer.viz.plot_distribution arrive en phase 11.")


def plot_orientation_polar(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("morphanalyzer.viz.plot_orientation_polar arrive en phase 11.")
