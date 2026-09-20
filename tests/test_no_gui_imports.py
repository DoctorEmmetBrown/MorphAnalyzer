"""Le noyau doit rester utilisable sans ecran.

C'est la contrainte d'architecture centrale du projet : on doit pouvoir appeler
les routines depuis un script ou un notebook, sur une machine de calcul sans
serveur graphique. Ce test echoue si quelqu'un glisse un import de napari, Qt
ou matplotlib dans le noyau.
"""

import subprocess
import sys

FORBIDDEN = (
    "napari",
    "qtpy",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "matplotlib",
    "pyvista",
    "vtk",
    "vispy",
    "magicgui",
)

CORE_MODULES = [
    "morphanalyzer",
    "morphanalyzer.core",
    "morphanalyzer.io",
    "morphanalyzer.filters",
    "morphanalyzer.metrics",
    "morphanalyzer.phantoms",
    "morphanalyzer.pipeline",
]


def test_core_imports_pull_no_gui():
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in CORE_MODULES)
        + f"bad=[m for m in {FORBIDDEN!r} if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    leaked = out.stdout.strip()
    assert not leaked, f"le noyau a importe des modules graphiques : {leaked}"


def test_core_works_without_display(monkeypatch):
    """Aucune routine du noyau ne doit dependre de DISPLAY."""
    monkeypatch.delenv("DISPLAY", raising=False)
    import morphanalyzer as ma

    vol = ma.phantoms.sphere(shape=(32, 32, 32), radius=8.0)
    assert 0.0 < ma.metrics.porosity(vol.solid) < 1.0


def test_viz_is_not_imported_by_default():
    import subprocess

    code = "import morphanalyzer, sys; print('morphanalyzer.viz' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
