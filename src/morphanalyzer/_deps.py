"""Gestion des dependances optionnelles.

Le noyau n'exige que numpy / scipy / scikit-image / tifffile / pandas.
Tout le reste est un extra, importe paresseusement et signale clairement
quand il manque.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# module importable -> extra pip qui le fournit
_EXTRA_OF: dict[str, str] = {
    "numba": "fast",
    "cc3d": "fast",
    "edt": "fast",
    "skfmm": "fmm",
    "zarr": "bigdata",
    "dask": "bigdata",
    "trimesh": "mesh",
    "fast_simplification": "mesh",
    "meshio": "mesh",
    "porespy": "network",
    "openpnm": "network",
    "networkx": "network",
    "matplotlib": "viz",
    "napari": "viz",
    "pyvista": "viz",
    "typer": "cli",
    "yaml": "cli",
    "fastapi": "web",
    "uvicorn": "web",
    "starlette": "web",
    "PIL": "web",
    "cupy": "gpu",
}


class MissingDependency(ImportError):
    """Une dependance optionnelle est requise mais absente."""


def require(name: str, *, reason: str = "") -> ModuleType:
    """Importe `name` ou leve une erreur qui dit quoi installer.

    >>> require("numba")            # doctest: +SKIP
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - depend de l'environnement
        extra = _EXTRA_OF.get(name)
        hint = f'pip install "morphanalyzer[{extra}]"' if extra else f"pip install {name}"
        msg = f"{name!r} est requis"
        if reason:
            msg += f" pour {reason}"
        raise MissingDependency(f"{msg}. Installer avec : {hint}") from exc


def have(name: str) -> bool:
    """True si le module optionnel est disponible, sans lever."""
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True
