"""Maillage de surface, export et decimation.

marching_cubes de scikit-image remplace DrawIsoSurface / Polygonise, trimesh
remplace les exports maison (STL binaire et ascii, OBJ, POV, vox) et la
decimation progressive de Hoppe.

Phase de portage : 1.
Origine iMorph : Thread/Mesh/mesh.cpp (3 254 l.), model.cpp ; Thread/Simplificator/progmesh.cpp

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = ["surface_mesh", "save_mesh", "decimate", "mesh_volume", "mesh_area"]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.mesh.{name} arrive en phase 1. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
