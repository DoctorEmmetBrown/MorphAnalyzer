"""Analyse de l'os cortical : profils radiaux et angulaires.

Porosite par secteurs radiaux et angulaires, distributions d'ouverture
angulaire, connectivite, Voronoi 2D. L'essentiel du volume C++ etait de
l'interface QCustomPlot ; le calcul se ramene a de la geometrie en coordonnees
cylindriques sur des cartes deja produites par les autres modules.

Phase de portage : 9.
Origine iMorph : Thread/Cortical/ (corticalModuleTab{Porosity,AngularAper,Connectivity,Voronoi2D}.cpp)

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = [
    "radial_profile",
    "angular_profile",
    "angular_aperture",
    "cortical_connectivity",
    "voronoi_2d",
]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.cortical.{name} arrive en phase 9. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
