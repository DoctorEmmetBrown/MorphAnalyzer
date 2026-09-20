"""Proprietes radiatives par lancer de rayons.

La physique est conservee (modele de spectrophotometre a sphere integrante,
reflexion speculaire, seuil d'extinction a 1 % de l'energie incidente), le
moteur change : un BVH Embree via trimesh traverse des millions de rayons par
seconde et rend inutile l'optimisation voxel-par-voxel d'iMorph, dont tout
l'interet etait d'eviter le test d'intersection contre tous les triangles.

Phase de portage : 8.
Origine iMorph : PhysicalModules/iMorph_Rad/ (rayTracing.cpp, exchangeFactor.cpp)

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = ["ray_trace", "transmittance", "reflectance", "absorbance", "exchange_factors"]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.radiative.{name} arrive en phase 8. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
