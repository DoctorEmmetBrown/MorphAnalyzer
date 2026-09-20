"""Proprietes radiatives par lancer de rayons — **hors perimetre**.

Phase de portage : 8, ecartee a la demande d'Emmanuel Brun (septembre 2026).
Origine iMorph : `PhysicalModules/iMorph_Rad/` (`rayTracing.cpp`,
`exchangeFactor.cpp` — ce dernier n'etait d'ailleurs pas compile en 3.2).

L'API reste declaree pour que le contrat de nommage tienne si le besoin
revient. Si c'est le cas, l'approche retenue serait de conserver la physique
(modele de spectrophotometre a sphere integrante, reflexion speculaire, seuil
d'extinction a 1 % de l'energie incidente) et de changer le moteur : un BVH
Embree via trimesh traverse des millions de rayons par seconde et rend inutile
l'optimisation voxel-par-voxel d'iMorph, dont tout l'interet etait d'eviter le
test d'intersection contre tous les triangles. Le maillage necessaire est
produit par :mod:`morphanalyzer.mesh`.
"""

from __future__ import annotations

__all__ = ["ray_trace", "transmittance", "reflectance", "absorbance", "exchange_factors"]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.radiative.{name} est hors perimetre du portage "
        "(phase 8, ecartee a la demande). Voir docs/PORTING_MAP.md."
    )
