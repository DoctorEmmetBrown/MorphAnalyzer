"""Granulometrie : carte d'ouverture locale, boules maximales, marqueurs.

Trois sorties, comme iMorph : la carte d'ouverture (rayon de la plus grande
boule incluse contenant le voxel), l'image d'identifiants de boules (~ ouverture
ultime) et l'histogramme des boules quasi entieres qui fournit les marqueurs de
la segmentation. Seule la premiere a un equivalent direct
(porespy.filters.local_thickness) ; les deux autres sont a ecrire.

Phase de portage : 3.
Origine iMorph : Thread/Granulometry/morphology.cpp : calc_Aperture_Map3DFAH*

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = ["aperture_map", "pore_size_distribution", "maximal_balls", "cell_markers"]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.granulometry.{name} arrive en phase 3. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
