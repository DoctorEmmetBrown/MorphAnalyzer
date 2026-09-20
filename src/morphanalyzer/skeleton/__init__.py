"""Squelettisation et graphe du solide.

Deux familles distinctes. L'amincissement homotopique ordonne par la distance
(DOHT) est couvert par skimage.morphology.skeletonize (Lee 1994, meme LUT
d'Euler) et par kimimaro. La squelettisation par **loi de Plateau** n'a pas
d'equivalent : elle propage les labels de cellules dans le solide, puis declare
noeud tout voxel voisin de 4 labels distincts et brin tout voxel voisin de 3.

Phase de portage : 5.
Origine iMorph : Thread/Skeleton/thin3D.cpp (classe Doht, 3 680 l.) ; Thread/Granulometry/graph3D.cpp (loi de Plateau)

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = ["skeletonize", "plateau_skeleton", "skeleton_graph", "prune", "medial_axis_flux"]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.skeleton.{name} arrive en phase 5. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
