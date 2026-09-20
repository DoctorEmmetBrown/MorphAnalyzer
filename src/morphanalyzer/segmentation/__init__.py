"""Segmentation des cellules et mesure des cols.

La variante a porter est le watershed a priorites reelles (tas binaire) avec
resolution des collisions par label majoritaire dans le voisinage deja traite :
c'est elle qui supprime les artefacts en marches d'escalier de la version de
Meyer quantifiee (these, fig. 3.4b vs 3.5b). skimage.segmentation.watershed ne
couvre pas ce cas.

Phase de portage : 4.
Origine iMorph : Thread/Granulometry/morphology.cpp : watershedBinarySearchTree ; cellsExtractionThread.cpp ; throatThread.cpp

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = ["watershed_cells", "cell_morphometry", "throats", "connectivity", "pore_network"]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.segmentation.{name} arrive en phase 4. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
