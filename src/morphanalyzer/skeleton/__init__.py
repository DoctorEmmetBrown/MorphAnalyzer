"""Squelettisation et graphe du solide.

Phase de portage : 5 (partiellement livre).
Origine iMorph : `Thread/Skeleton/thin3D.cpp` (classe `Doht`, 3 680 lignes,
amincissement homotopique ordonne par la distance) et
`Thread/Granulometry/graph3D.cpp` (squelettisation par loi de Plateau).

`skeletonize` s'appuie sur `skimage.morphology.skeletonize`, qui implemente
l'amincissement de Lee (1994) : meme famille que le DOHT d'iMorph, meme table
de caracteristique d'Euler, meme test de point simple. Les 3 680 lignes se
ramenent a un appel.

Restent a porter : la loi de Plateau (sans equivalent), le graphe de squelette,
l'elagage parametre et l'axe median par flux.
"""

from morphanalyzer.skeleton.thin import distance_ridge, skeletonize

__all__ = [
    "skeletonize",
    "distance_ridge",
    "plateau_skeleton",
    "skeleton_graph",
    "prune",
    "medial_axis_flux",
]


def plateau_skeleton(*args, **kwargs):
    raise NotImplementedError(
        "morphanalyzer.skeleton.plateau_skeleton arrive en phase 5 "
        "(propagation des cellules dans le solide, >= 4 labels voisins => noeud)."
    )


def skeleton_graph(*args, **kwargs):
    raise NotImplementedError("morphanalyzer.skeleton.skeleton_graph arrive en phase 5.")


def prune(*args, **kwargs):
    raise NotImplementedError("morphanalyzer.skeleton.prune arrive en phase 5.")


def medial_axis_flux(*args, **kwargs):
    raise NotImplementedError("morphanalyzer.skeleton.medial_axis_flux arrive en phase 5.")
