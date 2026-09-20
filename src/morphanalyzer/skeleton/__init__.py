"""Squelettisation et graphe du solide.

Phase de portage : 5 (livree, sauf l'axe median par flux).
Origine iMorph : `Thread/Skeleton/thin3D.cpp` (classe `Doht`, amincissement
homotopique ordonne par la distance) et
`Thread/Granulometry/graph3D.cpp` (loi de Plateau).

Deux familles, a ne pas confondre.

`skeletonize` amincit l'objet en preservant sa topologie. C'est l'equivalent du
DOHT d'iMorph : `skimage.morphology.skeletonize` implemente l'amincissement de
Lee (1994), meme table de caracteristique d'Euler, meme test de point simple.
Attention, sa mise en oeuvre 3D rend un squelette **vide** sur certains objets
compacts — voir `distance_ridge`.

`plateau_skeleton` ne fait aucun amincissement : il propage les labels de
cellules dans le solide et lit les jonctions dans un voisinage 2x2x2, d'apres la
loi de Plateau. Il produit directement un graphe de noeuds et de brins
physiquement interpretable. Sans equivalent en bibliotheque.
"""

from morphanalyzer.skeleton.plateau import (
    PlateauSkeleton,
    plateau_skeleton,
    skeleton_graph,
)
from morphanalyzer.skeleton.thin import distance_ridge, skeletonize

__all__ = [
    "skeletonize",
    "distance_ridge",
    "plateau_skeleton",
    "skeleton_graph",
    "PlateauSkeleton",
    "prune",
    "medial_axis_flux",
]


def prune(*args, **kwargs):
    raise NotImplementedError(
        "morphanalyzer.skeleton.prune arrive en phase 5b (elagage parametre du "
        "squelette d'amincissement). Pour le squelette de Plateau, le debruitage "
        "est deja dans plateau_skeleton(merge_distance=...)."
    )


def medial_axis_flux(*args, **kwargs):
    raise NotImplementedError(
        "morphanalyzer.skeleton.medial_axis_flux arrive en phase 5b "
        "(axe median par flux moyen sortant, Doht::TH_DOHT_FLUX)."
    )
