"""Granulometrie : carte d'ouverture locale et distribution de tailles.

Phase de portage : 3 (partiellement livre).
Origine iMorph : `Thread/Granulometry/morphology.cpp::calc_Aperture_Map3DFAH*`.

La carte d'ouverture donne, en chaque voxel, le rayon de la plus grande boule
incluse dans la phase qui contient ce voxel. C'est l'entree de la classification
de forme (elle fixe le rayon de propagation) et de l'extraction des marqueurs.

Restent a porter (phase 3) : l'image d'identifiants de boules (~ ouverture
ultime) et l'histogramme des boules quasi entieres, qui n'ont pas d'equivalent
en bibliotheque.
"""

from morphanalyzer.granulometry.aperture import aperture_map, pore_size_distribution

__all__ = ["aperture_map", "pore_size_distribution", "maximal_balls", "cell_markers"]


def maximal_balls(*args, **kwargs):
    raise NotImplementedError(
        "morphanalyzer.granulometry.maximal_balls arrive en phase 3 "
        "(image d'identifiants de boules)."
    )


def cell_markers(*args, **kwargs):
    raise NotImplementedError(
        "morphanalyzer.granulometry.cell_markers arrive en phase 3 "
        "(boules remplies a >= 75 % de leur volume theorique)."
    )
