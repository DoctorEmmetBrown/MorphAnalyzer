"""Granulometrie : ouverture locale, boules maximales, marqueurs de cellules.

Phase de portage : 3 (livree).
Origine iMorph : `Thread/Granulometry/morphology.cpp::calc_Aperture_Map3DFAH*`,
`Thread/Granulometry/utility.cpp::createBallsFromIdMap*`.

La chaine complete de la these : carte de distance -> carte d'ouverture et image
d'identifiants de boules -> boules quasi entieres -> marqueurs, un par cellule.
Ce sont ces marqueurs qui alimentent le watershed de la phase 4.
"""

from morphanalyzer.granulometry.aperture import (
    BallTable,
    aperture_map,
    cell_markers,
    maximal_balls,
    pore_size_distribution,
)

__all__ = [
    "aperture_map",
    "pore_size_distribution",
    "maximal_balls",
    "cell_markers",
    "BallTable",
]
