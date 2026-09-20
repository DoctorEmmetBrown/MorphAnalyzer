"""Cartes de distance et propagation.

Regle de conception : **la distance euclidienne passe par la transformee
exacte**, pas par le fast marching. La these mesure une erreur maximale de
2,77 voxels au 1er ordre et 2,00 au 2nd (fig. 3.19) ; `distance_transform_edt`
est exacte et plus rapide. Le fast marching (phase 2) reste necessaire pour ce
qu'il fait seul : geodesiques a vitesse non uniforme et propagation etiquetee.

Phase de portage : 2 (partiellement livre).
Origine iMorph : `Thread/Granulometry/fastMarchManu.cpp::distFastMarching`,
`calc_fdmapFast`, `fastMarchLimitedBlock`.
"""

from morphanalyzer.distance.edt import (
    distance_transform,
    geodesic_ball,
    nearest_seed_propagation,
)

__all__ = [
    "distance_transform",
    "nearest_seed_propagation",
    "geodesic_ball",
    "geodesic_distance",
    "travel_time",
    "label_propagation",
]


def geodesic_distance(*args, **kwargs):
    raise NotImplementedError(
        "morphanalyzer.distance.geodesic_distance arrive en phase 2 "
        "(fast marching a vitesse non uniforme). Pour une boule geodesique "
        "locale, voir geodesic_ball."
    )


def travel_time(*args, **kwargs):
    raise NotImplementedError("morphanalyzer.distance.travel_time arrive en phase 2.")


def label_propagation(*args, **kwargs):
    raise NotImplementedError(
        "morphanalyzer.distance.label_propagation arrive en phase 2. "
        "Pour une affectation au germe le plus proche, voir nearest_seed_propagation."
    )
