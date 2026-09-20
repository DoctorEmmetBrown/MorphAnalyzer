"""Cartes de distance, propagation de front, geodesiques.

Regle de conception : **la distance euclidienne passe par la transformee
exacte**, pas par le fast marching. La these mesure une erreur maximale de
2,77 voxels au 1er ordre et 2,00 au 2nd (fig. 3.19) ; `distance_transform` est
exacte et plus rapide. Le fast marching sert a ce qu'il fait seul : geodesiques
qui contournent les obstacles, et propagation a vitesse non uniforme.

Phase de portage : 2 (livree, schema du 1er ordre).
Origine iMorph : `Thread/Granulometry/fastMarchManu.cpp`.
"""

from morphanalyzer.distance.edt import (
    distance_transform,
    geodesic_ball,
    nearest_seed_propagation,
)
from morphanalyzer.distance.fmm import geodesic_distance, travel_time

__all__ = [
    "distance_transform",
    "nearest_seed_propagation",
    "geodesic_ball",
    "travel_time",
    "geodesic_distance",
    "label_propagation",
]


def label_propagation(*args, **kwargs):
    raise NotImplementedError(
        "morphanalyzer.distance.label_propagation (ManuLabelFastMarching) n'est pas "
        "portee : pour propager des labels, utiliser segmentation.watershed_cells, "
        "ou nearest_seed_propagation pour une affectation au plus proche."
    )
