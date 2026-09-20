"""Cartes de distance et propagation de front (fast marching).

Regle de conception retenue : la distance euclidienne passe par la transformee
exacte (scipy / edt), pas par le fast marching. La these mesure une erreur
maximale de 2,77 voxels au 1er ordre et 2,00 au 2nd (fig. 3.19) ; la transformee
exacte est a la fois juste et plus rapide. Le fast marching reste utilise pour
ce qu'il fait seul : geodesiques, champs de vitesse non uniformes (Poiseuille),
propagation etiquetee.

Phase de portage : 2.
Origine iMorph : Thread/Granulometry/fastMarchManu.cpp (3 718 l.), calc_fdmapFast

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = ["distance_transform", "geodesic_distance", "travel_time", "label_propagation"]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.distance.{name} arrive en phase 2. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
