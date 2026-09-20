"""Tortuosites geometrique, directionnelle, sur graphe et de type Poiseuille.

Definition de Carman conservee : rapport au carre de la longueur geodesique a la
distance euclidienne. La variante Poiseuille propage avec un champ de vitesse
v = 1 - d^2/R^2 (d = distance a la paroi, R = rayon d'ouverture local), ce qui
donne des chemins centres dans les constrictions, plus proches d'une ligne de
courant que du chemin topologiquement le plus court.

Phase de portage : 6.
Origine iMorph : Thread/Tortuosity/ (graph.cpp 2 363 l.) ; fastMarchManu.cpp : tortuosityFastMarch*

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = [
    "point_tortuosity",
    "plane_tortuosity",
    "directional_tortuosity",
    "graph_tortuosity",
    "poiseuille_tortuosity",
    "shortest_path",
]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.tortuosity.{name} arrive en phase 6. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
