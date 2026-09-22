"""Tortuosites geometrique, de plan, directionnelle, sur graphe et de Poiseuille.

Phase de portage : 6 (livree).
Origine iMorph : `Thread/Tortuosity/` (graph.cpp, les `*TortuosityModule`),
`Thread/Granulometry/fastMarchManu.cpp::tortuosityFastMarch*`.

Definition de Carman [Carm 37], conservee telle quelle : pour deux points de la
meme phase, connectes,

$$ \\tau(p_1, p_2) = \\left(\\frac{L_{min}(p_1, p_2)}{\\lVert p_1 - p_2 \\rVert}\\right)^2 $$

ou `L_min` est la longueur de la geodesique. C'est donc un **carre** de rapport de
longueurs : une tortuosite de 1,21 correspond a un chemin 10 % plus long.
"""

from morphanalyzer.tortuosity.geodesic import (
    PoiseuilleResult,
    TortuosityResult,
    directional_tortuosity,
    plane_tortuosity,
    point_tortuosity,
    poiseuille_speed,
    poiseuille_tortuosity,
    shortest_path,
)
from morphanalyzer.tortuosity.graph import graph_tortuosity

__all__ = [
    "point_tortuosity",
    "plane_tortuosity",
    "directional_tortuosity",
    "poiseuille_tortuosity",
    "poiseuille_speed",
    "graph_tortuosity",
    "shortest_path",
    "TortuosityResult",
    "PoiseuilleResult",
]
