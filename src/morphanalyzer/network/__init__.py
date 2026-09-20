"""Reseau de pores, drainage morphologique et percolation d'invasion.

Deux familles d'algorithmes de drainage, toutes deux dans iMorph : Hazlett
(percolation sur la carte d'ouverture) et Hilpert-Miller (suite d'erosions et
de dilatations). Plus une percolation d'invasion sur le graphe cellules-cols,
avec piegeage de la phase defendante et cols deformables.

Phase de portage : 7 (livree).
Origine iMorph : `PhysicalModules/PoreNetworkModelling/` (`fullMorphoThread.cpp`,
`invasionIPThread.cpp`), `Thread/Granulometry/utility.cpp::saveFractionFile`.

Conversion rayon <-> pression : :func:`capillary_pressure`. Attention, iMorph
ecrivait `Pc = 4 sigma / r` avec `r` un rayon, soit deux fois la loi de
Young-Laplace ; la convention historique reste accessible.
"""

from morphanalyzer.network.capillarity import (
    MERCURY_CONTACT_ANGLE,
    MERCURY_SURFACE_TENSION,
    WATER_AIR_SURFACE_TENSION,
    capillary_pressure,
    capillary_radius,
)
from morphanalyzer.network.drainage import (
    DrainageResult,
    drainage,
    face_slab,
    saturation_curve,
)
from morphanalyzer.network.export import save_network_text, to_openpnm
from morphanalyzer.network.invasion import (
    InvasionResult,
    deformed_throat_radius,
    face_cells,
    filling_map,
    invasion_percolation,
)

__all__ = [
    "drainage",
    "DrainageResult",
    "saturation_curve",
    "face_slab",
    "face_cells",
    "invasion_percolation",
    "InvasionResult",
    "filling_map",
    "deformed_throat_radius",
    "capillary_pressure",
    "capillary_radius",
    "WATER_AIR_SURFACE_TENSION",
    "MERCURY_SURFACE_TENSION",
    "MERCURY_CONTACT_ANGLE",
    "to_openpnm",
    "save_network_text",
]
