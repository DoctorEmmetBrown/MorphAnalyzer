"""Analyse de l'os cortical : profils radiaux et angulaires, connectivite.

Le contexte de ce module est une diaphyse : un tube dont les proprietes varient
surtout avec l'angle autour de l'axe et avec la distance au centre. On decoupe
donc chaque coupe en parts (« camembert ») et en couronnes, et on moyenne dans
chacune.

Phase de portage : 9 (livree).
Origine iMorph : `Thread/Cortical/` — `corticalModuleTabPorosity.cpp`,
`corticalModuleTabAngularAper.cpp`, `corticalModuleTabConnectivity.cpp`,
`corticalModuleTabVoronoi2D.cpp`. L'essentiel du volume C++ etait de
l'interface QCustomPlot ; le calcul se ramene a de la geometrie en coordonnees
cylindriques sur des cartes produites par les autres modules.
"""

from morphanalyzer.cortical.angular import (
    SectorProfile,
    angular_aperture,
    angular_profile,
    iso_area_angles,
    radial_profile,
    sector_bounds,
    sector_map,
)
from morphanalyzer.cortical.structure import (
    ConnectivityProfile,
    cortical_connectivity,
    voronoi_2d,
)

__all__ = [
    "angular_profile",
    "angular_aperture",
    "radial_profile",
    "sector_map",
    "sector_bounds",
    "iso_area_angles",
    "SectorProfile",
    "cortical_connectivity",
    "ConnectivityProfile",
    "voronoi_2d",
]
