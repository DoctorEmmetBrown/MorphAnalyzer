"""Reseau de pores, drainage morphologique et percolation d'invasion.

Deux algorithmes dans iMorph : Hazlett (percolation sur la carte d'ouverture) et
Hilpert-Miller (suite d'erosions-dilatations), plus une percolation d'invasion
sur le reseau. Les trois existent dans PoreSpy : porosimetry, drainage et ibip.
Ce module est donc essentiellement une couche d'adaptation, avec la conversion
rayon -> pression capillaire par Young-Laplace et les courbes de saturation.

Phase de portage : 7.
Origine iMorph : PhysicalModules/PoreNetworkModelling/ (fullMorphoThread.cpp, invasionIPThread.cpp)

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = [
    "drainage",
    "invasion_percolation",
    "capillary_pressure",
    "saturation_curve",
    "to_openpnm",
]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.network.{name} arrive en phase 7. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
