"""morphanalyzer — analyse morphologique 3D de milieux cellulaires et poreux.

Portage Python d'iMorph (J. Vicente & E. Brun, IUSTI).

Principe directeur : **bibliotheque d'abord**. Toutes les routines sont des
fonctions qui prennent et rendent des tableaux NumPy, utilisables depuis un
script, un notebook ou la ligne de commande, sans ecran ni interface graphique.
La visualisation (`morphanalyzer.viz`) est un extra strictement optionnel :
rien dans le noyau ne l'importe.

    import numpy as np
    import morphanalyzer as ma

    vol = ma.phantoms.sphere_pack(shape=(128, 128, 128), radius=12, n=40, seed=0)
    print(ma.metrics.porosity(vol.solid))
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from morphanalyzer import (  # noqa: F401
    cortical,
    distance,
    filters,
    granulometry,
    io,
    mesh,
    metrics,
    network,
    phantoms,
    radiative,
    segmentation,
    shape,
    skeleton,
    tortuosity,
)
from morphanalyzer.core import Roi, Volume  # noqa: F401

__all__ = [
    "__version__",
    "Volume",
    "Roi",
    "io",
    "filters",
    "distance",
    "granulometry",
    "segmentation",
    "skeleton",
    "shape",
    "tortuosity",
    "mesh",
    "network",
    "radiative",
    "cortical",
    "metrics",
    "phantoms",
]
