"""Volumes synthetiques a verite terrain analytique.

Chaque fantome rend un `Volume` binaire (True = solide) dont `meta["truth"]`
contient les grandeurs exactes du continu : porosite, surface specifique,
diametre de pore, tortuosite, labels de cellules attendus... C'est le socle de
validation de la bibliotheque : contrairement a un tomogramme, on connait la
reponse.

    vol = ma.phantoms.voronoi_foam((128,)*3, n_cells=24, strut=2.0, seed=0)
    vol.meta["truth"]["n_cells"]        # nombre exact de cellules
    vol.meta["truth"]["cell_labels"]    # partition de Voronoi de reference

Les tolerances de comparaison doivent tenir compte de la discretisation :
l'erreur relative sur une surface mesuree par marching cubes decroit comme
1/r, il faut donc des rayons >= 8 voxels pour viser le pourcent.
"""

from morphanalyzer.phantoms.basic import (
    cylinders,
    plate,
    sinusoidal_tube,
    sphere,
    sphere_pack,
    straight_tube,
)
from morphanalyzer.phantoms.bone import cortical_tube
from morphanalyzer.phantoms.foam import voronoi_foam

__all__ = [
    "sphere",
    "sphere_pack",
    "cylinders",
    "plate",
    "straight_tube",
    "sinusoidal_tube",
    "voronoi_foam",
    "cortical_tube",
]
