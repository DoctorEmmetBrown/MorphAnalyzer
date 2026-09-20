"""Segmentation des cellules et mesure des cols.

Phase de portage : 4 (livree).
Origine iMorph : `Thread/Granulometry/morphology.cpp::watershedBinarySearchTree`,
`cellsExtractionThread.cpp`, `throatThread.cpp`,
`Thread/Granulometry/morphometry.cpp`, `graph3D.cpp`.

La chaine de la these, de bout en bout :

    dist  = distance.distance_transform(fluide)
    mk    = granulometry.cell_markers(fluide)        # boules quasi entieres
    cells = segmentation.watershed_cells(dist, mk, mask=fluide)
    morph = segmentation.cell_morphometry(cells)
    cols  = segmentation.throats(cells)

`watershed_cells` porte la variante **a priorites reelles** avec resolution des
collisions par label majoritaire, qui n'existe pas en bibliotheque et qui est ce
qui distingue iMorph d'un watershed de Meyer standard.
"""

from morphanalyzer.segmentation.cells import (
    cell_morphometry,
    connectivity,
    pore_network,
    throats,
)
from morphanalyzer.segmentation.watershed import watershed as watershed_cells

__all__ = [
    "watershed_cells",
    "cell_morphometry",
    "throats",
    "connectivity",
    "pore_network",
]
