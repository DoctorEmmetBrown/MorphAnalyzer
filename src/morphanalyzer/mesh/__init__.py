"""Maillage de surface, mesures sur maillage, export, decimation.

Phase de portage : 1 (livree).
Origine iMorph : `Thread/Mesh/mesh.cpp` (3 254 lignes : marching cubes,
`calcSpecificSurface`, exports STL / OBJ / POV / vox), `model.cpp`, et
`Thread/Simplificator/progmesh.cpp` pour la decimation — laquelle n'etait pas
compilee dans la version 3.2.

Les ecritures STL, OBJ et PLY sont natives : pas de dependance pour le cas
courant. `meshio` (extra `mesh`) ouvre les autres formats, `trimesh` ou
`fast-simplification` la decimation.
"""

from morphanalyzer.mesh.surface import (
    Mesh,
    decimate,
    mesh_area,
    mesh_volume,
    save_mesh,
    surface_mesh,
)

__all__ = ["surface_mesh", "save_mesh", "decimate", "mesh_volume", "mesh_area", "Mesh"]
