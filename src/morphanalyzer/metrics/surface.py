"""Surface specifique par marching cubes.

iMorph : `Mesh::DrawIsoSurface` + `Mesh::calcSpecificSurface` (3 250 lignes de
`mesh.cpp`). Ici : `skimage.measure.marching_cubes` + `mesh_surface_area`.

La precision sous-voxelique vient de l'interpolation lineaire sur les aretes,
exactement comme dans l'implementation d'origine. Sur un volume **binaire**
l'iso-surface a 0.5 est la seule option sensee ; si les niveaux de gris sont
disponibles, les passer via `grey` ameliore nettement la precision, car
l'interpolation retrouve l'interface reelle au lieu de l'escalier du masque.
"""

from __future__ import annotations

import numpy as np
from skimage import measure

from morphanalyzer.core.volume import Volume, as_array

__all__ = ["specific_surface", "surface_mesh"]


def surface_mesh(solid, *, grey=None, level: float | None = None, spacing=(1.0, 1.0, 1.0)):
    """Maillage triangulaire de l'interface solide/fluide.

    Rend `(verts, faces, normals, values)` comme `skimage.measure.marching_cubes`.
    """
    if grey is not None:
        field = as_array(grey).astype(np.float32, copy=False)
        if level is None:
            raise ValueError("`level` (seuil de binarisation) est requis avec `grey`")
        iso = float(level)
    else:
        field = as_array(solid).astype(np.float32, copy=False)
        iso = 0.5 if level is None else float(level)
    return measure.marching_cubes(field, level=iso, spacing=tuple(spacing))


def specific_surface(solid, *, grey=None, level: float | None = None, voxel_size=None) -> float:
    """Surface specifique `S_v` = aire de l'interface / volume total.

    Unite : inverse de l'unite de `voxel_size` (m^2/m^3 = m^-1 si les voxels
    sont en metres). Le tableau 2.3 de la these donne les valeurs de reference
    pour les mousses ERG et Recemat.
    """
    if voxel_size is None:
        voxel_size = solid.voxel_size if isinstance(solid, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    verts, faces, _n, _v = surface_mesh(solid, grey=grey, level=level, spacing=voxel_size)
    area = measure.mesh_surface_area(verts, faces)
    shape = as_array(solid).shape
    total = float(np.prod([n * d for n, d in zip(shape, voxel_size, strict=True)]))
    return float(area) / total
