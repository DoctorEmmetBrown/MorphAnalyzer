"""Transformee en distance exacte, propagation au plus proche, boule geodesique."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["distance_transform", "nearest_seed_propagation", "geodesic_ball"]


def distance_transform(
    mask,
    *,
    voxel_size=None,
    inside: bool = True,
    return_indices: bool = False,
):
    """Distance euclidienne **exacte** au complementaire de `mask`.

    Parameters
    ----------
    mask
        Volume ou tableau booleen. Par defaut (`inside=True`) la distance est
        calculee *dans* `mask` : chaque voxel recoit sa distance au premier
        voxel hors masque. C'est la carte de distance au solide de la these
        quand on passe le fluide, et la carte dans le solide quand on passe le
        solide.
    voxel_size
        Anisotropie. Repris du `Volume` si absent.
    return_indices
        Rend aussi les indices du plus proche voxel hors masque, ce qui donne
        gratuitement la transformee de caracteristique.

    Notes
    -----
    iMorph obtenait cette carte par fast marching, avec une erreur mesuree
    jusqu'a 2,77 voxels. Ici elle est exacte, et toute la chaine aval
    (granulometrie, marqueurs, watershed, morphometrie) en beneficie.
    """
    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    field = m if inside else ~m
    out = ndi.distance_transform_edt(field, sampling=voxel_size, return_indices=return_indices)
    if return_indices:
        dist, ind = out
    else:
        dist, ind = out, None
    dist = dist.astype(np.float32, copy=False)
    result = mask.with_data(dist, name="distance") if isinstance(mask, Volume) else dist
    return (result, ind) if return_indices else result


def nearest_seed_propagation(values, seeds, target, *, voxel_size=(1.0, 1.0, 1.0)):
    """Etend a `target` les valeurs portees par `seeds`, au plus proche voisin.

    C'est l'etape finale de la classification de forme d'iMorph : le tenseur
    n'est calcule qu'aux voxels du squelette, puis chaque voxel du solide
    herite du voxel de squelette le plus proche. iMorph le faisait par une
    triple boucle de recherche dans une boite ; ici un seul appel a
    `distance_transform_edt(..., return_indices=True)` rend la transformee de
    caracteristique, donc l'affectation complete.

    Parameters
    ----------
    values
        Tableau de meme forme que le volume, defini sur `seeds`.
    seeds
        Masque booleen des voxels porteurs (le squelette).
    target
        Masque booleen des voxels a remplir (le solide).
    """
    vals = np.asarray(values)
    s = np.asarray(seeds, dtype=bool)
    t = np.asarray(target, dtype=bool)
    if not s.any():
        raise ValueError("aucun germe : impossible de propager")
    _d, ind = ndi.distance_transform_edt(~s, sampling=voxel_size, return_indices=True)
    out = np.zeros_like(vals)
    idx = tuple(i[t] for i in ind)
    out[t] = vals[idx]
    return out


def geodesic_ball(
    mask: np.ndarray,
    centre: tuple[int, int, int],
    radius: float,
    *,
    connectivity: int = 26,
    euclidean_clip: bool = True,
) -> np.ndarray:
    """Voxels de `mask` atteignables depuis `centre` sans sortir de `mask`.

    Rend un tableau `(n, 3)` de coordonnees **absolues**.

    iMorph utilisait `fastMarchLimitedBlock` : un fast marching borne a
    `radius`, confine a la phase. On reproduit la meme selection par dilatation
    geodesique — `scipy.ndimage.binary_dilation` avec `mask=` et `iterations=`
    fait exactement cela, en C.

    Une dilatation iteree mesure une distance de damier (26-connexite) ou de
    cite (6-connexite), pas une distance euclidienne : la boule obtenue serait
    un cube ou un octaedre, ce qui biaiserait le tenseur vers les axes de
    l'image. `euclidean_clip` (actif par defaut) intersecte donc le resultat
    avec la boule euclidienne de meme rayon. Ce qui reste est l'ensemble des
    voxels a la fois **connectes dans la phase** et **a moins de `radius`** :
    la meme chose que le fast marching partout ou le voisinage local est
    convexe, c'est-a-dire presque partout a cette echelle.
    """
    r = int(np.ceil(radius))
    lo = [max(0, c - r) for c in centre]
    hi = [min(n, c + r + 1) for c, n in zip(centre, mask.shape, strict=True)]
    box = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]

    seed = np.zeros(box.shape, dtype=bool)
    local = tuple(c - lo_i for c, lo_i in zip(centre, lo, strict=True))
    seed[local] = True

    st = ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    reached = ndi.binary_dilation(seed, structure=st, iterations=r, mask=box)

    if euclidean_clip:
        grids = np.ogrid[tuple(slice(0, n) for n in box.shape)]
        d2 = sum((g - c) ** 2 for g, c in zip(grids, local, strict=True))
        reached &= d2 <= radius * radius

    coords = np.argwhere(reached)
    coords += np.asarray(lo, dtype=coords.dtype)
    return coords
