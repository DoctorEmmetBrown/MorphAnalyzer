"""Nettoyage : composante unique, petits objets, bouchage de trous.

`keep_largest_component` reproduit `filterSingleComponent` d'iMorph, la recette
qui eliminait le bruit en ilots de la binarisation (these §2.1.2) : la phase
solide etant physiquement connexe, on ne garde que la plus grosse composante.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["complement", "keep_largest_component", "remove_small_objects", "fill_holes"]

_RANK = {6: 1, 18: 2, 26: 3}


def complement(mask):
    """Passe d'une phase a l'autre : `True` devient `False` et inversement.

    La convention de la bibliotheque est **`True` = solide**. Or presque toute
    la chaine morphologique — distance a la paroi, ouverture, granulometrie,
    marqueurs, cellules — se calcule dans le **fluide**. Il faut donc pouvoir
    changer de phase, et c'est tout ce que fait cette fonction.

        fluide = filters.complement(solide)
        dist   = distance.distance_transform(fluide)   # distance a la paroi

    En Python on ecrit plus court `~volume.solid`, ou `volume.fluid`. Cette
    fonction existe pour que le changement de phase soit **une etape de
    pipeline**, donc accessible depuis un fichier YAML ou depuis l'interface,
    ou l'operateur `~` n'a pas de place.

    iMorph n'en avait pas besoin : sa base de donnees portait la notion de
    *phase*, et chaque module s'appliquait a la phase selectionnee. Ici le
    changement de phase est explicite dans la chaine, ce qui la rend lisible.
    """
    m = as_array(mask)
    if m.dtype != bool:
        raise TypeError(
            f"complement attend un masque booleen, recu {m.dtype}. "
            "Binariser d'abord (filters.threshold_otsu)."
        )
    out = ~m
    return mask.with_data(out, name="complement") if isinstance(mask, Volume) else out


def keep_largest_component(mask, *, connectivity: int = 26, n: int = 1):
    """Ne conserve que les `n` plus grosses composantes connexes."""
    m = as_array(mask).astype(bool, copy=False)
    st = ndi.generate_binary_structure(3, _RANK[connectivity])
    lab, nlab = ndi.label(m, structure=st)
    if nlab <= n:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes)[::-1][:n]
    out = np.isin(lab, keep)
    return mask.with_data(out) if isinstance(mask, Volume) else out


def remove_small_objects(mask, min_size: int, *, connectivity: int = 26):
    """Supprime les composantes de moins de `min_size` voxels."""
    m = as_array(mask).astype(bool, copy=False)
    st = ndi.generate_binary_structure(3, _RANK[connectivity])
    lab, _ = ndi.label(m, structure=st)
    sizes = np.bincount(lab.ravel())
    small = np.flatnonzero(sizes < min_size)
    out = m & ~np.isin(lab, small)
    return mask.with_data(out) if isinstance(mask, Volume) else out


def fill_holes(mask, *, connectivity: int = 6):
    """Bouche les cavites fermees (meso-porosite des brins creux, §2.1.2)."""
    m = as_array(mask).astype(bool, copy=False)
    st = ndi.generate_binary_structure(3, _RANK[connectivity])
    out = ndi.binary_fill_holes(m, structure=st)
    return mask.with_data(np.asarray(out)) if isinstance(mask, Volume) else np.asarray(out)
