"""Carte d'ouverture locale (epaisseur locale)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from morphanalyzer.core import Volume
from morphanalyzer.core.neighborhood import ball_structure
from morphanalyzer.core.volume import as_array
from morphanalyzer.distance.edt import distance_transform

__all__ = ["aperture_map", "pore_size_distribution"]


def aperture_map(
    mask,
    *,
    voxel_size=None,
    radii=None,
    n_radii: int = 32,
    min_radius: float = 1.0,
    as_diameter: bool = False,
):
    """Rayon de la plus grande boule incluse dans `mask` contenant chaque voxel.

    Algorithme : pour chaque rayon `r` par ordre decroissant, les centres
    admissibles sont les voxels dont la distance au complementaire vaut au moins
    `r` ; dilater cet ensemble par une boule de rayon `r` donne l'union des
    boules de rayon `r` incluses dans `mask`. Le premier `r` qui couvre un voxel
    est son ouverture. C'est la formulation classique de l'epaisseur locale, et
    elle donne la meme carte que la file d'attente hierarchique d'iMorph
    (`calc_Aperture_Map3DFAH`), a la discretisation des rayons pres.

    Parameters
    ----------
    radii, n_radii, min_radius
        Rayons testes. Par defaut `n_radii` valeurs reparties lineairement entre
        `min_radius` et le maximum de la carte de distance. Plus il y en a, plus
        la carte est fine et plus le calcul est long — chaque rayon coute une
        dilatation.
    as_diameter
        Rend `2 * r` plutot que `r`. Le « diametre de pore » de la these.

    Notes
    -----
    Le parametre `apertureErrorPrecision` d'iMorph, qui elaguait les boules
    incluses dans une plus grande et faisait gagner un facteur 10, **est
    commente dans les sources 3.2** : la version livree est la force brute.
    Ici c'est la discretisation des rayons qui joue ce role, avec le meme
    compromis entre vitesse et finesse.
    """
    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    if not np.allclose(voxel_size, voxel_size[0]):
        raise NotImplementedError(
            "carte d'ouverture anisotrope non supportee : reechantillonner en voxels "
            "isotropes, ou passer voxel_size=1 et convertir apres coup"
        )
    scale = float(voxel_size[0])

    dist = np.asarray(distance_transform(m, voxel_size=(1.0, 1.0, 1.0)))
    dmax = float(dist.max())
    if dmax <= 0:
        out = np.zeros(m.shape, dtype=np.float32)
        return mask.with_data(out, name="aperture") if isinstance(mask, Volume) else out

    if radii is None:
        radii = np.linspace(min_radius, dmax, num=n_radii)
    radii = np.asarray(sorted({float(r) for r in radii if r >= min_radius}, reverse=True))

    aper = np.zeros(m.shape, dtype=np.float32)
    for r in radii:
        centres = dist >= r
        if not centres.any():
            continue
        covered = ndi.binary_dilation(centres, structure=ball_structure(r))
        covered &= m
        np.maximum(aper, np.float32(r), out=aper, where=covered & (aper == 0))
    # les voxels trop fins pour la plus petite boule gardent leur distance
    thin = m & (aper == 0)
    aper[thin] = dist[thin]

    aper *= scale
    if as_diameter:
        aper *= 2.0
    name = "aperture_diameter" if as_diameter else "aperture"
    return mask.with_data(aper, name=name) if isinstance(mask, Volume) else aper


def pore_size_distribution(aperture, *, mask=None, bins: int = 30, as_diameter: bool = True):
    """Histogramme volumique des tailles, pondere par le volume occupe.

    Rend un `DataFrame` avec `size`, `count`, `fraction` et `cumulative`.
    C'est la figure 2.20 de la these : la fraction du volume de la phase
    couverte par des boules de chaque taille.
    """
    a = as_array(aperture)
    sel = a > 0 if mask is None else (a > 0) & as_array(mask).astype(bool)
    vals = a[sel]
    if as_diameter and not str(getattr(aperture, "name", "")).endswith("diameter"):
        vals = vals * 2.0
    counts, edges = np.histogram(vals, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    frac = counts / total if total else counts.astype(float)
    return pd.DataFrame(
        {
            "size": centres,
            "count": counts,
            "fraction": frac,
            "cumulative": np.cumsum(frac),
        }
    )
