"""Amincissement homotopique 3D."""

from __future__ import annotations

import numpy as np

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["skeletonize", "distance_ridge"]


def skeletonize(mask, *, method: str = "lee"):
    """Squelette curviligne preservant la topologie.

    Parameters
    ----------
    method
        `"lee"` — amincissement de Lee (1994), l'equivalent du DOHT d'iMorph.
        `"medial_axis"` — axe median par transformee de distance : plus epais,
        mais plus stable sur les objets massifs.

    Notes
    -----
    Le squelette sert ici de **support d'echantillonnage** : la classification
    de forme n'y calcule le tenseur qu'aux voxels du squelette, puis propage au
    reste du solide. Sa qualite influe donc sur la densite des points de mesure,
    pas directement sur la valeur mesuree en chaque point — un squelette un peu
    trop fourni coute du temps, pas de la justesse.
    """
    from skimage.morphology import medial_axis
    from skimage.morphology import skeletonize as _sk

    m = as_array(mask).astype(bool, copy=False)
    if method == "lee":
        out = _sk(m, method="lee").astype(bool)
    elif method == "medial_axis":
        if m.ndim != 3:
            raise ValueError("volume 3D attendu")
        out = np.zeros_like(m)
        for k in range(m.shape[0]):  # medial_axis est 2D : applique par coupe
            out[k] = medial_axis(m[k])
    else:
        raise ValueError("method doit valoir 'lee' ou 'medial_axis'")
    return mask.with_data(out, name="skeleton") if isinstance(mask, Volume) else out


def distance_ridge(mask, *, distance=None, h: float = 1.0, voxel_size=(1.0, 1.0, 1.0)):
    """Crete de la carte de distance : les centres de boules maximales.

    Rend le masque des maxima locaux de la carte de distance a l'interieur de
    `mask`, au sens des h-maxima (les plateaux sont conserves entiers, ce qui
    est souhaitable : le « squelette » d'une plaque est son plan median, pas une
    courbe).

    Pourquoi ce n'est pas un gadget. `skimage.morphology.skeletonize` est
    l'equivalent du DOHT d'iMorph, mais sa mise en oeuvre 3D **supprime
    entierement certains objets compacts** : verifie en 0.25.2, un cube de
    4x4x4 voxels, une barre 10x2x2 et une sphere de rayon 12 rendent tous un
    squelette vide, alors qu'une sphere de rayon 6 rend 2 voxels. Pour un
    support d'echantillonnage, un squelette vide est un echec silencieux. La
    crete de distance, elle, n'est jamais vide sur un objet non vide.

    Cette amorce correspond aussi a une variante d'iMorph :
    `Doht::TH_DOHT_BALLSCENTERS`, qui partait des centres de boules maximales.
    """
    from skimage.morphology import h_maxima

    from morphanalyzer.distance.edt import distance_transform

    m = as_array(mask).astype(bool, copy=False)
    d = (
        np.asarray(distance)
        if distance is not None
        else np.asarray(distance_transform(m, voxel_size=voxel_size))
    )
    peaks = h_maxima(d.astype(np.float32), float(h)) > 0
    peaks &= m
    if not peaks.any():  # objet minuscule : tout garder plutot que rien
        peaks = m.copy()
    return mask.with_data(peaks, name="ridge") if isinstance(mask, Volume) else peaks
