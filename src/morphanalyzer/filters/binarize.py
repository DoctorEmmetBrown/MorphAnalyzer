"""Binarisation.

iMorph utilisait le critere d'Otsu (these §2.1.2, [Otsu 79]) pour tous les
echantillons, afin que les comparaisons inter-mousses soient coherentes. On
conserve ce defaut, et on expose le seuil retenu pour qu'il soit tracable.
"""

from __future__ import annotations

import numpy as np
from skimage import filters as skfilters

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["threshold_otsu", "threshold_value", "threshold_hysteresis"]


def _wrap(source, mask: np.ndarray, name: str) -> Volume:
    if isinstance(source, Volume):
        return source.with_data(mask, name=f"{source.name}:{name}" if source.name else name)
    return Volume(mask, name=name)


def threshold_otsu(grey, *, mask=None, nbins: int = 256, return_threshold: bool = False):
    """Binarise par le critere d'Otsu. True = solide (valeurs hautes).

    Notes
    -----
    Sur les mousses tres poreuses (> 80 %), l'histogramme est domine par la
    gaussienne du fluide et Otsu place le seuil dans une zone large et plate :
    la these estime l'incertitude resultante sur la porosite a environ 2 %
    (§2.1.2). Ce n'est pas un defaut de l'implementation mais une limite des
    donnees — le noter dans les metadonnees plutot que de chercher a l'affiner.
    """
    g = as_array(grey)
    values = g[as_array(mask).astype(bool)] if mask is not None else g
    t = float(skfilters.threshold_otsu(np.asarray(values), nbins=nbins))
    solid = g > t
    if mask is not None:
        solid &= as_array(mask).astype(bool, copy=False)
    out = _wrap(grey, solid, "otsu")
    out.meta["threshold"] = t
    out.meta["threshold_method"] = "otsu"
    return (out, t) if return_threshold else out


def threshold_value(grey, low: float, high: float | None = None, *, mask=None) -> Volume:
    """Binarise par intervalle `low <= v <= high` (fenetre de phase d'iMorph)."""
    g = as_array(grey)
    solid = g >= low if high is None else (g >= low) & (g <= high)
    if mask is not None:
        solid &= as_array(mask).astype(bool, copy=False)
    out = _wrap(grey, solid, "threshold")
    out.meta["threshold"] = (low, high)
    out.meta["threshold_method"] = "value"
    return out


def threshold_hysteresis(grey, low: float, high: float, *, mask=None) -> Volume:
    """Seuillage par hysteresis (`filterHysteresisThreshold` d'iMorph).

    Retient les voxels au-dessus de `high`, et ceux au-dessus de `low` connectes
    a un voxel fort. Plus robuste qu'un seuil unique sur des interfaces floues,
    typiquement en contraste de phase.
    """
    g = as_array(grey)
    solid = skfilters.apply_hysteresis_threshold(g, low, high)
    if mask is not None:
        solid &= as_array(mask).astype(bool, copy=False)
    out = _wrap(grey, np.asarray(solid), "hysteresis")
    out.meta["threshold"] = (low, high)
    out.meta["threshold_method"] = "hysteresis"
    return out
