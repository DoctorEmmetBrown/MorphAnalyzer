"""Porosite totale, par coupe, et porosite ouverte."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from morphanalyzer.core.volume import as_array

__all__ = ["porosity", "porosity_per_slice", "open_porosity", "phase_fraction"]


def porosity(solid, mask=None) -> float:
    """Porosite totale = fraction de voxels non solides.

    Parameters
    ----------
    solid
        Volume ou tableau booleen, True = solide.
    mask
        Restriction (typiquement `Roi.mask(...)` pour un echantillon
        cylindrique). Les voxels hors masque ne comptent ni au numerateur ni au
        denominateur — c'est ce qui evite de compter les coins vides.
    """
    s = as_array(solid).astype(bool, copy=False)
    if mask is None:
        return 1.0 - float(s.sum()) / s.size
    m = as_array(mask).astype(bool, copy=False)
    n = int(m.sum())
    if n == 0:
        raise ValueError("masque vide")
    return 1.0 - float((s & m).sum()) / n


def porosity_per_slice(solid, axis: int = 0, mask=None) -> np.ndarray:
    """Porosite coupe par coupe le long de `axis` (fig. 2.8 de la these)."""
    s = as_array(solid).astype(bool, copy=False)
    axes = tuple(a for a in range(3) if a != axis)
    if mask is None:
        n = np.prod([s.shape[a] for a in axes])
        return 1.0 - s.sum(axis=axes) / float(n)
    m = as_array(mask).astype(bool, copy=False)
    n = m.sum(axis=axes).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 1.0 - (s & m).sum(axis=axes) / n
    return np.where(n > 0, out, np.nan)


def open_porosity(solid, mask=None, connectivity: int = 26) -> tuple[float, np.ndarray]:
    """Porosite ouverte = fraction de la plus grande composante connexe fluide.

    Suit iMorph : le fluide est etiquete en composantes connexes et la plus
    volumineuse est declaree ouverte. Rend `(fraction, masque_de_cette_composante)`.

    Notes
    -----
    Cette definition est une approximation commode : elle suppose que la
    porosite percolante est d'un seul tenant et majoritaire. Pour une definition
    stricte (« connectee a une face donnee »), etiqueter et retenir les
    composantes touchant la face d'entree.
    """
    s = as_array(solid).astype(bool, copy=False)
    fluid = ~s
    if mask is not None:
        fluid &= as_array(mask).astype(bool, copy=False)
    structure = ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    lab, n = ndi.label(fluid, structure=structure)
    if n == 0:
        return 0.0, np.zeros_like(fluid)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = int(sizes.argmax())
    keep = lab == biggest
    denom = fluid.size if mask is None else int(as_array(mask).astype(bool).sum())
    return float(keep.sum()) / denom, keep


def phase_fraction(labels, mask=None) -> dict[int, float]:
    """Fraction volumique de chaque valeur presente dans un volume etiquete."""
    a = as_array(labels)
    if mask is not None:
        a = a[as_array(mask).astype(bool, copy=False)]
    vals, counts = np.unique(a, return_counts=True)
    total = float(counts.sum())
    return {int(v): float(c) / total for v, c in zip(vals, counts, strict=True)}
