"""Debruitage : median et moyennes non locales."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["median", "denoise_nl_means"]


def median(volume, radius: int = 1, *, footprint: np.ndarray | None = None):
    """Filtre median 3D. `radius=1` = cube 3x3x3, le reglage d'iMorph (§2.1.2)."""
    a = as_array(volume)
    if footprint is None:
        size = 2 * radius + 1
        out = ndi.median_filter(a, size=size)
    else:
        out = ndi.median_filter(a, footprint=footprint)
    return volume.with_data(out) if isinstance(volume, Volume) else out


def denoise_nl_means(
    volume,
    *,
    sigma: float | None = None,
    patch_size: int = 3,
    patch_distance: int = 5,
    fast_mode: bool = True,
):
    """Moyennes non locales (Buades).

    Remplace `Thread/Segment/nlmeans_lib.cpp` (bibliotheque IPOL sous GPL,
    1 002 lignes) par `skimage.restoration.denoise_nl_means`, qui implemente le
    meme algorithme et supporte la 3D nativement.
    """
    from skimage.restoration import denoise_nl_means as _nl
    from skimage.restoration import estimate_sigma

    a = as_array(volume).astype(np.float32, copy=False)
    if sigma is None:
        sigma = float(estimate_sigma(a, channel_axis=None))
    out = _nl(
        a,
        h=1.15 * sigma,
        sigma=sigma,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=fast_mode,
        channel_axis=None,
    )
    return volume.with_data(out) if isinstance(volume, Volume) else out
