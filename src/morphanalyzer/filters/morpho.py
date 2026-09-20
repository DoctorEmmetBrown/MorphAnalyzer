"""Morphologie mathematique : erosion, dilatation, ouverture, fermeture.

Element structurant spherique par defaut, comme iMorph (`strElement.cpp`).
Attention : un cube est separable et donc beaucoup plus rapide, mais il
introduit une anisotropie qui biaise les mesures d'orientation.
"""

from __future__ import annotations

from scipy import ndimage as ndi

from morphanalyzer.core import Volume
from morphanalyzer.core.neighborhood import ball_structure
from morphanalyzer.core.volume import as_array

__all__ = ["erode", "dilate", "open_binary", "close_binary"]


def _struct(radius: float, shape: str):
    if shape == "ball":
        return ball_structure(radius)
    if shape == "cube":
        import numpy as np

        n = 2 * int(radius) + 1
        return np.ones((n, n, n), dtype=bool)
    raise ValueError("shape doit valoir 'ball' ou 'cube'")


def _apply(volume, fn, radius, shape):
    a = as_array(volume)
    st = _struct(radius, shape)
    out = fn(a, structure=st) if a.dtype == bool else fn(a, footprint=st)
    return volume.with_data(out) if isinstance(volume, Volume) else out


def erode(volume, radius: float = 1.0, *, shape: str = "ball"):
    a = as_array(volume)
    fn = ndi.binary_erosion if a.dtype == bool else ndi.grey_erosion
    return _apply(volume, fn, radius, shape)


def dilate(volume, radius: float = 1.0, *, shape: str = "ball"):
    a = as_array(volume)
    fn = ndi.binary_dilation if a.dtype == bool else ndi.grey_dilation
    return _apply(volume, fn, radius, shape)


def open_binary(volume, radius: float = 1.0, *, shape: str = "ball"):
    a = as_array(volume)
    fn = ndi.binary_opening if a.dtype == bool else ndi.grey_opening
    return _apply(volume, fn, radius, shape)


def close_binary(volume, radius: float = 1.0, *, shape: str = "ball"):
    a = as_array(volume)
    fn = ndi.binary_closing if a.dtype == bool else ndi.grey_closing
    return _apply(volume, fn, radius, shape)
