"""Pretraitement et binarisation.

Remplace `Sources/FilterModules/` (7 675 lignes) par des appels a scipy.ndimage
et scikit-image. Rien d'original n'y etait implemente : median, erosion,
dilatation, hysteresis, Hessien, composante unique.
"""

from morphanalyzer.filters.binarize import (
    threshold_hysteresis,
    threshold_otsu,
    threshold_value,
)
from morphanalyzer.filters.cleanup import (
    complement,
    fill_holes,
    keep_largest_component,
    remove_small_objects,
)
from morphanalyzer.filters.denoise import denoise_nl_means, median
from morphanalyzer.filters.morpho import close_binary, dilate, erode, open_binary
from morphanalyzer.filters.ridges import hessian_features

__all__ = [
    "complement",
    "threshold_otsu",
    "threshold_value",
    "threshold_hysteresis",
    "median",
    "denoise_nl_means",
    "erode",
    "dilate",
    "open_binary",
    "close_binary",
    "keep_largest_component",
    "remove_small_objects",
    "fill_holes",
    "hessian_features",
]
