"""Detecteurs de structures par le Hessien.

Remplace `filterHessianFeaturesDetector` + `imageEigHessian3D` +
`image3DVecteur3` d'iMorph (~1 600 lignes) par les filtres de crete de
scikit-image, qui calculent les memes valeurs propres du Hessien a plusieurs
echelles.
"""

from __future__ import annotations

import numpy as np

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["hessian_features", "hessian_eigenvalues"]


def hessian_features(volume, *, sigmas=(1, 2, 4), kind: str = "frangi", **kwargs):
    """Reponse d'un filtre de crete : `frangi`, `meijering`, `sato` ou `hessian`.

    `frangi` privilegie les structures tubulaires (brins, vaisseaux), `sato` les
    lignes, `meijering` les neurites. Sur un empilement de brins creux, c'est
    `frangi` qui correspond au detecteur d'iMorph.
    """
    from skimage import filters as skfilters

    fn = {
        "frangi": skfilters.frangi,
        "meijering": skfilters.meijering,
        "sato": skfilters.sato,
        "hessian": skfilters.hessian,
    }[kind]
    a = as_array(volume).astype(np.float32, copy=False)
    out = fn(a, sigmas=sigmas, **kwargs)
    return volume.with_data(out) if isinstance(volume, Volume) else out


def hessian_eigenvalues(volume, sigma: float = 1.0) -> np.ndarray:
    """Valeurs propres du Hessien, triees par valeur absolue croissante.

    Rend un tableau `(3, nz, ny, nx)`. Utile quand on veut appliquer son propre
    critere plutot qu'une reponse de filtre preconstruite.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    a = as_array(volume).astype(np.float32, copy=False)
    H = hessian_matrix(a, sigma=sigma, use_gaussian_derivatives=True)
    return np.asarray(hessian_matrix_eigvals(H))
