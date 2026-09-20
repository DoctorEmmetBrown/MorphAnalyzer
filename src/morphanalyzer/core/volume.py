"""Volume 3D + metadonnees physiques.

Remplace `Image3D<T>` / `OutputImage3D<T>` / `Phase` d'iMorph (17 800 lignes de
C++) par un `dataclass` mince autour d'un tableau NumPy. Le tableau reste
accessible et modifiable : aucune copie, aucune encapsulation opaque.

Convention d'axes : `(k, j, i)` == `(z, y, x)`, comme les piles d'images
tomographiques et comme scikit-image. Le `voxel_size` est en micrometres par
defaut, et suit le meme ordre d'axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["Volume", "as_array"]


def as_array(x: Any) -> np.ndarray:
    """Accepte un `Volume` ou un `ndarray` et rend toujours un `ndarray` 3D."""
    arr = x.data if isinstance(x, Volume) else np.asarray(x)
    if arr.ndim != 3:
        raise ValueError(f"volume 3D attendu, recu un tableau de dimension {arr.ndim}")
    return arr


@dataclass(slots=True)
class Volume:
    """Un volume 3D et sa resolution physique.

    Parameters
    ----------
    data
        Tableau `(nz, ny, nx)`. N'importe quel dtype : `bool` pour un masque,
        `uint8`/`uint16` pour des niveaux de gris, `int32`/`int64` pour des
        labels, `float32` pour une carte de distance ou d'ouverture.
    voxel_size
        Taille du voxel, scalaire (isotrope) ou triplet `(dz, dy, dx)`.
    unit
        Unite de `voxel_size`. Par defaut le micrometre.
    name
        Etiquette libre, utile dans les journaux et les tables de resultats.
    meta
        Metadonnees libres (echantillon, ligne de lumiere, seuil applique...).

    Notes
    -----
    Convention iMorph conservee pour la binarisation : **True = solide**.
    Les fonctions qui travaillent sur le fluide prennent explicitement le
    complementaire, de maniere visible dans la signature.
    """

    data: np.ndarray
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    unit: str = "um"
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 3:
            raise ValueError(f"volume 3D attendu, recu {self.data.ndim}D")
        vs = self.voxel_size
        if np.isscalar(vs):
            vs = (float(vs),) * 3  # type: ignore[assignment]
        vs = tuple(float(v) for v in vs)  # type: ignore[assignment]
        if len(vs) != 3:
            raise ValueError("voxel_size doit etre un scalaire ou un triplet (dz, dy, dx)")
        if any(v <= 0 for v in vs):
            raise ValueError("voxel_size doit etre strictement positif")
        self.voxel_size = vs  # type: ignore[assignment]

    # -- geometrie ---------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def is_isotropic(self) -> bool:
        return np.allclose(self.voxel_size, self.voxel_size[0])

    @property
    def voxel_volume(self) -> float:
        """Volume d'un voxel dans l'unite courante, au cube."""
        dz, dy, dx = self.voxel_size
        return dz * dy * dx

    @property
    def physical_shape(self) -> tuple[float, float, float]:
        return tuple(n * d for n, d in zip(self.shape, self.voxel_size, strict=True))  # type: ignore[return-value]

    # -- vues et derives ---------------------------------------------------
    def with_data(self, data: np.ndarray, *, name: str | None = None) -> Volume:
        """Nouveau `Volume` de meme geometrie, autre contenu."""
        return replace(self, data=np.asarray(data), name=self.name if name is None else name)

    @property
    def solid(self) -> np.ndarray:
        """Masque booleen du solide (True = solide)."""
        if self.data.dtype == bool:
            return self.data
        raise TypeError(
            "`.solid` n'a de sens que sur un volume binaire ; "
            "binariser d'abord (voir morphanalyzer.filters.threshold_otsu)"
        )

    @property
    def fluid(self) -> np.ndarray:
        """Masque booleen du fluide (complementaire du solide)."""
        return ~self.solid

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Rend le tableau sous-jacent : `np.asarray(volume)` fonctionne.

        Sans cette methode, numpy enveloppe la dataclass dans un tableau objet
        de dimension 0, ce qui casse silencieusement plus loin. Un `Volume`
        s'utilise donc partout ou une fonction numpy attend un tableau.
        """
        a = self.data
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        if copy:
            a = a.copy()
        return a

    def crop(self, zslice: slice, yslice: slice, xslice: slice) -> Volume:
        return self.with_data(self.data[zslice, yslice, xslice])

    def __repr__(self) -> str:  # pragma: no cover - agrement
        nz, ny, nx = self.shape
        vs = "x".join(f"{v:g}" for v in self.voxel_size)
        tag = f" {self.name!r}" if self.name else ""
        return f"<Volume{tag} {nz}x{ny}x{nx} {self.dtype} voxel={vs} {self.unit}>"
