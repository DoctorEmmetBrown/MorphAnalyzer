"""Region d'interet : boite ou cylindre inscrit.

iMorph portait la ROI partout dans les signatures (`Roi* myRoi`). Ici la ROI est
simplement un objet capable de rendre un masque booleen, que l'appelant combine
avec son volume s'il le souhaite. Aucune fonction ne l'exige.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Roi"]


@dataclass(slots=True)
class Roi:
    """Region d'interet parallelepipedique, optionnellement cylindrique.

    Parameters
    ----------
    z, y, x
        Bornes `(debut, fin)` exclusives en fin, comme des `slice`. `None` = tout.
    cylinder
        Si vrai, restreint au cylindre inscrit dans la boite, d'axe `axis`.
        C'est le cas courant en tomographie (echantillon cylindrique) et cela
        evite de compter les coins vides dans la porosite.
    axis
        Axe du cylindre : 0 = z, 1 = y, 2 = x.
    margin
        Marge en voxels retiree au rayon du cylindre. Utile pour s'eloigner des
        artefacts de bord de reconstruction.
    """

    z: tuple[int, int] | None = None
    y: tuple[int, int] | None = None
    x: tuple[int, int] | None = None
    cylinder: bool = False
    axis: int = 0
    margin: int = 0

    def slices(self, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
        out = []
        for bounds, n in zip((self.z, self.y, self.x), shape, strict=True):
            if bounds is None:
                out.append(slice(0, n))
            else:
                a, b = bounds
                out.append(slice(max(0, a), min(n, b)))
        return tuple(out)  # type: ignore[return-value]

    def mask(self, shape: tuple[int, int, int]) -> np.ndarray:
        """Masque booleen de la ROI, aux dimensions du volume complet."""
        m = np.zeros(shape, dtype=bool)
        sl = self.slices(shape)
        m[sl] = True
        if not self.cylinder:
            return m
        # rayon du cylindre inscrit dans la boite, dans les deux axes transverses
        others = [a for a in (0, 1, 2) if a != self.axis]
        centres, radii = {}, []
        for a in others:
            s = sl[a]
            centres[a] = (s.start + s.stop - 1) / 2.0
            radii.append((s.stop - s.start) / 2.0 - self.margin)
        r = min(radii)
        if r <= 0:
            raise ValueError("marge trop grande : le cylindre inscrit est vide")
        grids = np.ogrid[tuple(slice(0, n) for n in shape)]
        d2 = sum((grids[a] - centres[a]) ** 2 for a in others)
        return m & (d2 <= r * r)

    def apply(self, data: np.ndarray, fill: float | bool | int = 0) -> np.ndarray:
        """Copie de `data` remise a `fill` hors de la ROI."""
        out = np.array(data, copy=True)
        out[~self.mask(data.shape)] = fill
        return out

    def volume_voxels(self, shape: tuple[int, int, int]) -> int:
        """Nombre de voxels dans la ROI — denominateur des fractions volumiques."""
        return int(self.mask(shape).sum())
