"""Volume elementaire representatif par tirage de boites (these, §2.1.4)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from morphanalyzer.core.volume import as_array

__all__ = ["representative_volume"]


def representative_volume(
    volume,
    quantity: Callable[[np.ndarray], float] | None = None,
    *,
    sizes=(8, 16, 24, 32, 48, 64),
    n_boxes: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Statistique de `quantity` sur des boites cubiques tirees au hasard.

    Reproduit l'approche de la these : pour chaque demi-cote, on tire `n_boxes`
    centres, on mesure la grandeur dans chaque boite, et on suit l'ecart-type et
    l'ecart maximal a la moyenne. Le VER est le volume au-dela duquel la
    dispersion passe sous un seuil fixe (2 % pour la porosite, 5 % pour la
    surface specifique, d'apres les figures 2.15 et 2.16).

    Parameters
    ----------
    quantity
        Fonction appliquee a chaque sous-volume. Par defaut la porosite.

    Returns
    -------
    pandas.DataFrame
        Colonnes `half_size`, `n`, `mean`, `std`, `std_rel`, `max_abs_dev`,
        `max_rel_dev`.
    """
    arr = as_array(volume)
    if quantity is None:

        def quantity(sub: np.ndarray) -> float:  # noqa: ANN202
            return 1.0 - float(sub.astype(bool).sum()) / sub.size

    rng = np.random.default_rng(seed)
    rows = []
    for h in sizes:
        lo = np.array([h, h, h])
        hi = np.array(arr.shape) - h
        if np.any(hi <= lo):
            continue
        vals = np.empty(n_boxes, dtype=float)
        for b in range(n_boxes):
            c = rng.integers(lo, hi)
            sub = arr[
                c[0] - h : c[0] + h,
                c[1] - h : c[1] + h,
                c[2] - h : c[2] + h,
            ]
            vals[b] = quantity(sub)
        m = float(vals.mean())
        rows.append(
            {
                "half_size": int(h),
                "n": int(n_boxes),
                "mean": m,
                "std": float(vals.std(ddof=1)),
                "std_rel": float(vals.std(ddof=1) / m) if m else np.nan,
                "max_abs_dev": float(np.abs(vals - m).max()),
                "max_rel_dev": float(np.abs(vals - m).max() / m) if m else np.nan,
            }
        )
    return pd.DataFrame(rows)
