"""Fantome d'os cortical : tube epais perce de canaux longitudinaux.

Verite terrain analytique pour le module :mod:`morphanalyzer.cortical` :
porosite par secteur angulaire et par couronne radiale connues par
construction, et nombre exact de canaux distincts.
"""

from __future__ import annotations

import numpy as np

from morphanalyzer.core import Volume

__all__ = ["cortical_tube"]


def cortical_tube(
    shape: tuple[int, int, int] = (64, 128, 128),
    *,
    r_inner: float = 24.0,
    r_outer: float = 52.0,
    n_canals: int = 24,
    canal_radius: float = 3.0,
    sector_weights=None,
    seed: int = 0,
    voxel_size: float = 1.0,
    unit: str = "um",
) -> Volume:
    """Tube epais (la matrice osseuse) perce de canaux paralleles a l'axe z.

    Parameters
    ----------
    r_inner, r_outer
        Rayons endostal et periostal, en voxels.
    n_canals
        Nombre de canaux. Ils sont places au hasard dans l'epaisseur, sans se
        recouvrir.
    sector_weights
        Poids relatifs par secteur angulaire, en partant de `theta = 0` et en
        tournant dans le sens trigonometrique. Par exemple `[3, 1, 1, 1]` met
        trois fois plus de canaux dans le premier quadrant. Sert a fabriquer
        une anisotropie angulaire dont on connait la reponse attendue.

    Returns
    -------
    Volume
        `True` = matrice osseuse. `meta["truth"]` contient `canal_centres`
        (en `(y, x)`), `canal_radius`, `n_canals`, `porosity` (fraction de
        canaux dans l'anneau), `ring_area`, `r_inner`, `r_outer` et
        `sector_weights`.
    """
    nz, ny, nx = (int(v) for v in shape)
    rng = np.random.default_rng(seed)
    y0, x0 = (ny - 1) / 2.0, (nx - 1) / 2.0

    if sector_weights is None:
        probs = None
        n_sec = 1
    else:
        w = np.asarray(sector_weights, dtype=float)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("sector_weights doit etre positif et non nul")
        probs = w / w.sum()
        n_sec = len(w)

    centres: list[tuple[float, float]] = []
    lo = r_inner + canal_radius + 1.0
    hi = r_outer - canal_radius - 1.0
    if hi <= lo:
        raise ValueError("epaisseur insuffisante pour des canaux de ce rayon")
    for _ in range(200000):
        if len(centres) >= n_canals:
            break
        if probs is None:
            theta = rng.uniform(0.0, 2 * np.pi)
        else:
            b = rng.choice(n_sec, p=probs)
            theta = (b + rng.uniform()) * (2 * np.pi / n_sec)
        r = np.sqrt(rng.uniform(lo**2, hi**2))
        cy, cx = y0 - r * np.sin(theta), x0 + r * np.cos(theta)
        if all(
            (cy - oy) ** 2 + (cx - ox) ** 2 >= (2 * canal_radius + 2.0) ** 2 for oy, ox in centres
        ):
            centres.append((cy, cx))
    if len(centres) < n_canals:
        raise ValueError(
            f"seulement {len(centres)} canaux places sur {n_canals} : "
            "elargir l'anneau ou reduire canal_radius"
        )

    j = np.arange(ny, dtype=float)[:, None]
    i = np.arange(nx, dtype=float)[None, :]
    r2 = (j - y0) ** 2 + (i - x0) ** 2
    ring = (r2 >= r_inner**2) & (r2 <= r_outer**2)

    canals = np.zeros((ny, nx), dtype=bool)
    for cy, cx in centres:
        canals |= ((j - cy) ** 2 + (i - cx) ** 2) <= canal_radius**2

    plane = ring & ~canals
    solid = np.broadcast_to(plane, (nz, ny, nx)).copy()

    ring_area = int(ring.sum())
    truth = {
        "canal_centres": np.asarray(centres, dtype=float),
        "canal_radius": float(canal_radius),
        "n_canals": int(n_canals),
        "r_inner": float(r_inner),
        "r_outer": float(r_outer),
        "centre": (float(y0), float(x0)),
        "ring_area": ring_area,
        "ring_mask": ring,
        "canal_mask": ring & canals,
        "porosity": float((ring & canals).sum() / ring_area),
        "sector_weights": None if sector_weights is None else np.asarray(sector_weights),
    }
    return Volume(
        solid,
        voxel_size=voxel_size,
        unit=unit,
        name="cortical_tube",
        meta={"truth": truth},
    )
