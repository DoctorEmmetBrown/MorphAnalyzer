"""Fantomes elementaires : sphere, empilement, cylindres, plaque, tubes."""

from __future__ import annotations

import numpy as np

from morphanalyzer.core import Volume

__all__ = ["sphere", "sphere_pack", "cylinders", "plate", "straight_tube", "sinusoidal_tube"]


def _grid(shape: tuple[int, int, int]):
    return np.ogrid[tuple(slice(0, n) for n in shape)]


def sphere(
    shape: tuple[int, int, int] = (64, 64, 64),
    radius: float = 16.0,
    centre: tuple[float, float, float] | None = None,
    voxel_size: float = 1.0,
) -> Volume:
    """Une sphere solide unique.

    Verite terrain : volume `4/3 pi r^3`, surface `4 pi r^2`, et une carte de
    distance au solide dont le maximum vaut exactement la distance du coin le
    plus eloigne — utile pour verifier une transformee en distance.
    """
    if centre is None:
        centre = tuple((n - 1) / 2.0 for n in shape)
    kk, jj, ii = _grid(shape)
    d2 = (kk - centre[0]) ** 2 + (jj - centre[1]) ** 2 + (ii - centre[2]) ** 2
    dist = np.sqrt(np.broadcast_to(d2, shape).astype(np.float64))
    solid = dist <= radius
    nvox = int(np.prod(shape))
    vol_c = 4.0 / 3.0 * np.pi * radius**3
    surf_c = 4.0 * np.pi * radius**2
    return Volume(
        solid,
        voxel_size=voxel_size,
        name="sphere",
        meta={
            "truth": {
                "solid_volume_continuum": vol_c * voxel_size**3,
                "porosity": 1.0 - vol_c / nvox,
                "specific_surface": surf_c / (nvox * voxel_size),
                "radius": radius,
                "centre": centre,
                # champ continu (> 0 dans le solide) : marcher a l'iso-0 dessus
                # rend l'interface exacte au sous-voxel pres, contrairement au
                # masque binaire dont l'iso-0.5 est un escalier
                "signed_distance": (radius - dist).astype(np.float32),
            }
        },
    )


def sphere_pack(
    shape: tuple[int, int, int] = (128, 128, 128),
    radius: float = 12.0,
    n: int = 40,
    seed: int = 0,
    min_gap: float = 2.0,
    voxel_size: float = 1.0,
    max_tries: int = 20000,
) -> Volume:
    """Empilement de spheres **disjointes** (rejet aleatoire).

    `min_gap` est l'ecart minimal entre surfaces, en voxels. La valeur par
    defaut (2) garantit que les spheres restent **disjointes en 26-connexite** :
    avec un ecart de 1 voxel, deux spheres peuvent se toucher en diagonale et
    l'analyse en composantes connexes les fusionne.

    Les spheres ne se recouvrent pas et ne touchent pas les bords, donc le
    volume et la surface du solide sont la somme exacte des contributions
    individuelles. C'est le test de reference pour la porosite, la surface
    specifique (marching cubes) et la granulometrie : la distribution des
    diametres est un pic de Dirac a `2 * radius`.
    """
    rng = np.random.default_rng(seed)
    lo = radius + 1.0
    hi = np.array(shape, dtype=float) - radius - 1.0
    if np.any(hi <= lo):
        raise ValueError("rayon trop grand pour la boite")
    centres: list[np.ndarray] = []
    tries = 0
    while len(centres) < n and tries < max_tries:
        tries += 1
        c = rng.uniform(lo, hi)
        if all(np.linalg.norm(c - o) >= 2 * radius + min_gap for o in centres):
            centres.append(c)
    if len(centres) < n:
        raise RuntimeError(
            f"seulement {len(centres)}/{n} spheres placees en {max_tries} essais ; "
            "reduire n ou radius"
        )
    kk, jj, ii = _grid(shape)
    dmin = np.full(shape, np.inf)
    for c in centres:
        d = np.sqrt((kk - c[0]) ** 2 + (jj - c[1]) ** 2 + (ii - c[2]) ** 2)
        np.minimum(dmin, d, out=dmin)
    solid = dmin <= radius
    nvox = int(np.prod(shape))
    vol_c = n * 4.0 / 3.0 * np.pi * radius**3
    surf_c = n * 4.0 * np.pi * radius**2
    return Volume(
        solid,
        voxel_size=voxel_size,
        name="sphere_pack",
        meta={
            "truth": {
                "n_spheres": n,
                "radius": radius,
                "centres": np.asarray(centres),
                "porosity": 1.0 - vol_c / nvox,
                "solid_volume_continuum": vol_c * voxel_size**3,
                "specific_surface": surf_c / (nvox * voxel_size),
                "solid_granulometry_diameter": 2 * radius * voxel_size,
                "signed_distance": (radius - dmin).astype(np.float32),
            }
        },
    )


def cylinders(
    shape: tuple[int, int, int] = (96, 96, 96),
    radius: float = 6.0,
    n: int = 6,
    axis: int = 0,
    seed: int = 0,
    voxel_size: float = 1.0,
) -> Volume:
    """Cylindres paralleles traversant la boite selon `axis`.

    Verite terrain pour la **classification de forme** : localement, un cylindre
    doit donner `a/b` grand (poutre) et une orientation alignee sur `axis`.
    """
    rng = np.random.default_rng(seed)
    others = [a for a in (0, 1, 2) if a != axis]
    lo = radius + 1.0
    hi = np.array([shape[a] for a in others], dtype=float) - radius - 1.0
    centres: list[np.ndarray] = []
    for _ in range(20000):
        if len(centres) >= n:
            break
        c = rng.uniform(lo, hi)
        if all(np.linalg.norm(c - o) >= 2 * radius + 1.0 for o in centres):
            centres.append(c)
    grids = _grid(shape)
    solid = np.zeros(shape, dtype=bool)
    r2 = radius * radius
    for c in centres:
        d2 = (grids[others[0]] - c[0]) ** 2 + (grids[others[1]] - c[1]) ** 2
        solid |= d2 <= r2
    nvox = int(np.prod(shape))
    length = shape[axis]
    vol_c = len(centres) * np.pi * radius**2 * length
    surf_c = len(centres) * 2 * np.pi * radius * length  # faces terminales exclues
    direction = np.zeros(3)
    direction[axis] = 1.0
    return Volume(
        solid,
        voxel_size=voxel_size,
        name="cylinders",
        meta={
            "truth": {
                "n_cylinders": len(centres),
                "radius": radius,
                "axis": axis,
                "direction": direction,
                "porosity": 1.0 - vol_c / nvox,
                "specific_surface": surf_c / (nvox * voxel_size),
                "solid_granulometry_diameter": 2 * radius * voxel_size,
                "expected_shape_class": "strut",
            }
        },
    )


def plate(
    shape: tuple[int, int, int] = (64, 64, 64),
    thickness: float = 6.0,
    axis: int = 0,
    voxel_size: float = 1.0,
) -> Volume:
    """Plaque solide perpendiculaire a `axis`.

    Verite terrain pour la classification : `a ~ b >> c` (plaque), orientation
    normale a `axis`.
    """
    grids = _grid(shape)
    c = (shape[axis] - 1) / 2.0
    solid = np.abs(grids[axis] - c) <= thickness / 2.0
    solid = np.broadcast_to(solid, shape).copy()
    nvox = int(np.prod(shape))
    face = nvox / shape[axis]
    return Volume(
        solid,
        voxel_size=voxel_size,
        name="plate",
        meta={
            "truth": {
                "thickness": thickness,
                "axis": axis,
                "porosity": 1.0 - float(solid.sum()) / nvox,
                "specific_surface": 2 * face / (nvox * voxel_size),
                "solid_granulometry_diameter": thickness * voxel_size,
                "expected_shape_class": "plate",
            }
        },
    )


def straight_tube(
    shape: tuple[int, int, int] = (64, 64, 64),
    radius: float = 8.0,
    axis: int = 0,
    voxel_size: float = 1.0,
) -> Volume:
    """Tube fluide rectiligne dans un solide plein.

    Verite terrain : **tortuosite geometrique exactement 1**. C'est le premier
    test de tout calcul de geodesique.
    """
    grids = _grid(shape)
    others = [a for a in (0, 1, 2) if a != axis]
    cs = [(shape[a] - 1) / 2.0 for a in others]
    d2 = (grids[others[0]] - cs[0]) ** 2 + (grids[others[1]] - cs[1]) ** 2
    fluid = d2 <= radius * radius
    solid = ~np.broadcast_to(fluid, shape).copy()
    return Volume(
        solid,
        voxel_size=voxel_size,
        name="straight_tube",
        meta={
            "truth": {
                "radius": radius,
                "axis": axis,
                "tortuosity": 1.0,
                "fluid_granulometry_diameter": 2 * radius * voxel_size,
            }
        },
    )


def sinusoidal_tube(
    shape: tuple[int, int, int] = (128, 64, 64),
    radius: float = 6.0,
    amplitude: float = 10.0,
    n_periods: float = 2.0,
    axis: int = 0,
    voxel_size: float = 1.0,
) -> Volume:
    """Tube fluide sinusoidal : tortuosite geometrique connue analytiquement.

    L'axe suit `y(z) = A sin(2 pi n z / L)`. La tortuosite au sens de Carman
    (rapport de longueurs au carre) vaut `(s / L)^2` ou `s` est la longueur
    d'arc, integree numeriquement a la precision machine.
    """
    grids = _grid(shape)
    others = [a for a in (0, 1, 2) if a != axis]
    L = shape[axis]
    t = np.arange(L, dtype=float)
    k = 2.0 * np.pi * n_periods / L
    offset = amplitude * np.sin(k * t)

    centre_a = (shape[others[0]] - 1) / 2.0
    centre_b = (shape[others[1]] - 1) / 2.0
    # centre du tube decale le long du 1er axe transverse, en fonction de `axis`
    shp_off = [1, 1, 1]
    shp_off[axis] = L
    off = offset.reshape(shp_off)
    d2 = (grids[others[0]] - (centre_a + off)) ** 2 + (grids[others[1]] - centre_b) ** 2
    fluid = d2 <= radius * radius
    solid = ~np.broadcast_to(fluid, shape).copy()

    dy = amplitude * k * np.cos(k * t)
    arc = (
        float(np.trapezoid(np.sqrt(1.0 + dy**2), t))
        if hasattr(np, "trapezoid")
        else float(
            np.trapz(np.sqrt(1.0 + dy**2), t)  # noqa: NPY201
        )
    )
    ratio = arc / (L - 1)
    return Volume(
        solid,
        voxel_size=voxel_size,
        name="sinusoidal_tube",
        meta={
            "truth": {
                "radius": radius,
                "axis": axis,
                "amplitude": amplitude,
                "n_periods": n_periods,
                "arc_length": arc,
                "geodesic_over_euclidean": ratio,
                "tortuosity": ratio**2,
            }
        },
    )
