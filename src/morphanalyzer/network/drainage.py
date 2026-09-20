"""Drainage morphologique : Hazlett (carte d'ouverture) et Hilpert (erosion-dilatation).

Portage de `PhysicalModules/PoreNetworkModelling/fullMorphoThread.cpp` et de
`Thread/Granulometry/utility.cpp::boundaryFill3DNajib` / `saveFractionFile`.

On simule l'intrusion d'un fluide non mouillant par une face, a pression
croissante — donc a rayon de courbure decroissant. Deux familles d'algorithmes,
toutes deux presentes dans iMorph :

`"hazlett"` (Hazlett, *Transp. Porous Media* 1995)
    Un voxel est envahi au rayon `R` s'il existe un **chemin de voxels
    d'ouverture locale >= R** le reliant a la face d'injection. C'est la carte
    d'ouverture qui porte le critere.

`"hilpert"` (Hilpert & Miller, *Adv. Water Resour.* 2001)
    Une boule de rayon `R` doit pouvoir **se deplacer continument** depuis la
    face d'injection : erosion par la boule, selection des composantes qui
    touchent la face, puis dilatation par la meme boule.

La difference n'est pas cosmetique. Hazlett autorise un passage des que
l'ouverture locale le permet ; Hilpert exige en plus que le *centre* de la
boule ait un chemin continu. Hilpert est donc toujours plus restrictif ou egal,
et c'est lui qui correspond a la definition usuelle du drainage morphologique.
Sans contrainte d'acces, les deux se reduisent a l'ouverture morphologique,
c'est-a-dire a la carte d'ouverture seuillee — c'est ce que verifie un test.

Convention d'entree : `mask` est l'espace **poreux** (le fluide). Avec un
:class:`~morphanalyzer.core.Volume`, passer `volume.fluid`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array
from morphanalyzer.distance.edt import distance_transform
from morphanalyzer.network.capillarity import capillary_pressure

__all__ = ["DrainageResult", "drainage", "saturation_curve", "face_slab"]

_TOL = 1e-6


def face_slab(shape, face: int) -> tuple[slice, ...]:
    """Tranche d'epaisseur 1 correspondant a une face du volume.

    `face` suit la convention du module :mod:`morphanalyzer.distance.fmm` :
    `0` = z minimal, `1` = z maximal, `2` = y minimal, `3` = y maximal,
    `4` = x minimal, `5` = x maximal.
    """
    face = int(face)
    if not 0 <= face <= 5:
        raise ValueError("un numero de face doit etre dans 0..5")
    axis, side = divmod(face, 2)
    sl: list[slice] = [slice(None)] * len(shape)
    sl[axis] = slice(-1, None) if side else slice(0, 1)
    return tuple(sl)


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=bool)
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndi.generate_binary_structure(3, 2)
    raise ValueError("connectivity doit valoir 6, 18 ou 26")


def _keep_touching(mask: np.ndarray, slab, structure) -> np.ndarray:
    """Composantes connexes de `mask` qui rencontrent la tranche `slab`."""
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=structure)
    if n == 0:
        return mask
    seeds = np.unique(lab[slab])
    seeds = seeds[seeds > 0]
    if seeds.size == 0:
        return np.zeros_like(mask)
    keep = np.zeros(n + 1, dtype=bool)
    keep[seeds] = True
    return keep[lab]


def _keep_spanning(mask: np.ndarray, axis: int, structure) -> np.ndarray:
    """Composantes qui touchent les deux faces opposees selon `axis`.

    Reproduit le filtre d'iMorph, qui comparait la boite englobante de chaque
    objet aux bornes du volume (`zmin == 0 && zmax == limite - 1`).
    """
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=structure)
    if n == 0:
        return mask
    lo = np.unique(np.take(lab, [0], axis=axis))
    hi = np.unique(np.take(lab, [-1], axis=axis))
    both = np.intersect1d(lo[lo > 0], hi[hi > 0])
    keep = np.zeros(n + 1, dtype=bool)
    keep[both] = True
    return keep[lab]


def _radii_grid(dmax: float, radii, step: float, min_radius: float) -> np.ndarray:
    if radii is not None:
        g = np.asarray(sorted({float(r) for r in np.ravel(radii)}, reverse=True))
        return g[g > 0]
    if dmax < min_radius:
        return np.asarray([], dtype=float)
    n = int(np.floor((dmax - min_radius) / step)) + 1
    return dmax - step * np.arange(n, dtype=float)


@dataclass
class DrainageResult:
    """Sortie d'un drainage morphologique.

    Attributes
    ----------
    filling_radius
        Carte du rayon d'envahissement : pour chaque voxel poreux, le plus
        grand rayon auquel il est atteint. `0` = jamais envahi, `-1` = hors de
        la phase consideree (solide, ou porosite exclue par `restrict`).
        iMorph initialisait les voxels non envahis a `1.0` et non a `0`, ce qui
        les faisait compter comme envahis au rayon 1 dans la courbe : c'est
        corrige ici.
    curve
        Courbe de retention, un point par rayon atteint.
    """

    filling_radius: np.ndarray
    curve: pd.DataFrame
    method: str
    face: int
    voxel_size: tuple[float, float, float]
    params: dict = field(default_factory=dict)

    @property
    def pore_volume(self) -> float:
        """Volume de l'espace poreux retenu, en unites physiques."""
        return float((self.filling_radius >= 0).sum() * np.prod(self.voxel_size))

    def as_volume(self, name: str = "filling_radius") -> Volume:
        return Volume(self.filling_radius, voxel_size=self.voxel_size, name=name)


def drainage(
    mask,
    *,
    face: int = 0,
    method: str = "hazlett",
    aperture=None,
    radii=None,
    step: float = 0.5,
    min_radius: float = 1.0,
    voxel_size=None,
    restrict: str | None = "spanning",
    connectivity: int = 26,
    surface_tension: float | None = None,
    contact_angle: float = 0.0,
    unit: str | None = None,
    convention: str = "laplace",
) -> DrainageResult:
    """Drainage morphologique par une face, courbe de retention associee.

    Parameters
    ----------
    mask
        Espace poreux (`True` = pore). Pour un :class:`Volume`, passer
        `volume.fluid`.
    face
        Face d'injection, `0..5` (`0` = z minimal). Meme convention que
        :func:`morphanalyzer.distance.travel_time`.
    method
        `"hazlett"` ou `"hilpert"`. Voir le docstring du module.
    aperture
        Carte d'ouverture deja calculee (rayons, meme unite que `voxel_size`).
        Uniquement utile pour `"hazlett"` : evite de la recalculer.
    radii
        Grille de rayons explicite. Par defaut, de l'ouverture maximale a
        `min_radius` par pas de `step`. iMorph utilisait un pas de 0,1 voxel
        pour Hazlett et de 0,5 pour Hilpert ; le defaut retenu ici est 0,5 pour
        les deux, car un pas de 0,1 multiplie le cout par cinq sans changer
        visiblement la courbe.
    restrict
        `"spanning"` (defaut, comportement d'iMorph) ne garde que les
        composantes de pore qui relient les deux faces opposees a `face` ;
        `"inlet"` garde celles qui touchent la face d'injection ; `None` garde
        tout.
    surface_tension, contact_angle, unit, convention
        Si `surface_tension` est fourni, la courbe recoit une colonne
        `pressure` en pascals. `unit` vaut par defaut l'unite du `Volume`, ou
        `"um"`.

    Returns
    -------
    DrainageResult

    Notes
    -----
    Le cout est domine par la boucle sur les rayons : une transformee de
    distance et un etiquetage par rayon. Pour un volume de 512^3 et 40 rayons,
    compter quelques minutes.
    """
    if method not in ("hazlett", "hilpert"):
        raise ValueError("method doit valoir 'hazlett' ou 'hilpert'")
    m = as_array(mask).astype(bool, copy=False)
    if m.ndim != 3:
        raise ValueError("drainage attend un volume 3D")
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    vs = tuple(float(v) for v in voxel_size)
    if unit is None:
        unit = mask.unit if isinstance(mask, Volume) else "um"

    axis, _side = divmod(int(face), 2)
    slab = face_slab(m.shape, face)
    structure = _structure(connectivity)

    if restrict == "spanning":
        m = _keep_spanning(m, axis, structure)
    elif restrict == "inlet":
        m = _keep_touching(m, slab, structure)
    elif restrict is not None:
        raise ValueError("restrict doit valoir 'spanning', 'inlet' ou None")

    fill = np.full(m.shape, -1.0, dtype=np.float32)
    fill[m] = 0.0
    if not m.any():
        return DrainageResult(
            fill,
            saturation_curve(fill, axis=axis, voxel_size=vs),
            method,
            int(face),
            vs,
            {"restrict": restrict, "empty": True},
        )

    dist = distance_transform(m, voxel_size=vs)
    grid = _radii_grid(float(dist.max()), radii, step, min_radius)

    if method == "hazlett":
        if aperture is None:
            from morphanalyzer.granulometry import aperture_map

            aper = aperture_map(m, voxel_size=vs, radii=grid, min_radius=min_radius)
        else:
            aper = as_array(aperture).astype(np.float32, copy=False)
        for r in grid:
            cand = m & (aper >= r - _TOL)
            if not cand[slab].any():
                continue
            inv = _keep_touching(cand, slab, structure)
            np.maximum(fill, np.where(inv, np.float32(r), np.float32(-np.inf)), out=fill)
    else:
        for r in grid:
            er = m & (dist >= r - _TOL)
            if not er[slab].any():
                continue
            er = _keep_touching(er, slab, structure)
            if not er.any():
                continue
            back = ndi.distance_transform_edt(~er, sampling=vs)
            inv = m & (back <= r + _TOL)
            np.maximum(fill, np.where(inv, np.float32(r), np.float32(-np.inf)), out=fill)

    curve = saturation_curve(
        fill,
        axis=axis,
        voxel_size=vs,
        surface_tension=surface_tension,
        contact_angle=contact_angle,
        unit=unit,
        convention=convention,
    )
    return DrainageResult(
        fill,
        curve,
        method,
        int(face),
        vs,
        {
            "restrict": restrict,
            "connectivity": connectivity,
            "radii": grid,
            "unit": unit,
        },
    )


def saturation_curve(
    filling_radius,
    *,
    axis: int = 0,
    voxel_size=None,
    surface_tension: float | None = None,
    contact_angle: float = 0.0,
    unit: str = "um",
    convention: str = "laplace",
) -> pd.DataFrame:
    """Courbe de retention deduite d'une carte de rayon d'envahissement.

    Pour chaque rayon `R` present dans la carte, la saturation en fluide non
    mouillant est la fraction du volume poreux dont le rayon d'envahissement
    est `>= R`. Les colonnes `min_depth`, `max_depth` et `penetration`
    reprennent la profondeur atteinte selon `axis`, comme le fichier
    `*_Zmin.txt` d'iMorph.

    Returns
    -------
    pandas.DataFrame
        `radius`, `diameter`, `volume`, `saturation`, `min_depth`, `max_depth`,
        `penetration`, et `pressure` si `surface_tension` est fourni. Triee par
        rayon decroissant.
    """
    f = as_array(filling_radius)
    if voxel_size is None:
        voxel_size = filling_radius.voxel_size if isinstance(filling_radius, Volume) else (1.0,) * 3
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    vvox = float(np.prod(voxel_size))

    pore = f >= 0
    total = int(pore.sum())
    values = np.unique(f[f > 0])
    cols = [
        "radius",
        "diameter",
        "volume",
        "saturation",
        "min_depth",
        "max_depth",
        "penetration",
    ]
    if total == 0 or values.size == 0:
        df = pd.DataFrame(columns=cols)
        if surface_tension is not None:
            df["pressure"] = []
        return df

    coord = np.arange(f.shape[axis])
    shape = [1, 1, 1]
    shape[axis] = -1
    coord = coord.reshape(shape)

    rows = []
    for r in values[::-1]:
        sel = f >= r - _TOL
        n = int(sel.sum())
        if n == 0:
            continue
        proj = np.broadcast_to(coord, f.shape)[sel]
        rows.append(
            {
                "radius": float(r),
                "diameter": 2.0 * float(r),
                "volume": n * vvox,
                "saturation": n / total,
                "min_depth": int(proj.min()),
                "max_depth": int(proj.max()),
                "penetration": int(proj.max() - proj.min()),
            }
        )
    df = pd.DataFrame(rows, columns=cols)
    if surface_tension is not None:
        df["pressure"] = capillary_pressure(
            df["radius"].to_numpy(),
            surface_tension=surface_tension,
            contact_angle=contact_angle,
            unit=unit,
            convention=convention,
        )
    df.attrs["pore_volume"] = total * vvox
    return df
