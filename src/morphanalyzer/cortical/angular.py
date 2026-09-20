"""Decoupage angulaire et radial d'un volume cylindrique, profils associes.

Portage de `Thread/Cortical/corticalModuleTabPorosity.cpp` (`buildAngularMapper`,
`refreshCamembert`, `computePorosity`) et de
`Thread/Cortical/corticalModuleTabAngularAper.cpp` (`computeAngularAper`,
`computeApparentVolumeMapper`).

Le contexte est l'os cortical : une diaphyse est un tube, et ses proprietes
varient surtout avec l'angle autour de l'axe et avec la distance au centre. On
decoupe donc chaque coupe en parts (« camembert »), et on moyenne dans chaque
part.

Convention d'angle, reprise telle quelle d'iMorph :

.. math:: \\theta(P) = \\mathrm{atan2}(y_0 - j,\\; i - x_0)

c'est-a-dire `x` vers la droite et `y` vers le **haut** — le sens
trigonometrique usuel, malgre l'axe des lignes qui descend. L'angle est ensuite
ramene dans `[0, 2\\pi[` par rapport a `start_angle`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = [
    "SectorProfile",
    "sector_map",
    "sector_bounds",
    "iso_area_angles",
    "angular_profile",
    "angular_aperture",
    "radial_profile",
]

_TWO_PI = 2.0 * np.pi


def _wrap_02pi(a):
    return np.mod(a, _TWO_PI)


def _flatten_ellipse(angles, *, a: float, b: float, theta0: float):
    """Applique la correction d'ellipse d'iMorph (`EllipseAngleApplatisseur`).

    Sur une section elliptique, des parts d'angle egal n'ont pas la meme aire.
    iMorph corrige en aplatissant les angles par le rapport `b/a`, ce qui
    revient a decouper le cercle image de l'ellipse par l'affinite qui la rend
    circulaire. `theta0` est l'orientation du grand axe, en radians.
    """
    ratio = b / a
    if ratio > 1.0:
        ratio = 1.0 / ratio
        theta0 = theta0 + np.pi / 2.0
    ang = np.asarray(angles, dtype=float)
    return np.arctan2(ratio * np.sin(ang + theta0), np.cos(ang + theta0)) - theta0


def _expand_ellipse(angles, *, a: float, b: float, theta0: float):
    ratio = b / a
    if ratio > 1.0:
        ratio = 1.0 / ratio
        theta0 = theta0 + np.pi / 2.0
    ang = np.asarray(angles, dtype=float)
    return np.arctan2(np.sin(ang + theta0) / ratio, np.cos(ang + theta0)) - theta0


def sector_bounds(
    n_sectors: int,
    *,
    start_angle: float = 0.0,
    degrees: bool = True,
    ellipse: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Bornes angulaires des secteurs, en radians, croissantes depuis `start_angle`.

    Parameters
    ----------
    ellipse
        `(a, b, angle)` : demi-grand axe, demi-petit axe et orientation du
        grand axe (en degres si `degrees`). Les parts sont alors d'aire egale
        sur l'ellipse plutot que d'angle egal.
    """
    if n_sectors < 1:
        raise ValueError("n_sectors doit valoir au moins 1")
    t0 = np.deg2rad(start_angle) if degrees else float(start_angle)
    thetas = t0 + np.arange(n_sectors, dtype=float) * (_TWO_PI / n_sectors)
    if ellipse is not None:
        a, b, ang = ellipse
        ang = np.deg2rad(ang) if degrees else float(ang)
        kw = {"a": float(a), "b": float(b), "theta0": -ang}
        shift = float(_expand_ellipse(thetas[:1], **kw)[0]) - float(thetas[0])
        thetas = _flatten_ellipse(thetas + shift, **kw)
    return thetas


def iso_area_angles(
    mask_in,
    n_sectors: int,
    *,
    center,
    start_angle: float = 0.0,
    degrees: bool = True,
    tolerance: float = 1.0,
) -> np.ndarray:
    """Bornes de secteurs portant tous le **meme volume apparent**.

    C'est l'option « iso apparent volume » d'iMorph
    (`computeApparentVolumeMapper`) : on histogramme d'abord le volume utile
    par intervalle fin de `tolerance` degres, puis on coupe cet histogramme en
    `n_sectors` parts de somme egale. Utile quand la section est loin d'etre
    circulaire, ou quand un masque en retire une partie.

    Parameters
    ----------
    mask_in
        Volume booleen des voxels a compter (typiquement : l'interieur de l'os,
        phase comprise).
    """
    m = as_array(mask_in).astype(bool, copy=False)
    tol = abs(float(tolerance))
    tol = max(tol, 1e-4)
    n_bins = int(np.ceil(360.0 / tol))
    fine = sector_map(m.shape[1:], center=center, angles=np.arange(n_bins) * (_TWO_PI / n_bins))
    weight = np.bincount(fine.ravel(), weights=m.sum(axis=0).ravel(), minlength=n_bins)

    total = weight.sum()
    if total <= 0:
        raise ValueError("aucun voxel actif : impossible d'equilibrer les secteurs")
    t0 = np.deg2rad(start_angle) if degrees else float(start_angle)
    dtheta = _TWO_PI / n_bins
    j0 = int(np.floor(_wrap_02pi(t0) / dtheta))
    # cumul circulaire a partir de la borne de depart
    idx = (j0 + np.arange(n_bins)) % n_bins
    w = weight[idx].astype(float)
    frac_first = 1.0 - (_wrap_02pi(t0) / dtheta - j0)
    w[0] *= frac_first
    cum = np.concatenate([[0.0], np.cumsum(w)])
    targets = np.arange(1, n_sectors) * (cum[-1] / n_sectors)
    out = [_wrap_02pi(t0)]
    for tgt in targets:
        k = int(np.searchsorted(cum, tgt, side="left"))
        k = min(max(k, 1), n_bins)
        span = cum[k] - cum[k - 1]
        frac = 0.0 if span <= 0 else (tgt - cum[k - 1]) / span
        pos = (k - 1) + frac
        if k == 1:
            pos = frac * frac_first
        out.append(_wrap_02pi(t0 + dtheta * pos))
    return np.asarray(out)


def sector_map(
    shape,
    *,
    center,
    n_sectors: int | None = None,
    start_angle: float = 0.0,
    degrees: bool = True,
    ellipse: tuple[float, float, float] | None = None,
    angles=None,
) -> np.ndarray:
    """Numero de secteur angulaire de chaque pixel d'une coupe `(ny, nx)`.

    Reproduit exactement `buildAngularMapper` : les bornes sont ramenees a
    `[0, 2\\pi[` relativement a la premiere d'entre elles, et un pixel tombe
    dans le secteur `b` si son angle est dans `[angles[b], angles[b+1][`, le
    dernier secteur se refermant sur le premier.

    Parameters
    ----------
    center
        `(y0, x0)` en pixels.
    angles
        Bornes explicites en radians (voir :func:`sector_bounds`,
        :func:`iso_area_angles`). Sinon, `n_sectors` parts regulieres.
    """
    ny, nx = (int(v) for v in shape[-2:])
    if angles is None:
        if n_sectors is None:
            raise ValueError("fournir soit `n_sectors`, soit `angles`")
        angles = sector_bounds(n_sectors, start_angle=start_angle, degrees=degrees, ellipse=ellipse)
    ang = np.asarray(angles, dtype=float)
    theta0 = float(ang[0])
    ang = np.sort(_wrap_02pi(ang - theta0))
    y0, x0 = (float(v) for v in center)

    j = np.arange(ny, dtype=float)[:, None]
    i = np.arange(nx, dtype=float)[None, :]
    theta = np.arctan2(y0 - j, i - x0)
    theta = _wrap_02pi(theta - theta0)
    box = np.searchsorted(ang, theta, side="left") - 1
    return np.clip(box, 0, len(ang) - 1).astype(np.int32)


@dataclass
class SectorProfile:
    """Profil par coupe et par secteur.

    Attributes
    ----------
    table
        Table longue : une ligne par `(z, sector)`, avec `n_voxels`,
        `n_phase` et `porosity` (ou `sum` et `mean` pour une grandeur
        continue).
    sector_map
        Carte 2D des numeros de secteur, commune a toutes les coupes.
    angles
        Bornes angulaires en radians.
    """

    table: pd.DataFrame
    sector_map: np.ndarray
    angles: np.ndarray
    center: tuple[float, float]
    value_column: str = "porosity"
    params: dict = field(default_factory=dict)

    def by_sector(self) -> pd.Series:
        """Moyenne sur toutes les coupes, secteur par secteur.

        Moyenne ponderee par le nombre de voxels, comme `poro_AvgTotal`.
        """
        g = self.table.groupby("sector")
        num = g["_num"].sum()
        den = g["n_voxels"].sum()
        return (num / den).rename(self.value_column)

    def by_slice(self) -> pd.Series:
        """Moyenne sur tous les secteurs, coupe par coupe (`poroAvg_Z`)."""
        g = self.table.groupby("z")
        return (g["_num"].sum() / g["n_voxels"].sum()).rename(self.value_column)

    def pivot(self) -> pd.DataFrame:
        """Table large `z` x `sector`, prete a tracer."""
        return self.table.pivot(index="z", columns="sector", values=self.value_column)

    @property
    def overall(self) -> float:
        return float(self.table["_num"].sum() / self.table["n_voxels"].sum())


def _accumulate(
    values: np.ndarray,
    boxes: np.ndarray,
    n_boxes: int,
    mask_out: np.ndarray | None,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Somme et effectif par (coupe, secteur)."""
    nz = values.shape[0]
    num = np.zeros((nz, n_boxes), dtype=np.float64)
    den = np.zeros((nz, n_boxes), dtype=np.float64)
    flat = boxes.ravel()
    for k in range(nz):
        v = values[k].ravel()
        if mask_out is not None:
            keep = ~mask_out[k].ravel()
        else:
            keep = None
        if weights is not None:
            w = weights[k].ravel()
            sel = w if keep is None else (w & keep)
        else:
            sel = keep
        if sel is None:
            den[k] = np.bincount(flat, minlength=n_boxes)
            num[k] = np.bincount(flat, weights=v, minlength=n_boxes)
        else:
            b = flat[sel]
            den[k] = np.bincount(b, minlength=n_boxes)
            num[k] = np.bincount(b, weights=v[sel], minlength=n_boxes)
    return num, den


def _profile(
    values,
    *,
    center,
    n_sectors,
    start_angle,
    degrees,
    ellipse,
    angles,
    mask_out,
    weights,
    value_column,
) -> SectorProfile:
    v = as_array(values)
    boxes = sector_map(
        v.shape,
        center=center,
        n_sectors=n_sectors,
        start_angle=start_angle,
        degrees=degrees,
        ellipse=ellipse,
        angles=angles,
    )
    n_boxes = int(boxes.max()) + 1
    mo = None if mask_out is None else as_array(mask_out).astype(bool, copy=False)
    wt = None if weights is None else as_array(weights).astype(bool, copy=False)
    num, den = _accumulate(v.astype(np.float64, copy=False), boxes, n_boxes, mo, wt)

    nz = v.shape[0]
    z = np.repeat(np.arange(nz), n_boxes)
    s = np.tile(np.arange(n_boxes), nz)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(den.ravel() > 0, num.ravel() / np.maximum(den.ravel(), 1), np.nan)
    table = pd.DataFrame(
        {
            "z": z,
            "sector": s,
            "n_voxels": den.ravel(),
            "_num": num.ravel(),
            value_column: val,
        }
    )
    used = (
        angles
        if angles is not None
        else sector_bounds(n_sectors, start_angle=start_angle, degrees=degrees, ellipse=ellipse)
    )
    return SectorProfile(
        table,
        boxes,
        np.asarray(used, dtype=float),
        (float(center[0]), float(center[1])),
        value_column,
        {"ellipse": ellipse, "start_angle": start_angle},
    )


def angular_profile(
    phase,
    *,
    center,
    n_sectors: int = 12,
    start_angle: float = 0.0,
    degrees: bool = True,
    ellipse: tuple[float, float, float] | None = None,
    angles=None,
    mask_out=None,
) -> SectorProfile:
    """Porosite par coupe et par secteur angulaire (`computePorosity`).

    Parameters
    ----------
    phase
        Masque booleen de la phase mesuree — pour de l'os cortical, les pores.
        La « porosite » rendue est la fraction de voxels de `phase` dans le
        secteur.
    center
        `(y0, x0)`, centre du camembert, en pixels.
    mask_out
        Masque booleen des voxels a **exclure** (hors de l'os). iMorph appelait
        cela le masque : les voxels marques ne comptent ni au numerateur ni au
        denominateur.
    """
    p = as_array(phase)
    return _profile(
        p.astype(np.float64, copy=False),
        center=center,
        n_sectors=n_sectors,
        start_angle=start_angle,
        degrees=degrees,
        ellipse=ellipse,
        angles=angles,
        mask_out=mask_out,
        weights=None,
        value_column="porosity",
    )


def angular_aperture(
    aperture,
    *,
    center,
    n_sectors: int = 12,
    start_angle: float = 0.0,
    degrees: bool = True,
    ellipse: tuple[float, float, float] | None = None,
    angles=None,
    mask_out=None,
) -> SectorProfile:
    """Ouverture moyenne par coupe et par secteur (`computeAngularAper`).

    `aperture` est une carte d'ouverture
    (:func:`morphanalyzer.granulometry.aperture_map`). Les voxels de valeur
    negative — hors phase — sont ignores, comme dans iMorph.
    """
    a = as_array(aperture).astype(np.float64, copy=False)
    return _profile(
        a,
        center=center,
        n_sectors=n_sectors,
        start_angle=start_angle,
        degrees=degrees,
        ellipse=ellipse,
        angles=angles,
        mask_out=mask_out,
        weights=(a >= 0),
        value_column="mean_aperture",
    )


def radial_profile(
    phase,
    *,
    center,
    n_bins: int = 20,
    r_max: float | None = None,
    equal_area: bool = False,
    mask_out=None,
    voxel_size=None,
) -> pd.DataFrame:
    """Porosite par coupe et par couronne radiale.

    Complement naturel du profil angulaire : sur une diaphyse, la porosite
    corticale varie fortement entre endoste et periose.

    Parameters
    ----------
    equal_area
        Decoupe en couronnes de meme aire plutot que de meme epaisseur. Les
        couronnes exterieures contiennent bien plus de pixels a epaisseur
        egale, ce qui ecrase leur barre d'erreur par rapport aux interieures.

    Returns
    -------
    pandas.DataFrame
        `z`, `ring`, `r_min`, `r_max`, `n_voxels`, `n_phase`, `porosity`.
        Les rayons sont en unites physiques si `voxel_size` est fourni.
    """
    p = as_array(phase)
    if voxel_size is None:
        voxel_size = phase.voxel_size if isinstance(phase, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    _dz, dy, dx = (float(v) for v in voxel_size)
    ny, nx = p.shape[1:]
    y0, x0 = (float(v) for v in center)
    j = np.arange(ny, dtype=float)[:, None]
    i = np.arange(nx, dtype=float)[None, :]
    r = np.hypot((j - y0) * dy, (i - x0) * dx)

    if r_max is None:
        r_max = float(r.max())
    if equal_area:
        edges = r_max * np.sqrt(np.arange(n_bins + 1) / n_bins)
    else:
        edges = np.linspace(0.0, r_max, n_bins + 1)
    ring = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_bins - 1)
    ring = np.where(r <= r_max, ring, -1)

    mo = None if mask_out is None else as_array(mask_out).astype(bool, copy=False)
    rows = []
    flat_ring = ring.ravel()
    valid = flat_ring >= 0
    for k in range(p.shape[0]):
        keep = valid if mo is None else (valid & ~mo[k].ravel())
        b = flat_ring[keep]
        den = np.bincount(b, minlength=n_bins)
        num = np.bincount(b, weights=p[k].ravel()[keep].astype(float), minlength=n_bins)
        for t in range(n_bins):
            rows.append(
                {
                    "z": k,
                    "ring": t,
                    "r_min": float(edges[t]),
                    "r_max": float(edges[t + 1]),
                    "n_voxels": int(den[t]),
                    "n_phase": float(num[t]),
                    "porosity": float(num[t] / den[t]) if den[t] else np.nan,
                }
            )
    return pd.DataFrame(rows)
