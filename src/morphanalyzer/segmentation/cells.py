"""Morphometrie des cellules, cols, connectivite, reseau de pores.

Portage de `Thread/Granulometry/morphometry.cpp` (`calculElipses`, `calcCell`,
`calcCol`), `throatThread.cpp` et `graph3D.cpp`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["cell_morphometry", "throats", "connectivity", "pore_network"]


def _spherical(v: np.ndarray) -> tuple[float, float]:
    vz, vy, vx = v
    if vz < 0:
        vz, vy, vx = -vz, -vy, -vx
    r = float(np.sqrt(vx * vx + vy * vy + vz * vz))
    if r == 0.0:
        return 0.0, 0.0
    return (
        float(np.degrees(np.arctan2(vy, vx))) % 360.0,
        float(np.degrees(np.arcsin(np.clip(vz / r, -1.0, 1.0)))),
    )


def cell_morphometry(labels, *, voxel_size=None, min_volume: int = 1) -> pd.DataFrame:
    """Volume, diametre equivalent, ellipsoide d'inertie et orientation par cellule.

    Returns
    -------
    pandas.DataFrame
        Une ligne par cellule : `label`, `volume`, `d_equivalent`, `centroid_*`,
        `a`, `b`, `c`, `a_on_b`, `b_on_c`, `theta`, `phi`, `touches_border`.

    Notes
    -----
    `d_equivalent` est le diametre de la sphere de meme volume — la definition du
    « diametre de pore » de la these (§3.1.4), et la grandeur de ses figures 3.6
    et 3.7.

    `a`, `b`, `c` suivent la convention d'iMorph : `2*sqrt(lambda)` des valeurs
    propres de la matrice de covariance des coordonnees. Attention a
    l'interpretation : pour un ellipsoide plein de demi-axes `A >= B >= C`, la
    covariance vaut `A^2/5`, donc `a = 2A/sqrt(5) ~ 0,894 A`. Les **rapports**
    `a/b` et `b/c`, eux, sont exacts. Pour retrouver les demi-axes geometriques,
    multiplier par `sqrt(5)/2`.

    Les statistiques de la these sont etablies sur les cellules entierement
    incluses dans l'echantillon ; `touches_border` permet de reproduire ce
    filtrage, indispensable car une cellule coupee par le bord a un volume et
    une forme tronques.
    """
    lab = as_array(labels)
    if voxel_size is None:
        voxel_size = labels.voxel_size if isinstance(labels, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    spacing = np.asarray(voxel_size, dtype=float)
    vvox = float(np.prod(spacing))

    ids = np.unique(lab)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return pd.DataFrame(
            columns=[
                "label",
                "volume",
                "d_equivalent",
                "centroid_z",
                "centroid_y",
                "centroid_x",
                "a",
                "b",
                "c",
                "a_on_b",
                "b_on_c",
                "theta",
                "phi",
                "touches_border",
            ]
        )

    nz, ny, nx = lab.shape
    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    lab[[0, -1], :, :].ravel(),
                    lab[:, [0, -1], :].ravel(),
                    lab[:, :, [0, -1]].ravel(),
                ]
            )
        ).tolist()
    )

    order = np.argsort(lab.ravel(), kind="stable")
    sorted_lab = lab.ravel()[order]
    starts = np.searchsorted(sorted_lab, ids, side="left")
    ends = np.searchsorted(sorted_lab, ids, side="right")

    rows = []
    for lid, a0, a1 in zip(ids, starts, ends, strict=True):
        flat = order[a0:a1]
        n = len(flat)
        if n < min_volume:
            continue
        kk, jj, ii = np.unravel_index(flat, lab.shape)
        pts = np.stack([kk, jj, ii], axis=1).astype(np.float64) * spacing
        centroid = pts.mean(axis=0)
        if n >= 4:
            lam, vec = np.linalg.eigh(np.cov(pts, rowvar=False, bias=True))
            lam = np.clip(lam[::-1], 0.0, None)
            vec = vec[:, ::-1]
            a, b, c = 2.0 * np.sqrt(lam)
            theta, phi = _spherical(vec[:, 0])
        else:
            a = b = c = float(spacing.min())
            theta = phi = 0.0
        vol = n * vvox
        rows.append(
            {
                "label": int(lid),
                "volume": vol,
                "d_equivalent": 2.0 * (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0),
                "centroid_z": centroid[0],
                "centroid_y": centroid[1],
                "centroid_x": centroid[2],
                "a": a,
                "b": b,
                "c": c,
                "a_on_b": a / b if b > 0 else np.nan,
                "b_on_c": b / c if c > 0 else np.nan,
                "theta": theta,
                "phi": phi,
                "touches_border": int(lid) in border_labels,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["voxel_size"] = tuple(spacing)
    return df


def throats(labels, *, voxel_size=None, min_area_voxels: int = 1) -> pd.DataFrame:
    """Cols entre cellules adjacentes : surface, diametre equivalent, barycentre.

    Un col est l'interface entre deux cellules segmentees. Sa surface est
    comptee en **faces de voxels** partagees (6-connexite), et son diametre
    equivalent est celui du disque de meme surface — la definition de la these
    (§3.1.4, figures 3.11 et 3.12, qui etablissent `Dcol = 0,53 Dpore`).

    Returns
    -------
    pandas.DataFrame
        `label_a`, `label_b` (avec `label_a < label_b`), `n_faces`, `area`,
        `d_equivalent`, `centroid_*`.
    """
    lab = as_array(labels)
    if voxel_size is None:
        voxel_size = labels.voxel_size if isinstance(labels, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    dz, dy, dx = (float(v) for v in voxel_size)
    face_area = {0: dy * dx, 1: dz * dx, 2: dz * dy}

    records: dict[tuple[int, int], dict] = {}
    for axis in (0, 1, 2):
        a = np.moveaxis(lab, axis, 0)
        left, right = a[:-1], a[1:]
        sel = (left > 0) & (right > 0) & (left != right)
        if not sel.any():
            continue
        la, lb = left[sel], right[sel]
        lo = np.minimum(la, lb)
        hi = np.maximum(la, lb)
        idx = np.argwhere(sel)
        # coordonnees du milieu de la face, ramenees dans l'ordre d'origine
        mid = idx.astype(np.float64)
        mid[:, 0] += 0.5
        mid = np.moveaxis(
            mid.reshape(-1, 3), 0, 0
        )  # pas de permutation a faire sur les colonnes : on remet plus bas
        perm = [0, 1, 2]
        perm.insert(axis, perm.pop(0))
        mid = mid[:, np.argsort(perm)]
        area1 = face_area[axis]
        for pair in np.unique(np.stack([lo, hi], axis=1), axis=0):
            k = (int(pair[0]), int(pair[1]))
            m = (lo == pair[0]) & (hi == pair[1])
            rec = records.setdefault(
                k, {"n_faces": 0, "area": 0.0, "sz": 0.0, "sy": 0.0, "sx": 0.0}
            )
            cnt = int(m.sum())
            rec["n_faces"] += cnt
            rec["area"] += cnt * area1
            sub = mid[m]
            rec["sz"] += float(sub[:, 0].sum())
            rec["sy"] += float(sub[:, 1].sum())
            rec["sx"] += float(sub[:, 2].sum())

    rows = []
    for (la, lb), rec in records.items():
        if rec["n_faces"] < min_area_voxels:
            continue
        n = rec["n_faces"]
        rows.append(
            {
                "label_a": la,
                "label_b": lb,
                "n_faces": n,
                "area": rec["area"],
                "d_equivalent": 2.0 * np.sqrt(rec["area"] / np.pi),
                "centroid_z": rec["sz"] / n * dz,
                "centroid_y": rec["sy"] / n * dy,
                "centroid_x": rec["sx"] / n * dx,
            }
        )
    df = pd.DataFrame(rows).sort_values(["label_a", "label_b"], ignore_index=True)
    df.attrs["voxel_size"] = (dz, dy, dx)
    return df


def connectivity(labels=None, *, throat_table: pd.DataFrame | None = None) -> pd.Series:
    """Nombre de cellules voisines par cellule (figure 3.13 de la these).

    Passer `throat_table` evite de recalculer les cols.
    """
    if throat_table is None:
        if labels is None:
            raise ValueError("fournir `labels` ou `throat_table`")
        throat_table = throats(labels)
    if throat_table.empty:
        return pd.Series(dtype=int, name="connectivity")
    both = pd.concat([throat_table["label_a"], throat_table["label_b"]])
    s = both.value_counts().sort_index()
    s.name = "connectivity"
    s.index.name = "label"
    return s


def pore_network(labels, *, voxel_size=None, as_networkx: bool = False):
    """Reseau de pores : cellules en noeuds, cols en aretes.

    Rend `(cells, throats)` — deux `DataFrame` — ou un graphe networkx si
    `as_networkx`, ce qui demande l'extra `network`.

    Les noeuds portent le barycentre, le volume et le diametre equivalent de la
    cellule ; les aretes portent la surface et le diametre equivalent du col,
    plus la distance entre barycentres (figure 3.15 de la these : la longueur
    des aretes du reseau fluide).
    """
    cells = cell_morphometry(labels, voxel_size=voxel_size)
    th = throats(labels, voxel_size=voxel_size)

    if not th.empty and not cells.empty:
        pos = cells.set_index("label")[["centroid_z", "centroid_y", "centroid_x"]]
        pa = pos.reindex(th["label_a"]).to_numpy()
        pb = pos.reindex(th["label_b"]).to_numpy()
        th = th.assign(length=np.linalg.norm(pa - pb, axis=1))

    if not as_networkx:
        return cells, th

    from morphanalyzer._deps import require

    nx = require("networkx", reason="l'export du reseau de pores en graphe")
    g = nx.Graph()
    for row in cells.itertuples(index=False):
        g.add_node(
            int(row.label),
            volume=float(row.volume),
            d_equivalent=float(row.d_equivalent),
            pos=(float(row.centroid_z), float(row.centroid_y), float(row.centroid_x)),
            touches_border=bool(row.touches_border),
        )
    for row in th.itertuples(index=False):
        g.add_edge(
            int(row.label_a),
            int(row.label_b),
            area=float(row.area),
            d_equivalent=float(row.d_equivalent),
            length=float(getattr(row, "length", np.nan)),
        )
    return g
