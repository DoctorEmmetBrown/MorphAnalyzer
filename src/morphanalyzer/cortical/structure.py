"""Connectivite par empilement de coupes et partition de Voronoi 2D.

Portage de `Thread/Cortical/corticalModuleTabConnectivity.cpp` (853 lignes) et
`Thread/Cortical/corticalModuleTabVoronoi2D.cpp`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array
from morphanalyzer.distance.edt import distance_transform
from morphanalyzer.segmentation.watershed import watershed

__all__ = ["ConnectivityProfile", "cortical_connectivity", "voronoi_2d"]


class _DSU:
    __slots__ = ("parent", "volume")

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.volume: list[float] = []

    def add(self, volume: float) -> int:
        i = len(self.parent)
        self.parent.append(i)
        self.volume.append(float(volume))
        return i

    def find(self, i: int) -> int:
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            p[i], i = root, p[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.volume[ra] < self.volume[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.volume[ra] += self.volume[rb]


@dataclass
class ConnectivityProfile:
    """Evolution des objets connectes au fur et a mesure de l'empilement.

    Attributes
    ----------
    table
        Une ligne par coupe : `z`, `n_objects`, `total_volume`,
        `largest_volume`, `largest_fraction`, `n_new`, `n_merged`.
    volumes
        `volumes[k]` : volumes des objets connus a la coupe `k`, decroissants.
    labels
        Etiquetage 3D final, apres toutes les fusions (`0` = fond).
    """

    table: pd.DataFrame
    volumes: list[np.ndarray]
    labels: np.ndarray
    params: dict = field(default_factory=dict)

    def n_objects_above(self, fraction: float, *, cumulative: bool = False) -> pd.Series:
        """Nombre d'objets depassant une fraction du volume total, par coupe.

        Reproduit `numberIndividualVolGreaterThan_pct_z` (`cumulative=False`)
        et `numberCumulVolGreaterThan_pct_z` (`cumulative=True`, nombre des
        plus gros objets qu'il faut reunir pour atteindre cette fraction).
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction doit etre dans [0, 1]")
        out = []
        for v in self.volumes:
            if v.size == 0:
                out.append(0)
                continue
            total = v.sum()
            if total <= 0:
                out.append(0)
                continue
            if cumulative:
                out.append(int(np.searchsorted(np.cumsum(v) / total, fraction, "left") + 1))
            else:
                out.append(int((v / total > fraction).sum()))
        return pd.Series(out, index=self.table["z"].to_numpy(), name="n_objects")


def cortical_connectivity(
    mask,
    *,
    axis: int = 0,
    start: int = 0,
    in_plane_connectivity: int = 8,
) -> ConnectivityProfile:
    """Suit les objets connectes coupe apres coupe, en empilant.

    A la coupe `k`, on ne connait que le sous-volume `[start, k]`. Deux canaux
    qui se rejoignent plus haut sont donc comptes separement tant que la
    jonction n'est pas atteinte. C'est tout l'interet de la mesure pour l'os :
    elle dit a quelle hauteur le reseau de canaux se referme sur lui-meme.

    L'implementation reprend la logique d'iMorph — etiquetage 2D de la coupe,
    recherche des contacts avec la coupe precedente dans un voisinage 3x3,
    fusion des objets ponte — mais la remplace par une structure union-find,
    ce qui evite le traitement explicite des « ponts » et la renumerotation
    globale a chaque fusion.

    Parameters
    ----------
    mask
        Masque booleen de la phase suivie (les pores, pour de l'os cortical).
    axis
        Axe d'empilement. `0` par defaut.
    start
        Premiere coupe prise en compte.

    Returns
    -------
    ConnectivityProfile
    """
    m = as_array(mask).astype(bool, copy=False)
    m = np.moveaxis(m, axis, 0)
    nz, ny, nx = m.shape
    if not 0 <= start < nz:
        raise ValueError(f"start doit etre dans [0, {nz - 1}]")
    struct2d = (
        np.ones((3, 3), dtype=bool)
        if in_plane_connectivity == 8
        else ndi.generate_binary_structure(2, 1)
    )

    dsu = _DSU()
    glob_stack = np.full((nz, ny, nx), -1, dtype=np.int64)
    prev = None
    rows = []
    for k in range(start, nz):
        lab2d, n2d = ndi.label(m[k], structure=struct2d)
        base = len(dsu.parent)
        if n2d:
            area = np.bincount(lab2d.ravel(), minlength=n2d + 1)[1:]
            for a in area:
                dsu.add(float(a))
        glob = np.where(lab2d > 0, lab2d.astype(np.int64) - 1 + base, -1)
        n_merged = 0
        if prev is not None and n2d:
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    a = glob[
                        max(0, dj) : ny + min(0, dj),
                        max(0, di) : nx + min(0, di),
                    ]
                    b = prev[
                        max(0, -dj) : ny + min(0, -dj),
                        max(0, -di) : nx + min(0, -di),
                    ]
                    sel = (a >= 0) & (b >= 0)
                    if not sel.any():
                        continue
                    pairs = np.unique(np.stack([a[sel], b[sel]], axis=1), axis=0)
                    for x, y in pairs:
                        if dsu.find(int(x)) != dsu.find(int(y)):
                            n_merged += 1
                        dsu.union(int(x), int(y))
        glob_stack[k] = glob
        prev = glob

        roots = {dsu.find(i) for i in range(len(dsu.parent))}
        vols = np.sort(np.array([dsu.volume[r] for r in roots], dtype=float))[::-1]
        total = float(vols.sum())
        rows.append(
            {
                "z": k,
                "n_objects": int(vols.size),
                "total_volume": total,
                "largest_volume": float(vols[0]) if vols.size else 0.0,
                "largest_fraction": float(vols[0] / total) if total else 0.0,
                "n_new": int(n2d),
                "n_merged": n_merged,
            }
        )

    volumes = []
    table = pd.DataFrame(rows)
    # relabel final : racine -> 1..n
    roots = sorted({dsu.find(i) for i in range(len(dsu.parent))})
    remap = np.zeros(len(dsu.parent) + 1, dtype=np.int64)
    order = {r: i + 1 for i, r in enumerate(roots)}
    for i in range(len(dsu.parent)):
        remap[i] = order[dsu.find(i)]
    labels = np.where(glob_stack >= 0, remap[np.clip(glob_stack, 0, None)], 0)
    labels = np.moveaxis(labels, 0, axis)

    # on reconstruit les listes de volumes par coupe a partir de l'historique
    # (identique a `volumes_i_z`), en rejouant l'union-find
    dsu2 = _DSU()
    prev = None
    for k in range(start, nz):
        lab2d, n2d = ndi.label(m[k], structure=struct2d)
        base = len(dsu2.parent)
        if n2d:
            area = np.bincount(lab2d.ravel(), minlength=n2d + 1)[1:]
            for a in area:
                dsu2.add(float(a))
        glob = np.where(lab2d > 0, lab2d.astype(np.int64) - 1 + base, -1)
        if prev is not None and n2d:
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    a = glob[max(0, dj) : ny + min(0, dj), max(0, di) : nx + min(0, di)]
                    b = prev[max(0, -dj) : ny + min(0, -dj), max(0, -di) : nx + min(0, -di)]
                    sel = (a >= 0) & (b >= 0)
                    if sel.any():
                        pairs = np.unique(np.stack([a[sel], b[sel]], axis=1), axis=0)
                        for x, y in pairs:
                            dsu2.union(int(x), int(y))
        prev = glob
        rts = {dsu2.find(i) for i in range(len(dsu2.parent))}
        volumes.append(np.sort(np.array([dsu2.volume[r] for r in rts], dtype=float))[::-1])

    return ConnectivityProfile(
        table,
        volumes,
        labels,
        {"axis": axis, "start": start, "in_plane_connectivity": in_plane_connectivity},
    )


def voronoi_2d(
    phase,
    *,
    mask_out=None,
    voxel_size=None,
    method: str = "auto",
    return_table: bool = False,
):
    """Partitionne chaque coupe de la matrice entre ses pores voisins.

    Pour chaque coupe : les composantes connexes du **complementaire** de
    `phase` servent de germes, et la matrice est inondee par distance
    geodesique croissante depuis ces germes. Chaque voxel de matrice revient
    donc au pore le plus proche *dans la coupe*, en contournant les obstacles.
    Pour de l'os cortical, cela decoupe la matrice en territoires d'osteones.

    iMorph faisait la meme chose (`computeVoronoi2D`) mais inondait un relief
    inverse `max(d) - d`, c'est-a-dire en partant des cretes. Ici on inonde la
    distance elle-meme depuis les germes de bord, ce qui est la formulation
    directe du probleme — et le meme choix que pour le squelette de Plateau,
    ou il avait ramene le decalage des noeuds de 1,4 voxel a 0.

    Parameters
    ----------
    phase
        Masque de la matrice a decouper (`True` = matrice).
    mask_out
        Voxels a exclure completement (hors de l'os).

    Returns
    -------
    Volume
        Etiquettes, renumerotees independamment dans chaque coupe. Avec
        `return_table=True`, un couple `(labels, table)` ou `table` donne
        l'aire de chaque territoire par coupe.
    """
    p = as_array(phase).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = phase.voxel_size if isinstance(phase, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    dz, dy, dx = (float(v) for v in voxel_size)
    mo = None if mask_out is None else as_array(mask_out).astype(bool, copy=False)

    out = np.zeros(p.shape, dtype=np.int64)
    rows = []
    struct = np.ones((3, 3), dtype=bool)
    for k in range(p.shape[0]):
        sl = p[k]
        if mo is not None:
            sl = sl & ~mo[k]
        if not sl.any():
            continue
        pores = ~sl
        if mo is not None:
            pores = pores & ~mo[k]
        markers, n = ndi.label(pores, structure=struct)
        if n == 0:
            out[k] = sl.astype(np.int64)
            continue
        dist = distance_transform(sl[None, ...], voxel_size=(dz, dy, dx))
        lab = watershed(
            dist,
            markers[None, ...],
            mask=sl[None, ...],
            method=method,
            invert=False,
        )
        lab = np.asarray(lab)[0]
        out[k] = lab
        if return_table:
            ids, counts = np.unique(lab[lab > 0], return_counts=True)
            for i, c in zip(ids, counts, strict=True):
                rows.append({"z": k, "label": int(i), "area": float(c) * dy * dx})

    vol = Volume(out, voxel_size=(dz, dy, dx), name="voronoi_2d")
    if return_table:
        return vol, pd.DataFrame(rows, columns=["z", "label", "area"])
    return vol
