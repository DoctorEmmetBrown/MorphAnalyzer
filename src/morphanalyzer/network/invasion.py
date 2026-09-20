"""Percolation d'invasion sur le reseau de pores.

Portage de `PhysicalModules/PoreNetworkModelling/invasionIPThread.cpp`
(786 lignes) : amas liquides, coalescence, piegeage de la phase defendante, et
le « taux de deformation » des cols propre a iMorph.

Le reseau attendu est celui que produit
:func:`morphanalyzer.segmentation.pore_network` : une table de cellules et une
table de cols.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array
from morphanalyzer.network.capillarity import capillary_pressure
from morphanalyzer.network.drainage import face_slab

__all__ = [
    "InvasionResult",
    "invasion_percolation",
    "face_cells",
    "filling_map",
    "deformed_throat_radius",
]

_TOL = 1e-3


def face_cells(labels, face: int, *, min_voxels: int = 1) -> np.ndarray:
    """Etiquettes de cellules presentes sur une face du volume.

    `face` suit la convention `0..5` (`0` = z minimal).
    """
    lab = as_array(labels)
    slab = lab[face_slab(lab.shape, face)]
    ids, counts = np.unique(slab, return_counts=True)
    keep = (ids > 0) & (counts >= min_voxels)
    return ids[keep]


def deformed_throat_radius(radius, *, deformation_rate: float, r_max: float):
    """Rayon de col effectif sous le modele de deformation d'iMorph.

    iMorph multipliait le rayon de chaque col par un facteur `k` qui croit
    lineairement avec la pression capillaire : `k = 1` a la pression la plus
    basse (celle du plus grand pore d'entree, rayon `r_max`) et
    `k = deformation_rate` a la pression la plus haute (conventionnellement un
    pore de rayon 0,5 voxel). Le milieu se laisse donc d'autant plus forcer que
    la pression monte — un modele de mousse elastique.

    Comme `Pc` varie en `1/r`, on a `k(r) = A/r + B` avec

    .. math:: A = \\frac{D - 1}{2 - 1/r_{max}}, \\qquad B = D - 2A

    ou `D = deformation_rate`. Ni la tension superficielle ni la taille de
    voxel n'interviennent : elles se simplifient.

    Un col de rayon `R` laisse passer au rayon courant `r` des que
    `k(r) R >= r`, c'est-a-dire `AR/r + BR >= r`, soit `r <= r_eff` avec

    .. math:: r_{eff} = \\frac{BR + \\sqrt{B^2R^2 + 4AR}}{2}

    Ce rayon effectif est calcule une fois pour toutes, ce qui evite la boucle
    de pression pas a pas d'iMorph tout en donnant le meme ordre d'invasion.

    Avec `deformation_rate=1` (defaut), `A = 0`, `B = 1` et `r_eff = R`.

    Warnings
    --------
    Le rayon de reference 0,5 est exprime en **voxels**. Si les rayons sont
    donnes en unites physiques, convertir `deformation_rate` n'a pas de sens :
    travailler en voxels.
    """
    r = np.asarray(radius, dtype=float)
    d = float(deformation_rate)
    if d == 1.0:
        return r
    if d < 1.0:
        raise ValueError("deformation_rate doit valoir au moins 1")
    if not np.isfinite(r_max) or r_max <= 0:
        raise ValueError("r_max doit etre fini et strictement positif")
    a = (d - 1.0) / (2.0 - 1.0 / float(r_max))
    b = d - 2.0 * a
    return 0.5 * (b * r + np.sqrt((b * r) ** 2 + 4.0 * a * r))


@dataclass
class InvasionResult:
    """Sortie d'une percolation d'invasion.

    Attributes
    ----------
    cells
        Table indexee par `label` : `volume`, `radius`, `filling_radius`
        (`NaN` si jamais envahie), `invaded`, `trapped`, `order` (rang
        d'invasion, `-1` si non envahie).
    curve
        Saturation cumulee en fonction du rayon, du plus grand au plus petit.
    """

    cells: pd.DataFrame
    curve: pd.DataFrame
    params: dict = field(default_factory=dict)

    @property
    def n_invaded(self) -> int:
        return int(self.cells["invaded"].sum())

    @property
    def final_saturation(self) -> float:
        v = self.cells["volume"].to_numpy(dtype=float)
        return float(v[self.cells["invaded"].to_numpy()].sum() / v.sum()) if v.sum() else 0.0


def _radius_column(table: pd.DataFrame, spec, n: int, what: str) -> np.ndarray:
    if spec is None:
        if "d_equivalent" not in table.columns:
            raise KeyError(
                f"pas de colonne 'd_equivalent' pour deduire le rayon des {what} ; "
                f"passer {what[:-1]}_radius explicitement"
            )
        return table["d_equivalent"].to_numpy(dtype=float) / 2.0
    if isinstance(spec, str):
        return table[spec].to_numpy(dtype=float)
    arr = np.asarray(spec, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"le rayon des {what} doit avoir {n} valeurs, pas {arr.shape}")
    return arr


def invasion_percolation(
    cells: pd.DataFrame,
    throats: pd.DataFrame,
    *,
    inlet,
    outlet=None,
    trapping: bool = False,
    deformation_rate: float = 1.0,
    cell_radius=None,
    throat_radius=None,
    volume: str = "volume",
    surface_tension: float | None = None,
    contact_angle: float = 0.0,
    unit: str = "um",
    convention: str = "laplace",
) -> InvasionResult:
    """Invasion quasi-statique d'un fluide non mouillant dans le reseau.

    A chaque etape, l'amas envahisseur franchit le **plus grand col
    accessible** ; la pression correspondante ne fait donc que croitre et le
    rayon enregistre ne fait que decroitre. C'est exactement l'ordre que
    produisait la boucle en pression d'iMorph (`r_current -= 0.2`), a ceci pres
    qu'une file de priorite evite de balayer une grille de pressions.

    L'entree dans une cellule de la face d'injection est commandee par le rayon
    de la **cellule** ; le passage d'une cellule a l'autre, par le rayon du
    **col**. C'est la regle d'`initClustersAtFace` / `foundToBeEmbeded`.

    Parameters
    ----------
    cells
        Table des cellules, avec au minimum `label` et `volume`
        (:func:`morphanalyzer.segmentation.cell_morphometry`).
    throats
        Table des cols, avec `label_a`, `label_b`
        (:func:`morphanalyzer.segmentation.throats`).
    inlet, outlet
        Etiquettes des cellules d'injection et de sortie. Voir
        :func:`face_cells`. `outlet` n'est requis que si `trapping=True`.
    trapping
        Piege la phase defendante : une cellule non envahie dont aucun chemin
        de cellules non envahies ne mene a la sortie devient inaccessible.
        Cout : un calcul de composantes connexes a chaque baisse de pression.
    deformation_rate
        Modele de col deformable d'iMorph, voir
        :func:`deformed_throat_radius`. `1.0` = cols rigides.
    cell_radius, throat_radius
        Nom de colonne, tableau, ou `None`. Par defaut `d_equivalent / 2`.
        Attention : le rayon equivalent d'une cellule (sphere de meme volume)
        n'est pas son rayon de boule inscrite, qui est ce que stockait iMorph
        (`distMax`). Pour reproduire iMorph, passer le rayon issu de la carte
        d'ouverture.

    Returns
    -------
    InvasionResult
    """
    if volume not in cells.columns:
        raise KeyError(f"la table des cellules n'a pas de colonne {volume!r}")
    lab = cells["label"].to_numpy()
    n = len(lab)
    index = {int(v): i for i, v in enumerate(lab)}
    vol = cells[volume].to_numpy(dtype=float)
    r_cell = _radius_column(cells, cell_radius, n, "cellules")

    inlet_idx = np.array([index[int(v)] for v in np.ravel(inlet) if int(v) in index], dtype=int)
    if inlet_idx.size == 0:
        raise ValueError("aucune cellule d'injection valide : verifier `inlet`")
    if trapping:
        if outlet is None:
            raise ValueError("`outlet` est obligatoire quand trapping=True")
        outlet_idx = np.array(
            [index[int(v)] for v in np.ravel(outlet) if int(v) in index], dtype=int
        )
        if outlet_idx.size == 0:
            raise ValueError("aucune cellule de sortie valide : verifier `outlet`")
    else:
        outlet_idx = np.array([], dtype=int)

    r_max = float(r_cell[inlet_idx].max())
    if len(throats):
        ta = np.array([index.get(int(v), -1) for v in throats["label_a"]], dtype=int)
        tb = np.array([index.get(int(v), -1) for v in throats["label_b"]], dtype=int)
        ok = (ta >= 0) & (tb >= 0)
        ta, tb = ta[ok], tb[ok]
        r_throat = _radius_column(throats, throat_radius, len(throats), "cols")[ok]
        r_throat = np.asarray(
            deformed_throat_radius(r_throat, deformation_rate=deformation_rate, r_max=r_max),
            dtype=float,
        )
    else:
        ta = tb = np.array([], dtype=int)
        r_throat = np.array([], dtype=float)

    # listes d'adjacence : pour chaque cellule, (voisin, rayon effectif du col)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b, r in zip(ta, tb, r_throat, strict=True):
        adj[a].append((b, float(r)))
        adj[b].append((a, float(r)))

    invaded = np.zeros(n, dtype=bool)
    trapped = np.zeros(n, dtype=bool)
    fill = np.full(n, np.nan, dtype=float)
    order = np.full(n, -1, dtype=int)

    heap: list[tuple[float, int]] = [(-float(r_cell[i]), int(i)) for i in inlet_idx]
    heapq.heapify(heap)

    level = np.inf
    rank = 0
    curve_r: list[float] = []
    curve_v: list[float] = []
    invaded_volume = 0.0

    while heap:
        neg_r, i = heapq.heappop(heap)
        if invaded[i] or trapped[i]:
            continue
        r = -neg_r
        if r < level:
            level = r
            if trapping:
                _update_trapped(n, adj, invaded, trapped, outlet_idx)
                if trapped[i]:
                    continue
        invaded[i] = True
        fill[i] = level
        order[i] = rank
        rank += 1
        invaded_volume += float(vol[i])
        curve_r.append(level)
        curve_v.append(invaded_volume)
        for j, rt in adj[i]:
            if not invaded[j] and not trapped[j]:
                heapq.heappush(heap, (-rt, j))

    total = float(vol.sum())
    table = pd.DataFrame(
        {
            "label": lab,
            "volume": vol,
            "radius": r_cell,
            "filling_radius": fill,
            "invaded": invaded,
            "trapped": trapped,
            "order": order,
        }
    ).set_index("label")

    curve = pd.DataFrame({"radius": curve_r, "invaded_volume": curve_v})
    curve = curve.groupby("radius", as_index=False)["invaded_volume"].max()
    curve = curve.sort_values("radius", ascending=False, ignore_index=True)
    curve["diameter"] = 2.0 * curve["radius"]
    curve["saturation"] = curve["invaded_volume"] / total if total else 0.0
    if surface_tension is not None:
        curve["pressure"] = capillary_pressure(
            curve["radius"].to_numpy(),
            surface_tension=surface_tension,
            contact_angle=contact_angle,
            unit=unit,
            convention=convention,
        )
    curve = curve[
        ["radius", "diameter", "invaded_volume", "saturation"]
        + (["pressure"] if surface_tension is not None else [])
    ]

    return InvasionResult(
        table,
        curve,
        {
            "trapping": trapping,
            "deformation_rate": deformation_rate,
            "n_inlet": int(inlet_idx.size),
            "n_outlet": int(outlet_idx.size),
            "r_max": r_max,
            "total_volume": total,
        },
    )


def _update_trapped(n, adj, invaded, trapped, outlet_idx) -> None:
    """Marque comme piegees les cellules non envahies coupees de la sortie."""
    reach = np.zeros(n, dtype=bool)
    stack = [int(i) for i in outlet_idx if not invaded[i]]
    for i in stack:
        reach[i] = True
    while stack:
        i = stack.pop()
        for j, _r in adj[i]:
            if not invaded[j] and not reach[j]:
                reach[j] = True
                stack.append(j)
    np.logical_or(trapped, ~invaded & ~reach, out=trapped)


def filling_map(labels, result: InvasionResult, *, voxel_size=None, fill_value: float = 0.0):
    """Peint le rayon d'envahissement de chaque cellule sur le volume segmente.

    Les voxels de fond et les cellules jamais envahies recoivent `fill_value`.
    """
    lab = as_array(labels)
    if voxel_size is None:
        voxel_size = labels.voxel_size if isinstance(labels, Volume) else (1.0, 1.0, 1.0)
    lut = np.full(int(lab.max()) + 1, fill_value, dtype=np.float32)
    idx = result.cells.index.to_numpy()
    val = result.cells["filling_radius"].to_numpy(dtype=float)
    good = np.isfinite(val) & (idx <= lab.max())
    lut[idx[good]] = val[good].astype(np.float32)
    out = lut[np.clip(lab, 0, len(lut) - 1)]
    out[lab <= 0] = fill_value
    return Volume(out, voxel_size=voxel_size, name="filling_radius")
