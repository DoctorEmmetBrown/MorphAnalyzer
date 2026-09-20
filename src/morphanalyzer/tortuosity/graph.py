"""Tortuosite sur graphe, par plus courts chemins de Dijkstra.

Portage de `Thread/Tortuosity/graph.cpp` (`tortuosityFromGraph`,
`tortuosityDeOufPlusNodes`) et de son usage dans `graphTortuosityModule`.

iMorph a du modifier Dijkstra pour accepter **plusieurs sources** — un plan de
depart plutot qu'un point — afin de pouvoir comparer aux tortuosites volumiques.
`scipy.sparse.csgraph.dijkstra` le fait nativement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["graph_tortuosity"]


def graph_tortuosity(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    axis: int = 0,
    margin: float | None = None,
    node_col: str = "node",
    coords=("z", "y", "x"),
) -> dict:
    """Tortuosite le long d'un graphe, d'une face a l'opposee.

    Parameters
    ----------
    nodes, edges
        Les tables rendues par `skeleton.plateau_skeleton` (ou toute table ayant
        un identifiant de noeud, des coordonnees, et des aretes `node_a`,
        `node_b`, `length`).
    axis
        Direction traversee : `0` pour z, etc.
    margin
        Epaisseur des couches d'entree et de sortie, dans l'unite des
        coordonnees. Par defaut un dixieme de l'etendue selon `axis`.

    Returns
    -------
    dict
        `tortuosity`, `std`, `n_sources`, `n_targets`, `n_reached`,
        `geodesic_over_euclidean`, et `paths_length` (la table des longueurs).

    Notes
    -----
    La these observe que les tortuosites sur graphe sont **superieures** aux
    tortuosites volumiques, « du fait que le squelette est centre sur les brins
    et que l'on force de ce fait le passage par des centres de noeuds alors qu'il
    y a surement des geodesiques plus courtes » (fig. 3.28). Les deux mesures
    suivent les memes variations angulaires mais ne se superposent pas : ne pas
    les comparer en valeur absolue.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    if nodes.empty or edges.empty:
        raise ValueError("graphe vide : rien a parcourir")

    ids = nodes[node_col].to_numpy()
    index = {int(v): n for n, v in enumerate(ids)}
    pos = nodes[list(coords)].to_numpy(dtype=float)

    a = np.array([index[int(v)] for v in edges["node_a"]])
    b = np.array([index[int(v)] for v in edges["node_b"]])
    if "length" in edges:
        w = edges["length"].to_numpy(dtype=float)
    else:
        w = np.linalg.norm(pos[a] - pos[b], axis=1)
    if np.any(w <= 0):
        raise ValueError("une arete de longueur nulle ou negative : Dijkstra l'interdit")

    n = len(ids)
    g = coo_matrix((w, (a, b)), shape=(n, n))
    g = g + g.T

    along = pos[:, axis]
    lo, hi = float(along.min()), float(along.max())
    if margin is None:
        margin = 0.1 * (hi - lo)
    sources = np.flatnonzero(along <= lo + margin)
    targets = np.flatnonzero(along >= hi - margin)
    if len(sources) == 0 or len(targets) == 0:
        raise ValueError("aucun noeud dans la couche d'entree ou de sortie : augmenter `margin`")

    dmat = dijkstra(g.tocsr(), directed=False, indices=sources, min_only=True)
    reach = dmat[targets]
    euclid = along[targets][:, None] - along[sources][None, :]
    # distance euclidienne minimale a une source, selon l'axe traverse
    euclid = np.abs(euclid).min(axis=1)

    ok = np.isfinite(reach) & (euclid > 0)
    if not ok.any():
        raise ValueError("aucun noeud de sortie atteignable depuis l'entree")
    ratio = reach[ok] / euclid[ok]
    tau = ratio**2
    return {
        "tortuosity": float(tau.mean()),
        "std": float(tau.std(ddof=0)),
        "geodesic_over_euclidean": float(ratio.mean()),
        "n_sources": int(len(sources)),
        "n_targets": int(len(targets)),
        "n_reached": int(ok.sum()),
        "paths_length": pd.DataFrame(
            {
                "node": ids[targets][ok],
                "geodesic": reach[ok],
                "euclidean": euclid[ok],
                "tortuosity": tau,
            }
        ),
    }
