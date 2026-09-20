"""Mousse a cellules ouvertes par diagramme de Voronoi.

Pourquoi ce fantome est le plus utile du lot : par construction, la **loi de
Plateau est exacte**. Les brins sont les aretes de Voronoi (3 germes
equidistants), les noeuds sont les sommets (4 germes equidistants), les cols
sont les faces (2 germes). La partition de Voronoi donne donc la verite terrain
exacte pour :

  - la segmentation des cellules (watershed sur la carte de distance) ;
  - le nombre de cellules et leur volume ;
  - la squelettisation par loi de Plateau (positions des noeuds et des brins) ;
  - la classification de forme (brin vs noeud) ;
  - la connectivite et le graphe pores-cols.

Aucun tomogramme ne permet cela.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from morphanalyzer.core import Volume

__all__ = ["voronoi_foam"]


def voronoi_foam(
    shape: tuple[int, int, int] = (128, 128, 128),
    n_cells: int = 30,
    strut: float = 4.0,
    seed: int = 0,
    min_seed_gap: float = 16.0,
    voxel_size: float = 1.0,
    periodic_pad: bool = True,
) -> Volume:
    """Mousse a cellules ouvertes : solide sur les aretes de Voronoi.

    Parameters
    ----------
    n_cells
        Nombre de germes *dans* la boite. Des germes images sont ajoutes autour
        (si `periodic_pad`) pour que les cellules de bord soient correctes.
    strut
        Epaisseur caracteristique des brins, en voxels. Un voxel est solide si
        `d3 - d1 < strut`, ou `d1 <= d2 <= d3` sont les distances aux trois
        germes les plus proches : la condition selectionne le voisinage des
        aretes de Voronoi.
    min_seed_gap
        Distance minimale entre germes, pour eviter des cellules degenerees.

    Returns
    -------
    Volume
        Volume binaire (True = solide). `meta["truth"]` contient :
        `seeds`, `n_cells`, `cell_labels` (partition de Voronoi restreinte au
        fluide, labels 1..n), `cell_volumes`, `porosity`,
        `node_mask` et `strut_mask`, la classification de forme exacte deduite
        de la loi de Plateau, et `node_fraction`.
    """
    rng = np.random.default_rng(seed)
    box = np.array(shape, dtype=float)

    seeds: list[np.ndarray] = []
    for _ in range(200000):
        if len(seeds) >= n_cells:
            break
        c = rng.uniform(np.zeros(3), box)
        if all(np.linalg.norm(c - o) >= min_seed_gap for o in seeds):
            seeds.append(c)
    if len(seeds) < n_cells:
        raise RuntimeError(
            f"seulement {len(seeds)}/{n_cells} germes places ; reduire n_cells ou min_seed_gap"
        )
    seeds_arr = np.asarray(seeds)

    # germes images : replique les germes dans les 26 boites voisines pour que
    # les cellules coupees par le bord aient la bonne geometrie
    if periodic_pad:
        shifts = np.array(
            [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], dtype=float
        )
        all_seeds = np.concatenate([seeds_arr + s * box for s in shifts], axis=0)
        owner = np.tile(np.arange(len(seeds_arr)), len(shifts))
    else:
        all_seeds = seeds_arr
        owner = np.arange(len(seeds_arr))

    tree = cKDTree(all_seeds)
    coords = np.stack(
        np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    d, idx = tree.query(coords, k=4, workers=-1)

    d1, _d2, d3, d4 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    solid = (d3 - d1) < strut
    # Loi de Plateau, exacte par construction : 3 germes equidistants = une
    # arete de Voronoi donc un brin, 4 germes equidistants = un sommet donc un
    # noeud. C'est la verite terrain de la classification de forme.
    node_truth = solid & ((d4 - d1) < strut)
    strut_truth = solid & ~node_truth

    # Cellules = regions de Voronoi restreintes au fluide. On etiquette par
    # *image* de germe et non par germe source : deux images d'un meme germe
    # sont deux regions disjointes de la boite, les confondre produirait des
    # cellules fragmentees. Les labels sont renumerotes 1..n.
    raw = np.where(solid, -1, idx[:, 0])
    present = np.unique(raw[raw >= 0])
    remap = np.zeros(len(all_seeds), dtype=np.int32)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    labels = np.where(raw >= 0, remap[np.maximum(raw, 0)], 0)

    solid = solid.reshape(shape)
    node_truth = node_truth.reshape(shape)
    strut_truth = strut_truth.reshape(shape)
    labels = labels.reshape(shape).astype(np.int32)

    # Aux coins ou trois regions se rejoignent, les egalites de distance
    # produisent quelques voxels isoles portant le label d'une cellule voisine.
    # On ne garde que la composante principale de chaque label et on reattribue
    # les eclats au label le plus proche, pour que la verite terrain soit
    # exactement ce qu'elle annonce : des cellules convexes et connexes.
    labels, n_specks = _keep_main_component(labels, solid)

    uniq, counts = np.unique(labels[labels > 0], return_counts=True)
    cell_volumes = dict(zip(uniq.tolist(), (counts * voxel_size**3).tolist(), strict=True))

    # Cellules entierement incluses dans la boite : la convention de la these
    # pour toutes les statistiques morphometriques (evite les effets de bord).
    border = np.zeros(shape, dtype=bool)
    border[[0, -1], :, :] = True
    border[:, [0, -1], :] = True
    border[:, :, [0, -1]] = True
    touching = set(np.unique(labels[border & (labels > 0)]).tolist())
    interior = sorted(set(uniq.tolist()) - touching)

    seed_of_label = np.zeros(len(present) + 1, dtype=np.int64)
    seed_of_label[1:] = owner[present]
    nvox = int(np.prod(shape))

    return Volume(
        solid,
        voxel_size=voxel_size,
        name="voronoi_foam",
        meta={
            "truth": {
                "seeds": seeds_arr,
                "n_cells": int(len(uniq)),
                "n_interior_cells": len(interior),
                "interior_cells": interior,
                "n_seeds_requested": n_cells,
                "cell_labels": labels,
                "cell_volumes": cell_volumes,
                "seed_of_label": seed_of_label,
                "seed_positions": all_seeds[present],
                "porosity": 1.0 - float(solid.sum()) / nvox,
                "strut": strut,
                "n_specks_reassigned": int(n_specks),
                "node_mask": node_truth,
                "strut_mask": strut_truth,
                "node_fraction": float(node_truth.sum()) / max(int(solid.sum()), 1),
                "strut_condition": "d3 - d1 < strut  (3 germes equidistants -> arete)",
                "node_condition": "d4 - d1 < strut  (4 germes equidistants -> sommet)",
                "plateau_law_exact": True,
            }
        },
    )


def _keep_main_component(labels: np.ndarray, solid: np.ndarray) -> tuple[np.ndarray, int]:
    """Ne garde que la composante principale de chaque label ; recolle le reste.

    Les eclats sont reattribues par **dilatation morphologique** des labels
    nettoyes, et non par plus proche voisin euclidien : la dilatation ne peut
    attribuer un voxel qu'a un label auquel il est adjacent, ce qui garantit par
    construction que chaque cellule reste d'un seul tenant. Une reattribution
    euclidienne, elle, peut recoller un eclat a une cellule qu'il ne touche pas
    et recreer une fragmentation.
    """
    from scipy import ndimage as ndi

    out = labels.copy()
    n_specks = 0
    for lab in np.unique(labels[labels > 0]):
        m = labels == lab
        cc, n = ndi.label(m)
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        drop = m & (cc != int(sizes.argmax()))
        n_specks += int(drop.sum())
        out[drop] = 0

    st = ndi.generate_binary_structure(3, 1)
    for _ in range(64):
        orphan = (out == 0) & ~solid
        if not orphan.any():
            break
        grown = ndi.grey_dilation(out, footprint=st)
        if not (orphan & (grown > 0)).any():
            break
        out = np.where(orphan & (grown > 0), grown, out)

    # Ce qui reste orphelin est une poche de fluide entierement close par du
    # solide : c'est une cellule fermee a part entiere, pas un eclat. On lui
    # attribue un label neuf plutot que de la laisser a zero.
    orphan = (out == 0) & ~solid
    if orphan.any():
        cc, n = ndi.label(orphan)
        out = np.where(cc > 0, cc + int(out.max()), out)
    return out, n_specks
