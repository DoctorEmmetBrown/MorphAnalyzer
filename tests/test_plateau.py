"""Squelettisation par la loi de Plateau.

Le fantome de Voronoi rend cette loi **exacte** : les brins sont les aretes du
diagramme, les noeuds ses sommets. On peut donc verifier non seulement que le
squelette est plausible, mais qu'il tombe au bon endroit.
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

import morphanalyzer as ma


@pytest.fixture(scope="module")
def plateau(segmented):
    return ma.skeleton.plateau_skeleton(segmented["vol"].solid, segmented["cells"])


def test_nodes_and_struts_are_in_the_solid(plateau, segmented):
    solid = segmented["vol"].solid
    assert plateau.node_mask.sum() > 0
    assert plateau.strut_mask.sum() > 0
    assert (solid[plateau.node_mask]).all()
    assert (solid[plateau.strut_mask]).all()
    assert not (plateau.node_mask & plateau.strut_mask).any()


def test_labels_are_propagated_through_the_whole_solid(plateau, segmented):
    solid = segmented["vol"].solid
    assert (plateau.labels_in_solid[solid] > 0).all()
    assert (plateau.labels_in_solid[~solid] == 0).all()


def test_node_degree_concentrates_on_three_and_four(plateau):
    """A l'interface de quatre cellules il y a un noeud, donc au plus quatre brins.

    Le degre se concentre sur 3 et 4. Il ne vaut pas 4 partout parce qu'un noeud
    de degre 4 exige que ses quatre voisins existent aussi : au bord de la boite
    il manque des cellules, donc des noeuds, donc des aretes. C'est la limite que
    `plateau_skeleton` documente, et que la these signale egalement (§3.3.1).
    """
    assert len(plateau.nodes) > 20
    assert len(plateau.edges) > 20
    deg = (
        plateau.edges["node_a"]
        .value_counts()
        .add(plateau.edges["node_b"].value_counts(), fill_value=0)
        .reindex(plateau.nodes["node"], fill_value=0)
    )
    assert deg.mode().iloc[0] in (3, 4)
    assert 2.8 < deg.mean() < 4.3
    assert deg.isin([3, 4]).mean() > 0.6
    assert (deg <= 4).mean() > 0.9


def test_each_node_carries_four_cells(plateau):
    sizes = plateau.nodes["cells"].map(len)
    assert sizes.max() <= 4
    assert (sizes == 4).mean() > 0.8


def test_nodes_land_on_the_voronoi_vertices(plateau, segmented):
    """La verite terrain : les sommets de Voronoi, ou 4 germes sont equidistants."""
    truth = segmented["vol"].meta["truth"]
    d_true = ndi.distance_transform_edt(~truth["node_mask"])
    off = np.array(
        [
            d_true[int(round(r.z)), int(round(r.y)), int(round(r.x))]
            for r in plateau.nodes.itertuples()
        ]
    )
    assert np.median(off) <= 1.0, f"ecart median {np.median(off):.2f} voxels"
    assert (off <= 2.0).mean() > 0.75


def test_struts_never_stray_outside_the_plateau_classification(plateau, segmented):
    """Les voxels de brin appartiennent tous au solide classe par Plateau.

    On ne peut pas comparer terme a terme le squelette et la verite terrain
    geometrique : le squelette est un **lieu** (une courbe), alors que
    `strut_mask` et `node_mask` du fantome sont des **regions** epaisses, definies
    par un critere metrique (`d3 - d1 < strut`). Un voxel du squelette situe pres
    d'une jonction tombe donc legitimement dans la region « noeud », plus large
    que le noeud topologique.

    Ce qui doit etre vrai : aucun voxel de brin predit ne tombe en dehors de la
    reunion des deux regions vraies. Mesure ici : exactement 0.
    """
    truth = segmented["vol"].meta["truth"]
    classified = truth["strut_mask"] | truth["node_mask"]
    stray = (plateau.strut_mask & ~classified).sum()
    assert stray == 0, f"{stray} voxels de brin hors de la classification vraie"
    on_edges = (plateau.strut_mask & truth["strut_mask"]).sum() / plateau.strut_mask.sum()
    assert on_edges > 0.6, f"seulement {on_edges:.2f} sur les aretes de Voronoi"


def test_strut_locus_is_thin(plateau, segmented):
    """Le lieu des brins doit rester mince devant le volume du solide."""
    solid = segmented["vol"].solid
    assert plateau.strut_mask.sum() < 0.15 * solid.sum()


def test_strut_length_matches_the_cell_spacing(plateau):
    """Les brins relient des cellules voisines : leur longueur suit l'espacement."""
    assert 5.0 < plateau.edges["length"].median() < 40.0


def test_merging_reduces_the_node_count(segmented):
    a = ma.skeleton.plateau_skeleton(segmented["vol"].solid, segmented["cells"], merge_distance=0.0)
    b = ma.skeleton.plateau_skeleton(segmented["vol"].solid, segmented["cells"], merge_distance=5.0)
    assert len(b.nodes) <= len(a.nodes)


def test_skeleton_graph_requires_three_shared_cells():
    import pandas as pd

    nodes = pd.DataFrame(
        {
            "node": [1, 2, 3],
            "z": [0.0, 10.0, 0.0],
            "y": [0.0, 0.0, 10.0],
            "x": [0.0, 0.0, 0.0],
            "cells": [(1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 6, 7)],
        }
    )
    edges = ma.skeleton.skeleton_graph(nodes)
    pairs = set(zip(edges["node_a"], edges["node_b"], strict=True))
    assert (1, 2) in pairs  # partagent 1, 2, 3
    assert (1, 3) not in pairs  # ne partagent que 1, 2
    assert edges.loc[0, "length"] == pytest.approx(10.0)


def test_plateau_on_a_single_wall_finds_no_node():
    """Deux cellules seulement : il y a un col, pas de noeud."""
    solid = np.zeros((20, 20, 20), dtype=bool)
    solid[:, 9:11, :] = True
    cells = np.zeros((20, 20, 20), dtype=np.int32)
    cells[:, :9, :] = 1
    cells[:, 11:, :] = 2
    sk = ma.skeleton.plateau_skeleton(solid, cells)
    assert sk.node_mask.sum() == 0
    assert sk.strut_mask.sum() == 0
    assert len(sk.nodes) == 0
    assert len(sk.edges) == 0
