"""Classification locale de forme — la piece originale d'iMorph.

La reference est la figure 3.35 de la these : un noeud a a~b~c, un brin a
a>>b~c, une plaque a a~b>>c. Les fantomes realisent exactement ces trois cas.
"""

import numpy as np
import pytest

import morphanalyzer as ma
from morphanalyzer.shape import NODE, PLATE, STRUT


@pytest.fixture(scope="module")
def cyl():
    return ma.phantoms.cylinders(shape=(64,) * 3, radius=5.0, n=4, axis=0, seed=0)


@pytest.fixture(scope="module")
def slab():
    return ma.phantoms.plate(shape=(48,) * 3, thickness=6.0, axis=0)


@pytest.fixture(scope="module")
def blob():
    return ma.phantoms.sphere(shape=(48,) * 3, radius=12.0)


# --- les trois formes canoniques ------------------------------------------


def test_cylinder_is_a_strut(cyl):
    st = ma.shape.local_shape_tensor(cyl, clip_ratios=False, propagate=False)
    s = st.seeds
    assert np.median(st.a_on_b[s]) > 2.0, "un cylindre doit etre franchement allonge"
    assert np.median(st.b_on_c[s]) == pytest.approx(1.0, abs=0.25), "sa section est ronde"
    cls = ma.shape.classify_solid(st, cyl.solid)
    assert (cls[st.seeds] == STRUT).mean() > 0.9


def _interior(shape, margin: int) -> np.ndarray:
    """Masque excluant une marge au bord du volume.

    La these restreint toutes ses statistiques morphometriques aux objets
    entierement inclus dans l'echantillon, pour la meme raison qu'ici : au bord,
    la boule de propagation est tronquee par la boite et la forme mesuree est
    celle de la troncature, pas celle de l'objet.
    """
    m = np.zeros(shape, dtype=bool)
    m[margin:-margin, margin:-margin, margin:-margin] = True
    return m


def test_plate_is_a_plate(slab):
    st = ma.shape.local_shape_tensor(slab, clip_ratios=False, propagate=False)
    s = st.seeds & _interior(slab.shape, 8)
    assert np.median(st.a_on_b[s]) < 1.6, "une plaque n'est pas allongee"
    assert np.median(st.b_on_c[s]) > 2.0, "mais elle est aplatie"
    cls = ma.shape.classify_solid(st, slab.solid)
    assert (cls[s] == PLATE).mean() > 0.95


def test_border_truncation_distorts_the_measure(slab):
    """Au bord de la boite, la boule est tronquee et la forme mesuree derive.

    Sur une plaque qui traverse tout le volume, les germes du bord sont classes
    brins au lieu de plaques : la troncature les allonge artificiellement. Ce
    n'est pas un defaut a corriger mais une limite a connaitre — d'ou la
    convention de la these, qui ecarte les objets touchant le bord.
    """
    st = ma.shape.local_shape_tensor(slab, clip_ratios=False, propagate=False)
    border = st.seeds & ~_interior(slab.shape, 8)
    inside = st.seeds & _interior(slab.shape, 8)
    assert border.sum() > 0 and inside.sum() > 0
    assert np.median(st.a_on_b[border]) > np.median(st.a_on_b[inside])


def test_sphere_is_a_node(blob):
    st = ma.shape.local_shape_tensor(blob, clip_ratios=False, propagate=False)
    s = st.seeds
    assert np.median(st.a_on_b[s]) == pytest.approx(1.0, abs=0.15)
    assert np.median(st.b_on_c[s]) == pytest.approx(1.0, abs=0.15)
    cls = ma.shape.classify_solid(st, blob.solid)
    assert (cls[st.seeds] == NODE).mean() > 0.9


# --- orientation ----------------------------------------------------------


def test_strut_orientation_follows_the_cylinder_axis(cyl):
    """Cylindres paralleles a z : elevation 90 degres."""
    st = ma.shape.local_shape_tensor(cyl, propagate=False)
    s = st.seeds
    assert np.median(st.phi[s]) == pytest.approx(90.0, abs=5.0)


def test_plate_principal_axis_lies_in_the_plane(slab):
    """Plaque perpendiculaire a z : le grand axe est dans le plan, elevation 0."""
    st = ma.shape.local_shape_tensor(slab, propagate=False)
    s = st.seeds & _interior(slab.shape, 8)
    assert np.median(st.phi[s]) == pytest.approx(0.0, abs=10.0)


def test_orientation_can_be_skipped(blob):
    st = ma.shape.local_shape_tensor(blob, with_orientation=False, propagate=False)
    assert st.direction is None
    with pytest.raises(ValueError, match="with_orientation"):
        ma.shape.strut_orientation(st)


# --- la mousse : brins et noeuds ensemble ---------------------------------


def test_foam_separates_struts_from_nodes(foam_struts):
    """Au point de mesure, a/b separe les brins des noeuds.

    La verite terrain vient de la loi de Plateau, exacte par construction dans
    ce fantome : 3 germes de Voronoi equidistants = une arete = un brin,
    4 germes equidistants = un sommet = un noeud.
    """
    truth = foam_struts.meta["truth"]
    st = ma.shape.local_shape_tensor(foam_struts, clip_ratios=False, propagate=False)
    struts = truth["strut_mask"] & st.seeds
    nodes = truth["node_mask"] & st.seeds
    assert struts.sum() > 100 and nodes.sum() > 50

    med_s = np.median(st.a_on_b[struts])
    med_n = np.median(st.a_on_b[nodes])
    assert med_s > 2.0, f"brins pas assez allonges (a/b median {med_s:.2f})"
    assert med_n < 1.5, f"noeuds trop allonges (a/b median {med_n:.2f})"
    assert med_s > med_n + 0.7


def test_imorph_threshold_is_high_precision(foam_struts):
    """Le seuil 1,6 de la these privilegie la precision sur le rappel.

    Mesure sur ce fantome : environ 95 % de ce qui est declare brin en est un,
    au prix d'un quart des brins manques. C'est le bon compromis quand la suite
    du traitement mesure des diametres et des orientations de brins : mieux
    vaut un echantillon plus petit que contamine par des noeuds.
    """
    truth = foam_struts.meta["truth"]
    st = ma.shape.local_shape_tensor(foam_struts, clip_ratios=False, propagate=False)
    struts = truth["strut_mask"] & st.seeds
    nodes = truth["node_mask"] & st.seeds
    pred = st.a_on_b >= 1.6
    tp = int(pred[struts].sum())
    fp = int(pred[nodes].sum())
    precision = tp / (tp + fp)
    recall = tp / int(struts.sum())
    assert precision > 0.90, f"precision {precision:.3f}"
    assert 0.6 < recall < 0.9, f"rappel {recall:.3f}"


def test_a_long_strut_far_from_any_node_is_strongly_elongated(foam_struts):
    """Controle direct : loin d'une jonction, a/b doit etre franchement grand."""
    truth = foam_struts.meta["truth"]
    st = ma.shape.local_shape_tensor(foam_struts, clip_ratios=False, propagate=False)
    d_node = np.asarray(ma.distance.distance_transform(~truth["node_mask"]))
    far = st.seeds & (d_node > 8)
    assert far.sum() > 50
    assert np.median(st.a_on_b[far]) > 3.0


# --- mecanique ------------------------------------------------------------


def test_propagation_covers_the_whole_solid(cyl):
    st = ma.shape.local_shape_tensor(cyl, propagate=True)
    assert (st.a_on_b[cyl.solid] > 0).all(), "des voxels de solide sans valeur"
    assert (st.a_on_b[~cyl.solid] == 0).all(), "des valeurs hors du solide"
    assert st.params["propagated"] is True


def test_clipping_bounds_the_ratios(cyl):
    st = ma.shape.local_shape_tensor(cyl, clip_ratios=True, expand_factor=3.0, propagate=False)
    assert st.a_on_b.max() <= 3.0 + 1e-6
    free = ma.shape.local_shape_tensor(cyl, clip_ratios=False, propagate=False)
    assert free.a_on_b.max() > 3.0


def test_axes_scale_with_voxel_size():
    """Les demi-axes sont des longueurs physiques, pas des comptes de voxels."""
    a = ma.phantoms.cylinders(shape=(48,) * 3, radius=4.0, n=2, seed=0, voxel_size=1.0)
    b = ma.phantoms.cylinders(shape=(48,) * 3, radius=4.0, n=2, seed=0, voxel_size=3.0)
    ta = ma.shape.local_shape_tensor(a, propagate=False)
    tb = ma.shape.local_shape_tensor(b, propagate=False)
    assert np.median(tb.a[tb.seeds]) == pytest.approx(3.0 * np.median(ta.a[ta.seeds]), rel=0.02)
    # les rapports, eux, sont sans dimension
    assert np.median(tb.a_on_b[tb.seeds]) == pytest.approx(np.median(ta.a_on_b[ta.seeds]), rel=0.1)


def test_two_class_mode_matches_imorph(cyl):
    """`plate_threshold=None` retrouve la classification binaire d'iMorph."""
    st = ma.shape.local_shape_tensor(cyl, propagate=False)
    cls = ma.shape.classify_solid(st, cyl.solid, plate_threshold=None)
    assert set(np.unique(cls)).issubset({0, NODE, STRUT})


def test_seed_strategy_is_recorded(blob):
    st = ma.shape.local_shape_tensor(blob, propagate=False)
    # sur une sphere, skimage rend un squelette vide : le repli doit s'activer
    assert st.params["seeds"] in ("auto->skeleton", "auto->ridge")
    assert st.n_seeds > 0


def test_explicit_seeds_are_honoured(cyl):
    seeds = np.zeros(cyl.shape, dtype=bool)
    idx = np.argwhere(cyl.solid)[::500]
    seeds[tuple(idx.T)] = True
    st = ma.shape.local_shape_tensor(cyl, seeds=seeds, propagate=False)
    assert st.n_seeds == int(seeds.sum())
    assert st.params["seeds"] == "explicite"


def test_bad_seed_strategy_is_rejected(cyl):
    with pytest.raises(ValueError, match="seeds doit valoir"):
        ma.shape.local_shape_tensor(cyl, seeds="magique")


def test_to_frame_gives_one_row_per_seed(cyl):
    st = ma.shape.local_shape_tensor(cyl, propagate=False)
    df = st.to_frame()
    assert len(df) == st.n_seeds
    assert {"a", "b", "c", "a_on_b", "b_on_c", "theta", "phi"} <= set(df.columns)
