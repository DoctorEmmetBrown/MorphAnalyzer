"""Tortuosites.

Definition de Carman : le **carre** du rapport de la longueur geodesique a la
distance euclidienne. Les fantomes de tube donnent la reponse exacte.
"""

import numpy as np
import pytest

import morphanalyzer as ma

# --- le fast marching lui-meme --------------------------------------------


def test_fast_marching_is_exact_along_an_axis():
    free = np.ones((32, 32, 32), dtype=bool)
    T = np.asarray(ma.distance.geodesic_distance(free, (16, 16, 16)))
    assert T[16, 16, 30] == pytest.approx(14.0, abs=1e-4)
    assert T[16, 16, 16] == 0.0


def test_first_order_scheme_overestimates_diagonals():
    """Le biais du schema du 1er ordre, figé pour qu'une regression se voie.

    Le fast marching du 1er ordre surestime les distances obliques : environ
    +2,7 % sur une diagonale 2D, +3,8 % sur une diagonale 3D, exact sur un axe.
    La these mesure le meme ordre de grandeur (jusqu'a 2,77 voxels, fig. 3.19).
    C'est pourquoi `point_tortuosity` normalise par le front libre plutot que par
    la distance euclidienne.
    """
    free = np.ones((64, 64, 64), dtype=bool)
    T = np.asarray(ma.distance.geodesic_distance(free, (32, 32, 32)))
    assert T[32, 32, 62] == pytest.approx(30.0, abs=1e-3)
    assert T[32, 62, 62] / (np.sqrt(2) * 30) == pytest.approx(1.027, abs=0.01)
    assert T[62, 62, 62] / (np.sqrt(3) * 30) == pytest.approx(1.038, abs=0.01)


def test_geodesic_goes_around_an_obstacle():
    """Une cloison percee : la geodesique passe par le trou, pas a travers."""
    m = np.ones((40, 40, 40), dtype=bool)
    m[20, :, :] = False
    m[20, 19:22, 19:22] = True  # une fenetre au centre
    T = np.asarray(ma.distance.geodesic_distance(m, (5, 5, 5)))
    direct = np.linalg.norm(np.array([35, 5, 5]) - np.array([5, 5, 5]))
    assert np.isfinite(T[35, 5, 5])
    assert T[35, 5, 5] > 1.5 * direct, "la geodesique a traverse la cloison"


def test_travel_time_rejects_sources_outside_the_mask():
    m = np.zeros((16, 16, 16), dtype=bool)
    m[8:, :, :] = True
    with pytest.raises(ValueError, match="aucune source"):
        ma.distance.travel_time(m, 0)  # face z=0, hors de la phase


def test_travel_time_rejects_both_speed_and_cost():
    m = np.ones((8, 8, 8), dtype=bool)
    with pytest.raises(ValueError, match="pas les deux"):
        ma.distance.travel_time(m, (0, 0, 0), speed=m.astype(float), cost=m.astype(float))


def test_face_and_point_sources_are_accepted():
    m = np.ones((16, 16, 16), dtype=bool)
    a = np.asarray(ma.distance.geodesic_distance(m, 0))
    b = np.asarray(ma.distance.geodesic_distance(m, np.argwhere(m)[:16]))
    assert a[0, 8, 8] == 0.0
    assert np.isfinite(b).all()
    with pytest.raises(ValueError, match="0\\.\\.5"):
        ma.distance.geodesic_distance(m, 9)


# --- tortuosite de plan ---------------------------------------------------


def test_straight_tube_has_plane_tortuosity_exactly_one():
    vol = ma.phantoms.straight_tube(shape=(64, 32, 32), radius=6.0, axis=0)
    res = ma.tortuosity.plane_tortuosity(~vol.solid, 0)
    assert res.value == pytest.approx(1.0, abs=1e-5)
    assert res.std == pytest.approx(0.0, abs=1e-5)


def test_sinusoidal_tube_is_more_tortuous_than_one():
    """La tortuosite de plan est inferieure a celle de l'axe : les trajets coupent.

    `truth["tortuosity"]` du fantome est la tortuosite de la **ligne centrale**.
    Un trajet dans un tube epais peut couper les virages, donc la tortuosite
    mesuree entre plans est plus faible. Ce n'est pas une erreur de mesure mais
    deux grandeurs differentes.
    """
    vol = ma.phantoms.sinusoidal_tube(shape=(96, 64, 64), radius=6.0, amplitude=8.0)
    res = ma.tortuosity.plane_tortuosity(~vol.solid, 0)
    assert 1.02 < res.value < vol.meta["truth"]["tortuosity"]


def test_tortuosity_grows_with_the_tube_amplitude():
    """Test de comportement : plus le tube serpente, plus il est tortueux."""
    values = []
    for amp in (4.0, 10.0, 16.0):
        vol = ma.phantoms.sinusoidal_tube(shape=(96, 72, 72), radius=5.0, amplitude=amp)
        values.append(ma.tortuosity.plane_tortuosity(~vol.solid, 0).value)
    assert values[0] < values[1] < values[2], values


def test_plane_tortuosity_reports_unreached_voxels():
    m = np.zeros((20, 20, 20), dtype=bool)
    m[:, 8:12, 8:12] = True  # un canal
    m[15:, 2:5, 2:5] = True  # une poche isolee
    res = ma.tortuosity.plane_tortuosity(m, 0)
    assert res.n_reached < int(m.sum())


# --- tortuosite de point --------------------------------------------------


def test_free_medium_has_tortuosity_one_with_the_free_front_reference():
    """La normalisation par le front libre annule le biais de discretisation."""
    free = np.ones((48,) * 3, dtype=bool)
    res = ma.tortuosity.point_tortuosity(free, (24, 24, 24), min_distance=3.0)
    assert res.value == pytest.approx(1.0, abs=1e-4)


def test_euclidean_reference_carries_the_scheme_bias():
    free = np.ones((48,) * 3, dtype=bool)
    biased = ma.tortuosity.point_tortuosity(
        free, (24, 24, 24), min_distance=3.0, reference="euclidean"
    )
    assert biased.value > 1.05, "le biais du 1er ordre devrait etre visible"


def test_bad_reference_is_rejected():
    free = np.ones((8,) * 3, dtype=bool)
    with pytest.raises(ValueError, match="reference"):
        ma.tortuosity.point_tortuosity(free, (4, 4, 4), reference="magique")


def test_solid_is_more_tortuous_than_fluid(foam_struts):
    """Constat de la these : « lorsque l'on est tortueux dans le fluide on ne
    l'est pas dans le solide » — et le solide d'une mousse est plus tortueux."""
    solid = foam_struts.solid
    fluid = ~solid
    cf = np.argwhere(fluid)
    cs = np.argwhere(solid)
    tf = ma.tortuosity.point_tortuosity(fluid, cf[len(cf) // 2], min_distance=3.0)
    ts = ma.tortuosity.point_tortuosity(solid, cs[len(cs) // 2], min_distance=3.0)
    assert ts.value > tf.value


# --- directionnelle -------------------------------------------------------


def test_directional_tortuosity_detects_anisotropy():
    """Des cylindres paralleles : traverser le long des axes est plus direct."""
    vol = ma.phantoms.cylinders(shape=(64,) * 3, radius=6.0, n=5, axis=0, seed=1)
    fluid = ~vol.solid
    df = ma.tortuosity.directional_tortuosity(fluid, angles=[0.0, 45.0, 90.0], axis=0)
    assert set(df.columns) == {"angle", "tortuosity", "std", "n_reached"}
    assert len(df) == 3
    assert df["tortuosity"].notna().any()


# --- Poiseuille -----------------------------------------------------------


def test_poiseuille_paths_stay_away_from_the_walls(foam_struts):
    """Le coeur du sujet : le chemin de Poiseuille est plus long mais plus centre.

    Les deux metriques partagent les memes points d'arrivee, sinon la comparaison
    ne dirait rien.
    """
    fluid = ~foam_struts.solid
    # L'effet ne depend pas du choix des points d'arrivee. Mesure sur cette
    # mousse : x1,74 et x1,64 en « fastest » (8 et 16 chemins), x1,43 et x1,54
    # en « spread ». Le seuil est pose sous le minimum observe.
    for ends in ("spread", "fastest"):
        res = ma.tortuosity.poiseuille_tortuosity(
            fluid, 0, variant="physical", n_paths=8, ends=ends
        )
        ecart = res["mean_wall_distance_poiseuille"] / res["mean_wall_distance_geometric"]
        assert ecart > 1.35, f"ends={ends} : x{ecart:.2f}"
        assert res["tortuosity"] >= res["geometric_tortuosity"] * 0.98


def test_the_fastest_arrivals_are_a_biased_sample():
    """Le piege du temoin apparie : les arrivees les plus rapides sont les plus droites.

    En prenant les `n` voxels atteints le plus tot, on tombe au bout des canaux
    les plus directs : la tortuosite geodesique qu'on leur associe vaut 1,000
    dans a peu pres n'importe quel milieu ouvert, et ne decrit plus le milieu.
    D'ou le defaut `ends="spread"`, qui repartit les arrivees sur la section.

    La tortuosite du milieu, elle, reste celle de `plane_tortuosity`, qui moyenne
    sur toute la face d'arrivee.
    """
    foam = ma.phantoms.voronoi_foam(
        shape=(128,) * 3, n_cells=64, strut=6.0, min_seed_gap=13.0, seed=3
    )
    fluid = ~foam.solid
    rapide = ma.tortuosity.poiseuille_tortuosity(fluid, 0, n_paths=16, ends="fastest")
    etale = ma.tortuosity.poiseuille_tortuosity(fluid, 0, n_paths=16, ends="spread")
    plan = ma.tortuosity.plane_tortuosity(fluid, 0)

    assert rapide["geometric_tortuosity"] == pytest.approx(1.0, abs=1e-3)
    assert etale["geometric_tortuosity"] > rapide["geometric_tortuosity"]
    assert plan.value > 1.0
    # les arrivees etalees couvrent la section : leurs chemins sont plus varies
    assert etale["std"] > rapide["std"]

    with pytest.raises(ValueError, match="ends doit valoir"):
        ma.tortuosity.poiseuille_tortuosity(fluid, 0, ends="au hasard")


def test_a_straight_tube_has_tortuosity_one_exactly():
    """Le controle qui manquait, et qui a revele un vrai biais.

    Dans un tube droit la tortuosite geometrique vaut 1 par construction. Elle
    valait 1,17 : la descente de gradient choisissait le voisin de plus petit
    `T`, sans regarder la longueur du pas. Le front etant quasi plan, un pas
    diagonal (longueur sqrt(3)) descend autant qu'un pas axial (longueur 1), et
    le chemin zigzaguait pour rien. Le critere est maintenant la **pente**,
    `(T - T_voisin) / longueur du pas`.
    """
    vol = ma.phantoms.straight_tube(shape=(64, 32, 32), radius=8.0, axis=0)
    fluid = ~vol.solid
    res = ma.tortuosity.poiseuille_tortuosity(fluid, 0, n_paths=8)
    assert res["geometric_tortuosity"] == pytest.approx(1.0, abs=1e-6)
    # le chemin de Poiseuille, lui, est legerement plus long : il rejoint l'axe
    assert 1.0 < res["tortuosity"] < 1.15


def test_a_path_is_never_longer_than_its_travel_time():
    """La longueur d'un chemin tracé doit valoir le temps de parcours, pas plus.

    A vitesse unite, le temps d'arrivee **est** la longueur geodesique. Un
    chemin extrait qui la depasse de 20 % n'est pas la geodesique.
    """
    foam = ma.phantoms.voronoi_foam(
        shape=(56,) * 3, n_cells=10, strut=3.0, min_seed_gap=16.0, seed=1
    )
    fluid = ~foam.solid
    T = np.asarray(ma.distance.geodesic_distance(fluid, 0))
    arrivee = np.zeros(fluid.shape, dtype=bool)
    arrivee[-1] = fluid[-1]
    ok = arrivee & np.isfinite(T)
    ends = np.argwhere(ok)[np.argsort(T[ok])[:12]]
    for e in ends:
        path = ma.tortuosity.shortest_path(T, e)
        ell = float(np.linalg.norm(np.diff(path.astype(float), axis=0), axis=1).sum())
        assert ell <= T[tuple(e)] * 1.02, f"chemin {100 * (ell / T[tuple(e)] - 1):.1f} % trop long"


def test_poiseuille_gives_fields_paths_and_a_table():
    """Le resultat doit etre exploitable, pas seulement affichable en une ligne.

    Il rendait un `dict` de scalaires : rien a afficher, rien a tracer. Il porte
    maintenant les champs et les chemins — tout en restant le meme `Mapping`,
    pour ne casser aucun appel existant.
    """
    vol = ma.phantoms.straight_tube(shape=(48, 32, 32), radius=8.0, axis=0)
    fluid = ~vol.solid
    res = ma.tortuosity.poiseuille_tortuosity(fluid, 0, n_paths=6)

    assert set(dict(res)) == {
        "variant",
        "separation",
        "n_paths",
        "tortuosity",
        "std",
        "mean_wall_distance_poiseuille",
        "geometric_tortuosity",
        "mean_wall_distance_geometric",
    }
    assert res["tortuosity"] == res.tortuosity  # les deux acces cohabitent

    assert set(res.paths["metric"]) == {"poiseuille", "geometric"}
    assert len(res.paths) == 12
    assert (res.paths["tortuosity"] >= 1.0).all()
    assert len(res.table) == 1

    for champ in (res.speed, res.travel_time, res.path_mask, res.geometric_path_mask):
        assert champ is not None and champ.shape == fluid.shape
    assert int(res.path_mask.max()) == 6  # une etiquette par chemin
    assert not fluid[res.path_mask > 0].__invert__().any(), "un chemin est sorti de la phase"

    leger = ma.tortuosity.poiseuille_tortuosity(fluid, 0, n_paths=6, keep_fields=False)
    assert leger.speed is None and leger.travel_time is None
    assert leger["tortuosity"] == pytest.approx(res["tortuosity"])


def test_poiseuille_speed_is_zero_at_the_wall_and_one_at_the_centre():
    vol = ma.phantoms.straight_tube(shape=(32, 32, 32), radius=10.0, axis=0)
    fluid = ~vol.solid
    v = ma.tortuosity.poiseuille_speed(fluid, variant="physical")
    assert v[16, 16, 16] == pytest.approx(1.0, abs=0.1)  # centre
    wall = fluid & (np.asarray(ma.distance.distance_transform(fluid)) <= 1.0)
    assert v[wall].max() < 0.5
    assert (v[~fluid] == 0).all()


def test_imorph_variant_is_an_inverted_profile():
    """L'equation 3.11 de la these est une **lenteur**, pas une vitesse.

    Elle vaut 0 au centre du canal et 1 a la paroi. Utilisee comme lenteur par le
    solveur d'iMorph (`|grad T| = F`), cela donne bien des chemins centres — mais
    lue comme une vitesse, elle ferait raser les parois.
    """
    vol = ma.phantoms.straight_tube(shape=(32, 32, 32), radius=10.0, axis=0)
    fluid = ~vol.solid
    c = ma.tortuosity.poiseuille_speed(fluid, variant="imorph")
    v = ma.tortuosity.poiseuille_speed(fluid, variant="physical")
    assert c[16, 16, 16] < 0.1, "la lenteur doit s'annuler au centre"
    assert v[16, 16, 16] > 0.9, "la vitesse doit etre maximale au centre"


def test_imorph_variant_is_refused_for_measurement():
    """On ne propose pas une mesure qu'on ne peut pas defendre.

    La lenteur de l'equation 3.11 s'annule au centre des canaux : les temps
    d'arrivee y dependent du plancher numerique employe pour l'inverser. Le champ
    reste accessible pour reproduire iMorph, mais pas via cette fonction.
    """
    vol = ma.phantoms.straight_tube(shape=(32, 24, 24), radius=8.0, axis=0)
    fluid = ~vol.solid
    with pytest.raises(ValueError, match="plancher numerique"):
        ma.tortuosity.poiseuille_tortuosity(fluid, 0, variant="imorph")
    # le champ, lui, reste utilisable directement
    c = ma.tortuosity.poiseuille_speed(fluid, variant="imorph")
    T = np.asarray(ma.distance.travel_time(fluid, 0, cost=c))
    assert np.isfinite(T[fluid]).all()


def test_bad_poiseuille_variant_is_rejected():
    m = np.ones((8,) * 3, dtype=bool)
    with pytest.raises(ValueError, match="variant"):
        ma.tortuosity.poiseuille_speed(m, variant="magique")


# --- plus court chemin ----------------------------------------------------


def test_shortest_path_reaches_the_source():
    vol = ma.phantoms.straight_tube(shape=(48, 24, 24), radius=5.0, axis=0)
    fluid = ~vol.solid
    T = np.asarray(ma.distance.geodesic_distance(fluid, (0, 12, 12)))
    path = ma.tortuosity.shortest_path(T, (47, 12, 12))
    assert len(path) >= 47
    assert tuple(path[-1]) == (0, 12, 12)
    assert fluid[tuple(path.T)].all(), "le chemin est sorti de la phase"


def test_shortest_path_rejects_an_unreached_start():
    m = np.zeros((16, 16, 16), dtype=bool)
    m[:8] = True
    T = np.asarray(ma.distance.geodesic_distance(m, (0, 8, 8)))
    with pytest.raises(ValueError, match="pas ete atteint"):
        ma.tortuosity.shortest_path(T, (15, 8, 8))


# --- sur graphe -----------------------------------------------------------


def test_graph_tortuosity_on_a_straight_chain():
    import pandas as pd

    n = 6
    nodes = pd.DataFrame({"node": range(1, n + 1), "z": np.arange(n) * 10.0, "y": 0.0, "x": 0.0})
    edges = pd.DataFrame(
        {
            "node_a": range(1, n),
            "node_b": range(2, n + 1),
            "length": [10.0] * (n - 1),
        }
    )
    res = ma.tortuosity.graph_tortuosity(nodes, edges, axis=0, margin=1.0)
    assert res["tortuosity"] == pytest.approx(1.0, abs=1e-9)


def test_graph_tortuosity_on_a_detour():
    """Une chaine qui fait un detour lateral : la tortuosite depasse 1."""
    import pandas as pd

    nodes = pd.DataFrame(
        {
            "node": [1, 2, 3],
            "z": [0.0, 5.0, 10.0],
            "y": [0.0, 12.0, 0.0],
            "x": [0.0, 0.0, 0.0],
        }
    )
    d = np.hypot(5.0, 12.0)
    edges = pd.DataFrame({"node_a": [1, 2], "node_b": [2, 3], "length": [d, d]})
    res = ma.tortuosity.graph_tortuosity(nodes, edges, axis=0, margin=1.0)
    assert res["tortuosity"] == pytest.approx((2 * d / 10.0) ** 2, rel=1e-9)


def test_graph_tortuosity_needs_both_layers(segmented):
    sk = ma.skeleton.plateau_skeleton(segmented["vol"].solid, segmented["cells"])
    res = ma.tortuosity.graph_tortuosity(sk.nodes, sk.edges, axis=0)
    assert res["tortuosity"] > 1.0
    assert res["n_reached"] > 0
    assert len(res["paths_length"]) == res["n_reached"]
    with pytest.raises(ValueError, match="margin"):
        ma.tortuosity.graph_tortuosity(sk.nodes, sk.edges, axis=0, margin=-1.0)


def test_directional_tortuosity_accepts_a_count():
    """Passer un entier doit vouloir dire « ce nombre d'angles », pas planter."""
    m = np.zeros((24, 40, 40), dtype=bool)
    m[:, 14:26, 14:26] = True
    df = ma.tortuosity.directional_tortuosity(m, angles=6, axis=0)
    assert len(df) == 6
    assert df["angle"].iloc[0] == 0.0
    assert df["angle"].max() < 180.0
    one = ma.tortuosity.directional_tortuosity(m, angles=0.0, axis=0)
    assert len(one) == 1
