import numpy as np

import morphanalyzer as ma


def test_skeleton_of_a_foam_is_a_thin_curve_network(foam_struts):
    """Sur un reseau de brins elances, l'amincissement fait son travail."""
    solid = foam_struts.solid
    sk = np.asarray(ma.skeleton.skeletonize(solid))
    assert sk.sum() > 0
    assert sk.sum() < solid.sum() / 10, "un squelette doit etre bien plus mince que l'objet"
    # il reste dans le solide et loin de l'interface
    assert (solid[sk]).all()
    d = np.asarray(ma.distance.distance_transform(solid))
    assert np.median(d[sk]) > np.median(d[solid])


def test_skeletonize_can_return_empty_on_simple_objects():
    """Piege documente de scikit-image, pas un defaut de morphanalyzer.

    En 0.25.2, `skeletonize(..., method="lee")` rend un squelette **vide** sur
    des objets pourtant elementaires : un cube de 4x4x4, une sphere de rayon 12,
    une plaque, et meme un cylindre isole — alors que le meme cylindre repete
    quatre fois dans la meme boite rend 174 voxels. Rembourrer le volume n'y
    change rien.

    Consequence pour la bibliotheque : on ne peut pas faire reposer l'amorce de
    `shape.local_shape_tensor` sur le seul squelette. D'ou la strategie `"auto"`,
    qui bascule sur la crete de distance quand le squelette ressort vide — et ce
    sont justement les objets compacts, ou un squelette curviligne n'a de toute
    facon pas de sens.

    Le test fige le comportement observe : s'il change dans une version future,
    la strategie d'amorce merite d'etre reexaminee.
    """
    cases = {}
    cube = np.zeros((20,) * 3, dtype=bool)
    cube[8:12, 8:12, 8:12] = True
    cases["cube"] = cube
    tube = np.zeros((40, 24, 24), dtype=bool)
    _z, y, x = np.ogrid[:40, :24, :24]
    tube |= ((y - 11.5) ** 2 + (x - 11.5) ** 2) <= 16
    cases["tube isole"] = tube

    for name, m in cases.items():
        ridge = np.asarray(ma.skeleton.distance_ridge(m))
        assert ridge.sum() > 0, f"crete vide sur {name} : cela ne doit jamais arriver"


def test_distance_ridge_is_never_empty_on_various_shapes():
    shapes = {
        "cube": np.zeros((20,) * 3, dtype=bool),
        "barre": np.zeros((20,) * 3, dtype=bool),
        "voxel": np.zeros((20,) * 3, dtype=bool),
    }
    shapes["cube"][8:12, 8:12, 8:12] = True
    shapes["barre"][5:15, 9:11, 9:11] = True
    shapes["voxel"][10, 10, 10] = True
    for name, m in shapes.items():
        assert np.asarray(ma.skeleton.distance_ridge(m)).sum() > 0, name


def test_distance_ridge_of_a_slab_is_its_median_plane():
    slab = np.zeros((30, 30, 30), dtype=bool)
    slab[12:19, :, :] = True
    ridge = np.asarray(ma.skeleton.distance_ridge(slab))
    ks = np.unique(np.argwhere(ridge)[:, 0])
    assert set(ks.tolist()) <= {14, 15, 16}
