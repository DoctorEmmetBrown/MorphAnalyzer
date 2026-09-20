"""Maillage de surface, mesures et exports."""

import numpy as np
import pytest

import morphanalyzer as ma


@pytest.fixture(scope="module")
def sphere_mesh(sphere):
    return ma.mesh.surface_mesh(sphere), ma.mesh.surface_mesh(
        sphere, grey=sphere.meta["truth"]["signed_distance"], level=0.0
    )


def test_mesh_area_and_volume_of_a_sphere(sphere_mesh, sphere):
    r = sphere.meta["truth"]["radius"]
    binary, grey = sphere_mesh
    area_c = 4 * np.pi * r**2
    vol_c = 4 / 3 * np.pi * r**3
    # le masque binaire surestime l'aire, le champ continu ne le fait pas
    assert binary.area / area_c == pytest.approx(1.093, rel=0.03)
    assert grey.area == pytest.approx(area_c, rel=0.01)
    # le volume est juste dans les deux cas : c'est une integrale, pas une aire
    assert binary.volume == pytest.approx(vol_c, rel=0.01)
    assert grey.volume == pytest.approx(vol_c, rel=0.01)


def test_marching_cubes_orientation_is_reported_by_the_sign(sphere_mesh):
    binary, _ = sphere_mesh
    signed = ma.mesh.mesh_volume(binary.vertices, binary.faces, signed=True)
    assert signed < 0, "marching_cubes oriente vers les valeurs decroissantes"
    assert abs(signed) == pytest.approx(binary.volume, rel=1e-12)


def test_mesh_scales_with_voxel_size():
    a = ma.phantoms.sphere(shape=(48,) * 3, radius=14.0, voxel_size=1.0)
    b = ma.phantoms.sphere(shape=(48,) * 3, radius=14.0, voxel_size=3.0)
    ma_, mb = ma.mesh.surface_mesh(a), ma.mesh.surface_mesh(b)
    assert mb.area == pytest.approx(9.0 * ma_.area, rel=1e-6)
    assert mb.volume == pytest.approx(27.0 * ma_.volume, rel=1e-6)


def test_grey_requires_a_level(sphere):
    with pytest.raises(ValueError, match="level"):
        ma.mesh.surface_mesh(sphere, grey=sphere.meta["truth"]["signed_distance"])


def test_cube_volume_is_accurate_and_area_is_bevelled():
    """Cube aligne sur la grille : le volume est juste, l'aire est rabotee.

    Le bloc occupe les voxels 6..17 sur chaque axe. L'iso-surface a 0.5 passe a
    un demi-voxel *au-dela* du dernier voxel plein, en 5.5 et 17.5 : le solide
    equivalent est donc un cube d'arete 12, pas 11 (l'ecart centre-a-centre).

    Marching cubes ne rend cependant pas un cube parfait : il biseaute les 12
    aretes et les 8 coins par des facettes a 45 degres. Un chanfrein de largeur
    w sur une arete de longueur L retire 2wL d'aire aux deux faces voisines et
    n'en rajoute que sqrt(2) wL, donc l'aire mesuree est *inferieure* a celle du
    cube parfait, et le volume lui aussi legerement deficitaire.
    """
    m = np.zeros((24, 24, 24), dtype=bool)
    m[6:18, 6:18, 6:18] = True
    mesh = ma.mesh.surface_mesh(m)
    # volume : 12^3 = 1728, moins les coins rabotes (~1 %)
    assert mesh.volume == pytest.approx(12**3, rel=0.02)
    assert mesh.volume < 12**3
    # aire : bornee par le cube d'arete 11 et celui d'arete 12
    assert 6 * 11**2 < mesh.area < 6 * 12**2


@pytest.mark.parametrize("ext", [".stl", ".obj", ".ply"])
def test_native_exports_roundtrip_through_meshio_or_size(sphere_mesh, tmp_path, ext):
    _, grey = sphere_mesh
    p = ma.mesh.save_mesh(grey, tmp_path / f"s{ext}")
    assert p.exists() and p.stat().st_size > 1000


def test_ascii_stl_is_readable_text(sphere_mesh, tmp_path):
    _, grey = sphere_mesh
    p = ma.mesh.save_mesh(grey, tmp_path / "s.stl", binary=False)
    head = p.read_text().splitlines()[:3]
    assert head[0].startswith("solid")
    assert "facet normal" in head[1]


def test_obj_indices_are_one_based(tmp_path):
    m = np.zeros((10, 10, 10), dtype=bool)
    m[3:7, 3:7, 3:7] = True
    mesh = ma.mesh.surface_mesh(m)
    p = ma.mesh.save_mesh(mesh, tmp_path / "c.obj")
    faces = [ln for ln in p.read_text().splitlines() if ln.startswith("f ")]
    idx = [int(v) for ln in faces for v in ln.split()[1:]]
    assert min(idx) == 1
    assert max(idx) == len(mesh.vertices)


def test_unknown_format_says_what_to_install(sphere_mesh, tmp_path):
    from morphanalyzer._deps import MissingDependency, have

    _, grey = sphere_mesh
    if have("meshio"):
        pytest.skip("meshio est installe : le chemin d'erreur n'est pas atteint")
    with pytest.raises(MissingDependency, match="meshio"):
        ma.mesh.save_mesh(grey, tmp_path / "s.vtk")


def test_decimate_rejects_bad_reduction(sphere_mesh):
    _, grey = sphere_mesh
    with pytest.raises(ValueError, match="target_reduction"):
        ma.mesh.decimate(grey, target_reduction=1.5)


def test_decimate_reduces_face_count(sphere_mesh):
    from morphanalyzer._deps import have

    if not (have("fast_simplification") or have("trimesh")):
        pytest.skip("ni fast-simplification ni trimesh")
    _, grey = sphere_mesh
    out = ma.mesh.decimate(grey, target_reduction=0.7)
    assert len(out) < len(grey)
    # la forme est preservee a quelques pourcents
    assert out.volume == pytest.approx(grey.volume, rel=0.05)


def test_padding_is_a_no_op_on_an_interior_object(sphere):
    """Un objet qui ne touche pas le bord ne doit rien devoir au rembourrage."""
    a = ma.mesh.surface_mesh(sphere.solid)
    b = ma.mesh.surface_mesh(sphere.solid, pad=True)
    assert b.area == pytest.approx(a.area, rel=1e-9)
    assert b.volume == pytest.approx(a.volume, rel=1e-9)


def test_open_surface_makes_the_volume_meaningless():
    """Un solide qui touche le bord donne une surface ouverte : volume faux.

    marching_cubes ne referme pas les sections coupees par les faces de la
    boite. C'est le bon comportement pour une aire — ces sections ne sont pas
    de l'interface solide/fluide — mais le volume enferme n'a alors aucun sens.
    `pad=True` referme et le volume redevient juste.
    """
    m = np.zeros((32, 32, 32), dtype=bool)
    m[:, 8:24, 8:24] = True  # barreau traversant selon z
    n_vox = int(m.sum())

    open_ = ma.mesh.surface_mesh(m)
    closed = ma.mesh.surface_mesh(m, pad=True)

    assert closed.volume == pytest.approx(n_vox, rel=0.05)
    assert abs(open_.volume - n_vox) / n_vox > 0.2
    # en revanche l'aire ouverte exclut les deux sections, l'aire fermee les ajoute
    assert closed.area > open_.area
    assert closed.area - open_.area == pytest.approx(2 * 16**2, rel=0.15)
