"""Maillage triangulaire de l'interface, mesures et exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["Mesh", "surface_mesh", "save_mesh", "decimate", "mesh_volume", "mesh_area"]


@dataclass(slots=True)
class Mesh:
    """Maillage triangulaire.

    Attributes
    ----------
    vertices
        `(n, 3)` en coordonnees physiques, ordre `(z, y, x)`.
    faces
        `(m, 3)` indices de sommets.
    normals
        Normales aux sommets, ou `None`.
    values
        Valeurs interpolees aux sommets, ou `None`.
    meta
        Metadonnees (`voxel_size`, `level`, `source`).
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    values: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.faces)

    @property
    def area(self) -> float:
        return mesh_area(self.vertices, self.faces)

    @property
    def volume(self) -> float:
        return mesh_volume(self.vertices, self.faces)

    def __repr__(self) -> str:  # pragma: no cover - agrement
        return f"<Mesh {len(self.vertices)} sommets, {len(self.faces)} triangles>"


def surface_mesh(
    solid,
    *,
    grey=None,
    level: float | None = None,
    voxel_size=None,
    pad: bool = False,
) -> Mesh:
    """Maillage de l'interface solide/fluide par marching cubes.

    Parameters
    ----------
    grey, level
        Passer les niveaux de gris et le seuil de binarisation donne une
        interface **sous-voxelique**. Sur un masque binaire, l'iso-surface a 0,5
        est un escalier : l'aire est surestimee d'environ 9 % sur une sphere,
        contre 0,05 % sur le champ continu. Le faire est gratuit et vaut un ordre
        de grandeur de precision.
    pad
        Entoure le volume d'une couche de fond avant de mailler, ce qui **ferme**
        la surface la ou le solide touche le bord de la boite. Voir la mise en
        garde ci-dessous. Les coordonnees rendues restent celles du volume
        d'origine.

    Warnings
    --------
    Quand le solide touche le bord de la boite — une mousse, un reseau de brins,
    presque tout echantillon reel — `marching_cubes` rend une surface **ouverte**
    : il ne ferme pas les sections coupees par les faces de la boite. C'est le
    bon comportement pour une **aire**, puisque ces sections ne sont pas de
    l'interface solide/fluide et ne doivent pas compter dans la surface
    specifique. Mais le **volume** enferme n'a alors aucun sens : sur une mousse
    de 62 862 voxels solides, le maillage non ferme donne 6 808.

    Regle : `pad=False` (defaut) pour mesurer une aire, `pad=True` pour mesurer
    un volume ou pour exporter un objet etanche vers un mailleur ou une
    imprimante 3D.

    Notes
    -----
    iMorph mutualisait les sommets entre cubes voisins pour eviter d'en creer en
    double (`mesh.cpp`, « notre implementation verifie dans les voxels voisins
    s'il n'y a pas des points assez proches »). `skimage.measure.marching_cubes`
    le fait nativement : le maillage rendu est sans triangle degenere.
    """
    from skimage import measure

    if voxel_size is None:
        voxel_size = solid.voxel_size if isinstance(solid, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3

    if grey is not None:
        if level is None:
            raise ValueError("`level` (seuil de binarisation) est requis avec `grey`")
        field_, iso = as_array(grey).astype(np.float32, copy=False), float(level)
    else:
        field_, iso = (
            as_array(solid).astype(np.float32, copy=False),
            0.5 if level is None else float(level),
        )

    if pad:
        below = float(field_.min())
        outside = min(below, iso - 1.0) if below >= iso else below
        field_ = np.pad(field_, 1, mode="constant", constant_values=outside)

    verts, faces, normals, values = measure.marching_cubes(
        field_, level=iso, spacing=tuple(voxel_size)
    )
    if pad:
        verts = verts - np.asarray(voxel_size, dtype=verts.dtype)

    return Mesh(
        vertices=verts,
        faces=faces,
        normals=normals,
        values=values,
        meta={
            "voxel_size": tuple(voxel_size),
            "level": iso,
            "from_grey": grey is not None,
            "padded": bool(pad),
        },
    )


def mesh_area(vertices, faces) -> float:
    """Aire totale, somme des aires des triangles."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def mesh_volume(vertices, faces, *, signed: bool = False) -> float:
    """Volume enferme, par le theoreme de la divergence.

    Suppose le maillage **ferme** et ses faces orientees de facon coherente, ce
    que garantit `marching_cubes`.

    Parameters
    ----------
    signed
        Par defaut on rend la valeur absolue. Avec `signed=True`, le signe
        renseigne sur l'orientation : `marching_cubes` oriente les normales vers
        les valeurs decroissantes du champ, ce qui donne un volume **negatif**
        pour un masque ou le solide vaut 1. Ce n'est pas une erreur de calcul,
        seulement une convention d'orientation.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    return vol if signed else abs(vol)


# ---------------------------------------------------------------- exports


def _write_stl_binary(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    import struct

    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    with path.open("wb") as fh:
        fh.write(b"morphanalyzer STL".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(f)))
        block = np.empty((len(f), 12), dtype="<f4")
        block[:, 0:3] = n
        block[:, 3:6] = a
        block[:, 6:9] = b
        block[:, 9:12] = c
        for row in block:
            fh.write(row.tobytes())
            fh.write(b"\0\0")


def _write_stl_ascii(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    with path.open("w") as fh:
        fh.write("solid morphanalyzer\n")
        for i in range(len(f)):
            fh.write(f"  facet normal {n[i, 0]:g} {n[i, 1]:g} {n[i, 2]:g}\n    outer loop\n")
            for p in (a[i], b[i], c[i]):
                fh.write(f"      vertex {p[0]:g} {p[1]:g} {p[2]:g}\n")
            fh.write("    endloop\n  endfacet\n")
        fh.write("endsolid morphanalyzer\n")


def _write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    with path.open("w") as fh:
        fh.write("# morphanalyzer\n")
        np.savetxt(fh, v, fmt="v %g %g %g")
        np.savetxt(fh, f + 1, fmt="f %d %d %d")


def _write_ply(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    with path.open("w") as fh:
        fh.write(
            "ply\nformat ascii 1.0\ncomment morphanalyzer\n"
            f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
            f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n"
        )
        np.savetxt(fh, v, fmt="%g %g %g")
        np.savetxt(fh, np.column_stack([np.full(len(f), 3), f]), fmt="%d %d %d %d")


def save_mesh(mesh: Mesh, path: str | Path, *, binary: bool = True) -> Path:
    """Ecrit le maillage. `.stl`, `.obj` et `.ply` sont natifs.

    Tout autre suffixe passe par `meshio` (extra `mesh`).

    !!! warning "Ordre des axes a l'export"
        Les sommets sont en `(z, y, x)`, l'ordre interne de la bibliotheque. La
        plupart des visualiseurs attendent `(x, y, z)` : le maillage y apparaitra
        donc en miroir sur la diagonale. Inverser avant d'exporter si cela
        importe :

        ```python
        mesh.vertices = mesh.vertices[:, ::-1]
        ```
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        (_write_stl_binary if binary else _write_stl_ascii)(path, v, f)
    elif suffix == ".obj":
        _write_obj(path, v, f)
    elif suffix == ".ply":
        _write_ply(path, v, f)
    else:
        from morphanalyzer._deps import require

        meshio = require("meshio", reason=f"l'ecriture du format {suffix}")
        meshio.write_points_cells(str(path), v, [("triangle", f)])
    return path


def decimate(mesh: Mesh, *, target_reduction: float = 0.5, method: str = "auto") -> Mesh:
    """Reduit le nombre de triangles en preservant la forme.

    Parameters
    ----------
    target_reduction
        Fraction de triangles a supprimer, entre 0 et 1.
    method
        `"auto"` : `fast-simplification` s'il est la, sinon `trimesh`.

    Notes
    -----
    Un maillage de marching cubes n'est pas econome : ses triangles font moins
    d'un voxel, et iMorph en comptait 2 a 10 millions par echantillon. La
    decimation est donc utile pour l'export vers un code de calcul ou un
    visualiseur — mais **elle change l'aire**, donc jamais avant de mesurer une
    surface specifique.
    """
    from morphanalyzer._deps import have

    if not 0.0 < target_reduction < 1.0:
        raise ValueError("target_reduction doit etre dans (0, 1)")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)

    use = method
    if use == "auto":
        use = "fast_simplification" if have("fast_simplification") else "trimesh"
    if use == "fast_simplification":
        import fast_simplification

        nv, nf = fast_simplification.simplify(v, f, target_reduction)
    elif use == "trimesh":
        from morphanalyzer._deps import require

        trimesh = require("trimesh", reason="la decimation de maillage")
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        keep = max(4, int(round(len(f) * (1.0 - target_reduction))))
        try:
            out = tm.simplify_quadric_decimation(face_count=keep)
        except ImportError as exc:  # pragma: no cover - depend de la version
            # trimesh >= 4.1 n'implemente plus la decimation : sa methode n'est
            # qu'un enrobage de `fast_simplification`. Sans lui les deux
            # branches echouent, et l'erreur brute nomme un module que
            # l'appelant n'a jamais demande.
            from morphanalyzer._deps import MissingDependency

            raise MissingDependency(
                "'fast_simplification' est requis pour la decimation de "
                "maillage : trimesh >= 4.1 ne fait qu'enrober cette "
                'bibliotheque. Installer avec : pip install "morphanalyzer[mesh]"'
            ) from exc
        nv, nf = np.asarray(out.vertices), np.asarray(out.faces)
    else:
        raise ValueError("method doit valoir 'auto', 'fast_simplification' ou 'trimesh'")

    return Mesh(
        vertices=np.asarray(nv, dtype=np.float64),
        faces=np.asarray(nf, dtype=np.int64),
        meta={**mesh.meta, "decimated_from": len(f), "method": use},
    )
