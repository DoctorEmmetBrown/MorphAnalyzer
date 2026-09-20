"""Tenseur de forme local, rapports d'elongation, classification.

Portage de `Thread/ShapeClassif/skullSolidThread.cpp` et
`shapeClassificationModule.cpp` d'iMorph 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array
from morphanalyzer.distance.edt import geodesic_ball, nearest_seed_propagation

__all__ = [
    "ShapeTensor",
    "local_shape_tensor",
    "shape_classification",
    "elongation_ratios",
    "classify_solid",
    "strut_orientation",
    "NODE",
    "STRUT",
    "PLATE",
]

#: Etiquettes de classe. iMorph n'en distinguait que deux (`rodes` / `plates`,
#: sur le seul rapport a/b) ; on ajoute la distinction noeud vs plaque via b/c,
#: conformement a la figure 3.35 de la these.
NODE = 1
STRUT = 2
PLATE = 3

CLASS_NAMES = {0: "fluide", NODE: "noeud", STRUT: "brin", PLATE: "plaque"}


@dataclass(slots=True)
class ShapeTensor:
    """Resultat de `local_shape_tensor`.

    Attributes
    ----------
    a, b, c
        Demi-axes de l'ellipsoide equivalent, `a >= b >= c`, en unites
        physiques. `a = 2*sqrt(lambda_max)` de la matrice de covariance.
    a_on_b, b_on_c
        Rapports d'elongation. `a/b` grand = allonge (brin) ; `b/c` grand avec
        `a/b` proche de 1 = aplati (plaque).
    direction
        Premier vecteur propre, `(nz, ny, nx, 3)`, ou `None` si non demande.
    theta, phi
        Azimut (0..360) et elevation (0..90) de `direction`, en degres. Les
        directions sont ramenees a l'hemisphere superieur : une direction et son
        opposee sont equivalentes pour un brin.
    seeds
        Masque des voxels ou le tenseur a reellement ete calcule.
    n_seeds, n_degenerate
        Nombre de germes traites, et parmi eux ceux dont la boule etait trop
        petite pour un tenseur (valeurs mises a 1, comme iMorph).
    params
        Parametres effectifs, pour tracabilite.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    a_on_b: np.ndarray
    b_on_c: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    seeds: np.ndarray
    direction: np.ndarray | None = None
    n_seeds: int = 0
    n_degenerate: int = 0
    params: dict = field(default_factory=dict)

    def to_frame(self, *, mask=None) -> pd.DataFrame:
        """Table voxel par voxel, restreinte a `mask` (par defaut les germes)."""
        sel = self.seeds if mask is None else np.asarray(mask, dtype=bool)
        idx = np.argwhere(sel)
        data = {
            "k": idx[:, 0],
            "j": idx[:, 1],
            "i": idx[:, 2],
            "a": self.a[sel],
            "b": self.b[sel],
            "c": self.c[sel],
            "a_on_b": self.a_on_b[sel],
            "b_on_c": self.b_on_c[sel],
            "theta": self.theta[sel],
            "phi": self.phi[sel],
        }
        return pd.DataFrame(data)


def _spherical(v: np.ndarray) -> tuple[float, float]:
    """(azimut, elevation) en degres, direction ramenee a l'hemisphere z >= 0."""
    vz, vy, vx = v
    if vz < 0:
        vz, vy, vx = -vz, -vy, -vx
    r = float(np.sqrt(vx * vx + vy * vy + vz * vz))
    if r == 0.0:
        return 0.0, 0.0
    elevation = float(np.degrees(np.arcsin(np.clip(vz / r, -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(vy, vx))) % 360.0
    return azimuth, elevation


def local_shape_tensor(
    solid,
    *,
    seeds="auto",
    aperture=None,
    expand_factor: float = 3.0,
    min_radius: float = 3.0,
    voxel_size=None,
    clip_ratios: bool = True,
    with_orientation: bool = True,
    propagate: bool = True,
    connectivity: int = 26,
    progress: bool = False,
) -> ShapeTensor:
    """Calcule le tenseur de forme local, puis le propage au solide.

    Parameters
    ----------
    solid
        Volume binaire, True = solide.
    seeds
        Ou mesurer. `"auto"` (defaut) : l'amincissement homotopique, le chemin
        d'iMorph, avec repli sur la crete de distance si le squelette ressort
        vide — ce qui arrive avec scikit-image sur les objets compacts (voir
        `skeleton.distance_ridge`), justement les cas ou un squelette curviligne
        n'a de toute facon pas de sens. `"skeleton"` et `"ridge"` forcent l'un
        ou l'autre. `"all"` : tous les voxels du solide, exact et tres lent,
        reserve aux petits volumes de reference. Un masque explicite est aussi
        accepte.

        Le choix de l'amorce ne change pas *ce qui est mesure* en un point,
        seulement *ou* on mesure. Mais une amorce trop clairsemee degrade la
        propagation : chaque voxel du solide herite du germe le plus proche, et
        si les germes sont rares ce voisin n'est plus representatif. Sur une
        mousse, le squelette donne environ 3 % du volume solide en germes, la
        crete de distance cinquante fois moins.
    aperture
        Carte d'ouverture **dans le solide**, qui fixe le rayon de propagation
        local. Calculee si absente (`granulometry.aperture_map`). Si elle est
        fournie, elle est supposee en unites physiques (comme `voxel_size`) et
        convertie en voxels en interne.
    expand_factor
        Le rayon de la boule vaut `expand_factor * ouverture locale`, avec un
        plancher a `min_radius`. Defaut 3, la valeur d'iMorph
        (`ParameterShapeClassifMethod::expandFactor`) et celle de la these
        (« le rayon d'ouverture le plus grand multiplie par trois »).
    clip_ratios
        Ecrete `a/b` et `b/c` a `expand_factor`, comme iMorph. Borne les
        valeurs aberrantes des boules degenerees sans toucher au regime utile
        (les seuils de classification sont tres en dessous).
    propagate
        Etend le resultat des germes a tout le solide, au plus proche germe.
        Mettre `False` pour ne garder que les valeurs mesurees.

    Returns
    -------
    ShapeTensor

    Notes
    -----
    Deux ecarts assumes par rapport a iMorph 3.2, tous deux documentes et
    testes :

    1. **La boule geodesique** est obtenue par dilatation contrainte puis
       intersection avec la boule euclidienne, au lieu d'un fast marching borne.
       Voir `distance.geodesic_ball` pour le detail et ses limites.
    2. **Les coordonnees sont mises a l'echelle physique** avant la covariance,
       ce qui rend `a`, `b`, `c` justes sur des voxels anisotropes. iMorph
       travaillait en voxels et supposait l'isotropie.
    """
    from morphanalyzer.granulometry.aperture import aperture_map
    from morphanalyzer.skeleton.thin import distance_ridge, skeletonize

    s = as_array(solid).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = solid.voxel_size if isinstance(solid, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    spacing = np.asarray(voxel_size, dtype=float)

    seed_kind = seeds if isinstance(seeds, str) else "explicite"
    if not isinstance(seeds, str):
        pass
    elif seeds == "auto":
        seeds = skeletonize(s)
        if not np.asarray(seeds).any():
            seeds = distance_ridge(s)
            seed_kind = "auto->ridge"
        else:
            seed_kind = "auto->skeleton"
    elif seeds == "ridge":
        seeds = distance_ridge(s)
    elif seeds == "skeleton":
        seeds = skeletonize(s)
    elif seeds == "all":
        seeds = s
    else:
        raise ValueError("seeds doit valoir 'auto', 'skeleton', 'ridge', 'all' ou un masque")
    seed_mask = as_array(seeds).astype(bool, copy=False) & s
    if not seed_mask.any():
        raise ValueError(
            f"aucun germe dans le solide (strategie {seed_kind!r}). "
            "Avec 'skeleton', c'est un piege connu de scikit-image sur les objets "
            "compacts ; essayer seeds='ridge'."
        )

    # Le rayon de propagation est exprime en voxels (c'est un nombre de pas de
    # dilatation), alors qu'une carte d'ouverture fournie par l'appelant est en
    # unites physiques. On calcule donc la notre directement en voxels, et on
    # convertit seulement celle qui vient de l'exterieur.
    if aperture is None:
        aper = np.asarray(aperture_map(s, voxel_size=(1.0, 1.0, 1.0)), dtype=np.float32)
    else:
        aper = as_array(aperture).astype(np.float32, copy=False)
        if not np.allclose(spacing, 1.0):
            aper = aper / float(spacing.min())

    shape = s.shape
    a = np.zeros(shape, dtype=np.float32)
    b = np.zeros(shape, dtype=np.float32)
    c = np.zeros(shape, dtype=np.float32)
    theta = np.zeros(shape, dtype=np.float32)
    phi = np.zeros(shape, dtype=np.float32)
    direction = np.zeros((*shape, 3), dtype=np.float32) if with_orientation else None

    coords = np.argwhere(seed_mask)
    n_degenerate = 0
    it = enumerate(coords)
    if progress:
        try:
            from tqdm.auto import tqdm

            it = enumerate(tqdm(coords, desc="tenseur local"))
        except ImportError:
            pass

    for _n, centre in it:
        ck, cj, ci = (int(x) for x in centre)
        local_r = float(aper[ck, cj, ci])
        if local_r <= 2.0:
            local_r = min_radius
        radius = expand_factor * local_r

        cloud = geodesic_ball(s, (ck, cj, ci), radius, connectivity=connectivity)
        if len(cloud) < 4:
            # boule degeneree : iMorph ecrit 1 partout et passe au suivant
            a[ck, cj, ci] = b[ck, cj, ci] = c[ck, cj, ci] = 1.0
            n_degenerate += 1
            continue

        pts = cloud.astype(np.float64) * spacing
        cov = np.cov(pts, rowvar=False, bias=True)
        lam, vec = np.linalg.eigh(cov)  # valeurs propres croissantes
        lam = np.clip(lam[::-1], 0.0, None)  # decroissantes, jamais negatives
        vec = vec[:, ::-1]
        axes = 2.0 * np.sqrt(lam)

        a[ck, cj, ci], b[ck, cj, ci], c[ck, cj, ci] = axes
        if with_orientation:
            v = vec[:, 0]
            direction[ck, cj, ci] = v
            th, ph = _spherical(v)
            theta[ck, cj, ci] = th
            phi[ck, cj, ci] = ph

    a_on_b = elongation_ratios(a, b, clip=expand_factor if clip_ratios else None)
    b_on_c = elongation_ratios(b, c, clip=expand_factor if clip_ratios else None)

    if propagate:
        fields = [a, b, c, a_on_b, b_on_c, theta, phi]
        a, b, c, a_on_b, b_on_c, theta, phi = (
            nearest_seed_propagation(f, seed_mask, s, voxel_size=voxel_size) for f in fields
        )
        if with_orientation:
            prop = np.zeros_like(direction)
            for comp in range(3):
                prop[..., comp] = nearest_seed_propagation(
                    direction[..., comp], seed_mask, s, voxel_size=voxel_size
                )
            direction = prop

    return ShapeTensor(
        a=a,
        b=b,
        c=c,
        a_on_b=a_on_b,
        b_on_c=b_on_c,
        theta=theta,
        phi=phi,
        seeds=seed_mask,
        direction=direction,
        n_seeds=len(coords),
        n_degenerate=n_degenerate,
        params={
            "expand_factor": expand_factor,
            "min_radius": min_radius,
            "voxel_size": tuple(spacing),
            "clip_ratios": clip_ratios,
            "connectivity": connectivity,
            "propagated": propagate,
            "seeds": seed_kind,
        },
    )


def elongation_ratios(num: np.ndarray, den: np.ndarray, *, clip: float | None = None):
    """`num / den` la ou `den > 0`, 0 ailleurs, eventuellement ecrete."""
    out = np.zeros_like(num, dtype=np.float32)
    ok = den > 0
    np.divide(num, den, out=out, where=ok)
    if clip is not None:
        np.minimum(out, np.float32(clip), out=out)
    return out


def classify_solid(
    tensor: ShapeTensor,
    solid=None,
    *,
    strut_threshold: float = 1.6,
    plate_threshold: float | None = 1.6,
):
    """Segmente le solide en noeuds, brins et plaques.

    Parameters
    ----------
    strut_threshold
        `a/b >= seuil` => brin. Defaut 1,6 : la valeur empirique de la these,
        stable sur toutes les mousses etudiees, et le defaut d'iMorph
        (`rodeThreshold`).
    plate_threshold
        Parmi les non-brins, `b/c >= seuil` => plaque, sinon noeud. Mettre
        `None` pour retrouver la classification binaire d'iMorph, qui ne
        distinguait pas noeuds et plaques.

    Returns
    -------
    numpy.ndarray
        Tableau `uint8` : 0 fluide, 1 noeud, 2 brin, 3 plaque.
    """
    s = np.asarray(solid, dtype=bool) if solid is not None else (tensor.a > 0) | (tensor.a_on_b > 0)
    out = np.zeros(tensor.a.shape, dtype=np.uint8)
    strut = s & (tensor.a_on_b >= strut_threshold)
    out[strut] = STRUT
    rest = s & ~strut
    if plate_threshold is None:
        out[rest] = NODE
    else:
        plate = rest & (tensor.b_on_c >= plate_threshold)
        out[plate] = PLATE
        out[rest & ~plate] = NODE
    return out


def strut_orientation(tensor: ShapeTensor, classes=None, *, bins: int = 36) -> pd.DataFrame:
    """Distribution des orientations, restreinte aux brins.

    Rend un `DataFrame` `azimuth`, `elevation`, `count` — la matiere des
    diagrammes polaires de la figure 3.43 de la these. L'orientation d'un noeud
    n'a pas de sens, d'ou la restriction.
    """
    if tensor.direction is None:
        raise ValueError("tenseur calcule sans orientation (with_orientation=False)")
    sel = tensor.seeds if classes is None else (np.asarray(classes) == STRUT)
    az = tensor.theta[sel]
    el = tensor.phi[sel]
    h_az, e_az = np.histogram(az, bins=bins, range=(0.0, 360.0))
    h_el, e_el = np.histogram(el, bins=bins // 2, range=(0.0, 90.0))
    n = max(len(h_az), len(h_el))

    def pad(x, m):
        return np.concatenate([x, np.full(m - len(x), np.nan)]) if len(x) < m else x

    return pd.DataFrame(
        {
            "azimuth": pad(0.5 * (e_az[:-1] + e_az[1:]), n),
            "azimuth_count": pad(h_az.astype(float), n),
            "elevation": pad(0.5 * (e_el[:-1] + e_el[1:]), n),
            "elevation_count": pad(h_el.astype(float), n),
        }
    )


def shape_classification(
    solid,
    *,
    strut_threshold: float = 1.6,
    plate_threshold: float | None = 1.6,
    **tensor_kwargs,
):
    """Tenseur puis classification, en un appel — rend la carte de classes.

    Commodite pour les pipelines et la ligne de commande, ou l'on veut la carte
    de classes et pas l'objet intermediaire. Equivaut a :

        st = local_shape_tensor(solid, **tensor_kwargs)
        classify_solid(st, solid, strut_threshold=..., plate_threshold=...)
    """
    from morphanalyzer.core.volume import as_array as _as

    st = local_shape_tensor(solid, **tensor_kwargs)
    return classify_solid(
        st,
        _as(solid).astype(bool),
        strut_threshold=strut_threshold,
        plate_threshold=plate_threshold,
    )
