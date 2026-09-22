"""Carte d'ouverture locale, boules maximales, marqueurs de cellules.

Portage de `Thread/Granulometry/morphology.cpp::calc_Aperture_Map3DFAH*` et
`Thread/Granulometry/utility.cpp::createBallsFromIdMap*`,
`computeMaxBallsHistoFromIdMap`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array
from morphanalyzer.distance.edt import distance_transform

__all__ = [
    "aperture_map",
    "pore_size_distribution",
    "maximal_balls",
    "cell_markers",
    "BallTable",
]


def _radii(dist: np.ndarray, radii, n_radii: int, min_radius: float) -> np.ndarray:
    if radii is None:
        dmax = float(dist.max())
        radii = np.linspace(min_radius, dmax, num=n_radii)
    return np.asarray(sorted({float(r) for r in np.ravel(radii) if r >= min_radius}, reverse=True))


def aperture_map(
    mask,
    *,
    voxel_size=None,
    radii=None,
    n_radii: int = 32,
    min_radius: float = 1.0,
    as_diameter: bool = False,
):
    """Rayon de la plus grande boule incluse dans `mask` contenant chaque voxel.

    Algorithme : pour chaque rayon `r` par ordre decroissant, les centres
    admissibles sont les voxels dont la distance au complementaire vaut au moins
    `r` ; un voxel appartient a une boule de rayon `r` incluse dans `mask` si sa
    distance au centre admissible le plus proche ne depasse pas `r`. Le premier
    `r` qui couvre un voxel est son ouverture. Meme resultat que la file
    d'attente hierarchique d'iMorph, a la discretisation des rayons pres.

    Parameters
    ----------
    as_diameter
        Rend `2 * r` plutot que `r`. Le « diametre de pore » de la these.

    Notes
    -----
    Le parametre `apertureErrorPrecision` d'iMorph, qui elaguait les boules
    incluses dans une plus grande et faisait gagner un facteur 10, **est
    commente dans les sources 3.2**. Ici c'est la discretisation des rayons qui
    joue ce role, avec le meme compromis entre vitesse et finesse.
    """
    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    if not np.allclose(voxel_size, voxel_size[0]):
        raise NotImplementedError(
            "carte d'ouverture anisotrope non supportee : reechantillonner en voxels "
            "isotropes, ou passer voxel_size=1 et convertir apres coup"
        )
    scale = float(voxel_size[0])

    dist = np.asarray(distance_transform(m, voxel_size=(1.0, 1.0, 1.0)))
    aper = np.zeros(m.shape, dtype=np.float32)

    if float(dist.max()) > 0:
        for r in _radii(dist, radii, n_radii, min_radius):
            centres = dist >= r
            if not centres.any():
                continue
            covered = m & (aper == 0) & (ndi.distance_transform_edt(~centres) <= r)
            if covered.any():
                aper[covered] = np.float32(r)

    thin = m & (aper == 0)
    aper[thin] = dist[thin]

    aper *= scale
    if as_diameter:
        aper *= 2.0
    name = "aperture_diameter" if as_diameter else "aperture"
    return mask.with_data(aper, name=name) if isinstance(mask, Volume) else aper


def pore_size_distribution(aperture, *, mask=None, bins: int = 30, as_diameter: bool = True):
    """Histogramme volumique des tailles, pondere par le volume occupe.

    Rend un `DataFrame` avec `size`, `count`, `fraction` et `cumulative`.
    C'est la figure 2.20 de la these : la fraction du volume de la phase
    couverte par des boules de chaque taille.
    """
    a = as_array(aperture)
    sel = a > 0 if mask is None else (a > 0) & as_array(mask).astype(bool)
    vals = a[sel]
    if as_diameter and not str(getattr(aperture, "name", "")).endswith("diameter"):
        vals = vals * 2.0
    counts, edges = np.histogram(vals, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    frac = counts / total if total else counts.astype(float)
    return pd.DataFrame(
        {"size": centres, "count": counts, "fraction": frac, "cumulative": np.cumsum(frac)}
    )


@dataclass(slots=True)
class BallTable:
    """Boules maximales extraites de l'image d'identifiants.

    Attributes
    ----------
    table
        `DataFrame` : `k`, `j`, `i` (centre), `radius`, `volume` (voxels
        reellement attribues a cette boule), `theoretical_volume`,
        `fill_ratio`, `touches_border`.
    ids
        L'image d'identifiants d'ou vient la table.
    """

    table: pd.DataFrame
    ids: np.ndarray

    def __len__(self) -> int:
        return len(self.table)


def _theoretical_volumes(
    radii: np.ndarray,
    coords: np.ndarray | None = None,
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Nombre de voxels qu'une boule discrete **pourrait** occuper, a sa place.

    On compte les voxels a distance strictement inferieure au rayon, comme
    `computeMaxBallsHistoFromIdMap` d'iMorph, pour que le taux de remplissage
    soit comparable a ses seuils.

    Avec `coords` et `shape`, le compte est **ecrete a la boite** : une boule
    coupee par le bord du volume est comparee a la part d'elle-meme qui tient
    dans l'image, pas a la sphere entiere.

    C'est ce qui rend le taux de remplissage interpretable partout. Sans
    ecretage, une boule de bord parfaitement inscrite dans un pore affiche un
    taux de 0,42 simplement parce que la moitie d'elle sort de l'image — et se
    fait rejeter comme si elle etait mal formee. Le contournement etait de
    l'exempter du test (`keep_border_balls`), donc de garder **toutes** les
    boules de bord, y compris les vraies incompletes : d'ou une cellule de bord
    decoupee en plusieurs morceaux. Avec l'ecretage, la mediane des boules de
    bord passe de 0,42 a 0,99 sur une mousse de Voronoi, le seuil retrouve son
    sens, et les fragments tombent de 33 a 11 (mousse de 212 cellules).
    """
    out = np.empty(len(radii), dtype=np.int64)
    cache: dict[float, tuple[np.ndarray, int]] = {}
    clip = coords is not None and shape is not None
    if clip:
        nz, ny, nx = shape
        kk, jj, ii = coords[:, 0], coords[:, 1], coords[:, 2]
    for n, r in enumerate(radii):
        key = float(r)
        if key not in cache:
            ri = int(np.ceil(r))
            g = np.arange(-ri, ri + 1)
            dz, dy, dx = np.meshgrid(g, g, g, indexing="ij")
            inside = (dz * dz + dy * dy + dx * dx) < r * r
            cache[key] = (np.stack([dz[inside], dy[inside], dx[inside]], axis=1), int(inside.sum()))
        offsets, full = cache[key]
        if not clip:
            out[n] = full
            continue
        z, y, x = kk[n] + offsets[:, 0], jj[n] + offsets[:, 1], ii[n] + offsets[:, 2]
        if (
            0 <= z.min()
            and z.max() < nz
            and 0 <= y.min()
            and y.max() < ny
            and 0 <= x.min()
            and x.max() < nx
        ):
            out[n] = full  # entierement dans la boite
        else:
            out[n] = int(((z >= 0) & (z < nz) & (y >= 0) & (y < ny) & (x >= 0) & (x < nx)).sum())
    return out


def _claim_territories(
    mask: np.ndarray, dist: np.ndarray, centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Attribue a chaque centre son territoire, par rayon decroissant.

    Reproduit la boucle d'iMorph : les voxels sont traites du plus eloigne du
    bord au plus proche, et chaque boule peint son voisinage la ou aucune boule
    plus grande n'est deja passee (`if imGranulo(v) < tailleOuverture`). Traiter
    les centres par rayon decroissant rend la regle equivalente a « le premier
    arrive garde », donc un seul passage suffit.

    Rend `(ids, aper)` : l'indice aplati du centre proprietaire et le rayon de
    la boule, `-1` et `0` hors de `mask`.
    """
    nz, ny, nx = mask.shape
    ids = np.full(mask.shape, -1, dtype=np.int64)
    aper = np.zeros(mask.shape, dtype=np.float32)

    radii = dist[centres[:, 0], centres[:, 1], centres[:, 2]]
    order = np.argsort(radii)[::-1]
    for n in order:
        ck, cj, ci = (int(x) for x in centres[n])
        r = float(radii[n])
        if r <= 0:
            continue
        ri = int(np.ceil(r))
        k0, k1 = max(0, ck - ri), min(nz, ck + ri + 1)
        j0, j1 = max(0, cj - ri), min(ny, cj + ri + 1)
        i0, i1 = max(0, ci - ri), min(nx, ci + ri + 1)
        zz = np.arange(k0, k1)[:, None, None] - ck
        yy = np.arange(j0, j1)[None, :, None] - cj
        xx = np.arange(i0, i1)[None, None, :] - ci
        inside = (zz * zz + yy * yy + xx * xx) <= r * r
        box_m = mask[k0:k1, j0:j1, i0:i1]
        box_a = aper[k0:k1, j0:j1, i0:i1]
        take = inside & box_m & (box_a < r)
        if take.any():
            box_a[take] = np.float32(r)
            ids[k0:k1, j0:j1, i0:i1][take] = ck * ny * nx + cj * nx + ci
    return ids, aper


def maximal_balls(
    mask,
    *,
    distance=None,
    candidates: str = "maxima",
    h: float = 0.5,
    min_radius: float = 3.0,
    voxel_size=None,
) -> BallTable:
    """Table des boules maximales, avec leur taux de remplissage.

    Une boule est « quasi entiere » quand la fraction de son volume theorique
    qui lui reste effectivement attribuee est elevee : les boules voisines plus
    grandes lui ont pris peu de terrain, donc elle occupe bien une cavite
    propre. C'est le critere d'iMorph pour reconnaitre un centre de cellule, et
    il a le bon comportement : la these montre (fig. 3.3) que le nombre de
    marqueurs est stable entre 60 et 80 % de remplissage, avec sous-segmentation
    au-dela.

    Parameters
    ----------
    candidates
        `"maxima"` (defaut) : seuls les maxima regionaux de la carte de distance
        sont candidats — ce sont les centres de boules maximales. `"all"` : tous
        les voxels de `mask`, la force brute d'iMorph.

        iMorph parcourait tous les voxels, mais elaguait ceux dont la boule est
        circonscrite a une plus grande (`apertureErrorPrecision`, code commente
        dans la version 3.2) ; la these observe que « les points restants se
        situent pour la majorite sur le squelette des boules maximales »
        (fig. 2.19). Partir des maxima donne donc le meme ensemble utile, sans
        les 34 minutes de calcul que la these rapporte pour la force brute.
    h
        Profondeur des h-maxima retenus comme candidats. Plus petit = plus de
        candidats, donc des territoires plus fragmentes.
    min_radius
        Rayon minimal retenu dans la table, en voxels. Defaut 3, la valeur
        d'iMorph (`minimalDistToSolid`).
    """
    from skimage.morphology import h_maxima

    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3

    if distance is None:
        distance = distance_transform(m, voxel_size=(1.0, 1.0, 1.0))
    dist = np.asarray(distance, dtype=np.float32)

    if candidates == "maxima":
        cand = (h_maxima(dist, float(h)) > 0) & m
        if not cand.any():
            cand = m
    elif candidates == "all":
        cand = m
    else:
        raise ValueError("candidates doit valoir 'maxima' ou 'all'")

    ids, _aper = _claim_territories(m, dist, np.argwhere(cand))

    flat_ids = ids[m]
    uniq, volumes = np.unique(flat_ids[flat_ids >= 0], return_counts=True)
    cols = ["k", "j", "i", "radius", "volume", "theoretical_volume", "fill_ratio", "touches_border"]
    if len(uniq) == 0:
        return BallTable(pd.DataFrame(columns=cols), ids)

    kk, jj, ii = np.unravel_index(uniq, m.shape)
    radius = dist[kk, jj, ii]
    theo = _theoretical_volumes(radius, np.stack([kk, jj, ii], axis=1), m.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        fill = np.where(theo > 0, volumes / theo, 0.0)

    nz, ny, nx = m.shape
    r_ceil = np.ceil(radius)
    border = (
        (kk - r_ceil < 0)
        | (kk + r_ceil >= nz)
        | (jj - r_ceil < 0)
        | (jj + r_ceil >= ny)
        | (ii - r_ceil < 0)
        | (ii + r_ceil >= nx)
    )

    keep = radius >= min_radius
    table = pd.DataFrame(
        {
            "k": kk[keep],
            "j": jj[keep],
            "i": ii[keep],
            "radius": radius[keep],
            "volume": volumes[keep],
            "theoretical_volume": theo[keep],
            "fill_ratio": fill[keep],
            "touches_border": border[keep],
        }
    ).sort_values("radius", ascending=False, ignore_index=True)
    table.attrs["voxel_size"] = tuple(voxel_size)
    table.attrs["candidates"] = candidates
    return BallTable(table, ids)


def cell_markers(
    mask,
    *,
    balls: BallTable | None = None,
    fill_ratio: float = 0.65,
    min_radius: float = 3.0,
    keep_border_balls: bool = False,
    return_table: bool = False,
    **kwargs,
):
    """Marqueurs de cellules : les centres des boules quasi entieres.

    Rend une image de labels (`int32`, `0` = pas un marqueur, `1..n`), un
    marqueur par boule retenue. Ce sont les germes du watershed de la phase 4.

    Parameters
    ----------
    fill_ratio
        Taux de remplissage minimal. Defaut **0,65**, la valeur d'iMorph 3.2
        (`thresholdVolumeBouleEntire = 35`, soit `> (100-35)/100`). La these
        cite 75 % ; les deux sont dans le palier 60–80 % qu'elle identifie
        comme stable (fig. 3.3). Monter au-dela de 0,8 sous-segmente.
    keep_border_balls
        Garder les boules coupees par le bord **sans leur appliquer le test de
        remplissage**. Defaut `False`, et ce defaut a change : depuis que le
        volume theorique est ecrete a la boite
        (:func:`_theoretical_volumes`), une boule de bord bien formee obtient un
        taux eleve toute seule et passe le test comme les autres. L'exemption
        ne servait qu'a compenser un taux artificiellement bas, et elle laissait
        passer les vraies boules incompletes — d'ou des cellules de bord
        decoupees en morceaux.

        Mesure sur trois mousses de Voronoi (128 cube, seuil 0,65), fragments =
        cellules predites de moins de 15 % du volume median :

        | mousse | exemption (ancien) | ecretage (nouveau) |
        |---|---:|---:|
        | 80 cellules | 68 marqueurs, 12 fragments | 54 marqueurs, **7** |
        | 106 cellules | 84 marqueurs, 17 fragments | 68 marqueurs, **9** |
        | 212 cellules | 172 marqueurs, 33 fragments | 136 marqueurs, **11** |

        L'IoU median des cellules interieures ne bouge pas (0,907 -> 0,908 sur
        la derniere). On perd donc des faux germes, pas des cellules.

    Notes
    -----
    Avec `fill_ratio` eleve **et** `keep_border_balls=False`, les cellules de
    bord peuvent perdre leur germe et fusionner avec leurs voisines. C'est le
    compromis que decrit la figure 3.3 de la these : au-dela de 80 % le nombre
    de marqueurs chute et on sous-segmente.
    """
    m = as_array(mask).astype(bool, copy=False)
    if balls is None:
        balls = maximal_balls(m, min_radius=min_radius, **kwargs)
    t = balls.table
    sel = (t["radius"] >= min_radius) & (t["fill_ratio"] > fill_ratio)
    if keep_border_balls:
        sel |= (t["radius"] >= min_radius) & t["touches_border"]
    chosen = t[sel]

    markers = np.zeros(m.shape, dtype=np.int32)
    if len(chosen):
        markers[chosen["k"].to_numpy(), chosen["j"].to_numpy(), chosen["i"].to_numpy()] = np.arange(
            1, len(chosen) + 1, dtype=np.int32
        )
    out = mask.with_data(markers, name="markers") if isinstance(mask, Volume) else markers
    return (out, chosen.reset_index(drop=True)) if return_table else out
