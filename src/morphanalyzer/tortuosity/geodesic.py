"""Tortuosites obtenues par propagation de front."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array
from morphanalyzer.distance.fmm import travel_time

__all__ = [
    "TortuosityResult",
    "point_tortuosity",
    "plane_tortuosity",
    "directional_tortuosity",
    "poiseuille_speed",
    "poiseuille_tortuosity",
    "shortest_path",
]


@dataclass(slots=True)
class TortuosityResult:
    """Resultat d'un calcul de tortuosite.

    Attributes
    ----------
    value
        La tortuosite agregee, moyenne sur les points d'arrivee retenus.
    std
        Ecart-type sur ces memes points — la dispersion compte autant que la
        moyenne, les histogrammes de la figure 3.21 en temoignent.
    tau
        Carte de tortuosite point par point, `nan` ou l'on n'a pas propage.
    travel_time
        Temps d'arrivee brut, utile pour extraire un plus court chemin.
    n_reached
        Nombre de voxels atteints. S'il est tres inferieur au volume de la
        phase, c'est que celle-ci n'est pas connexe depuis la source.
    params
        Parametres effectifs.
    """

    value: float
    std: float
    tau: np.ndarray
    travel_time: np.ndarray
    n_reached: int
    params: dict = field(default_factory=dict)


def _finish(tau, T, sel, params) -> TortuosityResult:
    vals = tau[sel]
    vals = vals[np.isfinite(vals)]
    return TortuosityResult(
        value=float(vals.mean()) if vals.size else float("nan"),
        std=float(vals.std(ddof=0)) if vals.size else float("nan"),
        tau=tau,
        travel_time=T,
        n_reached=int(np.isfinite(T).sum()),
        params=params,
    )


def point_tortuosity(
    mask,
    source,
    *,
    voxel_size=None,
    min_distance: float = 1.0,
    reference: str = "free_front",
    **fmm,
) -> TortuosityResult:
    """Tortuosite depuis un point unique, mesuree en tout point atteint.

    La these propage depuis le centre de l'echantillon et regarde la
    distribution des tortuosites (figures 3.20 et 3.21) : 1,02 en moyenne pour
    la mousse d'aluminium, 1,09 pour le fritte de polyethylene, plus disperse.

    Parameters
    ----------
    min_distance
        Les voxels a moins de cette distance de la source sont ignores : tout
        pres, le rapport est domine par la discretisation et diverge.
    reference
        Ce qui sert de longueur de reference au denominateur.

        `"free_front"` (defaut) : le temps d'arrivee du **meme front dans le meme
        volume sans obstacle**. `"euclidean"` : la distance euclidienne exacte.

        La difference n'est pas cosmetique. Le schema du 1er ordre du fast
        marching surestime les distances obliques : mesure en milieu libre, le
        biais median est de **+4 %** sur la distance, donc **+8,8 %** sur la
        tortuosite qui en est le carre (+2,7 % sur une diagonale 2D, +3,8 % sur
        une diagonale 3D, exact sur un axe). Rapporter le trajet geodesique a la
        distance euclidienne melange donc la tortuosite du milieu et l'erreur de
        l'instrument.

        En divisant par le front libre, les deux mesures partagent la meme
        discretisation et le biais s'annule : en milieu libre la tortuosite vaut
        exactement 1, au lieu de 1,088.
    """
    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    spacing = np.asarray(voxel_size, dtype=float)

    src = np.asarray(source, dtype=float)
    if src.size != 3:
        raise ValueError("source doit etre un point (k, j, i)")

    point = tuple(int(v) for v in src)
    T = np.asarray(travel_time(m, point, voxel_size=spacing, **fmm))

    if reference == "free_front":
        ref = np.asarray(
            travel_time(np.ones(m.shape, dtype=bool), point, voxel_size=spacing, **fmm)
        )
    elif reference == "euclidean":
        grids = np.ogrid[tuple(slice(0, n) for n in m.shape)]
        ref = np.sqrt(sum(((g - c) * s) ** 2 for g, c, s in zip(grids, src, spacing, strict=True)))
    else:
        raise ValueError("reference doit valoir 'free_front' ou 'euclidean'")

    tau = np.full(m.shape, np.nan, dtype=np.float32)
    sel = m & np.isfinite(T) & np.isfinite(ref) & (ref >= min_distance)
    tau[sel] = (T[sel] / ref[sel]) ** 2
    return _finish(
        tau,
        T,
        sel,
        {"source": point, "min_distance": min_distance, "reference": reference},
    )


def plane_tortuosity(mask, face: int = 0, *, voxel_size=None, **fmm) -> TortuosityResult:
    """Tortuosite entre deux plans : propagation depuis une face, lue sur l'opposee.

    `face` designe la face de depart : `0` = z minimal, `1` = z maximal, `2` = y
    minimal, etc. La tortuosite est moyennee sur la face d'arrivee, comme dans la
    these (§3.2.3).

    Notes
    -----
    Les valeurs sont **plus faibles** qu'en propageant depuis un point : « propager
    depuis un plan aura tendance a minimiser les valeurs derriere un obstacle ».
    La these mesure 1,012 depuis un point contre 1,0014 depuis un plan sur la
    mousse d'aluminium. Les deux grandeurs ne sont pas interchangeables.
    """
    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    spacing = np.asarray(voxel_size, dtype=float)

    axis, side = face // 2, face % 2
    T = np.asarray(travel_time(m, face, voxel_size=spacing, **fmm))

    n = m.shape[axis]
    length = (n - 1) * spacing[axis]
    arrival = [slice(None)] * 3
    arrival[axis] = slice(0, 1) if side else slice(-1, None)
    arrival = tuple(arrival)

    tau = np.full(m.shape, np.nan, dtype=np.float32)
    ok = m & np.isfinite(T)
    tau[ok] = (T[ok] / length) ** 2

    sel = np.zeros(m.shape, dtype=bool)
    sel[arrival] = True
    sel &= ok
    return _finish(tau, T, sel, {"face": face, "separation": length})


def _inscribed_box(shape, angle_deg: float, axis: int) -> tuple[slice, slice, slice]:
    """Parallelepipede inscrit au cylindre, pour une rotation donnee.

    La these extrait les parallelepipedes inscrits a l'image cylindrique avant
    de tourner (fig. 3.23) : sans cela, la rotation fait entrer du vide dans le
    volume et la tortuosite mesuree n'a plus de sens.
    """
    others = [a for a in (0, 1, 2) if a != axis]
    radius = min(shape[a] for a in others) / 2.0
    half = radius / np.sqrt(2.0)
    out = [slice(None)] * 3
    for a in others:
        c = (shape[a] - 1) / 2.0
        out[a] = slice(int(np.ceil(c - half)), int(np.floor(c + half)) + 1)
    return tuple(out)


def directional_tortuosity(
    mask, *, angles=None, axis: int = 0, voxel_size=None, inscribed: bool = True, **fmm
) -> pd.DataFrame:
    """Tortuosite de plan pour une revolution autour de `axis`.

    Le volume est tourne d'un angle autour de `axis`, un parallelepipede inscrit
    est extrait, et la tortuosite de plan y est mesuree dans une direction
    transverse. C'est la demarche des figures 3.23 a 3.25 de la these, qui
    revele une sinusoide de periode 180 degres — et le fait que la tortuosite du
    solide et celle du fluide sont en opposition de phase.

    Parameters
    ----------
    angles
        Liste d'angles en degres, ou un **entier** : le nombre d'angles
        regulierement repartis sur une demi-revolution. Par defaut, un angle
        tous les 22,5 degres.

    Returns
    -------
    pandas.DataFrame
        `angle`, `tortuosity`, `std`, `n_reached`.
    """
    from scipy import ndimage as ndi

    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    if angles is None:
        angles = np.arange(0.0, 180.0, 22.5)
    elif np.isscalar(angles) and float(angles) == int(angles) and int(angles) > 1:
        angles = np.linspace(0.0, 180.0, int(angles), endpoint=False)
    else:
        angles = np.atleast_1d(np.asarray(angles, dtype=float))

    others = [a for a in (0, 1, 2) if a != axis]
    rows = []
    for ang in np.asarray(angles, dtype=float):
        rot = (
            m
            if ang == 0.0
            else ndi.rotate(
                m.astype(np.uint8), ang, axes=tuple(others), order=0, reshape=False
            ).astype(bool)
        )
        sub = rot[_inscribed_box(m.shape, ang, axis)] if inscribed else rot
        face = 2 * others[0]
        try:
            res = plane_tortuosity(sub, face, voxel_size=voxel_size, **fmm)
            rows.append(
                {
                    "angle": ang,
                    "tortuosity": res.value,
                    "std": res.std,
                    "n_reached": res.n_reached,
                }
            )
        except ValueError:  # la phase ne touche pas la face apres rotation
            rows.append({"angle": ang, "tortuosity": np.nan, "std": np.nan, "n_reached": 0})
    return pd.DataFrame(rows)


def poiseuille_speed(
    mask, *, distance=None, aperture=None, variant: str = "physical", n_radii: int = 24
):
    """Champ de vitesse a profil parabolique dans les canaux.

    Parameters
    ----------
    variant
        `"physical"` (defaut) : le profil de Poiseuille reel,
        `v = 1 - (r/R)^2` avec `r = R - d` la distance a l'axe, donc `v = 0` a la
        paroi et `v = 1` au centre. Le temps d'arrivee est alors le temps de
        transit d'un traceur suivant la ligne de courant la plus rapide.
        `"imorph"` : reproduit l'equation 3.11 de la these, `1 - d^2/R^2`,
        **utilisee comme lenteur** — voir `distance.travel_time`. La lenteur
        s'annule au centre des canaux, donc les temps d'arrivee y dependent du
        plancher numerique employe pour l'inverser : ils ne sont pas
        interpretables, et `poiseuille_tortuosity` refuse cette variante.
        Conservee pour reproduire le champ d'iMorph, pas pour mesurer.

    Returns
    -------
    numpy.ndarray
        Le champ, a passer a `travel_time(speed=...)` pour `"physical"` et a
        `travel_time(cost=...)` pour `"imorph"`.
    """
    from morphanalyzer.distance.edt import distance_transform
    from morphanalyzer.granulometry.aperture import aperture_map

    m = as_array(mask).astype(bool, copy=False)
    d = (
        np.asarray(distance_transform(m, voxel_size=(1.0, 1.0, 1.0)), dtype=np.float32)
        if distance is None
        else np.asarray(distance, dtype=np.float32)
    )
    R = (
        np.asarray(aperture_map(m, voxel_size=(1.0, 1.0, 1.0), n_radii=n_radii), dtype=np.float32)
        if aperture is None
        else np.asarray(aperture, dtype=np.float32)
    )
    ratio = np.zeros(m.shape, dtype=np.float32)
    np.divide(d, R, out=ratio, where=R > 0)
    np.clip(ratio, 0.0, 1.0, out=ratio)

    if variant == "imorph":
        out = 1.0 - ratio**2
    elif variant == "physical":
        out = 1.0 - (1.0 - ratio) ** 2  # = ratio * (2 - ratio)
    else:
        raise ValueError("variant doit valoir 'physical' ou 'imorph'")
    out = out.astype(np.float32)
    out[~m] = 0.0
    return out


def poiseuille_tortuosity(
    mask,
    face: int = 0,
    *,
    variant: str = "physical",
    distance=None,
    aperture=None,
    voxel_size=None,
    n_paths: int = 16,
    **fmm,
) -> dict:
    """Tortuosite du chemin choisi par une propagation de type Poiseuille.

    « Si le fluide est newtonien, lors d'un ecoulement laminaire le chemin pris
    par le fluide ne sera pas forcement le chemin topologiquement le plus court »
    (these §3.2.3). Le champ de vitesse parabolique fait passer le front par les
    axes des canaux au lieu de raser les parois, ce qui donne des trajets qui
    ressemblent a des lignes de courant.

    Construction, celle de la these : le champ de Poiseuille **choisit** le
    chemin, puis on mesure la **longueur geometrique** de ce chemin. Moyenner la
    carte des temps d'arrivee n'aurait pas de sens — ces temps sont exprimes dans
    une metrique ou la paroi coute infiniment cher.

    Returns
    -------
    dict
        `tortuosity` et `std` (sur la longueur geometrique des chemins),
        `geometric_tortuosity` (les memes chemins choisis par la metrique
        geodesique, pour comparaison), `mean_wall_distance` pour chacune des deux
        familles de chemins — c'est la que se voit l'effet : les chemins de
        Poiseuille restent plus loin des parois.
    """
    from morphanalyzer.distance.edt import distance_transform

    if variant == "imorph":
        raise ValueError(
            "variant='imorph' n'est pas utilisable pour mesurer une tortuosite. "
            "L'equation 3.11 de la these s'annule au centre des canaux, donc les "
            "temps d'arrivee y dependent du plancher numerique employe pour "
            "inverser la lenteur, et l'extraction de chemin suit ce bruit. "
            "Le champ reste disponible pour reproduire iMorph :\n"
            "    c = tortuosity.poiseuille_speed(mask, variant='imorph')\n"
            "    T = distance.travel_time(mask, face, cost=c)\n"
            "Pour mesurer, utiliser variant='physical'."
        )

    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    spacing = np.asarray(voxel_size, dtype=float)

    d_wall = (
        np.asarray(distance_transform(m, voxel_size=(1.0, 1.0, 1.0)), dtype=np.float32)
        if distance is None
        else np.asarray(distance, dtype=np.float32)
    )
    field = poiseuille_speed(m, distance=d_wall, aperture=aperture, variant=variant)
    kw = {"cost": field} if variant == "imorph" else {"speed": field}

    axis, side = face // 2, face % 2
    length = (m.shape[axis] - 1) * spacing[axis]
    arrival = [slice(None)] * 3
    arrival[axis] = slice(0, 1) if side else slice(-1, None)
    arrival = tuple(arrival)

    # Les deux metriques doivent partager les **memes** points d'arrivee, sinon
    # on compare des trajets entre points differents et la comparaison ne dit
    # rien. On choisit les sorties naturelles : les voxels de la face d'arrivee
    # les plus vite atteints en metrique geodesique.
    T_geo = np.asarray(travel_time(m, face, voxel_size=spacing, **fmm))
    reach = np.zeros(m.shape, dtype=bool)
    reach[arrival] = True
    reach &= m & np.isfinite(T_geo)
    if not reach.any():
        raise ValueError("la phase ne relie pas les deux faces")
    ends = np.argwhere(reach)
    ends = ends[np.argsort(T_geo[reach])[: max(1, n_paths)]]

    T_poi = np.asarray(travel_time(m, face, voxel_size=spacing, **kw, **fmm))

    out: dict = {"variant": variant, "separation": length, "n_paths": len(ends)}
    for name, T in (("poiseuille", T_poi), ("geometric", T_geo)):
        lengths, walls = [], []
        for e in ends:
            if not np.isfinite(T[tuple(e)]):
                continue
            path = shortest_path(T, e, voxel_size=spacing)
            if len(path) < 2:
                continue
            steps = np.diff(path.astype(float) * spacing, axis=0)
            lengths.append(float(np.linalg.norm(steps, axis=1).sum()))
            walls.append(float(d_wall[tuple(path.T)].mean()))
        if not lengths:
            raise ValueError(f"aucun chemin exploitable en metrique {name}")
        tau = (np.asarray(lengths) / length) ** 2
        if name == "poiseuille":
            out["tortuosity"] = float(tau.mean())
            out["std"] = float(tau.std(ddof=0))
        else:
            out["geometric_tortuosity"] = float(tau.mean())
        out[f"mean_wall_distance_{name}"] = float(np.mean(walls))

    if out["tortuosity"] < 1.0:
        import warnings

        warnings.warn(
            f"tortuosite de Poiseuille {out['tortuosity']:.3f} < 1, ce qui est "
            "geometriquement impossible pour une longueur de chemin : la descente "
            "de gradient s'est arretee trop tot, probablement sur un plateau de la "
            "carte des temps. Verifier que le champ de vitesse ne s'annule pas sur "
            "une region etendue.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def shortest_path(
    travel_time_map, start, *, voxel_size=None, max_steps: int = 100_000
) -> np.ndarray:
    """Plus court chemin par descente de gradient sur la carte des temps.

    Rend un tableau `(n, 3)` de coordonnees, de `start` vers la source (le
    minimum de la carte). C'est la methode de la figure 3.31 de la these.

    Notes
    -----
    On descend de voisin en voisin sur la grille (26-connexite) plutot que par
    integration continue du gradient : le chemin est donc un chemin de voxels,
    pas une courbe lisse. C'est suffisant pour mesurer une longueur et pour
    visualiser, et cela ne peut pas sortir du domaine ni boucler.
    """
    T = np.asarray(travel_time_map, dtype=np.float64)
    if voxel_size is None:
        voxel_size = (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3

    from morphanalyzer.core.neighborhood import connectivity_offsets

    off = connectivity_offsets(26)
    cur = np.asarray(start, dtype=int)
    if not np.isfinite(T[tuple(cur)]):
        raise ValueError("le point de depart n'a pas ete atteint par la propagation")

    path = [cur.copy()]
    for _ in range(max_steps):
        best, best_t = None, T[tuple(cur)]
        for o in off:
            nxt = cur + o
            if np.any(nxt < 0) or np.any(nxt >= np.asarray(T.shape)):
                continue
            t = T[tuple(nxt)]
            if t < best_t:
                best_t, best = t, nxt
        if best is None or best_t <= 0.0:
            if best is not None:
                path.append(best)
            break
        cur = best
        path.append(cur.copy())
    return np.asarray(path)
