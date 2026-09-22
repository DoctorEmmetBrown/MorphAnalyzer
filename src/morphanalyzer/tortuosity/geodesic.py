"""Tortuosites obtenues par propagation de front."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array
from morphanalyzer.distance.fmm import travel_time

__all__ = [
    "TortuosityResult",
    "PoiseuilleResult",
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


@dataclass
class PoiseuilleResult(Mapping):
    """Resultat d'une tortuosite de Poiseuille — et un `Mapping` par-dessus.

    La fonction rendait un `dict` de scalaires. C'etait suffisant a la ligne de
    commande et inexploitable ailleurs : ni carte, ni chemins, ni tableau, donc
    rien a afficher, rien a tracer, et dans un pipeline une seule ligne de
    journal ou tout est ecrase en texte.

    Cet objet porte les memes cles — `res["tortuosity"]` marche toujours, et
    `dict(res)` redonne exactement l'ancien dictionnaire — plus ce qu'il fallait
    garder : les deux champs qui ont servi, les chemins traces, et un tableau par
    chemin.

    Attributes
    ----------
    tortuosity, std
        Tortuosite de Carman des chemins **choisis par le champ de Poiseuille**,
        mesuree sur leur longueur geometrique, et sa dispersion.
    geometric_tortuosity
        Les memes points d'arrivee, mais les chemins choisis par la metrique
        geodesique. C'est la reference a laquelle comparer.
    mean_wall_distance_poiseuille, mean_wall_distance_geometric
        Distance moyenne a la paroi le long de chaque famille de chemins. C'est
        la que se voit l'effet : le fluide passe plus loin des parois.
    paths
        Une ligne par chemin : `metric`, `path`, `length`, `tortuosity`,
        `mean_wall_distance`, et le point d'arrivee `k`, `j`, `i`.
    speed
        Le champ de vitesse parabolique (`poiseuille_speed`).
    travel_time
        Les temps d'arrivee dans la metrique de Poiseuille. **A ne pas moyenner**
        pour en tirer une tortuosite : ces temps vivent dans une metrique ou la
        paroi coute infiniment cher. Ils servent a extraire les chemins et a
        voir le front.
    path_mask, geometric_path_mask
        Images d'etiquettes des chemins traces (`0` ailleurs), une etiquette par
        chemin — la figure 3.31 de la these.
    """

    variant: str
    separation: float
    n_paths: int
    tortuosity: float
    std: float
    geometric_tortuosity: float
    mean_wall_distance_poiseuille: float
    mean_wall_distance_geometric: float
    paths: pd.DataFrame = field(default_factory=pd.DataFrame)
    speed: np.ndarray | None = None
    travel_time: np.ndarray | None = None
    path_mask: np.ndarray | None = None
    geometric_path_mask: np.ndarray | None = None

    #: Les cles de l'ancien dictionnaire, dans l'ordre ou il les produisait.
    SCALARS = (
        "variant",
        "separation",
        "n_paths",
        "tortuosity",
        "std",
        "mean_wall_distance_poiseuille",
        "geometric_tortuosity",
        "mean_wall_distance_geometric",
    )

    def __getitem__(self, key: str):
        if key not in self.SCALARS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self.SCALARS)

    def __len__(self) -> int:
        return len(self.SCALARS)

    @property
    def table(self) -> pd.DataFrame:
        """Le resume en une ligne, pret a etre ecrit en CSV."""
        return pd.DataFrame([{k: self[k] for k in self.SCALARS}])


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
    ends: str = "spread",
    keep_fields: bool = True,
    **fmm,
) -> PoiseuilleResult:
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

    Parameters
    ----------
    ends
        Comment choisir les points d'arrivee, partages par les deux metriques.

        `"spread"` (defaut) : la face d'arrivee est decoupee en une grille, et
        l'on retient dans chaque case le voxel atteint le plus tot. Les chemins
        couvrent alors la section.

        `"fastest"` : les `n_paths` voxels atteints le plus tot, tous confondus.
        C'est le choix le plus simple, et c'est un piege pour la mesure : ces
        voxels sont au bout des canaux les plus directs, si bien que la
        tortuosite geodesique qu'on leur associe vaut 1,000 dans a peu pres
        n'importe quel milieu ouvert — elle ne decrit plus le milieu. Utile si
        l'on veut precisement les trajets preferentiels.

        Dans les deux cas, `geometric_tortuosity` reste un **temoin apparie**,
        sur les memes points d'arrivee : pour la tortuosite du milieu, c'est
        `plane_tortuosity` qu'il faut, qui moyenne sur toute la face.
    keep_fields
        Garder le champ de vitesse, les temps d'arrivee et les images des
        chemins dans le resultat. Defaut `True` : c'est ce qui rend le calcul
        affichable. Passer `False` sur un tres gros volume, ou ces trois
        tableaux pesent chacun le volume en float32.

    Returns
    -------
    PoiseuilleResult
        `tortuosity` et `std` (sur la longueur geometrique des chemins),
        `geometric_tortuosity` (les memes chemins choisis par la metrique
        geodesique, pour comparaison), `mean_wall_distance` pour chacune des deux
        familles de chemins — c'est la que se voit l'effet : les chemins de
        Poiseuille restent plus loin des parois — plus, avec `keep_fields`, les
        champs et les chemins eux-memes.

        L'objet se comporte comme l'ancien dictionnaire : `res["tortuosity"]`
        marche, et `dict(res)` redonne les memes cles.
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
    want = max(1, n_paths)
    coords = np.argwhere(reach)
    order = np.argsort(T_geo[reach])
    if ends == "fastest":
        picked = coords[order[:want]]
    elif ends == "spread":
        # Une grille sur la face d'arrivee, le voxel le plus tot atteint par
        # case. Les chemins couvrent la section au lieu de s'entasser dans le
        # canal le plus direct.
        plane = [a for a in (0, 1, 2) if a != axis]
        side = max(1, int(np.ceil(np.sqrt(want))))
        keys = np.empty(len(coords), dtype=np.int64)
        for n, a in enumerate(plane):
            cell = np.minimum((coords[:, a] * side) // m.shape[a], side - 1)
            keys = keys * side + cell if n else cell.astype(np.int64)
        best: dict[int, int] = {}
        for idx in order:  # du plus tot au plus tard
            best.setdefault(int(keys[idx]), int(idx))
        chosen = sorted(best.values(), key=lambda i: T_geo[reach][i])[:want]
        picked = coords[chosen]
    else:
        raise ValueError("ends doit valoir 'spread' ou 'fastest'")
    ends_xyz = picked

    T_poi = np.asarray(travel_time(m, face, voxel_size=spacing, **kw, **fmm))

    agg: dict = {}
    rows: list[dict] = []
    masks: dict[str, np.ndarray] = {}
    for name, T in (("poiseuille", T_poi), ("geometric", T_geo)):
        lengths, walls = [], []
        mask_img = np.zeros(m.shape, dtype=np.int32)
        for n, e in enumerate(ends_xyz, start=1):
            if not np.isfinite(T[tuple(e)]):
                continue
            path = shortest_path(T, e, voxel_size=spacing)
            if len(path) < 2:
                continue
            steps = np.diff(path.astype(float) * spacing, axis=0)
            ell = float(np.linalg.norm(steps, axis=1).sum())
            wall = float(d_wall[tuple(path.T)].mean())
            lengths.append(ell)
            walls.append(wall)
            mask_img[tuple(path.T)] = n
            rows.append(
                {
                    "metric": name,
                    "path": n,
                    "length": ell,
                    "tortuosity": (ell / length) ** 2,
                    "mean_wall_distance": wall,
                    "k": int(e[0]),
                    "j": int(e[1]),
                    "i": int(e[2]),
                }
            )
        if not lengths:
            raise ValueError(f"aucun chemin exploitable en metrique {name}")
        masks[name] = mask_img
        tau = (np.asarray(lengths) / length) ** 2
        if name == "poiseuille":
            agg["tortuosity"] = float(tau.mean())
            agg["std"] = float(tau.std(ddof=0))
        else:
            agg["geometric_tortuosity"] = float(tau.mean())
        agg[f"mean_wall_distance_{name}"] = float(np.mean(walls))

    out = PoiseuilleResult(
        variant=variant,
        separation=float(length),
        n_paths=len(ends_xyz),
        paths=pd.DataFrame(rows),
        speed=field if keep_fields else None,
        travel_time=T_poi if keep_fields else None,
        path_mask=masks["poiseuille"] if keep_fields else None,
        geometric_path_mask=masks["geometric"] if keep_fields else None,
        **agg,
    )

    if out.tortuosity < 1.0:
        import warnings

        warnings.warn(
            f"tortuosite de Poiseuille {out.tortuosity:.3f} < 1, ce qui est "
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

    **Le critere est la pente, pas la valeur.** On prend le voisin qui maximise
    `(T(courant) - T(voisin)) / longueur du pas`, et non celui de plus petit
    `T`. La difference n'est pas cosmetique : dans un canal large, le front est
    quasi plan et beaucoup de voisins ont presque le meme `T`, si bien que le
    critere « plus petit `T` » choisit indifferemment un pas axial (longueur 1)
    ou un pas diagonal (longueur `sqrt(3)`) qui descend d'autant. Le chemin
    zigzague sans que rien ne le penalise, et sa longueur gonfle.

    Mesure : dans un tube droit, ou la tortuosite vaut exactement 1, le critere
    « plus petit `T` » donnait des chemins 7 a 10 % trop longs, soit une
    tortuosite de 1,17 au lieu de 1,00 ; dans une mousse, +20,8 % de longueur et
    une tortuosite de 1,46 pour des trajets rigoureusement droits. Avec la
    pente, le tube droit redonne 1,0000.
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

    spacing = np.asarray(voxel_size, dtype=float)
    step_len = np.linalg.norm(off.astype(float) * spacing, axis=1)
    shape = np.asarray(T.shape)

    path = [cur.copy()]
    for _ in range(max_steps):
        here = T[tuple(cur)]
        best, best_rate, best_t = None, 0.0, here
        for o, ell in zip(off, step_len, strict=True):
            nxt = cur + o
            if np.any(nxt < 0) or np.any(nxt >= shape):
                continue
            t = T[tuple(nxt)]
            if not np.isfinite(t) or t >= here:
                continue
            rate = (here - t) / ell  # pente, pas valeur
            if rate > best_rate:
                best_rate, best, best_t = rate, nxt, t
        if best is None:
            break
        cur = best
        path.append(cur.copy())
        if best_t <= 0.0:
            break
    return np.asarray(path)
