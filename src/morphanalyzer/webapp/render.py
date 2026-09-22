"""Rendu d'une coupe en PNG, cote serveur.

Envoyer un volume entier au navigateur n'a pas de sens : un 1024^3 en float32
pese 4 Go. On envoie **une image par coupe affichee**, composee ici, ou le
tableau est deja memmappe. Le cout est independant de la taille du volume, et
l'interface marche aussi bien sur une machine de calcul distante.

Les rampes de couleur suivent la meme regle que les figures du tutoriel : la
**clarte** doit varier de facon monotone d'un bout a l'autre. Une rampe qui
repart en arriere — l'arc-en-ciel — fabrique des frontieres qui n'existent pas
dans les donnees. Les cinq rampes a teinte unique vont du clair au fonce ; les
deux rampes perceptuelles (`tortuosite`, `fire`) tournent en teinte mais leur
L* CIELAB croit strictement, ce qui est la condition qui compte.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

__all__ = [
    "RAMPS",
    "NODATA_COLOR",
    "colormap",
    "label_colors",
    "render_slice",
    "LayerView",
    "png_bytes",
]

#: Couleur des voxels hors domaine (temps infini, valeur manquante). Choisie
#: pour etre a plus de 19 unites de dE00 de **tous** les paliers des deux
#: rampes perceptuelles, en vision normale comme en deuteranopie et en
#: protanopie : elle ne peut pas passer pour une valeur.
NODATA_COLOR = "#ff00ff"

#: Rampes sequentielles, une teinte chacune, du clair au fonce.
RAMPS: dict[str, list[str]] = {
    "blue": [
        "#f2f7fe",
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
    "orange": [
        "#fdf4ef",
        "#fbdcc9",
        "#f7bb99",
        "#f29566",
        "#eb6834",
        "#c44e22",
        "#993a17",
        "#6b280f",
    ],
    "teal": [
        "#eefaf5",
        "#c8efe0",
        "#95e0c4",
        "#5ecda3",
        "#1baf7a",
        "#128a60",
        "#0d6748",
        "#08462f",
    ],
    "violet": [
        "#f3f2fb",
        "#d9d5f1",
        "#b9b2e5",
        "#948ad6",
        "#6f63c4",
        "#4a3aa7",
        "#372b7d",
        "#241c54",
    ],
    # Rampe perceptuelle a 16 paliers, construite en LCh : L* de 11 a 96 par pas
    # reguliers, teinte de 282 deg (bleu nuit) a 100 deg (jaune). Pensee pour les
    # champs monotones qu'on lit par leurs iso-valeurs — la carte de temps de
    # parcours de la tortuosite en premier lieu. Mesures : L* strictement
    # croissant (pas minimal 5,2), dE00 entre paliers voisins >= 4,9, >= 3,7 en
    # deuteranopie, >= 3,2 en protanopie.
    "tortuosite": [
        "#001c49",
        "#002956",
        "#003862",
        "#00466d",
        "#005577",
        "#00637f",
        "#007487",
        "#00848a",
        "#00948b",
        "#2ba388",
        "#54b283",
        "#70c27f",
        "#8fd278",
        "#b3e071",
        "#daed6b",
        "#fff767",
    ],
    # Thermique (corps noir) : L* de 3 a 99, teinte de 25 a 100 deg, chroma
    # maximale au milieu et nulle aux deux bouts. dE00 entre paliers voisins
    # >= 6,1 (>= 4,5 en deuteranopie, >= 3,9 en protanopie). Pour les champs a
    # grande dynamique : carte d'ouverture, distance, temps de parcours.
    "fire": [
        "#140807",
        "#301110",
        "#4e1315",
        "#6e1219",
        "#8c151c",
        "#a32c1f",
        "#ba4222",
        "#cf5823",
        "#e07021",
        "#ea8a1c",
        "#f1a51b",
        "#f5c021",
        "#f7d259",
        "#f9e18f",
        "#fbefc3",
        "#fefcf6",
    ],
    "grey": [
        "#ffffff",
        "#e0e0dd",
        "#c1c1bd",
        "#a2a29d",
        "#83837e",
        "#646460",
        "#454543",
        "#111111",
    ],
}
DEFAULT_RAMP = "blue"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def colormap(name: str = DEFAULT_RAMP, n: int = 256, bands: int | None = None) -> np.ndarray:
    """Rampe `(n, 3)` uint8, interpolee entre les paliers documentes.

    Avec `bands`, la rampe devient **escalier** : `bands` couleurs constantes par
    morceaux, prises aux bouts et aux nœuds de la rampe continue. Les frontieres
    tombent alors a des valeurs connues — `vmin + k (vmax - vmin) / bands` — et
    l'image devient une carte d'iso-valeurs.

    C'est la seule discretisation honnete : la frontiere est annoncee, elle n'est
    pas un artefact de la rampe. Pour `tortuosite`, `bands=16` redonne exactement
    les seize couleurs documentees, et les bandes se lisent comme les isochrones
    d'un front — leur ecart a un plan **est** la tortuosite.
    """
    if name not in RAMPS:
        raise ValueError(f"rampe inconnue : {name!r}. Disponibles : {sorted(RAMPS)}")
    anchors = np.array([_hex_to_rgb(h) for h in RAMPS[name]], dtype=float)
    x = np.linspace(0.0, 1.0, len(anchors))
    if bands is None or bands <= 0:
        xi = np.linspace(0.0, 1.0, n)
    else:
        bands = int(min(bands, n))
        levels = np.linspace(0.0, 1.0, bands)
        xi = levels[np.minimum((np.arange(n) * bands) // n, bands - 1)]
    return np.stack([np.interp(xi, x, anchors[:, c]) for c in range(3)], axis=1).astype(np.uint8)


def label_colors(n: int, *, seed: int = 0) -> np.ndarray:
    """`(n + 1, 3)` uint8 : l'indice 0 est le fond, les suivants sont distincts.

    Les teintes avancent du nombre d'or, ce qui evite que deux etiquettes
    voisines se ressemblent, et la luminosite reste dans une bande etroite pour
    qu'aucune etiquette ne disparaisse dans le fond.
    """
    out = np.zeros((max(n, 0) + 1, 3), dtype=np.uint8)
    if n <= 0:
        return out
    idx = np.arange(1, n + 1, dtype=float)
    hue = np.mod(seed * 0.113 + idx * 0.61803398875, 1.0)
    sat = 0.45 + 0.25 * np.mod(idx * 0.37, 1.0)
    val = 0.55 + 0.30 * np.mod(idx * 0.71, 1.0)
    h6 = hue * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = val * (1 - sat), val * (1 - sat * f), val * (1 - sat * (1 - f))
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [val, q, p, p, t, val])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, val, val, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, val, val, q])
    out[1:] = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)
    return out


@dataclass
class LayerView:
    """Comment afficher un calque : ce que l'interface envoie au serveur."""

    data: np.ndarray
    kind: str = "scalar"
    ramp: str = DEFAULT_RAMP
    vmin: float | None = None
    vmax: float | None = None
    nodata: float | None = None
    bands: int | None = None
    #: Couleur des voxels non finis **autres que `nan`** : `inf` (dans le
    #: domaine, jamais atteint) et la valeur sentinelle `nodata`. `nan` reste
    #: transparent, car c'est le hors-domaine.
    nodata_color: str | None = None
    alpha: float = 1.0
    color: str = "#2a78d6"
    visible: bool = True


def _take(arr: np.ndarray, axis: int, index: int, step: int = 1) -> np.ndarray:
    index = int(np.clip(index, 0, arr.shape[axis] - 1))
    sl = np.take(arr, index, axis=axis)
    if step > 1:
        sl = sl[::step, ::step]
    return np.asarray(sl)


def _to_rgba(sl: np.ndarray, view: LayerView) -> np.ndarray:
    """Une coupe 2D -> RGBA uint8, transparent la ou il n'y a rien a montrer."""
    h, w = sl.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if view.kind == "binary":
        mask = sl.astype(bool)
        rgba[..., :3] = np.array(_hex_to_rgb(view.color), dtype=np.uint8)
        rgba[..., 3] = np.where(mask, 255, 0)
        return rgba

    if view.kind == "labels":
        lab = sl.astype(np.int64, copy=False)
        n = int(lab.max()) if lab.size else 0
        lut = label_colors(n)
        idx = np.clip(lab, 0, n)
        rgba[..., :3] = lut[idx]
        rgba[..., 3] = np.where(lab > 0, 255, 0)
        return rgba

    # grey et scalar : fenetrage puis rampe
    a = sl.astype(np.float32, copy=False)
    valid = np.isfinite(a)
    if view.nodata is not None:
        valid &= a != view.nodata
    lo = view.vmin if view.vmin is not None else (float(a[valid].min()) if valid.any() else 0.0)
    hi = view.vmax if view.vmax is not None else (float(a[valid].max()) if valid.any() else 1.0)
    if hi <= lo:
        hi = lo + 1.0
    with np.errstate(invalid="ignore"):
        t = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    # NaN et inf ne survivent pas a une conversion en entier : les neutraliser
    # ici evite un RuntimeWarning que l'interface remonterait comme un vrai
    # avertissement de calcul.
    t = np.where(valid, t, 0.0)
    lut = colormap(view.ramp, bands=view.bands)
    idx = np.where(valid, (t * 255).astype(np.int32), 0)
    rgba[..., :3] = lut[np.clip(idx, 0, 255)]
    rgba[..., 3] = np.where(valid, 255, 0)
    if view.nodata_color is not None:
        # `nan` et `inf` ne disent pas la meme chose. `nan` est hors domaine —
        # le solide — et reste transparent. `inf` est **dans** le domaine mais
        # jamais atteint : cul-de-sac, porosite fermee. Le peindre est tout
        # l'interet, car peint comme le solide il disparait, alors que c'est
        # justement ce qu'on cherche sur une carte de temps de parcours.
        paint = ~valid & ~np.isnan(a)
        if paint.any():
            rgba[..., :3][paint] = np.array(_hex_to_rgb(view.nodata_color), dtype=np.uint8)
            rgba[..., 3][paint] = 255
    return rgba


def _over(base: np.ndarray, top: np.ndarray, alpha: float) -> np.ndarray:
    """Composition « source over », en entiers, sans passer par du flottant."""
    a = (top[..., 3:4].astype(np.uint16) * int(np.clip(alpha, 0.0, 1.0) * 255)) // 255
    out = base.copy()
    out[..., :3] = (
        top[..., :3].astype(np.uint16) * a + base[..., :3].astype(np.uint16) * (255 - a)
    ) // 255
    out[..., 3] = np.maximum(base[..., 3:4], a)[..., 0]
    return out


def render_slice(
    views: list[LayerView],
    *,
    axis: int = 0,
    index: int = 0,
    max_size: int = 2048,
    background: str = "#f5f4f1",
) -> np.ndarray:
    """Compose les calques visibles en une image RGBA.

    Le premier calque visible sert de fond ; les suivants sont superposes avec
    leur propre transparence, dans l'ordre donne.
    """
    visible = [v for v in views if v.visible]
    if not visible:
        raise ValueError("aucun calque visible")
    shape = visible[0].data.shape
    n = shape[axis]
    step = max(1, int(np.ceil(max(s for i, s in enumerate(shape) if i != axis) / max_size)))
    index = int(np.clip(index, 0, n - 1))

    out = np.zeros((*_take(visible[0].data, axis, index, step).shape, 4), dtype=np.uint8)
    out[..., :3] = np.array(_hex_to_rgb(background), dtype=np.uint8)
    out[..., 3] = 255
    for view in visible:
        if view.data.shape != shape:
            raise ValueError(f"calques de formes differentes : {view.data.shape} contre {shape}")
        rgba = _to_rgba(_take(view.data, axis, index, step), view)
        out = _over(out, rgba, view.alpha)
    return out


def png_bytes(rgba: np.ndarray) -> bytes:
    """Encode en PNG. Pillow si disponible, sinon un encodeur minimal."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - chemin de secours
        return _png_fallback(rgba)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _png_fallback(rgba: np.ndarray) -> bytes:  # pragma: no cover - chemin de secours
    """Encodeur PNG minimal, pour ne pas dependre de Pillow en dernier recours."""
    import struct
    import zlib

    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
