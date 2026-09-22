"""Fabrique `docs/images/poiseuille.png`.

Ce que la chaine de Poiseuille rend, et pourquoi elle ne se resume pas a un
nombre : le champ de vitesse parabolique, les isochrones du front qu'il induit,
et les deux familles de chemins — ceux que choisit ce champ, et les geodesiques
entre les memes points d'arrivee. Les chemins sont projetes en intensite
maximale le long de x, parce qu'ils sortent du plan de coupe.

    python tools/build_poiseuille_figure.py
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import morphanalyzer as ma
import morphanalyzer.webapp.render as R

SHAPE = 128
ZOOM = 3
DEST = pathlib.Path(__file__).resolve().parent.parent / "docs/images/poiseuille.png"
FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Medium.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def font(size: int):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def zoom(im: Image.Image) -> Image.Image:
    return im.resize((im.width * ZOOM, im.height * ZOOM), Image.NEAREST)


def main() -> None:
    warnings.simplefilter("ignore")
    foam = ma.phantoms.voronoi_foam(
        shape=(SHAPE,) * 3, n_cells=64, strut=6.0, min_seed_gap=13.0, seed=3
    )
    fluid = np.asarray(foam.fluid)
    res = ma.tortuosity.poiseuille_tortuosity(fluid, 0, n_paths=16)
    par = res.paths.groupby("metric")[["length", "mean_wall_distance"]].mean()
    print(
        f"tau Poiseuille {res.tortuosity:.3f} +- {res.std:.3f} | "
        f"tau geometrique {res.geometric_tortuosity:.3f}\n"
        f"longueur {par.loc['poiseuille', 'length']:.1f} contre "
        f"{par.loc['geometric', 'length']:.1f} voxels | "
        f"paroi {res.mean_wall_distance_poiseuille:.2f} contre "
        f"{res.mean_wall_distance_geometric:.2f}"
    )

    mid = SHAPE // 2
    temps = np.where(fluid, res.travel_time, np.nan)
    t_max = float(np.nanmax(temps[np.isfinite(temps)]))
    vues = [
        (
            "temps de Poiseuille — 16 paliers",
            R.LayerView(
                temps,
                kind="scalar",
                ramp="tortuosite",
                bands=16,
                vmin=0,
                vmax=t_max,
                nodata_color=R.NODATA_COLOR,
            ),
        ),
        (
            "champ de vitesse — 1 au centre, 0 à la paroi",
            R.LayerView(np.where(fluid, res.speed, np.nan), kind="scalar", ramp="fire"),
        ),
    ]
    panneaux = [
        (label, zoom(Image.fromarray(R.render_slice([v], axis=2, index=mid), "RGBA")))
        for label, v in vues
    ]

    # Les chemins sortent du plan : projection en intensite maximale le long de x.
    solide = (~fluid).any(axis=2)
    fond = np.where(solide, np.nan, 0.0).astype(np.float32)[None]
    chemins = Image.fromarray(
        R.render_slice(
            [
                R.LayerView(fond, kind="scalar", ramp="grey", vmin=0, vmax=1),
                R.LayerView(
                    (res.path_mask > 0).any(axis=2)[None], kind="binary", color="#2a78d6"
                ),
                R.LayerView(
                    (res.geometric_path_mask > 0).any(axis=2)[None],
                    kind="binary",
                    color="#eb6834",
                    alpha=0.85,
                ),
            ],
            axis=0,
            index=0,
        ),
        "RGBA",
    )
    panneaux.append(
        (
            "chemins projetés — Poiseuille / géodésique",
            zoom(chemins),
        )
    )

    big, small = font(15), font(13)
    w, h = panneaux[0][1].size
    pad, lab_h = 16, 22
    out = Image.new("RGB", (3 * w + 4 * pad, h + lab_h + 2 * pad + 22), "#ffffff")
    draw = ImageDraw.Draw(out)
    for k, (label, im) in enumerate(panneaux):
        x = pad + k * (w + pad)
        draw.text((x, pad), label, fill="#222222", font=big)
        out.paste(im, (x, pad + lab_h))
    draw.text(
        (pad, pad + lab_h + h + 6),
        f"τ Poiseuille {res.tortuosity:.3f} ± {res.std:.3f}   —   "
        f"τ géométrique {res.geometric_tortuosity:.3f}   —   "
        f"distance à la paroi {res.mean_wall_distance_poiseuille:.2f} contre "
        f"{res.mean_wall_distance_geometric:.2f} voxels",
        fill="#444444",
        font=small,
    )
    DEST.parent.mkdir(parents=True, exist_ok=True)
    out.save(DEST, optimize=True)
    print("ecrit", DEST, out.size)


if __name__ == "__main__":
    main()
