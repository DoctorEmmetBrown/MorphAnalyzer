"""Fabrique `docs/images/rampes-tortuosite-fire.png`.

Une rampe de couleur ne se juge pas sur une bande : elle se juge sur une vraie
carte. Cette figure montre les deux rampes perceptuelles sur les champs pour
lesquels elles sont faites — le temps de parcours de la tortuosite et la carte
d'ouverture — en escalier et en continu. La mousse porte six cellules fermees,
pour que le magenta du hors-domaine ait quelque chose a signaler.

    python tools/build_ramp_figure.py
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import morphanalyzer as ma
import morphanalyzer.webapp.render as R

SHAPE = 160
ZOOM = 3
DEST = pathlib.Path(__file__).resolve().parent.parent / "docs/images/rampes-tortuosite-fire.png"
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


def closed_cells(fluid: np.ndarray, n: int = 6, radius: int = 7, seed: int = 4) -> np.ndarray:
    """Scelle `n` boules de fluide derriere une coque solide.

    Une mousse reelle en a — ce sont ses cellules fermees. Le front ne les
    atteint jamais : leur temps de parcours vaut `+inf`, pas `nan`.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[:SHAPE, :SHAPE, :SHAPE]
    for _ in range(n):
        c = rng.integers(30, SHAPE - 30, 3)
        d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        fluid[d2 <= (radius + 3) ** 2] = False
        fluid[d2 <= radius * radius] = True
    return fluid


def main() -> None:
    warnings.simplefilter("ignore")
    foam = ma.phantoms.voronoi_foam(
        shape=(SHAPE,) * 3, n_cells=60, strut=6.0, min_seed_gap=16.0, seed=5
    )
    fluid = closed_cells(np.asarray(foam.fluid))

    src = np.zeros_like(fluid)
    src[0] = fluid[0]
    temps = np.asarray(ma.distance.travel_time(fluid, src))
    reached = np.isfinite(temps)
    lost = int((fluid & ~reached).sum())
    print(
        f"temps max {temps[reached].max():.1f} | fluide {fluid.sum()} | "
        f"inatteignable {lost} ({100 * lost / fluid.sum():.1f} % du fluide)"
    )
    ouverture = np.where(
        fluid, np.asarray(ma.granulometry.aperture_map(fluid, n_radii=24)), np.nan
    )
    t_max = float(temps[reached].max())

    def panel(view: R.LayerView, index: int = SHAPE // 2, axis: int = 2) -> Image.Image:
        im = Image.fromarray(
            R.render_slice([view], axis=axis, index=index, background="#f5f4f1"), "RGBA"
        )
        return im.resize((im.width * ZOOM, im.height * ZOOM), Image.NEAREST)

    panels = [
        (
            "tortuosité — 16 paliers : les isochrones",
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
            "tortuosité — la même, rampe continue",
            R.LayerView(
                temps,
                kind="scalar",
                ramp="tortuosite",
                vmin=0,
                vmax=t_max,
                nodata_color=R.NODATA_COLOR,
            ),
        ),
        ("fire — carte d'ouverture, continue", R.LayerView(ouverture, kind="scalar", ramp="fire")),
        ("fire — 16 paliers", R.LayerView(ouverture, kind="scalar", ramp="fire", bands=16)),
    ]
    imgs = [(label, panel(view)) for label, view in panels]

    big, small = font(15), font(12)
    w, h = imgs[0][1].size
    pad, lab_h, strip_h = 16, 22, 30
    width = 2 * w + 3 * pad
    height = 2 * pad + 2 * (strip_h + lab_h + 16) + 2 * (h + lab_h + pad)
    out = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(out)

    y = pad
    for name, top in (("tortuosite", t_max), ("fire", float(np.nanmax(ouverture)))):
        lut = R.colormap(name, n=width - 2 * pad, bands=16)
        strip = Image.fromarray(np.repeat(lut[None], strip_h, 0).astype(np.uint8), "RGB")
        draw.text((pad, y), f"rampe « {name} » — 16 paliers", fill="#222222", font=big)
        out.paste(strip, (pad, y + lab_h))
        band_w = (width - 2 * pad) / 16
        for i in range(17):
            x = pad + i * band_w
            if i:
                draw.line([(x, y + lab_h), (x, y + lab_h + strip_h)], fill="#ffffff")
            if i % 4 == 0:
                draw.text(
                    (x - 8 if i else x, y + lab_h + strip_h + 2),
                    f"{top * i / 16:.0f}",
                    fill="#555555",
                    font=small,
                )
        y += lab_h + strip_h + 16 + pad // 2
    y += pad // 2

    for k, (label, im) in enumerate(imgs):
        x = pad + (k % 2) * (w + pad)
        yy = y + (k // 2) * (h + lab_h + pad)
        draw.text((x, yy), label, fill="#222222", font=big)
        out.paste(im, (x, yy + lab_h))

    DEST.parent.mkdir(parents=True, exist_ok=True)
    out.save(DEST, optimize=True)
    print("ecrit", DEST, out.size)


if __name__ == "__main__":
    main()
