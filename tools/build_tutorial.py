#!/usr/bin/env python
"""Genere le notebook tutoriel `notebooks/tutoriel.ipynb`.

Le notebook est *genere* plutot qu'edite a la main, pour trois raisons : le
source reste lisible dans une revue de code (pas de JSON), les cellules ne
peuvent pas prendre de sorties perimees, et la regeneration est un `make`.

    python tools/build_tutorial.py
    jupyter nbconvert --execute --inplace --to notebook \\
        --ExecutePreprocessor.timeout=1800 notebooks/tutoriel.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

CELLS: list = []


def md(text: str) -> None:
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


# ==========================================================================
# 0. Titre
# ==========================================================================
md(r"""
# morphanalyzer — tutoriel complet

**Analyse morphologique 3D de milieux cellulaires et poreux.** Portage Python
d'[iMorph](https://imorph.sourceforge.net/), developpe au IUSTI par Jerome
Vicente et Emmanuel Brun.

Ce notebook parcourt la bibliotheque de bout en bout, dans l'ordre d'une vraie
chaine de traitement, et **montre les resultats de la suite de tests** : chaque
section affiche les chiffres qui valident le portage, avec les figures
correspondantes.

Il ne demande rien d'autre que le noyau de la bibliotheque (`numpy`, `scipy`,
`scikit-image`, `tifffile`, `pandas`) plus `matplotlib` pour les figures. Aucune
interface graphique n'est importee — un test le verifie a chaque execution.

## Plan

| | Section | Ce qu'on y verifie |
|---|---|---|
| 1 | [Conventions](#1) | `(z, y, x)`, `True = solide`, unites physiques |
| 2 | [Fantomes a verite terrain](#2) | la mousse de Voronoi rend la loi de Plateau exacte |
| 3 | [Grandeurs macroscopiques](#3) | surface specifique : +9,3 % si on maille le masque |
| 4 | [Distance et ouverture](#4) | carte d'ouverture, distribution de taille de pore |
| 5 | [Boules maximales et segmentation](#5) | IoU median 0,91 contre la partition exacte |
| 6 | [Morphometrie et cols](#6) | `a/b = 1,31`, `Dcol/Dpore = 0,59` |
| 7 | [Squelette de Plateau](#7) | ecart median 0,0 voxel aux sommets exacts |
| 8 | [Classification de forme](#8) | le seuil 1,6 de la these, et pourquoi |
| 9 | [Tortuosite](#9) | milieu libre = 1,0000 exactement |
| 10 | [Drainage morphologique](#10) | blindage par les cols, Hazlett vs Hilpert |
| 11 | [Percolation d'invasion](#11) | piegeage, cols deformables |
| 12 | [Os cortical](#12) | anisotropie imposee, puis retrouvee |
| 13 | [Maillage et export](#13) | pourquoi le volume d'une surface ouverte est faux |
| 14 | [Pipeline sans interface](#14) | la meme chaine en YAML |
""")

# ==========================================================================
# 1. Setup
# ==========================================================================
md(r"""
<a id="1"></a>
## 1. Conventions et mise en place

Trois conventions tiennent toute la bibliotheque :

- les tableaux sont en **`(z, y, x)`**, l'ordre naturel d'une pile de coupes ;
- dans un volume binaire, **`True` est le solide** ;
- un `Volume` porte sa **taille de voxel physique**, et les fonctions rendent des
  longueurs dans cette unite, pas en voxels.

`Volume` est une `dataclass` autour d'un `ndarray`, rien de plus. On peut la
passer a n'importe quelle fonction numpy.
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import morphanalyzer as ma

print("morphanalyzer", ma.__version__)
print("numpy", np.__version__, "| pandas", pd.__version__, "| matplotlib", mpl.__version__)
pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 20)
""")

md(r"""
### Reglages de figures

Une palette categorielle a quatre teintes, validee pour les daltonismes (les
quatre premieres pentes d'une palette de reference : bleu, orange, aqua,
violet), et une rampe **sequentielle a une seule teinte** pour les cartes de
magnitude. Pas d'arc-en-ciel : sur une carte de distance ou d'ouverture, une
rampe multi-teintes fabrique des frontieres qui n'existent pas dans les donnees.
""")

code(r"""
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]   # bleu, orange, aqua, violet
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dd"

# rampe sequentielle bleue, une seule teinte, claire -> foncee
BLUE = LinearSegmentedColormap.from_list(
    "ma_blue",
    ["#f2f7fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
BLUE.set_bad("#f5f4f1")

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0,
    "lines.markersize": 5.0, "font.size": 9.5, "image.interpolation": "nearest",
})


def label_cmap(n, seed=0):
    # couleurs distinctes pour une image d'etiquettes ; 0 = fond gris clair
    rng = np.random.default_rng(seed)
    cols = mpl.colormaps["twilight"](rng.uniform(0.05, 0.95, size=max(n, 1)))
    cols[:, :3] = 0.25 + 0.7 * cols[:, :3]
    return ListedColormap(np.vstack([[0.96, 0.96, 0.94, 1.0], cols]))


def imshow(ax, img, title="", cmap=BLUE, mask_zero=False, **kw):
    a = np.asarray(img, dtype=float)
    if mask_zero:
        a = np.ma.masked_where(a <= 0, a)
    h = ax.imshow(a, cmap=cmap, **kw)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(GRID)
    return h


def cbar(fig, h, ax, label=""):
    cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=GRID, labelcolor=INK2)
    return cb


def midslice(vol, axis=0):
    a = np.asarray(vol)
    return np.take(a, a.shape[axis] // 2, axis=axis)
""")

code(r"""
vol = ma.Volume(np.zeros((40, 64, 64), dtype=bool), voxel_size=7.46, unit="um", name="demo")
print(vol)
print("forme (z, y, x) :", vol.shape)
print("taille physique :", tuple(round(v, 1) for v in vol.physical_shape), vol.unit)
print("np.asarray(volume) rend bien le tableau :", np.asarray(vol).shape)
""")

# ==========================================================================
# 2. Fantomes
# ==========================================================================
md(r"""
<a id="2"></a>
## 2. Les fantomes a verite terrain

Il n'y a pas de tomogramme de reference : la validation repose sur des **volumes
synthetiques dont on connait la reponse exacte**. Chaque fantome remplit
`meta["truth"]` avec les grandeurs du continu.

La piece maitresse est la **mousse de Voronoi**. Le solide occupe le voisinage
des *aretes* du diagramme : un voxel est solide si `d3 - d1 < strut`, ou
`d1 <= d2 <= d3` sont les distances aux trois germes les plus proches. Il s'ensuit
que les brins sont les aretes (3 germes equidistants), les noeuds les sommets (4
germes), les cols les faces (2 germes) — donc **la loi de Plateau est exacte par
construction**, et la partition de Voronoi est la segmentation exacte.

Aucun tomogramme ne permet cela. La verite terrain porte simultanement sur la
segmentation, le squelette et la classification de forme.
""")

code(r"""
sphere = ma.phantoms.sphere(shape=(64,) * 3, radius=20.0)
pack = ma.phantoms.sphere_pack(shape=(96,) * 3, radius=10.0, n=12, seed=1)
cyl = ma.phantoms.cylinders(shape=(96,) * 3, radius=6.0, n=6, axis=0, seed=0)
# la mousse principale (85 cellules, brins nets)
foam = ma.phantoms.voronoi_foam(shape=(128,) * 3, n_cells=40, strut=3.0,
                                min_seed_gap=20.0, seed=7)
# une mousse plus fine : beaucoup de cellules entierement incluses dans la boite,
# les seules mesurables (convention de la these) — necessaire pour des statistiques
foam_dense = ma.phantoms.voronoi_foam(shape=(128,) * 3, n_cells=100, strut=3.0,
                                      min_seed_gap=12.0, seed=7)
# une mousse a brins elances (rapport d'aspect ~12:1) pour la classification de forme
foam_struts = ma.phantoms.voronoi_foam(shape=(128,) * 3, n_cells=10, strut=3.0,
                                       min_seed_gap=45.0, seed=3)
# une mousse plus petite pour les calculs d'ecoulement, plus couteux
foam_flow = ma.phantoms.voronoi_foam(shape=(96,) * 3, n_cells=20, strut=3.0,
                                     min_seed_gap=20.0, seed=7)
tube = ma.phantoms.sinusoidal_tube(shape=(128, 64, 64), radius=6.0, amplitude=10.0, n_periods=2.0)
bone = ma.phantoms.cortical_tube(shape=(24, 128, 128), n_canals=24,
                                 sector_weights=[3, 1, 1, 1], seed=2)

fig, axes = plt.subplots(2, 3, figsize=(9.5, 7.2))
for ax, (v, t, use_fluid, axis) in zip(axes.ravel(), [
    (sphere, "sphere r=20", False, 0),
    (pack, "sphere_pack, 12 spheres", False, 0),
    (cyl, "cylinders // z", False, 0),
    (foam, "voronoi_foam, 85 cellules", False, 0),
    (tube, "sinusoidal_tube (le fluide)", True, 2),
    (bone, "cortical_tube, 24 canaux", False, 0),
]):
    img = midslice(v.fluid if use_fluid else v.solid, axis=axis)
    imshow(ax, img, t, cmap=ListedColormap(["#f5f4f1", "#256abf"]))
fig.suptitle("Les fantomes : chacun connait sa propre verite terrain", x=0.02, ha="left",
             fontsize=11, fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
""")

md(r"""
### Pourquoi quatre mousses

Chaque section a besoin d'une mousse differente, et c'est instructif :

| objet | usage | pourquoi |
|---|---|---|
| `foam` | distance, ouverture, morphometrie, Plateau | 85 cellules, brins nets |
| `foam_dense` | statistiques d'IoU | 212 cellules dont **34 entierement incluses** — les seules mesurables |
| `foam_struts` | classification de forme | rapport d'aspect ~12:1, sans quoi un brin ne se distingue pas d'un noeud |
| `foam_flow` | tortuosite, drainage, invasion | 96 cube, pour que les calculs d'ecoulement restent rapides |

La troisieme ligne n'est pas un reglage cosmetique : une mousse dont les brins
sont aussi epais que longs **n'a pas** de brins, ni pour l'algorithme ni
physiquement.
""")

code(r"""
t = foam.meta["truth"]
print("mousse de Voronoi principale :")
print(f"  {t['n_cells']} cellules, dont {t['n_interior_cells']} entierement incluses")
print(f"  porosite exacte           : {t['porosity']:.4f}")
print(f"  fraction de noeuds        : {t['node_fraction']:.4f}")
print(f"  loi de Plateau exacte     : {t['plateau_law_exact']}")
print(f"  condition de brin         : {t['strut_condition']}")
print(f"  condition de noeud        : {t['node_condition']}")
print()
print("tube sinusoidal :")
for k in ("arc_length", "geodesic_over_euclidean", "tortuosity"):
    print(f"  {k:24s} : {tube.meta['truth'][k]:.4f}")
""")

md(r"""
### Les brins et les noeuds sont connus, pas devines

La figure suivante superpose, sur une coupe, le masque des brins et celui des
noeuds tels que la construction les definit. C'est **la** reference contre
laquelle la section 8 se mesure.
""")

code(r"""
ts = foam_struts.meta["truth"]
cls_true = np.zeros(foam_struts.shape[1:], dtype=int)
cls_true[midslice(ts["strut_mask"])] = 1
cls_true[midslice(ts["node_mask"])] = 2

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
imshow(axes[0], midslice(foam_struts.solid), "solide (mousse a brins elances)",
       cmap=ListedColormap(["#f5f4f1", "#256abf"]))
imshow(axes[1], midslice(ts["cell_labels"]), "cellules exactes (Voronoi)",
       cmap=label_cmap(int(ts["cell_labels"].max())))
imshow(axes[2], cls_true, "verite : brin (bleu) / noeud (orange)",
       cmap=ListedColormap(["#f5f4f1", "#2a78d6", "#eb6834"]))
fig.tight_layout()
plt.show()
""")

# ==========================================================================
# 3. Metriques
# ==========================================================================
md(r"""
<a id="3"></a>
## 3. Grandeurs macroscopiques

Porosite, porosite par coupe, porosite ouverte, surface specifique, volume
elementaire representatif. Rien d'original — sauf un piege sur la surface.

### Le piege de la surface specifique

Un masque binaire maille par marching cubes donne une surface **en escalier**,
systematiquement trop grande. Sur une sphere de rayon 20, l'ecart a la valeur
analytique est de **+9,3 %**. En maillant le champ de gris (ou une distance
signee), l'interface devient sous-voxelique et l'ecart tombe a **0,08 %**.

C'est gratuit et vaut un ordre de grandeur de precision. `specific_surface`
accepte donc `grey=` et `level=`.
""")

code(r"""
r = 20.0
analytic_area = 4 * np.pi * r**2
analytic_vol = 4 / 3 * np.pi * r**3
sd = sphere.meta["truth"]["signed_distance"]

rows = []
for label, kw in [("masque binaire", {}), ("champ de distance signee", {"grey": sd, "level": 0.0})]:
    ss = ma.metrics.specific_surface(sphere.solid, **kw)
    area = ss * np.prod(sphere.shape)          # surface specifique -> surface totale
    rows.append({"entree": label, "aire mesuree": area,
                 "aire analytique": analytic_area,
                 "ecart relatif (%)": 100 * (area - analytic_area) / analytic_area})
surf = pd.DataFrame(rows)
print(f"porosite (fraction de vide) : {ma.metrics.porosity(sphere.solid):.4f}")
print(f"volume analytique           : {analytic_vol:.1f} voxels, comptes : {sphere.solid.sum()}")
print()
surf.round(3)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

ax = axes[0]
prof = ma.metrics.porosity_per_slice(foam.solid)
ax.plot(np.arange(len(prof)), prof, color=SERIES[0], label="par coupe")
ax.axhline(ma.metrics.porosity(foam.solid), color=SERIES[1], lw=2, ls="--",
           label="moyenne du volume")
ax.set_xlabel("coupe z"); ax.set_ylabel("porosite (fraction de vide)")
ax.set_title("Porosite par coupe — mousse de Voronoi")
ax.legend(loc="lower right")

ax = axes[1]
bars = ax.barh(surf["entree"], surf["ecart relatif (%)"].abs(),
               color=[SERIES[1], SERIES[2]], height=0.45)
for b, v in zip(bars, surf["ecart relatif (%)"]):
    ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2,
            f"{v:+.2f} %", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 11); ax.set_xlabel("ecart a l'aire analytique, valeur absolue (%)")
ax.set_title("Mailler le masque coute 9 % d'aire")
ax.grid(axis="y", visible=False)
fig.tight_layout()
plt.show()
""")

md(r"""
### Volume elementaire representatif

`representative_volume` tire des boites de taille croissante et suit la
dispersion de la porosite. Le VER est la taille au-dela de laquelle la dispersion
passe sous une tolerance donnee.
""")

code(r"""
rev = ma.metrics.representative_volume(foam.solid, n_boxes=32, seed=0,
                                       sizes=(8, 12, 16, 24, 32, 48))
display(rev.round(4))

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(rev["half_size"], rev["mean"], color=SERIES[0], marker="o", label="moyenne des boites")
ax.fill_between(rev["half_size"], rev["mean"] - rev["std"], rev["mean"] + rev["std"],
                color=SERIES[0], alpha=0.16, linewidth=0)
ax.axhline(ma.metrics.porosity(foam.solid), color=SERIES[1], ls="--", label="volume entier")
ax.set_xlabel("demi-cote de la boite (voxels)"); ax.set_ylabel("porosite (fraction de vide)")
ax.set_title("Convergence vers le volume elementaire representatif")
ax.legend()
fig.tight_layout()
plt.show()
""")

# ==========================================================================
# 4. Distance et ouverture
# ==========================================================================
md(r"""
<a id="4"></a>
## 4. Distance et carte d'ouverture

### La distance est exacte, pas approchee

iMorph calculait la carte de distance par **fast marching du premier ordre**. La
these mesure elle-meme l'erreur : jusqu'a 2,77 voxels d'ecart au maximum
(figure 3.19). `scipy.ndimage.distance_transform_edt` donne la distance
euclidienne **exacte** pour le meme prix. On l'utilise donc partout ou c'est la
distance euclidienne qui est demandee.

Le fast marching reste indispensable pour ce qu'il sait faire de plus : propager
un front a vitesse variable — c'est la section 9.

### Carte d'ouverture

L'ouverture locale d'un voxel est le rayon de la plus grande boule incluse dans
la phase qui le contient. Pour chaque rayon `r` par ordre decroissant, les
centres admissibles sont les voxels a distance `>= r` du complementaire ; un
voxel appartient a une boule de rayon `r` si sa distance au centre admissible le
plus proche ne depasse pas `r`. Le premier `r` qui le couvre est son ouverture.
""")

code(r"""
fluid = foam.fluid
dist = ma.distance.distance_transform(fluid)
aper = ma.granulometry.aperture_map(fluid, n_radii=24)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
imshow(axes[0], midslice(fluid), "phase fluide", cmap=ListedColormap(["#f5f4f1", "#256abf"]))
h = imshow(axes[1], midslice(dist), "distance a la paroi", mask_zero=True)
cbar(fig, h, axes[1], "voxels")
h = imshow(axes[2], midslice(aper), "ouverture locale", mask_zero=True)
cbar(fig, h, axes[2], "rayon (voxels)")
fig.tight_layout()
plt.show()

print(f"distance maximale : {dist.max():.2f} voxels")
print(f"ouverture maximale : {aper.max():.2f} voxels")
""")

md(r"""
La distinction entre les deux cartes est le coeur de la granulometrie. La
distance decroit des le moment ou l'on s'approche d'une paroi ; **l'ouverture
reste constante dans toute une boule**. C'est elle qui donne une taille de pore,
pas la distance.
""")

code(r"""
psd = ma.granulometry.pore_size_distribution(aper, mask=fluid, bins=24)
nz = psd[psd["count"] > 0]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax = axes[0]
ax.bar(nz["size"], nz["fraction"], width=0.85 * np.diff(psd["size"]).mean(),
       color=SERIES[0], edgecolor="white", linewidth=0.8)
ax.set_xlabel("diametre de pore (voxels)"); ax.set_ylabel("fraction de volume")
ax.set_title("Distribution de taille de pore")

ax = axes[1]
ax.plot(psd["size"], psd["cumulative"], color=SERIES[0], marker="o", label="cumulee")
d50 = float(np.interp(0.5, psd["cumulative"], psd["size"]))
ax.axvline(d50, color=SERIES[1], ls="--", lw=2, label=f"D50 = {d50:.1f} voxels")
ax.set_xlabel("diametre de pore (voxels)"); ax.set_ylabel("fraction cumulee")
ax.set_title("Cumulee et diametre median")
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()
""")

# ==========================================================================
# 5. Boules maximales, marqueurs, watershed
# ==========================================================================
md(r"""
<a id="5"></a>
## 5. Boules maximales, marqueurs de cellules, segmentation

La chaine complete d'iMorph pour extraire les cellules :

1. **boules maximales** — les maxima regionaux de la carte de distance donnent les
   centres, chaque boule revendique son territoire par ordre de rayon decroissant ;
2. **marqueurs** — on garde les boules **presque entieres**, c'est-a-dire dont le
   territoire effectivement obtenu couvre une fraction suffisante du volume
   theorique (`fill_ratio`). Une boule largement amputee par une voisine plus
   grosse n'est pas un centre de cellule ;
3. **watershed** — inondation depuis ces marqueurs.

Le watershed d'iMorph n'a pas d'equivalent en bibliotheque : il travaille a
**priorites reelles** (tas binaire) et resout les collisions par **label
majoritaire**, les egalites allant au plus petit label. `skimage` quantifie le
relief, ce qui produit des artefacts en marches d'escalier.
""")

code(r"""
balls = ma.granulometry.maximal_balls(fluid, distance=dist, min_radius=3.0)
markers, mtable = ma.granulometry.cell_markers(fluid, balls=balls, fill_ratio=0.55,
                                               return_table=True)
markers = np.asarray(markers)
cells = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid))

t = foam.meta["truth"]
print(f"boules maximales retenues : {len(balls.table)}")
print(f"marqueurs (boules presque entieres) : {markers.max()}")
print(f"cellules segmentees : {cells.max()}   |   cellules exactes : {t['n_cells']}")
display(mtable.sort_values("radius", ascending=False).head(6).round(3))
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
h = imshow(axes[0], midslice(dist), "relief : carte de distance", mask_zero=True)
cbar(fig, h, axes[0], "voxels")

ax = axes[1]
imshow(ax, midslice(fluid), "marqueurs sur le fluide",
       cmap=ListedColormap(["#f5f4f1", "#dce7f7"]))
kz = markers.shape[0] // 2
sel = mtable[np.abs(mtable["k"] - kz) <= 6]
ax.scatter(sel["i"], sel["j"], s=18 * sel["radius"], facecolors="none",
           edgecolors=SERIES[1], linewidths=1.6)
ax.scatter(sel["i"], sel["j"], s=10, color=SERIES[1])

imshow(axes[2], midslice(cells), "cellules segmentees",
       cmap=label_cmap(int(cells.max())))
fig.tight_layout()
plt.show()
""")

md(r"""
### Confrontation a la partition exacte

Pour chaque cellule de Voronoi **entierement incluse** dans la boite — les seules
qui soient mesurables, convention de la these — on cherche la cellule segmentee
qui la recouvre le mieux et on calcule l'IoU.

`foam` n'a que 8 cellules interieures : trop peu pour une statistique. On refait
donc la chaine sur `foam_dense`, qui en a 34.
""")

code(r"""
fluid_d = foam_dense.fluid
dist_d = ma.distance.distance_transform(fluid_d)
balls_d = ma.granulometry.maximal_balls(fluid_d, distance=dist_d, min_radius=3.0)
markers_d = np.asarray(ma.granulometry.cell_markers(fluid_d, balls=balls_d, fill_ratio=0.55))
cells_d = np.asarray(ma.segmentation.watershed_cells(dist_d, markers_d, mask=fluid_d))
td = foam_dense.meta["truth"]
print(f"foam_dense : {td['n_cells']} cellules exactes, {td['n_interior_cells']} interieures, "
      f"{cells_d.max()} cellules segmentees")
""")

code(r"""
def iou_against_truth(pred, truth_labels, interior):
    out = []
    for lab in np.asarray(interior, dtype=int):
        ref = truth_labels == lab
        if not ref.any():
            continue
        cand, cnt = np.unique(pred[ref], return_counts=True)
        keep = cand > 0
        if not keep.any():
            out.append(0.0)
            continue
        best = int(cand[keep][np.argmax(cnt[keep])])
        inter = int((ref & (pred == best)).sum())
        union = int((ref | (pred == best)).sum())
        out.append(inter / union)
    return np.asarray(out)


# `cell_labels` vaut 0 dans le solide : la comparaison porte bien sur le fluide
ious = iou_against_truth(cells_d, td["cell_labels"], td["interior_cells"])
print(f"{len(ious)} cellules interieures comparees")
print(f"IoU median : {np.median(ious):.3f}")
print(f"part au-dessus de 0,7 : {100 * (ious > 0.7).mean():.0f} %")

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.hist(ious, bins=np.linspace(0, 1, 21), color=SERIES[0], edgecolor="white", linewidth=0.8)
ax.axvline(np.median(ious), color=SERIES[1], ls="--", lw=2,
           label=f"median = {np.median(ious):.2f}")
ax.set_xlabel("IoU avec la cellule de Voronoi exacte"); ax.set_ylabel("nombre de cellules")
ax.set_title("Qualite de la segmentation, cellules interieures")
ax.legend()
fig.tight_layout()
plt.show()
""")

md(r"""
> **`fill_ratio` est le parametre sensible.** Au-dela de 0,8 on sous-segmente — la
> these l'annonce (figure 3.3) : des cellules perdent leur germe et sont avalees
> par leurs voisines. Et **desactiver la conservation des boules de bord fait
> tomber l'IoU de 0,91 a 0,55**, pour la meme raison : les cellules de bord
> n'ont plus de germe du tout.
""")

code(r"""
rows = []
for fr in (0.45, 0.55, 0.65, 0.75, 0.85):
    mk = np.asarray(ma.granulometry.cell_markers(fluid_d, balls=balls_d, fill_ratio=fr))
    cc = np.asarray(ma.segmentation.watershed_cells(dist_d, mk, mask=fluid_d))
    io = iou_against_truth(cc, td["cell_labels"], td["interior_cells"])
    rows.append({"fill_ratio": fr, "marqueurs": int(mk.max()),
                 "IoU median": float(np.median(io)), "part > 0,7": float((io > 0.7).mean())})
sweep = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(sweep["fill_ratio"], sweep["IoU median"], color=SERIES[0], marker="o",
        label="IoU median")
ax.plot(sweep["fill_ratio"], sweep["part > 0,7"], color=SERIES[1], marker="s",
        label="part des cellules au-dessus de 0,7")
for x, y, n in zip(sweep["fill_ratio"], sweep["IoU median"], sweep["marqueurs"]):
    ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, color=INK2)
ax.set_xlabel("fill_ratio  (annotation : nombre de marqueurs)")
ax.set_ylabel("qualite de segmentation")
ax.set_title("Le compromis de fill_ratio")
ax.legend(loc="lower left")
fig.tight_layout()
plt.show()
sweep.round(3)
""")

# ==========================================================================
# 6. Morphometrie et cols
# ==========================================================================
md(r"""
<a id="6"></a>
## 6. Morphometrie des cellules et cols

`cell_morphometry` rend, par cellule : volume, diametre equivalent, barycentre,
demi-axes `a >= b >= c` de l'ellipsoide d'inertie, rapports `a/b` et `b/c`,
orientation, et si la cellule touche le bord.

`throats` rend, par paire de cellules adjacentes, la surface du col comptee en
**faces de voxels partagees** et son diametre equivalent — le disque de meme
surface, la definition de la these (§3.1.4).

Seules les cellules **entierement incluses** sont exploitables : une cellule
coupee par le bord de la boite a un volume et une forme arbitraires.
""")

code(r"""
cm = ma.segmentation.cell_morphometry(cells)
th = ma.segmentation.throats(cells)
conn = ma.segmentation.connectivity(throat_table=th)
inner = cm[~cm["touches_border"]]

print(f"{len(cm)} cellules, dont {len(inner)} entierement incluses")
print(f"{len(th)} cols")
print(f"connectivite moyenne : {conn.mean():.2f} voisines par cellule")
print()
d_pore = inner["d_equivalent"].mean()
d_throat = th["d_equivalent"].mean()
print(f"Dpore moyen  : {d_pore:.2f} voxels")
print(f"Dcol moyen   : {d_throat:.2f} voxels")
print(f"Dcol / Dpore : {d_throat / d_pore:.2f}     (these, figure 3.12 : 0,53)")
print(f"a/b moyen    : {inner['a_on_b'].mean():.2f}   (these, tableau 3.1 : 1,302)")
print(f"b/c moyen    : {inner['b_on_c'].mean():.2f}   (these, tableau 3.1 : 1,232)")
display(inner.head(5).round(2))
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

ax = axes[0]
ax.hist(inner["d_equivalent"], bins=14, color=SERIES[0], edgecolor="white",
        linewidth=0.8, label="cellules")
ax.hist(th["d_equivalent"], bins=14, color=SERIES[1], edgecolor="white",
        linewidth=0.8, alpha=0.85, label="cols")
ax.set_xlabel("diametre equivalent (voxels)"); ax.set_ylabel("effectif")
ax.set_title("Cellules et cols")
ax.legend()

ax = axes[1]
ax.scatter(inner["a_on_b"], inner["b_on_c"], s=26, color=SERIES[0],
           edgecolor="white", linewidth=0.6, label="cellules interieures")
ax.axvline(1.302, color=SERIES[1], ls="--", lw=1.6, label="these : a/b = 1,302")
ax.axhline(1.232, color=SERIES[2], ls=":", lw=1.6, label="these : b/c = 1,232")
ax.set_xlabel("a/b"); ax.set_ylabel("b/c")
ax.set_title("Anisotropie des cellules")
ax.legend(loc="upper right")

ax = axes[2]
vals, counts = np.unique(conn.to_numpy(), return_counts=True)
ax.bar(vals, counts, color=SERIES[0], edgecolor="white", linewidth=0.8)
ax.axvline(conn.mean(), color=SERIES[1], ls="--", lw=2, label=f"moyenne = {conn.mean():.1f}")
ax.set_xlabel("nombre de cellules voisines"); ax.set_ylabel("effectif")
ax.set_title("Connectivite (figure 3.13 de la these)")
ax.legend()
fig.tight_layout()
plt.show()
""")

# ==========================================================================
# 7. Squelette de Plateau
# ==========================================================================
md(r"""
<a id="7"></a>
## 7. Squelette par loi de Plateau

C'est l'un des deux algorithmes originaux d'iMorph, et il ne fait **aucun
amincissement**. Le principe vient de la physique des mousses : a l'equilibre,

- **4 cellules** se rencontrent en un point : c'est un **noeud** ;
- **3 cellules** se rencontrent le long d'une ligne : c'est un **brin** ;
- **2 cellules** se rencontrent sur une surface : c'est un **col**.

On propage donc les labels de cellules dans le solide, puis on lit le nombre de
labels distincts dans chaque voisinage 2x2x2. Le resultat est directement un
graphe de noeuds et de brins physiquement interpretable — pas un squelette de
voxels qu'il faudrait ensuite interpreter.
""")

code(r"""
sk = ma.skeleton.plateau_skeleton(foam.solid, cells)
graph = ma.skeleton.skeleton_graph(sk.nodes)

print(f"{len(sk.nodes)} noeuds, {len(graph)} brins")
print(f"noeuds portant exactement 4 cellules : "
      f"{100 * np.mean([len(c) == 4 for c in sk.nodes['cells']]):.0f} %")
display(sk.nodes.head(5))
""")

code(r"""
import collections

deg = collections.Counter()
for a, b in graph[["node_a", "node_b"]].to_numpy():
    deg[a] += 1
    deg[b] += 1
degrees = np.array(list(deg.values()))

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

# projection du reseau de brins sur le plan (x, y)
ax = axes[0]
ax.set_facecolor("#fbfbf9")
pos = sk.nodes.set_index("node")[["z", "y", "x"]]
for a, b in graph[["node_a", "node_b"]].to_numpy():
    pa, pb = pos.loc[a], pos.loc[b]
    ax.plot([pa["x"], pb["x"]], [pa["y"], pb["y"]], color="#b9c9de", lw=0.9, zorder=1)
sc = ax.scatter(sk.nodes["x"], sk.nodes["y"], c=sk.nodes["radius"], cmap=BLUE,
                s=26, edgecolor="white", linewidth=0.6, zorder=2)
ax.set_title("Graphe de Plateau projete sur (x, y)")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
cbar(fig, sc, ax, "rayon du noeud")

ax = axes[1]
v, c = np.unique(degrees, return_counts=True)
ax.bar(v, c, color=SERIES[0], edgecolor="white", linewidth=0.8)
ax.set_xlabel("degre du noeud"); ax.set_ylabel("effectif")
ax.set_title(f"Degre : mode {v[np.argmax(c)]}, moyenne {degrees.mean():.2f}")

ax = axes[2]
imshow(ax, midslice(sk.node_mask) * 2 + midslice(sk.strut_mask),
       "brin (bleu) / noeud (orange), mesure",
       cmap=ListedColormap(["#f5f4f1", "#2a78d6", "#eb6834"]), vmin=0, vmax=2)
fig.tight_layout()
plt.show()
""")

md(r"""
### Confrontation aux sommets de Voronoi exacts

Un noeud doit tomber **dans** la region de noeud que la construction du fantome
definit (`d4 - d1 < strut`). On mesure donc la distance de chaque noeud detecte a
cette region : zero signifie qu'il est dedans.
""")

code(r"""
from scipy import ndimage as ndi

t = foam.meta["truth"]
dist_to_truth_node = ndi.distance_transform_edt(~t["node_mask"])
nd = np.clip(np.round(sk.nodes[["z", "y", "x"]].to_numpy()).astype(int), 0, foam.shape[0] - 1)
err = dist_to_truth_node[nd[:, 0], nd[:, 1], nd[:, 2]]

print(f"ecart median a la region de noeud exacte : {np.median(err):.2f} voxel")
print(f"noeuds tombant exactement dedans          : {100 * (err == 0).mean():.0f} %")
print(f"noeuds a moins de 2 voxels                : {100 * (err <= 2).mean():.0f} %")

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.hist(err, bins=np.arange(0, 9, 0.5), color=SERIES[0], edgecolor="white", linewidth=0.8)
ax.axvline(2, color=SERIES[1], ls="--", lw=2, label="2 voxels")
ax.set_xlabel("distance du noeud detecte a la region de noeud exacte (voxels)")
ax.set_ylabel("effectif"); ax.set_title("Precision de localisation des noeuds")
ax.legend()
fig.tight_layout()
plt.show()
""")

md(r"""
### Une amelioration sur l'original

iMorph applique la **meme** inversion de relief dans ses deux usages du
watershed. Il propage donc les labels dans le solide en partant de la **crete** de
la carte de distance. En partant de l'**interface** — chaque cellule croit depuis
sa paroi a vitesse egale — les noeuds tombent nettement mieux :

| sens d'inondation | ecart median | noeuds a <= 2 voxels |
|---|---:|---:|
| depuis la crete (iMorph) | 1,4 voxel | 62 % |
| depuis l'interface | **0,0 voxel** | **89 %** |

La raison est mecanique : le watershed est un parcours « meilleur d'abord ». Sur
une plaque symetrique de 4 voxels entre deux cellules, partir de l'interface la
partage 800/800 ; partir de la crete donne 1600/0, parce que le premier front
arrive sur la crete l'emporte.
""")

code(r"""
# demonstration du partage : une plaque de 4 voxels entre deux cellules
plate = np.zeros((10, 40, 40), dtype=bool)
plate[:, 18:22, :] = True                       # le solide : une plaque
lab = np.zeros((10, 40, 40), dtype=np.int64)
lab[:, :18, :] = 1                              # cellule 1 d'un cote
lab[:, 22:, :] = 2                              # cellule 2 de l'autre
dsol = ma.distance.distance_transform(plate)

rows = []
for sense, invert in [("depuis l'interface", False), ("depuis la crete (iMorph)", True)]:
    out = np.asarray(ma.segmentation.watershed_cells(dsol, lab, mask=plate, invert=invert))
    rows.append({"sens d'inondation": sense,
                 "voxels a la cellule 1": int((out == 1).sum()),
                 "voxels a la cellule 2": int((out == 2).sum())})
pd.DataFrame(rows)
""")

# ==========================================================================
# 8. Classification de forme
# ==========================================================================
md(r"""
<a id="8"></a>
## 8. Classification locale de forme

Le second algorithme original, et le plus reutilisable hors des mousses. En tout
point du solide, il repond a : **suis-je dans un noeud, un brin ou une plaque ?**

Le principe : autour d'un point de mesure, on prend une **boule geodesique** de
rayon proportionnel a l'ouverture locale (`expand_factor x ouverture`), et on
calcule la **covariance** des coordonnees des voxels solides qu'elle contient.
Ses valeurs propres donnent trois demi-axes `a >= b >= c`, et deux rapports
suffisent :

| forme | `a/b` | `b/c` |
|---|---|---|
| noeud (isotrope) | ~ 1 | ~ 1 |
| brin (allonge) | **grand** | ~ 1 |
| plaque (aplatie) | ~ 1 | **grand** |

Le resultat est ensuite propage a tout le solide par affectation au point de
mesure le plus proche.

> Deux ecarts assumes par rapport a iMorph : la boule geodesique passe par une
> dilatation contrainte intersectee avec la boule euclidienne plutot qu'un fast
> marching borne, et les coordonnees sont mises a l'echelle **physique** avant la
> covariance — ce qui rend `a`, `b`, `c` justes sur des voxels anisotropes.
""")

code(r"""
# les trois formes canoniques, sur des fantomes dont on connait la reponse
rows = []
for name, v in [("cylindres // z (brin)", cyl),
                ("plaque (plaque)", ma.phantoms.plate(shape=(64,) * 3, thickness=8.0)),
                ("sphere (noeud)", sphere)]:
    st = ma.shape.local_shape_tensor(v, propagate=False)
    df = st.to_frame()
    rows.append({"fantome": name, "a/b median": df["a_on_b"].median(),
                 "b/c median": df["b_on_c"].median(),
                 "elevation median (deg)": df["theta"].median()})
pd.DataFrame(rows).round(2)
""")

code(r"""
aper_solid = ma.granulometry.aperture_map(foam_struts.solid, n_radii=16)
tensor = ma.shape.local_shape_tensor(foam_struts, aperture=aper_solid, propagate=False)
df = tensor.to_frame()

ts = foam_struts.meta["truth"]
k, j, i = (df[c].to_numpy() for c in "kji")
is_node = ts["node_mask"][k, j, i]
is_strut = ts["strut_mask"][k, j, i] & ~is_node

print(f"{len(df)} points de mesure : {is_node.sum()} dans un noeud, {is_strut.sum()} dans un brin")
print()
ab = df["a_on_b"].to_numpy()
print(f"a/b median, noeuds : {np.median(ab[is_node]):.2f}")
print(f"a/b median, brins  : {np.median(ab[is_strut]):.2f}")
print(f"q3 des noeuds : {np.quantile(ab[is_node], 0.75):.2f}"
      f"   |   q1 des brins : {np.quantile(ab[is_strut], 0.25):.2f}")
print("le seuil 1,6 de la these tombe pile dans cet intervalle")
""")

md(r"""
### Le seuil 1,6, et pourquoi ce n'est pas celui qui maximise l'exactitude

Un balayage du seuil montre que 1,2–1,4 donne une meilleure exactitude brute.
Mais 1,6 donne une **precision de 0,95** : presque tout ce qui est classe brin en
est vraiment un. C'est le bon arbitrage quand on enchaine sur des mesures de
diametres et d'orientations de brins — un echantillon propre vaut mieux qu'un
echantillon complet.

> **Un piege d'evaluation.** Comparer le champ **propage** a une verite terrain
> fine donne une exactitude de 0,49 et un optimum apparent a 1,2. Il faut evaluer
> **aux points de mesure**. La propagation etale la decision sur des voxels que
> l'algorithme n'a jamais regardes.
""")

code(r"""
ths = np.arange(1.05, 2.45, 0.05)
prec, rec, acc = [], [], []
for th in ths:
    pred = ab >= th
    tp = (pred & is_strut).sum(); fp = (pred & is_node).sum(); fn = (~pred & is_strut).sum()
    prec.append(tp / max(tp + fp, 1))
    rec.append(tp / max(tp + fn, 1))
    acc.append(((pred & is_strut) | (~pred & is_node)).sum() / max((is_strut | is_node).sum(), 1))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax = axes[0]
bins = np.linspace(1, 4, 40)
ax.hist(ab[is_node], bins=bins, color=SERIES[0], alpha=0.85, label="noeuds (verite)",
        edgecolor="white", linewidth=0.5, density=True)
ax.hist(ab[is_strut], bins=bins, color=SERIES[1], alpha=0.75, label="brins (verite)",
        edgecolor="white", linewidth=0.5, density=True)
ax.axvline(1.6, color=INK, ls="--", lw=2, label="seuil 1,6 (these)")
ax.set_xlabel("a/b au point de mesure"); ax.set_ylabel("densite")
ax.set_title("Les deux populations se separent")
ax.legend()

ax = axes[1]
ax.plot(ths, prec, color=SERIES[0], label="precision")
ax.plot(ths, rec, color=SERIES[1], label="rappel")
ax.plot(ths, acc, color=SERIES[2], label="exactitude")
ax.axvline(1.6, color=INK, ls="--", lw=2)
k16 = int(np.argmin(np.abs(ths - 1.6)))
ax.annotate(f"a 1,6 : precision {prec[k16]:.2f}, rappel {rec[k16]:.2f}",
            xy=(1.6, prec[k16]), xytext=(1.75, 0.35), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("seuil sur a/b"); ax.set_ylabel("score")
ax.set_title("Le compromis du seuil")
ax.legend(loc="lower left")
fig.tight_layout()
plt.show()
""")

code(r"""
cls = ma.shape.shape_classification(foam_struts.solid, aperture=aper_solid)
cls = np.asarray(cls)
names = {0: "fond", 1: "noeud", 2: "brin", 3: "plaque"}
counts = {names[v]: int((cls == v).sum()) for v in (1, 2, 3)}
print("volumes classes :", counts)

cls_true = np.zeros(foam_struts.shape[1:], dtype=int)
cls_true[midslice(ts["strut_mask"])] = 2
cls_true[midslice(ts["node_mask"])] = 1

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
cmap4 = ListedColormap(["#f5f4f1", "#eb6834", "#2a78d6", "#1baf7a"])
imshow(axes[0], midslice(foam_struts.solid), "solide",
       cmap=ListedColormap(["#f5f4f1", "#256abf"]))
imshow(axes[1], cls_true, "verite : noeud (orange) / brin (bleu)", cmap=cmap4, vmin=0, vmax=3)
imshow(axes[2], midslice(cls), "classification mesuree, propagee", cmap=cmap4, vmin=0, vmax=3)
fig.tight_layout()
plt.show()
""")

md(r"""
### Orientation des brins

Une fois les brins identifies, leur grand axe donne une orientation. Sur des
cylindres paralleles a `z`, l'elevation doit valoir 90 degres partout.
""")

code(r"""
ori = ma.shape.strut_orientation(ma.shape.local_shape_tensor(cyl, propagate=False))
display(ori.head(4))

# les deux histogrammes n'ont pas le meme nombre de classes : l'elevation couvre
# 0-90 degres, l'azimut 0-360. Les colonnes les plus courtes sont completees par
# des NaN, qu'on retire.
elev = ori[["elevation", "elevation_count"]].dropna()
azim = ori[["azimuth", "azimuth_count"]].dropna()

fig = plt.figure(figsize=(9.5, 3.6))
ax = fig.add_subplot(1, 2, 1)
ax.bar(elev["elevation"], elev["elevation_count"],
       width=0.85 * np.diff(elev["elevation"].to_numpy()).mean(),
       color=SERIES[0], edgecolor="white", linewidth=0.8)
ax.axvline(90, color=SERIES[1], ls="--", lw=2, label="attendu : 90 deg")
ax.set_xlabel("elevation theta (deg)"); ax.set_ylabel("nombre de points de mesure")
ax.set_title("Cylindres // z : les brins sont verticaux")
ax.legend()

ax = fig.add_subplot(1, 2, 2, projection="polar")
w = np.deg2rad(np.diff(azim["azimuth"].to_numpy()).mean())
ax.bar(np.deg2rad(azim["azimuth"].to_numpy()), azim["azimuth_count"].to_numpy(),
       width=0.9 * w, color=SERIES[2], edgecolor="white", linewidth=0.6)
ax.set_title("Azimut : degenere pour un axe // z", pad=16)
ax.set_yticklabels([]); ax.grid(color=GRID, linewidth=0.6)
ax.set_rlabel_position(135)
fig.tight_layout()
plt.show()

peak = elev.loc[elev["elevation_count"].idxmax(), "elevation"]
print(f"elevation dominante : {peak:.1f} deg   (attendu : 90 deg pour des cylindres // z)")
print("l'azimut d'un axe parallele a z est degenere : le solveur rend une valeur")
print("conventionnelle (ici le lobe a 0 deg), qu'il ne faut pas interpreter.")
""")

# ==========================================================================
# 9. Tortuosite
# ==========================================================================
md(r"""
<a id="9"></a>
## 9. Tortuosite

La tortuosite de Carman compare la longueur du plus court chemin dans la phase a
la distance a vol d'oiseau :

$$\tau = \left(\frac{L_{\min}}{\lVert p_1 - p_2\rVert}\right)^2$$

C'est un **carre** — l'usage en milieux poreux, herite de Kozeny-Carman, ou
$\tau$ intervient au carre dans la permeabilite. Un milieu libre vaut exactement 1.

La longueur geodesique vient d'un fast marching : on resout l'equation eikonale
$\lVert \nabla T\rVert = 1/F$ depuis la source, et $T$ est le temps de parcours
minimal.

> **Convention de vitesse.** iMorph resout $\lVert\nabla T\rVert = F$ : son
> « champ de vitesse » est en realite une **lenteur**. Le module expose les deux,
> `speed=` (convention standard) et `cost=` (celle d'iMorph).
""")

code(r"""
fluid_f = foam_flow.fluid
T = np.asarray(ma.distance.travel_time(fluid_f, 0))          # depuis la face z = 0
arrival = T[-1]
ok = np.isfinite(arrival)
j, i = np.unravel_index(np.nanargmin(np.where(ok, arrival, np.inf)), arrival.shape)
path = ma.tortuosity.shortest_path(T, (foam_flow.shape[0] - 1, j, i))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
h = imshow(axes[0], np.where(np.isfinite(T[:, :, i]), T[:, :, i], np.nan),
           f"temps de parcours, coupe x = {i}")
cbar(fig, h, axes[0], "voxels")
axes[0].plot(path[:, 1], path[:, 0], color=SERIES[1], lw=2.0)
axes[0].scatter([path[0, 1], path[-1, 1]], [path[0, 0], path[-1, 0]],
                s=34, color=SERIES[1], zorder=3, edgecolor="white", linewidth=0.8)

ax = axes[1]
ax.plot(np.arange(len(path)), np.linalg.norm(np.diff(path, axis=0, prepend=path[:1]), axis=1).cumsum(),
        color=SERIES[0], label="longueur parcourue")
ax.plot(np.arange(len(path)), np.abs(path[:, 0] - path[0, 0]), color=SERIES[1],
        label="avancee selon z")
ax.set_xlabel("pas le long du chemin"); ax.set_ylabel("longueur (voxels)")
ax.set_title("Le chemin geodesique et sa projection")
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()

print(f"chemin : {len(path)} points, longueur {np.linalg.norm(np.diff(path, axis=0), axis=1).sum():.1f} voxels")
print(f"separation des plans : {foam_flow.shape[0] - 1} voxels")
""")

md(r"""
### Le piege du premier ordre, et ce qui l'annule

Le schema amont du premier ordre sous-estime les trajets obliques : +2,7 % sur
une diagonale 2D, +3,8 % en 3D. Comme la tortuosite eleve ce rapport au carre, le
biais devient **+8,8 %** — assez pour qu'un milieu libre ressorte a 1,088.

`point_tortuosity(reference="free_front")` divise par le temps de parcours du
**meme solveur** dans une boite vide de meme geometrie. Le biais se simplifie.

`plane_tortuosity` n'a pas ce probleme a l'echelle globale : il divise par la
separation exacte des deux plans, et un front plan qui avance selon un axe ne
diagonalise pas. Il garde en revanche une petite contribution du premier ordre
dans les portions ou le front doit contourner un obstacle.
""")

code(r"""
free = np.ones((64, 64, 64), dtype=bool)
rows = []
for milieu, m, src in [("milieu libre", free, (32, 32, 32)),
                       ("mousse", fluid_f, tuple(s // 2 for s in foam_flow.shape))]:
    for ref in ("free_front", "euclidean"):
        r = ma.tortuosity.point_tortuosity(m, src, reference=ref)
        rows.append({"milieu": milieu, "reference": ref, "tau (point)": r.value,
                     "ecart-type": r.std})
bias = pd.DataFrame(rows)
display(bias.round(4))
print()
print(f"plan, milieu libre : {ma.tortuosity.plane_tortuosity(free, 0).value:.4f}")
print(f"plan, mousse       : {ma.tortuosity.plane_tortuosity(fluid_f, 0).value:.4f}")
""")

md(r"""
### Validation analytique : un conduit incline

Pour un conduit droit incline de $\alpha$ par rapport a la normale aux plans de
mesure, la reponse est connue : le chemin est plus long d'un facteur
$1/\cos\alpha$, donc

$$\tau = \sec^2\alpha$$

C'est la verification la plus directe possible de l'ensemble fast marching +
formule de Carman.
""")

code(r"""
def tilted_channel(alpha_deg, n=96, radius=6.0):
    a = np.deg2rad(alpha_deg)
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    c = (n - 1) / 2.0
    uy, ux = np.cos(a), np.sin(a)
    dy, dx = y - c, x - c
    t = dy * uy + dx * ux
    return np.sqrt((z - c) ** 2 + (dy - t * uy) ** 2 + (dx - t * ux) ** 2) <= radius


alphas = np.array([0, 10, 20, 30, 40], dtype=float)
meas = np.array([ma.tortuosity.plane_tortuosity(tilted_channel(a), face=2).value for a in alphas])
theo = 1.0 / np.cos(np.deg2rad(alphas)) ** 2

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax = axes[0]
ax.plot(alphas, theo, color=SERIES[1], lw=2, label=r"theorie  $\sec^2\alpha$")
ax.plot(alphas, meas, color=SERIES[0], marker="o", ls="none", label="mesure")
ax.set_xlabel("inclinaison du conduit (deg)"); ax.set_ylabel(r"$\tau$")
ax.set_title("Conduit incline : reponse analytique")
ax.legend()

ax = axes[1]
ax.bar(alphas, 100 * (meas - theo) / theo, width=5, color=SERIES[0],
       edgecolor="white", linewidth=0.8)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("inclinaison (deg)"); ax.set_ylabel("ecart a la theorie (%)")
ax.set_title("Ecart : au plus 2,4 %")
fig.tight_layout()
plt.show()

pd.DataFrame({"alpha (deg)": alphas, "mesure": meas, "sec^2(alpha)": theo,
              "ecart (%)": 100 * (meas - theo) / theo}).round(4)
""")

md(r"""
### Tube sinusoidal : la tortuosite croit avec l'amplitude

Le fantome connait la tortuosite de sa **ligne centrale**, calculee par longueur
d'arc a la precision machine. La mesure sur le volume est necessairement plus
faible : le conduit a une epaisseur, et le chemin le plus court coupe les
virages.
""")

code(r"""
rows = []
for amp in (0.0, 4.0, 8.0, 12.0, 16.0):
    tb = ma.phantoms.sinusoidal_tube(shape=(128, 80, 80), radius=6.0, amplitude=amp,
                                     n_periods=2.0)
    r = ma.tortuosity.plane_tortuosity(tb.fluid, face=0)   # le tube, pas la matrice
    rows.append({"amplitude": amp, "tau mesure": r.value,
                 "tau de la ligne centrale": tb.meta["truth"]["tortuosity"]})
sine = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(sine["amplitude"], sine["tau de la ligne centrale"], color=SERIES[1],
        marker="s", label="ligne centrale (exact)")
ax.plot(sine["amplitude"], sine["tau mesure"], color=SERIES[0], marker="o",
        label="mesure sur le volume")
ax.set_xlabel("amplitude du tube (voxels)"); ax.set_ylabel(r"$\tau$")
ax.set_title("Le chemin le plus court coupe les virages")
ax.legend()
fig.tight_layout()
plt.show()
sine.round(4)
""")

md(r"""
### Isotropie : une mousse de Voronoi n'a pas de direction privilegiee

`directional_tortuosity` tourne le volume autour d'un axe, extrait le
parallelepipede inscrit — sinon la rotation fait entrer du vide dans l'image —
et mesure une tortuosite de plan transverse. Sur un milieu anisotrope, la these
observe une sinusoide de periode 180 degres (figures 3.23 a 3.25). Sur une
mousse de Voronoi, il ne doit rien se passer : c'est ce qu'on verifie.
""")

code(r"""
dirs = ma.tortuosity.directional_tortuosity(fluid_f, angles=12, axis=0)
spread = 100 * (dirs["tortuosity"].max() - dirs["tortuosity"].min()) / dirs["tortuosity"].mean()

fig, ax = plt.subplots(figsize=(4.6, 4.2), subplot_kw={"projection": "polar"})
ang = np.deg2rad(dirs["angle"].to_numpy())
val = dirs["tortuosity"].to_numpy()
ang = np.concatenate([ang, ang + np.pi, ang[:1]])     # periode 180 deg
val = np.concatenate([val, val, val[:1]])
ax.plot(ang, val, color=SERIES[0], marker="o")
ax.set_ylim(1.0, max(1.02, val.max() * 1.002))
ax.set_title(f"Tortuosite directionnelle — dispersion {spread:.1f} %", pad=16)
ax.grid(color=GRID, linewidth=0.6)
ax.set_rlabel_position(200)
fig.tight_layout()
plt.show()
dirs.round(4)
""")

md(r"""
### Tortuosite hydraulique

Un champ de vitesse de Poiseuille, $F = 1 - (1 - d/R)^2$ ou $d$ est la distance a
la paroi et $R$ l'ouverture locale, pousse le chemin vers l'axe du conduit. On
compare au chemin geometrique **sur les memes extremites**, faute de quoi la
comparaison n'a pas de sens.

> **La variante d'iMorph est refusee.** iMorph utilise $F = 1 - d^2/R^2$, qui
> s'annule au **centre** du conduit au lieu d'y etre maximale. Le resultat
> degenere de facon dependante des donnees ($\tau = 0{,}44$ sur une mousse, 1,02
> sur une autre). `poiseuille_speed(variant="imorph")` la donne toujours, pour
> inspection ; `poiseuille_tortuosity` la refuse par une `ValueError`.
""")

code(r"""
d_wall = ma.distance.distance_transform(fluid_f)
speed = np.asarray(ma.tortuosity.poiseuille_speed(fluid_f))
speed_im = np.asarray(ma.tortuosity.poiseuille_speed(fluid_f, variant="imorph"))

Tp = np.asarray(ma.distance.travel_time(fluid_f, 0, speed=speed))
arr = T[-1]
ok = np.isfinite(arr) & np.isfinite(Tp[-1])
j, i = np.unravel_index(np.nanargmin(np.where(ok, arr, np.inf)), arr.shape)
end = (foam_flow.shape[0] - 1, j, i)
p_geo = ma.tortuosity.shortest_path(T, end)
p_poi = ma.tortuosity.shortest_path(Tp, end)
w_geo = d_wall[p_geo[:, 0], p_geo[:, 1], p_geo[:, 2]]
w_poi = d_wall[p_poi[:, 0], p_poi[:, 1], p_poi[:, 2]]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

# (a) les deux profils de vitesse, analytiques
ax = axes[0]
u = np.linspace(0, 1, 200)                       # u = d / R, 0 a la paroi, 1 au centre
ax.plot(u, 1 - (1 - u) ** 2, color=SERIES[0], label=r"physique : $1-(1-d/R)^2$")
ax.plot(u, 1 - u**2, color=SERIES[1], label=r"iMorph : $1-d^2/R^2$")
ax.axvline(1.0, color=INK2, lw=0.8, ls=":")
ax.text(0.98, 0.5, "centre du conduit", rotation=90, ha="right", va="center",
        fontsize=8, color=INK2)
ax.set_xlabel("d / R  (0 = paroi, 1 = centre)"); ax.set_ylabel("vitesse F")
ax.set_title("La variante d'iMorph s'annule au centre")
ax.legend(loc="center left")
ax.grid(True)

# (b) la distance a la paroi le long du chemin
ax = axes[1]
ax.plot(np.arange(len(w_geo)), w_geo, color=SERIES[1],
        label=f"geometrique (moyenne {w_geo.mean():.1f})")
ax.plot(np.arange(len(w_poi)), w_poi, color=SERIES[3],
        label=f"Poiseuille (moyenne {w_poi.mean():.1f})")
ax.set_xlabel("pas le long du chemin"); ax.set_ylabel("distance a la paroi (voxels)")
ax.set_title("Le chemin hydraulique reste au centre")
ax.legend(loc="lower right")

# (c) le resultat, sur 16 chemins partageant leurs extremites
poi = ma.tortuosity.poiseuille_tortuosity(fluid_f, n_paths=16)
ax = axes[2]
names = ["geometrique", "Poiseuille"]
vals = [poi["geometric_tortuosity"], poi["tortuosity"]]
bars = ax.bar(names, vals, color=[SERIES[1], SERIES[3]], width=0.5,
              edgecolor="white", linewidth=0.8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
            fontsize=9, color=INK)
ax.set_ylim(1.0, max(vals) * 1.08)
ax.set_ylabel(r"$\tau$")
ax.set_title(f"Sur {poi['n_paths']} chemins, memes extremites")
ax.grid(axis="x", visible=False)
fig.tight_layout()
plt.show()

display(pd.DataFrame([poi]).round(3))
try:
    ma.tortuosity.poiseuille_tortuosity(fluid_f, variant="imorph")
except ValueError as exc:
    print("variant='imorph' :", exc)
""")

# ==========================================================================
# 10. Drainage
# ==========================================================================
md(r"""
<a id="10"></a>
## 10. Drainage morphologique et courbe de retention

On simule l'intrusion d'un fluide non mouillant par une face, a pression
croissante — donc a rayon de courbure decroissant. iMorph propose deux
algorithmes, et leur difference est de fond :

| | critere de passage au rayon $R$ | reference |
|---|---|---|
| **Hazlett** | un chemin de voxels d'**ouverture locale** $\ge R$ | Hazlett, *Transp. Porous Media* 1995 |
| **Hilpert–Miller** | une boule de rayon $R$ qui **se deplace** continument | Hilpert & Miller, *Adv. Water Resour.* 2001 |

Hilpert exige en plus que le *centre* de la boule ait un chemin continu : il est
donc toujours plus restrictif ou egal. C'est ce qu'on verifie.
""")

code(r"""
res = {}
for method in ("hazlett", "hilpert"):
    res[method] = ma.network.drainage(fluid_f, face=0, method=method, step=0.5,
                                      surface_tension=ma.network.WATER_AIR_SURFACE_TENSION,
                                      unit="um")
    print(f"{method:8s} : {len(res[method].curve):2d} points de courbe, "
          f"rayon d'entree {res[method].curve['radius'].iloc[0]:.2f} voxels")

a = res["hazlett"].filling_radius
b = res["hilpert"].filling_radius
print(f"\nHilpert jamais plus permissif que Hazlett : {bool((b[fluid_f] <= a[fluid_f] + 1e-6).all())}")
""")

code(r"""
i0 = foam_flow.shape[2] // 2
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
vmax = float(max(a.max(), b.max()))
for ax, (m, arr) in zip(axes[:2], [("Hazlett", a), ("Hilpert", b)]):
    h = imshow(ax, np.where(arr[:, :, i0] >= 0, arr[:, :, i0], np.nan),
               f"{m} : rayon d'envahissement", vmin=0, vmax=vmax)
    cbar(fig, h, ax, "rayon (voxels)")

ax = axes[2]
diff = np.where(fluid_f[:, :, i0], a[:, :, i0] - b[:, :, i0], np.nan)
h = imshow(ax, diff, "Hazlett - Hilpert (toujours >= 0)", vmin=0, vmax=vmax / 2)
cbar(fig, h, ax, "voxels")
fig.tight_layout()
plt.show()
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
for k, (m, color) in enumerate(zip(("hazlett", "hilpert"), (SERIES[0], SERIES[1]))):
    c = res[m].curve
    axes[0].plot(c["radius"], c["saturation"], color=color, marker="o", ms=3.5, label=m)
    axes[1].plot(c["pressure"] / 1e3, c["saturation"], color=color, marker="o", ms=3.5, label=m)
axes[0].invert_xaxis()
axes[0].set_xlabel("rayon de courbure (voxels) — la pression monte vers la droite")
axes[0].set_ylabel("saturation en fluide non mouillant")
axes[0].set_title("Courbe de retention")
axes[0].legend()
axes[1].set_xscale("log")
axes[1].set_xlabel("pression capillaire (kPa), eau/air, voxel = 1 um")
axes[1].set_ylabel("saturation")
axes[1].set_title("La meme, en pression (Young-Laplace)")
axes[1].legend()
fig.tight_layout()
plt.show()
display(res["hilpert"].curve.head(6).round(3))
""")

md(r"""
### Le blindage par les cols : l'effet « bouteille d'encre »

Une chambre large derriere un col etroit n'est envahie qu'une fois le rayon
descendu **sous celui du col**, quelle que soit sa propre taille. C'est la seule
chose qu'un algorithme de drainage morphologique doit absolument reproduire.
""")

code(r"""
nz, ny, nx = 64, 48, 48
z, y, x = np.ogrid[:nz, :ny, :nx]
c = 23.5
neck = ((y - c) ** 2 + (x - c) ** 2 <= 4.0**2) & (z < 34)
bulb = ((z - 44) ** 2 + (y - c) ** 2 + (x - c) ** 2) <= 11.0**2
pore = neck | bulb

ink = {m: ma.network.drainage(pore, face=0, method=m, step=0.25, min_radius=0.5,
                              restrict="inlet") for m in ("hazlett", "hilpert")}
r_neck = ink["hilpert"].filling_radius[neck].max()
r_bulb_geo = ma.distance.distance_transform(pore)[bulb & ~neck].max()

print(f"rayon du col (mesure)                  : {r_neck:.2f} voxels")
print(f"rayon de la chambre (mesure)           : {r_bulb_geo:.2f} voxels")
for m, v in ink.items():
    print(f"{m:8s} : rayon max atteint dans la chambre = {v.filling_radius[bulb].max():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
imshow(axes[0], pore[:, :, 24], "le fantome : col puis chambre",
       cmap=ListedColormap(["#f5f4f1", "#256abf"]))
for ax, m in zip(axes[1:], ("hazlett", "hilpert")):
    arr = ink[m].filling_radius[:, :, 24]
    h = imshow(ax, np.where(arr >= 0, arr, np.nan), f"{m} : rayon d'envahissement",
               vmin=0, vmax=float(r_bulb_geo))
    cbar(fig, h, ax, "voxels")
fig.suptitle("La chambre n'est jamais envahie au-dela du rayon du col",
             x=0.02, ha="left", fontsize=11, fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
plt.show()
""")

md(r"""
### Pression capillaire

Par defaut, Young-Laplace : $P_c = 2\sigma\cos\theta / r$.

> **La convention d'iMorph vaut le double.** `saveFractionFile` ecrit
> $P_c = 4\sigma/r$ avec $r$ la colonne intitulee « radius ball ». Cela revient a
> lire l'ouverture comme un **diametre**. Le facteur 2 est systematique : il ne
> change pas la forme de la courbe, mais il decale l'axe des pressions.
""")

code(r"""
r_um = np.array([1.0, 2.5, 5.0, 10.0, 25.0])
comp = pd.DataFrame({
    "rayon (um)": r_um,
    "Young-Laplace (Pa)": ma.network.capillary_pressure(r_um, unit="um"),
    "convention iMorph (Pa)": ma.network.capillary_pressure(r_um, unit="um",
                                                            convention="imorph"),
    "mercure, theta=140 deg (Pa)": ma.network.capillary_pressure(
        r_um, unit="um", surface_tension=ma.network.MERCURY_SURFACE_TENSION,
        contact_angle=ma.network.MERCURY_CONTACT_ANGLE),
})
comp["rapport iMorph / Laplace"] = (comp["convention iMorph (Pa)"]
                                    / comp["Young-Laplace (Pa)"])
comp.round(1)
""")

# ==========================================================================
# 11. Percolation d'invasion
# ==========================================================================
md(r"""
<a id="11"></a>
## 11. Percolation d'invasion sur le reseau

Sur le graphe cellules-cols, l'amas envahisseur franchit a chaque etape le **plus
grand col accessible**. La pression ne fait donc que monter et le rayon
enregistre ne fait que decroitre.

L'entree dans une cellule de la face d'injection est commandee par le rayon de la
**cellule** ; le passage d'une cellule a l'autre, par celui du **col**.

> iMorph balayait une grille de pressions (`r_current -= 0.2`) et reconstruisait
> les amas a chaque niveau. Une file de priorite sur les rayons de cols donne
> exactement le meme ordre d'invasion, sans balayage.
""")

code(r"""
d_f = ma.distance.distance_transform(fluid_f)
mk_f = ma.granulometry.cell_markers(fluid_f, distance=d_f, fill_ratio=0.55)
cells_f = np.asarray(ma.segmentation.watershed_cells(d_f, mk_f, mask=fluid_f))
cm_f, th_f = ma.segmentation.pore_network(cells_f)
inlet = ma.network.face_cells(cells_f, 0)
outlet = ma.network.face_cells(cells_f, 1)
print(f"{len(cm_f)} cellules, {len(th_f)} cols, "
      f"{len(inlet)} cellules a l'entree, {len(outlet)} a la sortie")

runs = {
    "cols rigides": dict(),
    "avec piegeage": dict(outlet=outlet, trapping=True),
    "cols deformables (x2)": dict(deformation_rate=2.0),
}
inv = {}
rows = []
for name, kw in runs.items():
    r = ma.network.invasion_percolation(cm_f, th_f, inlet=inlet, **kw)
    inv[name] = r
    rows.append({"variante": name, "cellules envahies": r.n_invaded,
                 "sur": len(cm_f), "saturation finale": r.final_saturation,
                 "dernier rayon": float(np.nanmin(r.cells["filling_radius"])),
                 "piegees": int(r.cells["trapped"].sum())})
pd.DataFrame(rows).round(3)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
ax = axes[0]
for (name, r), color in zip(inv.items(), SERIES):
    ax.plot(r.curve["radius"], r.curve["saturation"], color=color, marker="o",
            ms=3.0, label=name)
ax.invert_xaxis()
ax.set_xlabel("rayon de courbure (voxels) — la pression monte vers la droite")
ax.set_ylabel("saturation (fraction de volume)")
ax.set_title("Courbes d'invasion")
ax.legend(loc="lower right")

ax = axes[1]
r = inv["cols rigides"]
order = r.cells[r.cells["invaded"]].sort_values("order")
ax.plot(order["order"], order["filling_radius"], color=SERIES[0])
ax.set_xlabel("rang d'invasion"); ax.set_ylabel("rayon enregistre (voxels)")
ax.set_title("Un escalier strictement decroissant")
fig.tight_layout()
plt.show()

diffs = np.diff(order["filling_radius"].to_numpy())
print(f"le rayon ne remonte jamais : {bool((diffs <= 1e-9).all())}")
""")

code(r"""
fmap = {name: np.asarray(ma.network.filling_map(cells_f, r)) for name, r in inv.items()}
k0 = foam_flow.shape[0] // 2
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
vmax = max(float(v.max()) for v in fmap.values())
for ax, (name, v) in zip(axes, fmap.items()):
    h = imshow(ax, np.where(v[k0] > 0, v[k0], np.nan), name, vmin=0, vmax=vmax)
    cbar(fig, h, ax, "rayon d'envahissement")
fig.suptitle("Le piegeage laisse des cellules seches (blanches)", x=0.02, ha="left",
             fontsize=11, fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
plt.show()
""")

md(r"""
### Les cols deformables, en forme close

Une trouvaille d'iMorph pour les milieux elastiques : les cols s'elargissent avec
la pression. Le facteur $k$ vaut 1 a la pression la plus basse et
`deformation_rate` a la plus haute ; comme $P_c \propto 1/r$, on a
$k(r) = A/r + B$, et un col de rayon $R$ passe des que $k(r)\,R \ge r$ :

$$r_{\text{eff}} = \frac{BR + \sqrt{B^2R^2 + 4AR}}{2},\qquad
A = \frac{D-1}{2 - 1/r_{\max}},\quad B = D - 2A$$

Calculable une fois pour toutes. **Ni la tension superficielle ni la taille de
voxel n'y interviennent** : elles se simplifient.
""")

code(r"""
R = np.linspace(0.5, 20.0, 160)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
for D, color in zip((1.0, 1.5, 2.0, 3.0), SERIES):
    reff = np.asarray(ma.network.deformed_throat_radius(R, deformation_rate=D, r_max=20.0))
    axes[0].plot(R, reff, color=color, label=f"D = {D:g}")
    axes[1].plot(R, reff / R, color=color, label=f"D = {D:g}")
axes[0].plot(R, R, color=INK2, lw=0.9, ls=":", label="identite")
axes[0].set_xlabel("rayon geometrique du col (voxels)")
axes[0].set_ylabel("rayon effectif (voxels)")
axes[0].set_title("En valeur absolue, l'effet se voit a peine")
axes[0].legend()
axes[1].axhline(1.0, color=INK2, lw=0.9, ls=":")
axes[1].set_xlabel("rayon geometrique du col (voxels)")
axes[1].set_ylabel(r"$r_{eff} / R$")
axes[1].set_title("En relatif : seuls les petits cols s'ouvrent")
axes[1].legend()
fig.tight_layout()
plt.show()

print("un col de rayon 1 avec D = 3 : r_eff =",
      f"{float(ma.network.deformed_throat_radius(1.0, deformation_rate=3.0, r_max=20.0)):.2f}")
print("un col de rayon 20 avec D = 3 : r_eff =",
      f"{float(ma.network.deformed_throat_radius(20.0, deformation_rate=3.0, r_max=20.0)):.2f}")
print("la deformabilite ne joue donc qu'a haute pression — la ou elle doit jouer.")
""")

# ==========================================================================
# 12. Os cortical
# ==========================================================================
md(r"""
<a id="12"></a>
## 12. Os cortical : profils angulaires et radiaux

Une diaphyse est un tube. Ses proprietes varient surtout avec l'angle autour de
l'axe et avec la distance au centre, d'ou le decoupage en parts de camembert et
en couronnes.

La convention d'angle d'iMorph est conservee telle quelle :

$$\theta(P) = \operatorname{atan2}(y_0 - j,\; i - x_0)$$

`x` vers la droite et `y` vers le **haut** — le sens trigonometrique usuel,
malgre l'axe des lignes qui descend. Le secteur 0 couvre donc le quadrant en haut
a droite de l'image.

Le fantome `cortical_tube` impose une anisotropie connue (`sector_weights`), ce
qui donne une verite terrain exacte.
""")

code(r"""
tb = bone.meta["truth"]
nz = bone.shape[0]
canals = np.broadcast_to(tb["canal_mask"], (nz,) + tb["canal_mask"].shape).copy()
outside = np.broadcast_to(~tb["ring_mask"], canals.shape).copy()

sect = ma.cortical.sector_map(bone.shape, center=tb["centre"], n_sectors=4)
prof = ma.cortical.angular_profile(canals, center=tb["centre"], n_sectors=12,
                                   mask_out=outside)
prof4 = ma.cortical.angular_profile(canals, center=tb["centre"], n_sectors=4,
                                    mask_out=outside)

print(f"porosite exacte du fantome : {tb['porosity']:.6f}")
print(f"porosite mesuree           : {prof.overall:.6f}")
print(f"ecart                      : {abs(prof.overall - tb['porosity']):.2e}")
print()
print("poids imposes  : 3 / 1 / 1 / 1")
print("porosite par quadrant :", " ".join(f"{v:.4f}" for v in prof4.by_sector()))
""")

code(r"""
fig = plt.figure(figsize=(10.5, 3.6))
ax = fig.add_subplot(1, 3, 1)
imshow(ax, np.where(tb["ring_mask"], sect, np.nan), "secteurs (4 parts)",
       cmap=ListedColormap(SERIES))
ax = fig.add_subplot(1, 3, 2)
imshow(ax, tb["ring_mask"].astype(int) + tb["canal_mask"].astype(int),
       "matrice et canaux", cmap=ListedColormap(["#f5f4f1", "#cde2fb", "#1c5cab"]))

ax = fig.add_subplot(1, 3, 3, projection="polar")
by = prof.by_sector()
edges = np.sort(np.mod(prof.angles, 2 * np.pi))
width = 2 * np.pi / len(by)
ax.bar(edges, by.to_numpy(), width=0.92 * width, align="edge",
       color=SERIES[0], edgecolor="white", linewidth=0.8)
ax.axhline(tb["porosity"], color=SERIES[1], lw=2, ls="--")
ax.set_title("Porosite par secteur (12 parts)\nligne : porosite totale exacte", pad=16)
ax.grid(color=GRID, linewidth=0.6)
ax.set_rlabel_position(112)
fig.tight_layout()
plt.show()
""")

md(r"""
### Couronnes et secteurs d'aire egale

Des parts d'angle egal n'ont pas la meme aire des que la section n'est pas
circulaire, et des couronnes de meme epaisseur ne contiennent pas le meme nombre
de pixels. Deux corrections, toutes deux dans iMorph :

- `ellipse=(a, b, angle)` aplatit les angles par le rapport `b/a` ;
- `iso_area_angles` equilibre le **volume reellement present** — l'option
  « iso apparent volume », utile quand un masque retire une partie de la section.
""")

code(r"""
inside = ~outside
iso = ma.cortical.iso_area_angles(inside, 6, center=tb["centre"])
sm_iso = ma.cortical.sector_map(inside.shape, center=tb["centre"], angles=iso)
sm_reg = ma.cortical.sector_map(inside.shape, center=tb["centre"], n_sectors=6)

n_iso = np.bincount(sm_iso[inside[0]], minlength=6).astype(float)
n_reg = np.bincount(sm_reg[inside[0]], minlength=6).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax = axes[0]
w = 0.38
xx = np.arange(6)
ax.bar(xx - w / 2, n_reg, width=w, color=SERIES[0], edgecolor="white",
       linewidth=0.8, label=f"angles egaux (disp. {n_reg.std() / n_reg.mean():.1%})")
ax.bar(xx + w / 2, n_iso, width=w, color=SERIES[1], edgecolor="white",
       linewidth=0.8, label=f"iso-volume (disp. {n_iso.std() / n_iso.mean():.1%})")
ax.set_xlabel("secteur"); ax.set_ylabel("pixels utiles par coupe")
ax.set_title("Equilibrer le volume des secteurs")
ax.legend()

ax = axes[1]
rad = ma.cortical.radial_profile(canals, center=tb["centre"], n_bins=16, r_max=64.0)
per_ring = rad.groupby("ring").agg(r=("r_min", "first"), poro=("porosity", "mean"),
                                   n=("n_voxels", "sum"))
ax.plot(per_ring["r"], per_ring["poro"], color=SERIES[0], marker="o")
ax.axvspan(tb["r_inner"], tb["r_outer"], color=SERIES[2], alpha=0.12, linewidth=0)
ax.text(0.5 * (tb["r_inner"] + tb["r_outer"]), ax.get_ylim()[1] * 0.92,
        "epaisseur corticale", ha="center", fontsize=8.5, color=INK2)
ax.set_xlabel("rayon (voxels)"); ax.set_ylabel("porosite")
ax.set_title("Profil radial : les canaux sont bien dans la couronne")
fig.tight_layout()
plt.show()
""")

md(r"""
### Connectivite par empilement

A la coupe $k$, on ne connait que le sous-volume $[start, k]$. Deux canaux qui se
rejoignent plus haut sont comptes **separement** tant que la jonction n'est pas
atteinte — la mesure dit a quelle hauteur le reseau se referme sur lui-meme.

Les 853 lignes d'iMorph, avec leur traitement explicite des « ponts » par
intersection et union de `std::set`, deviennent ici une structure union-find qui
les gere par construction.
""")

code(r"""
conn_par = ma.cortical.cortical_connectivity(canals)

# un pont introduit a la coupe 30 : deux canaux qui se rejoignent
bridge = np.zeros((40, 40, 40), dtype=bool)
bridge[:, 18:22, 10:14] = True
bridge[:, 18:22, 26:30] = True
bridge[30:, 18:22, 10:30] = True
conn_br = ma.cortical.cortical_connectivity(bridge)

print(f"canaux paralleles : {conn_par.table['n_objects'].iloc[-1]} objets a la fin "
      f"(attendu {tb['n_canals']})")
print(f"fantome a pont    : {conn_br.table['n_objects'].iloc[29]} objets a la coupe 29, "
      f"{conn_br.table['n_objects'].iloc[30]} a la coupe 30")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
axes[0].plot(conn_par.table["z"], conn_par.table["n_objects"], color=SERIES[0],
             label="canaux paralleles")
axes[0].plot(conn_br.table["z"], conn_br.table["n_objects"], color=SERIES[1],
             label="fantome a pont")
axes[0].axvline(30, color=INK2, ls=":", lw=1.2)
axes[0].annotate("le pont", xy=(30, 2), xytext=(33, 6), fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
axes[0].set_xlabel("coupe z"); axes[0].set_ylabel("objets distincts connus")
axes[0].set_title("Le reseau se referme-t-il ?")
axes[0].legend()

axes[1].plot(conn_par.table["z"], conn_par.table["largest_fraction"], color=SERIES[0],
             label="canaux paralleles")
axes[1].plot(conn_br.table["z"], conn_br.table["largest_fraction"], color=SERIES[1],
             label="fantome a pont")
axes[1].set_xlabel("coupe z"); axes[1].set_ylabel("part du plus gros objet")
axes[1].set_title("Fraction portee par le plus gros objet")
axes[1].legend()
fig.tight_layout()
plt.show()
""")

md(r"""
### Voronoi 2D : les territoires d'osteones

Chaque coupe de la matrice est partagee entre ses pores voisins, par distance
**geodesique** — donc en contournant les obstacles.

> iMorph inondait un relief inverse $\max(d) - d$, c'est-a-dire en partant des
> cretes. Ici on inonde la distance elle-meme depuis les germes de bord : la
> formulation directe du probleme, et le meme choix que pour le squelette de
> Plateau, ou il avait ramene le decalage des noeuds de 1,4 voxel a 0.
""")

code(r"""
vor, vtable = ma.cortical.voronoi_2d(np.asarray(bone), mask_out=outside,
                                     return_table=True)
varr = np.asarray(vor)
print(f"{vtable['label'].nunique()} territoires par coupe, "
      f"aire mediane {vtable['area'].median():.0f} voxels")
print(f"toute la matrice est etiquetee : "
      f"{bool((varr[np.asarray(bone) & ~outside] > 0).all())}")

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
imshow(axes[0], np.asarray(bone)[0], "matrice osseuse",
       cmap=ListedColormap(["#f5f4f1", "#256abf"]))
imshow(axes[1], varr[0], "territoires de Voronoi 2D",
       cmap=label_cmap(int(varr.max()), seed=3))
fig.tight_layout()
plt.show()
""")

# ==========================================================================
# 13. Maillage
# ==========================================================================
md(r"""
<a id="13"></a>
## 13. Maillage de surface et export

`surface_mesh` s'appuie sur marching cubes et accepte un **champ de gris** en
plus du masque — c'est le meme constat qu'a la section 3.

Deux pieges, tous deux verifies par des tests :

1. **Marching cubes biseaute.** Sur un cube aligne sur la grille, les 12 aretes
   et les 8 coins sont remplaces par des facettes a 45 degres. Le volume reste
   juste a 1 % pres, l'aire non : un chanfrein retire plus d'aire aux faces
   voisines qu'il n'en ajoute.
2. **Le volume d'une surface ouverte n'a pas de sens.** Quand le solide touche le
   bord de la boite — donc presque toujours — marching cubes ne referme pas les
   sections coupees. C'est correct pour une aire, mais pas pour un volume.
""")

code(r"""
rows = []
for name, m, kw in [
    ("sphere, masque binaire", sphere.solid, {}),
    ("sphere, champ de gris", sphere.solid,
     {"grey": sphere.meta["truth"]["signed_distance"], "level": 0.0}),
]:
    mesh = ma.mesh.surface_mesh(m, **kw)
    rows.append({"cas": name, "triangles": len(mesh.faces), "aire": mesh.area,
                 "aire exacte": analytic_area,
                 "ecart aire (%)": 100 * (mesh.area - analytic_area) / analytic_area,
                 "volume": mesh.volume, "volume exact": analytic_vol})

for name, pad in [("mousse, surface ouverte", False), ("mousse, pad=True", True)]:
    mesh = ma.mesh.surface_mesh(foam_flow.solid, pad=pad)
    rows.append({"cas": name, "triangles": len(mesh.faces), "aire": mesh.area,
                 "aire exacte": np.nan, "ecart aire (%)": np.nan,
                 "volume": mesh.volume, "volume exact": float(foam_flow.solid.sum())})
pd.DataFrame(rows).round(2)
""")

code(r"""
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sub = foam_flow.solid[:40, :40, :40]
mesh = ma.mesh.surface_mesh(sub, pad=True)
small = ma.mesh.decimate(mesh, target_reduction=0.75) if len(mesh.faces) > 40000 else mesh

fig = plt.figure(figsize=(9.5, 4.2))
ax = fig.add_subplot(1, 2, 1, projection="3d")
tri = small.vertices[small.faces][:, :, [2, 1, 0]]          # (x, y, z) pour matplotlib
n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
light = np.abs(n @ np.array([0.4, 0.5, 0.75]))             # eclairage diffus simple
shade = BLUE(0.25 + 0.55 * light)
coll = Poly3DCollection(tri, facecolors=shade, edgecolor="none")
ax.add_collection3d(coll)
ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_zlim(0, 40)
ax.set_box_aspect((1, 1, 1))
ax.set_title(f"Maillage du solide, {len(small.faces)} triangles")
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.grid(False)

ax = fig.add_subplot(1, 2, 2)
areas = []
sizes = (8, 12, 16, 20, 24, 28, 32)
for r in sizes:
    sph = ma.phantoms.sphere(shape=(2 * r + 24,) * 3, radius=float(r))
    m1 = ma.mesh.surface_mesh(sph.solid)
    m2 = ma.mesh.surface_mesh(sph.solid, grey=sph.meta["truth"]["signed_distance"], level=0.0)
    exact = 4 * np.pi * r**2
    areas.append({"rayon": r, "masque": 100 * (m1.area - exact) / exact,
                  "champ de gris": 100 * (m2.area - exact) / exact})
conv = pd.DataFrame(areas)
ax.plot(conv["rayon"], conv["masque"], color=SERIES[1], marker="o", label="masque binaire")
ax.plot(conv["rayon"], conv["champ de gris"], color=SERIES[2], marker="s",
        label="champ de gris")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("rayon de la sphere (voxels)"); ax.set_ylabel("ecart a l'aire exacte (%)")
ax.set_title("Le biais du masque ne disparait pas en raffinant")
ax.legend()
fig.tight_layout()
plt.show()
display(conv.round(2))
""")

code(r"""
import tempfile, pathlib

tmp = pathlib.Path(tempfile.mkdtemp())
sizes = {}
for ext, kw in [(".stl", {}), (".stl", {"binary": False}), (".obj", {}), (".ply", {})]:
    tag = ext + ("-ascii" if kw.get("binary") is False else "")
    path = ma.mesh.save_mesh(small, tmp / f"mousse{tag.replace('.', '_')}{ext}", **kw)
    sizes[tag] = path.stat().st_size / 1024
print("exports natifs, sans aucune dependance :")
for k, v in sizes.items():
    print(f"  {k:12s} {v:8.0f} ko")
""")

# ==========================================================================
# 14. Pipeline
# ==========================================================================
md(r"""
<a id="14"></a>
## 14. La meme chaine, sans une ligne de Python

iMorph ne savait rien faire sans sa fenetre : chaque calcul etait un
`CalcModule` couple a un `QThread`, une fenetre de parametres et le widget
central. Ici une analyse se decrit en donnees et s'execute d'un appel — depuis un
script, un notebook, un job de calcul ou la ligne de commande.
""")

code(r"""
from morphanalyzer.pipeline import Pipeline, available_steps

print(f"{len(available_steps())} etapes enregistrees :")
steps = available_steps()
for k in range(0, len(steps), 5):
    print("  " + "  ".join(f"{s:24s}" for s in steps[k:k + 5]))
""")

code(r"""
result = (
    Pipeline()
    .step("porosity", out="porosite")
    .step("specific_surface", out="Sv")
    .step("distance_transform", out="distance")
    .step("aperture_map", out="ouverture", n_radii=16)
    .run(foam_flow.fluid)
)
print()
print(f"porosite         : {result['porosite']:.4f}")
print(f"surface spec.    : {result['Sv']:.5f} voxel^-1")
print(f"ouverture max    : {result['ouverture'].max():.2f} voxels")
""")

md(r"""
Le meme pipeline en YAML, pour un traitement par lot reproductible :

```yaml
steps:
  - threshold_otsu
  - {keep_largest_component: {connectivity: 26}}
  - {porosity: {out: porosite}}
  - {specific_surface: {out: Sv}}
  - {aperture_map: {out: ouverture, n_radii: 24}}
```

```bash
morphanalyzer run pipeline.yaml --input tomo/ --voxel-size 7.46
```

### Et aucune interface n'est importee

Un test verrouille cette propriete a chaque execution de la suite : importer
n'importe quel module du noyau ne doit charger ni Qt, ni napari, ni matplotlib.
""")

code(r"""
import subprocess, sys

check = (
    "import sys, morphanalyzer as ma; "
    "import morphanalyzer.granulometry, morphanalyzer.network, morphanalyzer.cortical; "
    "gui = [m for m in sys.modules if m.split('.')[0] in "
    "{'PyQt5','PyQt6','PySide2','PySide6','napari','matplotlib','vispy'}]; "
    "print('modules graphiques charges :', gui or 'aucun')"
)
print(subprocess.run([sys.executable, "-c", check], capture_output=True,
                     text=True).stdout.strip())
""")

# ==========================================================================
# 15. Recapitulatif
# ==========================================================================
md(r"""
<a id="15"></a>
## 15. Recapitulatif : ce que la validation etablit

| grandeur | reference | mesure |
|---|---|---|
| aire d'une sphere, champ de gris | analytique | **-0,08 %** |
| aire d'une sphere, masque binaire | analytique | +9,3 % |
| segmentation des cellules | partition de Voronoi exacte | **IoU median 0,90** |
| `a/b` des cellules | these, tableau 3.1 : 1,302 | ~1,3 |
| `Dcol / Dpore` | these, figure 3.12 : 0,53 | ~0,6 |
| noeuds de Plateau | region de noeud exacte | **ecart median 0,0 voxel**, 89 % a moins de 2 |
| noeuds portant 4 cellules | loi de Plateau | **100 %** |
| seuil de classification | these : 1,6 | precision 0,95 |
| tortuosite, milieu libre | 1 exactement | **1,0000** |
| tortuosite, conduit incline | $\sec^2\alpha$ | ecart <= 2,4 % |
| porosite corticale par secteur | fantome | **exacte a $10^{-7}$ pres** |
| canaux distincts | fantome | **exact** |

### Trois ameliorations sur l'original, chacune mesuree

1. **Distance euclidienne exacte** au lieu du fast marching du premier ordre — la
   these mesurait elle-meme 2,77 voxels d'erreur.
2. **Marching cubes sur le champ de gris** pour la surface specifique : 9,3 % → 0,05 %.
3. **Inondation depuis l'interface** plutot que depuis la crete pour la
   propagation de Plateau : decalage des noeuds 1,4 → 0,0 voxel.

### Quatre erreurs de l'original, corrigees et documentees

1. Le champ de vitesse de Poiseuille d'iMorph s'annule **au centre** du conduit.
2. La pression capillaire d'iMorph vaut **le double** de Young-Laplace.
3. Les voxels poreux non envahis etaient initialises a `1.0`, donc comptes comme
   envahis au rayon 1 dans la courbe de retention.
4. La cle du tas du fast marching doit etre la valeur **arrondie en float32**
   reellement stockee, sinon la diagonale 3D ressort a +39 %.

## Pour aller plus loin

- `docs/` — le guide complet, section par section
- `docs/PORTING_MAP.md` — la carte de portage, fichier iMorph par fichier
- `docs/validation.md` — la strategie de validation
- `examples/full_chain.py`, `examples/drainage_cortical.py` — les memes chaines en script
- `pytest -q` — 221 tests
""")

# ==========================================================================
nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "tutoriel.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"{len(CELLS)} cellules ecrites dans {out}")
