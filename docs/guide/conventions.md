# Conventions

Sept décisions qui gouvernent toute la bibliothèque. Les connaître évite
l'essentiel des surprises.

## Ordre des axes : `(z, y, x)`

Un volume est un tableau `(nz, ny, nx)`, indexé `[k, j, i]`. C'est l'ordre des
piles tomographiques, celui de scikit-image, et celui d'iMorph, qui indexait
`data[k][j*width+i]`.

Le `voxel_size` suit le même ordre : `(dz, dy, dx)`.

## `True` = solide

Convention d'iMorph, conservée. Le fluide s'écrit `~solid`, explicitement, ce qui
rend visible dans chaque appel sur quelle phase on travaille.

```python
dist_fluide = ma.distance.distance_transform(~vol.solid)  # distance au solide
dist_solide = ma.distance.distance_transform(vol.solid)  # distance au fluide
```

!!! warning "La carte de distance est *dans* le masque qu'on lui passe"
    `distance_transform(m)` donne, pour chaque voxel de `m`, sa distance au
    premier voxel **hors** de `m`. Donc `distance_transform(~solid)` est la
    distance au solide dans le fluide — la carte de distance de la thèse.

## Le `Volume` est une commodité, pas une obligation

Toute fonction accepte indifféremment un `Volume` ou un `ndarray` nu. Le `Volume`
ne fait que transporter la taille de voxel et quelques métadonnées.

```python
vol = ma.Volume(array, voxel_size=7.46, unit="um", name="AL10")
ma.metrics.porosity(vol)  # marche
ma.metrics.porosity(array)  # marche aussi, voxel_size = 1
```

## Les grandeurs sont physiques

Les longueurs, surfaces et volumes rendus sont dans l'unité du `voxel_size`. Les
**rapports** — `a/b`, `b/c`, `Dcol/Dpore`, la tortuosité — sont sans dimension et
ne dépendent donc pas de la résolution.

```python
a = ma.segmentation.cell_morphometry(cells, voxel_size=1.0)
b = ma.segmentation.cell_morphometry(cells, voxel_size=2.0)
b["volume"].sum() == 8 * a["volume"].sum()  # vrai
b["a_on_b"].median() == a["a_on_b"].median()  # vrai aussi
```

## Labels : `0` = fond, `1..n` = objets

Convention `scipy.ndimage` / `skimage`. Les marqueurs suivent la même règle : `0`
n'est pas un marqueur.

## Les résultats tabulaires sont des `DataFrame`

Cellules, cols, brins, nœuds, boules : un `DataFrame` par famille d'objets,
écrivable en Parquet. Cela remplace la base XML et les fichiers `.txt` maison
d'iMorph.

```python
cells = ma.segmentation.cell_morphometry(labels)
ma.io.write_table(cells, "cells.parquet")
```

## Les objets touchant le bord sont signalés, pas retirés

`touches_border` apparaît dans les tables de cellules et de boules. Le filtrage
reste à l'appelant, parce qu'il dépend de la question posée.

!!! important "Pourquoi cela compte"
    Un objet coupé par le bord a un volume et une forme tronqués. La thèse
    établit **toutes** ses statistiques morphométriques sur les cellules
    entièrement incluses dans l'échantillon (tableaux 3.1 à 3.3). Reproduire ses
    chiffres sans ce filtrage ne fonctionne pas.

    ```python
    inner = cells[~cells.touches_border]
    ```

    L'effet vaut aussi pour la classification de forme : au bord, la boule de
    propagation est tronquée par la boîte, ce qui allonge artificiellement la
    forme mesurée. Une plaque traversante y est classée « brin ».
