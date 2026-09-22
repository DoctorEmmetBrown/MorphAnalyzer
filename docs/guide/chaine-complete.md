# La chaîne complète

De la pile d'images aux tables de cellules, de cols et de brins. C'est la chaîne
des chapitres 2 et 3 de la thèse, dans l'ordre.

```python
import numpy as np
import morphanalyzer as ma

# --- 1. lire et binariser -------------------------------------------------
vol = ma.io.read_stack("tomo/", voxel_size=7.46)
bin_ = ma.filters.threshold_otsu(vol)
bin_ = ma.filters.keep_largest_component(bin_)  # élimine le bruit en îlots
solid = bin_.solid
fluid = ~solid

# --- 2. grandeurs macroscopiques -----------------------------------------
phi = ma.metrics.porosity(bin_)
phi_ouvert, _ = ma.metrics.open_porosity(solid)
sv = ma.metrics.specific_surface(bin_, grey=vol.data, level=bin_.meta["threshold"])

# --- 3. distance et granulométrie ----------------------------------------
dist = ma.distance.distance_transform(fluid)  # distance au solide
aper = ma.granulometry.aperture_map(fluid)  # ouverture locale
psd = ma.granulometry.pore_size_distribution(aper)

# --- 4. segmentation des cellules ----------------------------------------
markers = ma.granulometry.cell_markers(fluid, distance=dist, fill_ratio=0.65)
cells = ma.segmentation.watershed_cells(dist, markers, mask=fluid)

# --- 5. morphométrie -----------------------------------------------------
cm = ma.segmentation.cell_morphometry(cells)
th = ma.segmentation.throats(cells)
cx = ma.segmentation.connectivity(throat_table=th)
inner = cm[~cm.touches_border]  # convention de la thèse

# --- 6. solide : forme locale et squelette -------------------------------
st = ma.shape.local_shape_tensor(solid)
cls = ma.shape.classify_solid(st, solid)  # 1 nœud, 2 brin, 3 plaque
sk = ma.skeleton.plateau_skeleton(solid, cells)

# --- 7. sauver -----------------------------------------------------------
ma.io.write_table(cm, "cells.parquet")
ma.io.write_table(th, "throats.parquet")
ma.io.write_table(sk.nodes, "nodes.parquet")
ma.io.write_table(sk.edges, "struts.parquet")
```

`examples/full_chain.py` exécute cette chaîne sur un fantôme et affiche les
chiffres à chaque étape.

## Ce que chaque étape apporte

### Binarisation

Sur une mousse très poreuse (> 80 %), l'histogramme est dominé par la gaussienne
du fluide et Otsu place le seuil dans une zone large et plate. La thèse estime
l'incertitude qui en résulte sur la porosité à **± 2 %** (§2.1.2). Ce n'est pas un
défaut d'implémentation mais une limite des données : le noter dans les
métadonnées plutôt que chercher à l'affiner.

`keep_largest_component` reproduit la recette d'iMorph : la phase solide étant
physiquement connexe, ne garder que la plus grosse composante élimine le bruit en
îlots.

!!! danger "Sauf si le solide n'est pas connexe"
    Sur un empilement de particules, un os trabéculaire fragmenté ou un milieu à
    plusieurs phases, `keep_largest_component` détruit l'échantillon. iMorph
    l'appliquait par défaut ; ici c'est un choix explicite.

### Surface spécifique

Mesurée sur le **champ de niveaux de gris**, pas sur le masque binaire. Sur une
sphère analytique, le masque surestime l'aire de **9,3 %** ; le champ continu
tombe à **0,05 %** d'erreur. Le gain est d'un ordre de grandeur pour un coût nul,
il suffit de passer `grey=` et `level=`.

### Granulométrie et marqueurs

La carte d'ouverture donne en chaque voxel le rayon de la plus grande boule
incluse qui le contient. Les **boules maximales** en sont extraites avec leur taux
de remplissage : la fraction de leur volume théorique qui leur reste attribuée.
Une boule bien remplie occupe une cavité propre — c'est un centre de cellule.

Le seuil par défaut est **0,65**, la valeur d'iMorph 3.2. La thèse cite 75 % ; les
deux sont dans le palier 60–80 % qu'elle identifie comme stable (fig. 3.3).
Au-delà de 0,8 on sous-segmente, ce qu'un test vérifie.

Les boules tronquées par le bord sont **comparées à la part d'elles-mêmes qui
tient dans l'image**, pas à la sphère entière. Sans cet écrêtage, une boule de
bord parfaitement inscrite dans un pore affichait un taux de 0,42 parce que la
moitié d'elle sortait du volume, et se faisait rejeter comme mal formée. iMorph
contournait le problème en exemptant ces boules du test (`isUseBallsAtFace`),
donc en gardant aussi les vraies boules incomplètes — d'où des cellules de bord
découpées en morceaux.

L'écrêtage rend le seuil interprétable partout, et `keep_border_balls` vaut
désormais `False` par défaut. Mesuré sur le fantôme de Voronoï (128³, 85
cellules, seuil 0,55) : l'IoU médian des cellules intérieures ne bouge pas
(0,908), les marqueurs passent de 85 à 70 et les fragments — cellules prédites
de moins de 15 % du volume médian — de 17 à 9. On perd des faux germes, pas des
cellules.

### Watershed

C'est la variante à **priorités réelles** avec résolution des collisions par label
majoritaire — celle qu'iMorph a écrite parce que la version quantifiée de Meyer
produit des lignes de partage en marches d'escalier (thèse, fig. 3.4b vs 3.5b).
Elle n'existe pas en bibliothèque.

Il n'y a **pas de ligne de partage matérialisée** : chaque voxel du masque reçoit
un label, la frontière est implicite. Cela évite d'avoir à décider à quelle
cellule appartient un voxel de digue.

## La même chaîne en YAML — et son piège

Dans un pipeline déclaratif, chaque étape reçoit par défaut **la sortie de la
précédente**. C'est ce qu'on veut pour une chaîne linéaire, et c'est faux dès
qu'une étape a besoin d'un résultat plus ancien. Le cas qui fait mal :

```
distance_transform  →  distance     le volume courant devient la distance
cell_markers        →  marqueurs    il devient les marqueurs
watershed_cells                     il lui faut la DISTANCE comme relief
```

Sans précaution, la dernière étape inonde l'image des marqueurs. Elle rend une
partition qui **ressemble** à des cellules — c'est à peu près un Voronoï
euclidien des marqueurs — et l'IoU contre la partition exacte tombe de **0,908 à
0,636**. Rien ne plante, rien n'avertit, et la figure a l'air correcte.

D'où le champ `input`, qui nomme le calque passé en premier argument :

```yaml
steps:
  - {complement: {out: fluide, input: volume}}
  - {distance_transform: {out: distance, input: fluide}}
  - {cell_markers: {out: marqueurs, input: fluide, distance: "@distance", fill_ratio: 0.55}}
  - {watershed_cells: {out: cellules, input: distance, markers: "@marqueurs", mask: "@fluide"}}
  - {cell_morphometry: {out: morphometrie, input: cellules}}
  - {throats: {out: cols, input: cellules}}
```

```bash
morphanalyzer run chaine.yaml volume.tif --project mon_echantillon/
```

Les valeurs `"@calque"` désignent un tableau du projet — c'est la seule façon de
passer une image en paramètre depuis un fichier de configuration.

Cette chaîne est celle que l'interface propose sous « cellules & cols », et un
test la rejoue à chaque exécution de la suite en vérifiant son IoU contre la
partition de Voronoï exacte. Une chaîne écrite quelque part sans être vérifiée
nulle part est une chaîne fausse en puissance.

!!! warning "Une poche de fluide sans marqueur reste à zéro"
    L'inondation ne peut pas l'atteindre. C'est correct, mais il faut le savoir
    avant de sommer des volumes. Vérifier :
    `cm["volume"].sum() / fluid.sum()`.

### Morphométrie

`d_equivalent` est le diamètre de la sphère de même volume — le « diamètre de
pore » de la thèse (figures 3.6 et 3.7).

`a`, `b`, `c` suivent la convention d'iMorph : `2·√λ` des valeurs propres de la
matrice de covariance des coordonnées. Attention à l'interprétation : pour un
ellipsoïde plein de demi-axes `A ≥ B ≥ C`, la covariance vaut `A²/5`, donc
`a = 2A/√5 ≈ 0,894 A`. Les **rapports** `a/b` et `b/c`, eux, sont exacts. Pour
retrouver les demi-axes géométriques, multiplier par `√5/2`.

Les cols sont comptés en **faces de voxels** partagées entre deux cellules, et
leur diamètre équivalent est celui du disque de même surface. La thèse établit
`Dcol = 0,53 · Dpore` (fig. 3.12) ; sur le fantôme de Voronoï on mesure 0,59.
