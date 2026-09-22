# L'interface

```bash
pip install -e ".[web,fast]"          # depuis le dossier du paquet

# a) a partir d'un vrai volume : chemin d'une pile TIFF ou d'un repertoire d'images
morphanalyzer new mousse/ --volume /chemin/vers/ma_tomo/ --voxel-size 7.46 --binarize

# b) ou, pour essayer tout de suite, a partir d'un fantome
morphanalyzer new essai/ --phantom-kind voronoi_foam --shape 128

morphanalyzer serve essai/
```

!!! warning "Deux pieges au premier lancement"
    `--volume` attend un **chemin existant** — une pile TIFF ou un répertoire
    d'images. `tomo/` dans un exemple est un emplacement à remplacer, pas un nom
    à taper.

    Et `pip install "morphanalyzer[web]"` répond `does not provide the extra
    'web'` si l'installation éditable date d'avant l'ajout de cet extra : les
    métadonnées sont figées à l'installation. Relancer `pip install -e
    ".[web,fast]"` depuis le dossier du paquet.

Le navigateur s'ouvre sur le projet. Trois colonnes : les **calques** et le
**pipeline** à gauche, le **slicer** au centre, les **courbes** et
l'**historique** à droite.

## Le choix d'architecture, à l'envers de celui d'iMorph

Dans iMorph, chaque calcul *était* un morceau d'interface : un `CalcModule`
couplé à un `QThread`, une fenêtre de paramètres et le widget central. Rien ne
s'exécutait sans fenêtre, donc rien ne se scriptait, rien ne se testait, rien ne
tournait sur une machine de calcul.

Ici, **le serveur ne calcule rien**. Il expose trois choses qui existent sans
lui :

| | à quoi ça sert | utilisable seul |
|---|---|---|
| [`Project`](#le-projet-est-un-dossier) | le dossier de projet | oui, depuis un script |
| [`pipeline`](chaine-complete.md) | le registre des 50 étapes | oui, en YAML ou en Python |
| `webapp.render` | la composition d'une coupe en PNG | oui, c'est une fonction |

Conséquence pratique : ce qu'on fait à la souris se rejoue en lot, et
réciproquement.

## Le projet est un dossier

Pas de base propriétaire. Un projet, c'est :

```
mousse/
    morphanalyzer.json     manifeste : taille de voxel, calques, historique
    layers/volume.npy      le volume d'entrée
    layers/distance.npy    les cartes calculées
    tables/granulometrie.csv
    exports/
```

Tout se relit sans morphanalyzer — `.npy`, JSON, CSV — et se versionne. Les
calques sont **memmappés** : ouvrir un projet ne charge rien, et afficher une
coupe ne lit que cette coupe.

```python
from morphanalyzer import Project

proj = Project.open("mousse/")
proj.layers                    # ce qui est calculé
proj.layer("distance")[64]     # une coupe, sans tout charger
proj.table("granulometrie")    # un DataFrame
proj.history                   # ce qui a été lancé, et avec quels réglages
```

## Le slicer

Le navigateur ne reçoit **jamais** le volume : il demande une coupe composée, et
le serveur la rend en PNG là où les données sont déjà memmappées. Le coût ne
dépend donc pas de la taille du volume — un 1024³ en float32 pèse 4 Go, la coupe
pèse quelques dizaines de kilo-octets.

C'est ce qui rend l'interface utilisable **sur une machine de calcul distante** :

```bash
# sur la machine de calcul
morphanalyzer serve /data/mousse/ --host 127.0.0.1 --no-browser
# depuis le poste de travail
ssh -L 8000:localhost:8000 la-machine
```

Les calques s'empilent du fond vers le dessus, chacun avec sa rampe, son
fenêtrage et son opacité. Les rampes sont **à une seule teinte** : sur une carte
de distance, une rampe multi-teintes fabrique des frontières qui n'existent pas
dans les données.

| geste | effet |
|---|---|
| molette | zoom autour du curseur |
| glisser | déplacer |
| double-clic | ajuster à la fenêtre |
| ↑ ↓ | coupe suivante / précédente |
| 1 2 3 | axe Z / Y / X |

Le survol affiche la valeur de **tous** les calques sous le curseur — c'est là
qu'on vérifie qu'un calcul est correct.

## Lancer un pipeline

Le panneau de gauche liste les 50 étapes enregistrées, groupées par module, avec
leurs paramètres introspectés depuis les signatures Python. On empile des étapes
dans une file, on choisit le calque d'entrée, on exécute.

Quatre **chaînes types** évitent de tout monter à la main : granulométrie
(carte d'ouverture, distribution de taille, table des boules maximales et leur
image d'identifiants), cellules & cols, squelette de Plateau, drainage. Elles sont servies par
`/api/presets`, donc définies **une seule fois**, côté serveur — et la suite de
tests les rejoue contre la vérité terrain d'un fantôme. Une chaîne qui ne vit
que dans le JavaScript de l'interface n'est vérifiée par personne.

Une chaîne dont il manque une entrée est grisée : le squelette de Plateau
attend que les cellules existent.

### La phase

Un bouton **fluide / solide** commande la phase sur laquelle la chaîne tourne.
C'est le même geste que dans iMorph, qui donnait les deux : la granulométrie du
fluide est une distribution de taille de pore, celle du solide une distribution
d'épaisseur de brin.

Les deux jeux de résultats **coexistent**. Toutes les sorties portent le suffixe
de leur phase — `distance_fluide` et `distance_solide`, `granulometrie_fluide`
et `granulometrie_solide` — donc on peut les empiler dans le slicer et
superposer leurs courbes. Sans ce suffixe, le second calcul écrasait le premier
et on ne pouvait regarder qu'une carte.

Deux chaînes n'existent que sur le fluide : on ne draine pas une matrice, et la
loi de Plateau décrit les parois d'une mousse — son squelette vit dans le
solide, mais les cellules qui s'y rencontrent sont celles du fluide. C'est la
seule chaîne qui touche les deux phases à la fois, et elle insère le
`complement` qu'il lui faut.

La convention de la bibliothèque est **`True` = solide**, alors que presque
toute la chaîne morphologique se calcule dans le **fluide**. Le projet retient
donc ce que contient son calque `volume` — `phase: solid` par défaut — et les
chaînes types préfixent au besoin une étape `complement` qui produit la phase
demandée.

### Le champ « entrée »

Chaque étape reçoit par défaut la sortie de la précédente. Le champ **entrée**
permet de la faire repartir d'un calque nommé, ce qui est indispensable dès
qu'une étape a besoin d'un résultat plus ancien — le watershed veut la carte de
distance comme relief, pas les marqueurs que l'étape d'avant vient de produire.

Les paramètres qui attendent un **tableau** — les marqueurs d'un watershed, un
masque — se désignent par `@nom_de_calque`. Le menu à droite du champ les
insère.

Chaque étape range son résultat selon sa nature :

- un tableau 3D devient un **calque** ;
- un `DataFrame` devient une **table** ;
- un scalaire va dans le **journal** ;
- un objet composite est **décomposé** — un drainage donne sa carte de rayon
  d'envahissement *et* sa courbe de rétention ; un squelette de Plateau donne
  ses masques de nœuds et de brins *et* ses deux tables.

Une étape qui échoue n'interrompt pas ce qui précède : le journal garde ce qui a
réussi et affiche le message d'erreur complet.

## Rejouer en lot ce qu'on a fait à la souris

L'historique du projet est une configuration de pipeline. Le bouton
**exporter** du panneau pipeline la télécharge :

```yaml
steps:
  - {distance_transform: {out: distance}}
  - {cell_markers: {fill_ratio: 0.55, distance: "@distance", out: marqueurs}}
  - {watershed_cells: {markers: "@marqueurs", mask: "@volume", out: cellules}}
  - {cell_morphometry: {out: morphometrie}}
```

et on l'applique aux vingt autres échantillons :

```bash
for d in echantillons/*/ ; do
  morphanalyzer run chaine.yaml "$d/volume.tif" --project "$d"
done
```

## Courbes

Le panneau de droite trace n'importe quelle table du projet : on choisit
l'abscisse et une ou plusieurs ordonnées. Courbe ou barres, `log X`, `X
décroissant` — l'axe d'une courbe de rétention se lit de droite à gauche, la
pression montant quand le rayon descend.

Le survol donne les valeurs exactes, `table` affiche le tableau sous le
graphique, `CSV` le télécharge. Il n'y a **jamais deux axes d'ordonnées** : deux
grandeurs d'échelles différentes font deux graphiques.

## Les replis silencieux ne le sont pas

Un calcul qui se dégrade sans le dire est pire qu'un calcul qui échoue. Les
avertissements émis pendant une étape sont capturés et remontés : dans le journal
de la tâche, dans une notification, et dans l'historique du projet.

Le cas qui a motivé ce choix : sans numba, `watershed_cells` se replie sur
`skimage`, qui quantifie le relief et redonne les frontières en marches
d'escalier de la figure 3.4 de la thèse au lieu de la figure 3.5 — +15 % de
surface d'interface. Dans un terminal l'avertissement se voit ; dans une
interface, il fallait le remonter.

## Ce qui n'y est pas encore

Le rendu 3D et le graphe de Plateau interactif. Ils viendront ; le slicer et les
courbes sont ce qui sert à vérifier qu'un calcul est juste, donc ce qui devait
exister d'abord.
