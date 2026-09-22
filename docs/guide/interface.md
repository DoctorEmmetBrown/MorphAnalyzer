# L'interface

```bash
pip install "morphanalyzer[web,fast]"
morphanalyzer new mousse/ --volume tomo/ --voxel-size 7.46 --binarize
morphanalyzer serve mousse/
```

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

Trois **chaînes types** évitent de tout monter à la main : granulométrie,
cellules & cols, drainage.

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

## Ce qui n'y est pas encore

Le rendu 3D et le graphe de Plateau interactif. Ils viendront ; le slicer et les
courbes sont ce qui sert à vérifier qu'un calcul est juste, donc ce qui devait
exister d'abord.
