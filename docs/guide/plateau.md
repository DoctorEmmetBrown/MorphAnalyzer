# Squelette par loi de Plateau

Une squelettisation qui ne fait **aucun amincissement**, et qui produit
directement un graphe physiquement interprétable.

## La loi

Plateau (1873) : à l'interface de **quatre** cellules il y a un nœud, de **trois**
un brin, de **deux** un col. La méthode en tire un squelette en deux temps :

1. propager les labels de cellules **à l'intérieur du solide**, par ligne de
   partage des eaux sur la carte de distance au fluide ;
2. balayer le volume et compter les labels distincts dans un voisinage 2×2×2 :
   quatre ou plus, c'est un nœud ; exactement trois, c'est un brin.

Les brins relient ensuite les nœuds qui partagent trois de leurs quatre cellules.

```python
cells = ma.segmentation.watershed_cells(dist, markers, mask=fluid)
sk = ma.skeleton.plateau_skeleton(solid, cells)

sk.nodes  # node, z, y, x, n_voxels, cells, radius
sk.edges  # node_a, node_b, shared_cells, length
sk.node_mask  # voxels de nœud
sk.strut_mask  # voxels de brin
sk.labels_in_solid
```

## Pourquoi ce n'est pas un amincissement

Un amincissement topologique — le DOHT d'iMorph, l'algorithme de Lee — produit une
courbe centrée dont les branchements sont des accidents de la géométrie
discrétisée. Ici les nœuds sont des **jonctions réelles** : quatre cellules s'y
rencontrent, et on sait lesquelles. Le graphe qui en sort porte donc une
information physique, pas seulement topologique.

C'est aussi ce qui le rend vérifiable. Sur le fantôme de Voronoï, la loi de
Plateau est exacte par construction — les brins sont les arêtes du diagramme, les
nœuds ses sommets. On peut donc comparer non pas une plausibilité, mais un lieu.

## Résultats mesurés

Sur une mousse de Voronoï 128³ à 40 germes :

| grandeur | mesure |
|---|---|
| nœuds trouvés | 194 |
| brins trouvés | 383 |
| cellules par nœud | exactement 4 pour 100 % des nœuds |
| degré des nœuds | mode 3–4, moyenne 3,4 |
| écart au sommet de Voronoï exact | **médiane 0,0 voxel**, p90 2,9 |
| nœuds à 2 voxels ou moins d'un vrai sommet | **89 %** |
| longueur de brin médiane | 15 voxels (espacement des germes : 20) |

Le degré ne vaut pas 4 partout parce qu'un nœud de degré 4 exige que ses quatre
voisins existent aussi : au bord de la boîte il manque des cellules, donc des
nœuds, donc des arêtes.

## Le sens d'inondation n'est pas une convention neutre

iMorph applique la même inversion de relief dans ses deux usages du watershed,
donc propage les labels dans le solide en partant de la **crête** de la carte de
distance. `plateau_skeleton` part de l'**interface** : chaque cellule croît depuis
sa propre paroi à vitesse égale, et deux cellules se rejoignent sur la surface
médiane du solide — qui est justement la surface de Plateau.

La différence est mesurable :

| sens d'inondation | écart médian au vrai sommet | nœuds à ≤ 2 voxels |
|---|---:|---:|
| depuis la crête | 1,4 voxel | 62 % |
| depuis l'interface | **0,0 voxel** | **89 %** |

Sur un cas symétrique le contraste est plus parlant encore : une plaque de
4 voxels entre deux cellules se partage 800/800 en partant de l'interface, et
**1600/0** en partant de la crête. La raison est que l'algorithme est un parcours
« meilleur d'abord » sur le relief, pas une immersion synchronisée par niveaux :
le premier front arrivé sur la crête l'emporte ensuite largement.

## Limites

**La qualité du squelette dépend entièrement de celle de la segmentation des
cellules.** La thèse le dit explicitement. Une sur-segmentation crée de faux
nœuds, une sous-segmentation en supprime. C'est le prix de la méthode, et son
avantage : le squelette hérite du sens physique des cellules.

**Au bord de l'échantillon, on ne peut pas trouver de nœud**, puisqu'il y manque
des cellules. La thèse propose deux remèdes : détecter sur les faces les points à
l'interface de trois cellules, ou travailler sur un volume plus grand puis
recouper. Aucun des deux n'est implémenté ; les nœuds de bord manquent, et cela se
voit dans la distribution des degrés.

**Le débruitage** fusionne les nœuds plus proches que `merge_distance` (défaut 3,
la valeur d'iMorph `cleanDistancePlateau`) **et portant exactement les mêmes
quatre cellules** — la condition compte, sans elle on fusionnerait des jonctions
distinctes.
