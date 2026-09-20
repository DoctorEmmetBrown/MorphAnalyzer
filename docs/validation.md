# Validation

Il n'y a pas de tomogramme de référence dans ce dépôt. La validation repose sur
des **fantômes à vérité terrain analytique** : des volumes synthétiques dont on
connaît la réponse exacte.

C'est un choix, pas un pis-aller. Sur un tomogramme, on ne peut comparer qu'à une
autre implémentation, donc propager ses erreurs. Sur un fantôme, on compare à la
géométrie.

## Les fantômes

```python
vol = ma.phantoms.voronoi_foam(shape=(128,) * 3, n_cells=40, strut=3.0, seed=7)
vol.meta["truth"]  # tout ce qui est connu exactement
```

| fantôme | vérité terrain |
|---|---|
| `sphere` | volume et surface du continu, champ de distance signée |
| `sphere_pack` | sphères disjointes : volume, surface, granulométrie = Dirac |
| `cylinders` | classe « brin » attendue, orientation connue |
| `plate` | classe « plaque » attendue |
| `straight_tube` | tortuosité exactement 1 |
| `sinusoidal_tube` | tortuosité par longueur d'arc, à la précision machine |
| `voronoi_foam` | **cellules, cols, brins et nœuds exacts** |

### La mousse de Voronoï

C'est la pièce maîtresse. Par construction, le solide occupe le voisinage des
**arêtes** du diagramme de Voronoï : un voxel est solide si `d₃ − d₁ < strut`, où
`d₁ ≤ d₂ ≤ d₃` sont les distances aux trois germes les plus proches. Il s'ensuit
que :

- les brins sont les arêtes (3 germes équidistants) ;
- les nœuds sont les sommets (4 germes équidistants, `d₄ − d₁ < strut`) ;
- les cols sont les faces (2 germes) ;
- **la loi de Plateau est exacte** ;
- la partition de Voronoï est la segmentation exacte des cellules.

Aucun tomogramme ne permet cela. La vérité terrain porte donc sur la segmentation
*et* sur le squelette *et* sur la classification de forme, simultanément.

## Résultats

### Grandeurs macroscopiques

| grandeur | référence | mesure | écart |
|---|---|---|---|
| porosité, sphère | analytique | — | < 0,2 % |
| surface spécifique, masque binaire | analytique | surestimée | **+ 9,3 %** |
| surface spécifique, champ de gris | analytique | — | **0,05 %** |

!!! tip "Conséquence directe"
    Mesurer la surface spécifique sur les niveaux de gris, jamais sur le masque.
    L'iso-surface à 0,5 d'un masque est un escalier, et l'interpolation linéaire
    de marching cubes ne peut pas retrouver une interface qui n'est plus dans les
    données.

### Segmentation des cellules

Sur les cellules entièrement incluses d'une mousse de Voronoï 128³ :

| taux de remplissage | marqueurs | IoU médian | IoU > 0,7 |
|---:|---:|---:|---:|
| 0,40 | 85 | 0,908 | 88 % |
| 0,55 | 85 | 0,908 | 88 % |
| 0,65 | 84 | 0,885 | 62 % |
| 0,85 | 82 | 0,731 | 50 % |

La dégradation au-delà de 0,8 est la sous-segmentation que la thèse annonce
(fig. 3.3). Désactiver `keep_border_balls` fait tomber l'IoU médian à 0,55 :
les cellules de bord perdent leur germe et avalent leurs voisines.

### Classification de forme

Les trois formes canoniques de la figure 3.35 :

| fantôme | `a/b` | `b/c` | élévation | classe |
|---|---:|---:|---:|---|
| cylindres ∥ z | 3,25 | 1,04 | 90° | brin |
| plaque ⟂ z | 1,00 | 2,61 | 0° | plaque |
| sphère | 1,00 | 1,00 | — | nœud |

Le seuil 1,6 sur la mousse : précision **0,95**, rappel 0,76.

### Squelette de Plateau

Écart médian au sommet de Voronoï exact : **0,0 voxel**. 89 % des nœuds à moins de
2 voxels. Degré de mode 3–4. Tous les nœuds portent exactement 4 cellules.

### Morphométrie

Comparaison de régime avec le tableau 3.1 de la thèse (Recemat NC-1723) :

| grandeur | thèse (mousse réelle) | fantôme de Voronoï |
|---|---:|---:|
| `a/b` des cellules | 1,302 | 1,31 |
| `b/c` des cellules | 1,232 | 1,17 |
| `Dcol / Dpore` | 0,53 | 0,59 |

Le fantôme n'a aucune raison de reproduire ces valeurs exactement : ce sont deux
matériaux différents. L'accord de régime indique en revanche que la mesure se
comporte comme celle de la thèse.

## Le harnais « oracle »

La seconde source de vérité terrain, prévue en phase 12 : compiler les fichiers de
calcul d'iMorph sans son interface et comparer voxel à voxel. L'étude de
faisabilité est faite, et `tools/oracle/README.md` en donne le résultat — le
couplage à l'interface est syntaxique, pas sémantique, donc l'opération est
possible mais demande un en-tête réduit.

Ce harnais trancherait les cas où une différence de convention (connexité,
arrondi, traitement des bords) fait diverger deux implémentations pourtant
correctes. Les fantômes, eux, tranchent les cas où l'une des deux est fausse.

## Reproduire

```bash
pytest -q                                  # 144 tests
python examples/shape_classification.py    # les chiffres de la classification
python examples/full_chain.py              # la chaîne complète
```
