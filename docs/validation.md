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
| `cortical_tube` | canaux longitudinaux : porosité par secteur et par couronne exactes, nombre de canaux exact |

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

| taux de remplissage | marqueurs | IoU médian | IoU > 0,7 | fragments |
|---:|---:|---:|---:|---:|
| 0,40 | 73 | 0,908 | 88 % | 10 |
| 0,55 | 70 | 0,908 | 88 % | 9 |
| 0,65 | 68 | 0,885 | 62 % | 9 |
| 0,85 | 56 | 0,730 | 50 % | 9 |

La dégradation au-delà de 0,8 est la sous-segmentation que la thèse annonce
(fig. 3.3).

### Le taux de remplissage ne servait à rien

Défaut mesuré tard, signalé par l'usage : la sur-segmentation persistait quelle
que soit la valeur de `fill_ratio`. La cause est que les boules coupées par le
bord étaient **exemptées du test** (`keep_border_balls=True`, comme
`isUseBallsAtFace` d'iMorph), et qu'elles sont nombreuses : le seuil ne
s'appliquait qu'à une minorité de candidats, et 0,45 comme 0,75 rendaient les
mêmes 68 marqueurs.

L'exemption compensait un artefact de mesure : le taux était calculé contre la
sphère entière, donc une boule de bord parfaitement inscrite dans son pore
affichait 0,42 parce que la moitié d'elle sortait de l'image. Le volume
théorique est maintenant écrêté à la boîte ; la médiane des boules de bord passe
de **0,42 à 0,99**, le seuil s'applique à tout le monde et le défaut de
`keep_border_balls` devient `False`.

À marqueurs mesurés sur trois mousses (128³, seuil 0,65), fragments = cellules
prédites de moins de 15 % du volume médian :

| mousse | exemption (ancien) | écrêtage (nouveau) |
|---|---:|---:|
| 80 cellules | 68 marqueurs, 12 fragments | 54 marqueurs, **7** |
| 106 cellules | 84 marqueurs, 17 fragments | 68 marqueurs, **9** |
| 212 cellules | 172 marqueurs, 33 fragments | 136 marqueurs, **11** |

L'IoU médian des cellules intérieures est inchangé (0,907 → 0,908). Ce sont des
faux germes qui disparaissent, pas des cellules.

### Le repli sans numba reproduit la figure 3.4

Contrôle involontaire mais concluant : exécuté dans un environnement sans numba,
le portage se replie sur `skimage.segmentation.watershed` — l'algorithme de Meyer
à file hiérarchique — et produit **exactement** l'artefact que la figure 3.4 de la
thèse documente, des frontières en marches d'escalier sur les lignes de partage
obliques. La figure 3.5, obtenue avec le tas binaire à priorités réelles et la
résolution de collisions par label majoritaire, est ce que rend le noyau numba.

Sur une mousse de Voronoï de 128³, à marqueurs et carte de distance identiques :

| | IoU médian | surface d'interface | voxels attribués autrement |
|---|---:|---:|---:|
| tas binaire, priorités réelles (fig. 3.5) | **0,908** | référence | — |
| file hiérarchique, relief quantifié (fig. 3.4) | 0,892 | **+15 %** | **9,1 %** |

L'IoU bouge peu parce qu'il mesure un recouvrement de volume, dominé par
l'intérieur des cellules. C'est la **surface d'interface** qui voit la
différence : +15 %, c'est la longueur ajoutée par les marches.

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

### Tortuosité

Le milieu libre, avec `reference="free_front"`, donne **exactement 1,0000** : la
normalisation par le même solveur dans une boîte vide annule le biais du premier
ordre, qui vaut sinon +8,8 % (le carré de +3,8 % sur la diagonale 3D).

Le tube sinusoïdal se compare à la tortuosité de sa ligne centrale, calculée par
longueur d'arc à la précision machine : la mesure reste entre 1,02 et cette
valeur, et croît avec l'amplitude.

### Drainage

L'effet « bouteille d'encre » est reproduit par les deux algorithmes : une
chambre de rayon 11 derrière un col de rayon 3,6 n'est jamais envahie au-delà du
rayon du col. Et Hilpert n'est jamais plus permissif que Hazlett, voxel par voxel
et rayon par rayon.

Sur un tube droit, le remplissage est uniforme et la courbe de rétention n'a
qu'un point, à la saturation 1.

### Percolation d'invasion

Le rayon d'envahissement décroît le long de l'ordre d'invasion — l'escalier
monotone que garantissait la boucle en pression d'iMorph. Le piégeage ne fait que
retirer des cellules, jamais en ajouter, et aucune cellule n'est à la fois
envahie et piégée. Des cols déformables font monter le rayon final : le milieu
s'envahit à plus basse pression.

### Os cortical

Sur `cortical_tube`, la porosité totale rendue par `angular_profile` retombe
**exactement** sur la valeur exacte du fantôme, et le secteur chargé est bien
celui dont le poids a été augmenté. Le profil est constant en `z` sur un fantôme
invariant en `z`, à la précision machine.

La correction d'ellipse divise par plus de deux la dispersion des aires de
secteurs sur une ellipse de rapport 60/25. Les angles « iso-volume » équilibrent
les secteurs à moins de 2 %.

La connectivité empilée compte exactement les canaux parallèles, et un pont
introduit à la coupe 30 fait bien passer le compte de 2 à 1 à cette coupe-là.

Le Voronoï 2D place la frontière entre deux canaux symétriques sur leur
médiatrice, et étiquette exactement la matrice — ni plus, ni moins.

### Maillage

Sphère de rayon 20 : aire à **−0,08 %** de l'analytique en maillant le champ de
distance signée, contre **+9,3 %** en maillant le masque binaire. Le maillage est
invariant par changement d'échelle (aire ×9, volume ×27 pour un voxel triple).

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

Le notebook `notebooks/tutoriel.ipynb` rejoue l'essentiel de ces mesures,
figures comprises.

```bash
pytest -q                                  # 276 tests
python examples/shape_classification.py    # les chiffres de la classification
python examples/full_chain.py              # la chaîne complète
python examples/drainage_cortical.py       # tortuosité, drainage, os cortical
python tools/build_tutorial.py             # régénère le notebook tutoriel
morphanalyzer serve mon_projet/            # les mêmes résultats dans le navigateur
```
