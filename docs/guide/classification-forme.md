# Classification de forme

L'algorithme original d'iMorph, et sa pièce la plus différenciante. Il répond à
une question que la segmentation ne pose pas : en **tout point du solide**, à quel
type d'objet appartient-on — un nœud, un brin, une plaque ?

## Principe

En chaque point d'amorce :

1. une **boule géodésique** de rayon `expand_factor × ouverture locale`, confinée
   au solide (plancher à 3 voxels) ;
2. la **matrice de covariance** du nuage de voxels atteints ;
3. ses valeurs propres, triées, donnent les demi-axes de l'ellipsoïde équivalent

$$ a = 2\sqrt{\lambda_1} \ge b = 2\sqrt{\lambda_2} \ge c = 2\sqrt{\lambda_3} $$

4. les rapports `a/b` et `b/c` discriminent les trois formes de la figure 3.35 de
   la thèse ;
5. le premier vecteur propre donne l'orientation ;
6. la valeur est **propagée** à tout le solide, au plus proche point d'amorce.

| forme | signature | `a/b` | `b/c` |
|---|---|---|---|
| nœud | `a ≈ b ≈ c` | ≈ 1 | ≈ 1 |
| brin | `a ≫ b ≈ c` | grand | ≈ 1 |
| plaque | `a ≈ b ≫ c` | ≈ 1 | grand |

```python
st = ma.shape.local_shape_tensor(solid)  # expand_factor=3, comme iMorph
cls = ma.shape.classify_solid(st, solid)  # seuil a/b = 1,6, comme iMorph
st.to_frame()  # une ligne par point de mesure
ma.shape.strut_orientation(st, cls)  # histogrammes polaires
```

## Le seuil 1,6

La thèse le trouve empiriquement et le dit stable sur toutes les mousses
étudiées ; c'est le défaut d'iMorph (`rodeThreshold`). Sur le fantôme de Voronoï,
où les brins et les nœuds sont connus exactement, `a/b` aux points de mesure vaut

| | p10 | q1 | médiane | q3 | p90 |
|---|---:|---:|---:|---:|---:|
| brins | 1,38 | **1,62** | 2,33 | 3,75 | 4,61 |
| nœuds | 1,11 | 1,19 | 1,32 | **1,49** | 1,85 |

Le seuil tombe pile entre le troisième quartile des nœuds et le premier quartile
des brins. À 1,6 : **précision 0,95**, rappel 0,76.

!!! tip "Ce n'est pas le seuil qui maximise l'exactitude brute"
    1,2 à 1,4 font mieux sur ce critère. 1,6 est un choix de **haute précision** :
    ce qui est déclaré brin en est un dans 95 % des cas, au prix d'un quart des
    brins manqués. C'est le bon arbitrage quand on enchaîne sur des mesures de
    diamètres et d'orientations de brins — mieux vaut un échantillon plus petit
    que contaminé par des nœuds.

## Contrôles

Les trois formes canoniques, mesurées sur des fantômes :

| fantôme | `a/b` | `b/c` | élévation | classe rendue |
|---|---:|---:|---:|---|
| cylindres ∥ z | 3,25 | 1,04 | 90° | brin |
| plaque ⟂ z | 1,00 | 2,61 | 0° | plaque |
| sphère | 1,00 | 1,00 | — | nœud |

Contrôle analytique : pour un cylindre de rayon 4 avec une boule de rayon 12, la
théorie donne `a = 2R/√3 = 13,86` ; on mesure 13,29. Les demi-axes suivent
exactement la taille du voxel (rapport 3,000 pour un voxel triplé) alors que les
rapports restent adimensionnels.

## Deux classes ou trois ?

iMorph n'en distinguait que deux, sur le seul rapport `a/b` : `rodes` (brins) et
`plates` (tout le reste). `classify_solid` en propose trois en utilisant aussi
`b/c`, conformément à la figure 3.35.

!!! warning "Sur une mousse, la troisième classe capte les jonctions planes"
    Mesuré sur le fantôme de Voronoï : 40 % de brins, 22 % de nœuds et **37 % de
    « plaques »** — alors qu'une mousse à cellules ouvertes n'a aucune membrane.

    Ce n'est pas une erreur. Trois brins qui se rejoignent en Y sont localement
    coplanaires, donc `a ≈ b ≫ c` : la jonction *est* localement une plaque. La
    classe « plaque » capte donc les jonctions planes autant que les vraies
    membranes.

    Pour retrouver le comportement d'iMorph sur une mousse :

    ```python
    cls = ma.shape.classify_solid(st, solid, plate_threshold=None)  # 2 classes
    ```

    Garder les trois classes a du sens sur un milieu qui a réellement des parois
    — mousse à cellules fermées, os cortical, tissu lamellaire.

## Écarts assumés par rapport à iMorph

**La boule géodésique** passe par une dilatation contrainte intersectée avec la
boule euclidienne, au lieu d'un fast marching borné. Une dilatation itérée mesure
une distance de damier, donc sa boule serait un cube, ce qui biaiserait le tenseur
vers les axes de l'image ; l'intersection corrige. Le résultat est l'ensemble des
voxels à la fois connectés dans la phase et à moins du rayon — la même chose que
le fast marching partout où le voisinage local est convexe.

**Les coordonnées sont mises à l'échelle physique** avant la covariance, ce qui
rend `a`, `b`, `c` justes sur des voxels anisotropes. iMorph travaillait en voxels
et supposait l'isotropie.

**C'est bien une covariance, pas la matrice d'inertie** de l'équation 3.1 de la
thèse. Le code 3.2 (`calc_Dispersion`) assemble les variances et covariances des
coordonnées. Les deux partagent les mêmes axes propres mais pas les mêmes valeurs
propres ; c'est la version 3.2 qui fait foi.

## Choix des points d'amorce

Par défaut `seeds="auto"` : l'amincissement homotopique, avec repli sur la crête
de distance s'il ressort vide.

!!! bug "Un piège de scikit-image"
    `skimage.morphology.skeletonize(method="lee")` rend un squelette **vide** sur
    des objets pourtant élémentaires : un cube de 4×4×4, une sphère de rayon 12,
    une plaque, et même un cylindre isolé — alors que le même cylindre répété
    quatre fois dans la même boîte rend 174 voxels. Rembourrer le volume n'y
    change rien. Vérifié en 0.25.2 et figé par un test.

    D'où le repli sur `skeleton.distance_ridge`, qui n'est jamais vide, et qui
    correspond d'ailleurs à une variante d'iMorph (`Doht::TH_DOHT_BALLSCENTERS`,
    qui partait des centres de boules maximales).

Le choix de l'amorce ne change pas *ce qui est mesuré* en un point, seulement *où*
on mesure. Mais une amorce trop clairsemée dégrade la propagation : sur une
mousse, le squelette donne environ 3 % du volume solide en germes, la crête de
distance cinquante fois moins.

## Coût

Environ 5 s pour 2 700 points d'amorce sur un volume 128³, en pur numpy/scipy. Le
coût est dominé par les dilatations contraintes, une par point. C'est le candidat
évident à une accélération Numba si le besoin s'en fait sentir.
