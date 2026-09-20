"""Classification locale de forme par tenseur d'inertie.

L'algorithme original d'iMorph, et sa piece la plus differenciante. En chaque
voxel du squelette : boule geodesique de rayon ``expand_factor x ouverture
locale``, matrice de **covariance** du nuage de voxels atteints, valeurs propres
triees, puis

    a = 2*sqrt(lambda1) >= b = 2*sqrt(lambda2) >= c = 2*sqrt(lambda3)

Les rapports a/b et b/c discriminent noeud (a~b~c), brin (a>>b~c) et plaque
(a~b>>c). Le seuil `a/b = 1,6` separe brins et noeuds sur toutes les mousses de
la these, et c'est la valeur par defaut d'iMorph (`rodeThreshold`, avec
`expandFactor = 3`). L'orientation vient du premier vecteur propre.

La valeur calculee sur le squelette est ensuite propagee a tout le solide par
plus proche voxel de squelette.

Attention en portant : iMorph 3.2 assemble une matrice de **covariance**
(`calc_Dispersion`), pas la matrice d'inertie de l'equation 3.1 de la these.
Les deux partagent les memes axes propres mais pas les memes valeurs propres.
C'est la version 3.2 qui fait foi.

Phase de portage : 5.
Origine iMorph : `Thread/ShapeClassif/skullSolidThread.cpp` (calcul),
`Thread/ShapeClassif/shapeClassificationModule.cpp` (pilotage et propagation).
"""

from morphanalyzer.shape.tensor import (
    NODE,
    PLATE,
    STRUT,
    ShapeTensor,
    classify_solid,
    elongation_ratios,
    local_shape_tensor,
    shape_classification,
    strut_orientation,
)

__all__ = [
    "local_shape_tensor",
    "classify_solid",
    "shape_classification",
    "strut_orientation",
    "elongation_ratios",
    "ShapeTensor",
    "NODE",
    "STRUT",
    "PLATE",
]
