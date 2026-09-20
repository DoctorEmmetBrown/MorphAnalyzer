"""Classification locale de forme par tenseur d'inertie.

L'algorithme original du logiciel, et sa piece la plus differenciante. En chaque
voxel du squelette : boule geodesique de rayon
``expand_factor * ouverture_locale`` (plancher a 3 voxels), matrice de
covariance du nuage de voxels atteints, valeurs propres triees, puis

    a = 2*sqrt(lambda1),  b = 2*sqrt(lambda2),  c = 2*sqrt(lambda3)

Les rapports a/b et b/c discriminent noeud (a~b~c), brin (a>>b~c) et plaque
(a~b>>c) ; le seuil empirique a/b = 1,6 separe brins et noeuds sur toutes les
mousses de la these. L'orientation vient du premier vecteur propre, exprimee en
azimut/elevation. La valeur calculee sur le squelette est ensuite propagee a
tout le solide par plus proche voxel de squelette — en Python cela se fait d'un
appel a ``scipy.ndimage.distance_transform_edt(..., return_indices=True)``.

Attention en portant : iMorph 3.2 assemble une matrice de **covariance**
(``calc_Dispersion``), pas la matrice d'inertie de l'equation 3.1 de la these.
Les deux partagent les memes axes propres mais pas les memes valeurs propres.
C'est la version 3.2 qui fait foi.

Phase de portage : 5.
Origine iMorph : Thread/ShapeClassif/solidSegmentationThread.cpp,
Thread/ShapeClassif/skullSolidThread.cpp

Ce module n'est pas encore implemente. Les signatures ci-dessous fixent le
contrat d'API : elles ne changeront pas sans raison, pour que les notebooks et
scripts ecrits maintenant restent valides.
"""

from __future__ import annotations

__all__ = ["local_shape_tensor", "classify_solid", "strut_orientation", "elongation_ratios"]


def _todo(name: str):
    raise NotImplementedError(
        f"morphanalyzer.shape.{name} arrive en phase 5. "
        "Voir docs/PORTING_MAP.md pour l'etat d'avancement."
    )
