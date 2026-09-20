# Carte de portage iMorph 3.2 → morphanalyzer

Une ligne par module Python : sa phase, son etat, les fichiers iMorph **reellement
compiles** dont il derive (verifies par `tools/inventory.py`, qui rejoue la
resolution `VPATH` de qmake), et la strategie retenue.

Etats : ✅ fait · 🔨 en cours · ⬜ a faire · 🚫 abandonne (ne pas porter)

## Socle — phase 0 et 1

| Module Python | Etat | Origine iMorph (compilee) | Strategie |
|---|:--:|---|---|
| `core.Volume`, `core.Roi`, `core.neighborhood` | ✅ | `DataType/image3D.h` (3 830 l.), `roi.cpp`, `phase.cpp`, `strElement.cpp` — 17 805 l. | `dataclass` autour d'un `ndarray`. Convention `(z, y, x)`, `voxel_size` physique. |
| `io.read_raw` / `write_raw` | ✅ | `Thread/Import/importRawThread.cpp`, `utility.cpp::saveEntireImage` | `np.memmap` pour le hors-memoire ; sidecar JSON de description. |
| `io.read_stack` / `write_stack` | ✅ | `importthread.cpp`, `newTsampledialog.cpp` | `tifffile` ; tri naturel des noms de coupes. |
| `io.read_table` / `write_table` | ✅ | base XML `iMorphDataBase.xml`, sorties `.txt` maison | `pandas` → Parquet/CSV. |
| `filters.*` | ✅ | `FilterModules/` (7 675 l.) | `scipy.ndimage` + `skimage.filters`. Rien d'original dans l'original. |
| `metrics.porosity` / `porosity_per_slice` / `open_porosity` | ✅ | `Thread/Granulometry/porosityThread.cpp` (597 l.), `fractionThread.cpp` | numpy + `scipy.ndimage.label`. |
| `metrics.specific_surface` | ✅ | `Thread/Mesh/mesh.cpp::DrawIsoSurface` + `calcSpecificSurface` | `skimage.measure.marching_cubes`. **Mesurer sur les niveaux de gris, pas sur le masque** : +9 % de biais sinon (cf. `tests/test_metrics.py`). |
| `metrics.representative_volume` | ✅ | these §2.1.4 (pas de fichier dedie) | tirage de boites + statistiques pandas. |
| `phantoms.*` | ✅ | — (nouveau) | Verite terrain analytique. `voronoi_foam` rend la loi de Plateau exacte par construction. |
| `pipeline`, `cli` | ✅ | `CalcModule` + `*Module`/`*Thread`/`Param*Window` (~8 000 l. d'orchestration Qt) | Registre de fonctions + YAML/JSON. Sans ecran. |
| `mesh.*` (export, decimation) | ⬜ P1 | `Thread/Mesh/mesh.cpp` (3 254 l.), `model.cpp` (1 398 l.) | `skimage` + `trimesh` (STL/OBJ/PLY, volume, aire, decimation quadrique). |

## Noyau algorithmique — phases 2 a 6

| Module Python | Etat | Origine iMorph (compilee) | Strategie |
|---|:--:|---|---|
| `distance.distance_transform` | ✅ | `Thread/Granulometry/fastMarchManu.cpp::distFastMarching`, `calc_fdmapFast` | **`scipy.ndimage.distance_transform_edt` (exacte)**, pas le fast marching : la these mesure 2,77 voxels d'erreur max au 1er ordre (fig. 3.19). |
| `distance.nearest_seed_propagation`, `geodesic_ball` | ✅ | idem (1er/2nd ordre, champ de vitesse, `ManuLabelFastMarching`, `distFastIterativeMarching`) | `pykonal` (2nd ordre) ou `scikit-fmm` ; propagation etiquetee en Numba. |
| `granulometry.aperture_map`, `pore_size_distribution`, `maximal_balls`, `cell_markers` | ✅ | `Thread/Granulometry/morphology.cpp::calc_Aperture_Map3DFAHWithBall` (1 819 l.) | `porespy.filters.local_thickness` pour la carte. **La tolerance `apertureErrorPrecision` est commentee dans 3.2** : la reactiver (×10 en vitesse, ~2 % d'erreur, tableaux 2.4–2.5). |
| — | — | `morphology.cpp` (image `id`), `utility.cpp::createBallsFromIdMap*`, `computeMaxBallsHistoFromIdMap` | **Sans equivalent** : l'image d'identifiants et le critere de boule quasi entiere (75 %) sont a ecrire (~400 l. Numba). |
| `segmentation.watershed_cells` | ✅ | `morphology.cpp::watershedBinarySearchTree` + `mostRepresenatedlabel` | **Sans equivalent** : priorites reelles (tas binaire) + collisions par label majoritaire. `skimage.segmentation.watershed` quantifie le relief → artefacts en marches d'escalier (fig. 3.4b vs 3.5b). |
| `segmentation.cell_morphometry` | ✅ | `Thread/Granulometry/morphometry.cpp` (2 198 l.) | `skimage.measure.regionprops` + `numpy.linalg.eigh`. **Supprimer `jacobi`/`eigsrt`/`svdcmp`** (Numerical Recipes, non redistribuable). |
| `segmentation.throats` / `connectivity` / `pore_network` | ✅ | `throatThread.cpp` (1 055 l.), `graph3D.cpp` (2 313 l.) | A arbitrer contre `porespy.networks.snow2` + OpenPNM apres comparaison numerique. |
| `skeleton.skeletonize`, `distance_ridge` | ✅ | `Thread/Skeleton/thin3D.cpp` (3 680 l., classe `Doht`) | `skimage.morphology.skeletonize` (Lee 1994, meme LUT d'Euler) ou `kimimaro`. 3 680 lignes → un appel. |
| `skeleton.plateau_skeleton`, `skeleton_graph` | ✅ | `graph3D.cpp::Graph3D(waterInSolid, distInSolid, …)` | **Sans equivalent** (~200 l.) : voisinage 2×2×2, ≥ 4 labels ⇒ noeud, 3 labels ⇒ brin. |
| `skeleton.medial_axis_flux` | ⬜ P5b | `thin3D.cpp::gradientX/Y/Z` + `get_flux` | `numpy.gradient` + divergence (~60 l.). |
| `shape.local_shape_tensor` / `classify_solid` / `strut_orientation` | ✅ | `Thread/ShapeClassif/skullSolidThread.cpp` (310 l.) + `shapeClassificationModule.cpp` (1 141 l.) | **Le cœur de valeur.** Covariance sur boule geodesique de rayon `expand_factor × ouverture locale`, `eigh`, a/b et b/c, seuil 1,6. Propagation au solide par `distance_transform_edt(return_indices=True)`. ~250 l. |
| `tortuosity.*` | ⬜ P6 | `Thread/Tortuosity/` (7 622 l.), `fastMarchManu.cpp::tortuosityFastMarch*` | `scipy.sparse.csgraph.dijkstra` pour le graphe ; fast marching avec champ de vitesse pour Poiseuille ; parallelepipedes inscrits pour la directionnelle. |

## Modules applicatifs — phases 7 a 9

| Module Python | Etat | Origine iMorph (compilee) | Strategie |
|---|:--:|---|---|
| `network.drainage` / `invasion_percolation` | ⬜ P7 | `PoreNetworkModelling/fullMorphoThread.cpp` (432 l., Hazlett + Hilpert), `invasionIPThread.cpp` (787 l.) | `porespy.filters.porosimetry` (= Hazlett), `porespy.simulations.drainage`, `porespy.filters.ibip`, OpenPNM. Essentiellement une couche d'adaptation. |
| `radiative.ray_trace` / `transmittance` / `reflectance` | ⬜ P8 | `iMorph_Rad/rayTracing.cpp` (1 054 l.), `rayTracingPileThread.cpp`, `utilityRayTracing.cpp` | `trimesh.ray` + `embreex`. Physique conservee (sphere integrante, reflexion speculaire, seuil a 1 %), moteur remplace par un BVH. |
| `cortical.*` | ⬜ P9 | `Thread/Cortical/` (5 491 l.) | numpy en coordonnees cylindriques + `scipy.spatial.Voronoi`. L'essentiel du volume C++ etait du QCustomPlot. |
| `viz.*` | ⬜ P11 | `Gui/` (35 081 l., dont 30 922 de tiers) | napari + matplotlib. **Jamais importe par le noyau** — verrouille par `tests/test_no_gui_imports.py`. |

## Ce que les phases 3, 4 et 5 ont etabli

### Granulometrie et segmentation (phases 3 et 4)

La chaine `distance -> boules maximales -> marqueurs -> watershed` est portee et
validee contre la partition de Voronoi exacte : **IoU median 0,91** sur les
cellules entierement incluses, 88 % au-dessus de 0,7. La degradation au-dela de
80 % de remplissage est bien la sous-segmentation que la these annonce
(fig. 3.3), et desactiver la conservation des boules de bord fait tomber l'IoU a
0,55 — les cellules de bord perdent leur germe et avalent leurs voisines.

Un ecart a signaler sur les **candidats** de la granulometrie : iMorph parcourait
tous les voxels du fluide, en elaguant ceux dont la boule est circonscrite a une
plus grande (code commente dans la version 3.2). On part des maxima regionaux de
la carte de distance, ce que la these decrit comme l'ensemble effectivement
retenu (« les points restants se situent pour la majorite sur le squelette des
boules maximales », fig. 2.19). Verifie sur un cas simple : les deux approches
donnent la meme boule maximale.

La morphometrie retrouve le regime de la these : `a/b = 1,31` contre 1,302 pour
la Recemat 1723 (tableau 3.1), `Dcol/Dpore = 0,59` contre 0,53 (fig. 3.12).

### Squelette de Plateau (phase 5)

Porte et valide : **ecart median 0,0 voxel** aux sommets de Voronoi exacts, 89 %
des noeuds a moins de 2 voxels, tous portant exactement 4 cellules, degre de
mode 3-4.

**Une amelioration sur l'original.** iMorph applique la meme inversion de relief
dans ses deux usages du watershed, donc propage les labels dans le solide en
partant de la crete de la carte de distance. En partant de l'interface — chaque
cellule croit depuis sa paroi a vitesse egale — les noeuds tombent nettement
mieux :

| sens d'inondation | ecart median | noeuds a <= 2 voxels |
|---|---:|---:|
| depuis la crete (iMorph) | 1,4 voxel | 62 % |
| depuis l'interface | **0,0 voxel** | **89 %** |

Sur un cas symetrique, une plaque de 4 voxels entre deux cellules se partage
800/800 en partant de l'interface et 1600/0 en partant de la crete : l'algorithme
est un parcours « meilleur d'abord », le premier front arrive sur la crete
l'emporte.

## Ce que la phase 5 a etabli pour le tenseur d'inertie

Le tenseur de forme local est porte et valide. Trois resultats.

**Les trois formes canoniques sortent justes** (`examples/shape_classification.py`) :

| fantome | a/b | b/c | elevation | classe rendue |
|---|---:|---:|---:|---|
| cylindres ∥ z | 3,25 | 1,04 | 90° | brin |
| plaque ⟂ z | 1,00 | 2,61 | 0° | plaque |
| sphere | 1,00 | 1,00 | — | noeud |

Controle analytique : pour un cylindre de rayon 4 avec une boule de rayon 12, la
theorie donne `a = 2R/sqrt(3) = 13,86` ; mesure 13,29. Les demi-axes suivent
exactement la taille du voxel (rapport 3,000 pour un voxel triple).

**Le seuil 1,6 de la these est confirme, et son role eclairci.** Sur une mousse
de Voronoi ou la loi de Plateau est exacte, `a/b` aux points de mesure vaut
2,33 en mediane sur les brins contre 1,32 sur les noeuds (q3 des noeuds : 1,49 ;
q1 des brins : 1,62 — le seuil tombe pile dans l'intervalle). A 1,6 :
**precision 0,95**, rappel 0,76. Ce n'est donc pas le seuil qui maximise
l'exactitude brute (1,2–1,4 fait mieux) mais celui qui garantit un echantillon
de brins propre — le bon arbitrage quand on enchaine sur des mesures de
diametres et d'orientations.

**Deux ecarts assumes par rapport a iMorph**, documentes et testes : la boule
geodesique passe par une dilatation contrainte intersectee avec la boule
euclidienne au lieu d'un fast marching borne (`distance.geodesic_ball`), et les
coordonnees sont mises a l'echelle physique avant la covariance, ce qui rend
`a`, `b`, `c` justes sur des voxels anisotropes.

**Un piege decouvert au passage.** `skimage.morphology.skeletonize(method="lee")`
rend un squelette **vide** sur des objets elementaires — cube de 4³, sphere de
rayon 12, plaque, cylindre isole — alors qu'il fonctionne sur un reseau de brins.
Rembourrer le volume n'y change rien. L'amorce par defaut est donc `"auto"` :
squelette, avec repli sur la crete de distance quand il ressort vide. Le
comportement de scikit-image est fige par un test, pour qu'un changement de
version se voie.

## 🚫 Ne pas porter — code present mais non compile

Verifie : ces fichiers ne sont ni dans le `.pro` ni inclus par un fichier compile.
Ce sont des experiences abandonnees. Les prendre pour du code vivant couterait
plusieurs centaines de milliers de tokens pour rien.

| Fichier | Lignes | Remarque |
|---|---:|---|
| `Thread/Segment/powerwatershedthread.cpp` | 665 | Power watershed (Couprie–Grady) — **jamais integre** |
| `Thread/Segment/maxflow.cpp` + `graphKolmo.cpp` + `arcs.cpp` | 873 | Graph cuts (Kolmogorov) — jamais integre |
| `Thread/Segment/graphsegmentationthread.cpp` | 441 | jamais integre |
| `Thread/Segment/watershedsegmentationthread.cpp` | 194 | jamais integre |
| `Thread/Simplificator/progmesh.cpp` + `simplificator.cpp` + `heapReductor.cpp` | 1 360 | Decimation progressive (Hoppe) — jamais integree |
| `iMorph_Rad/exchangeFactor.cpp` + `exchangeFactorModule.cpp` + `ExchangeFactorPileThread.cpp` | 1 531 | Facteurs d'echange radiatifs — jamais integres |
| `Thread/ShapeClassif/solidSegmentationThread.cpp` | 1 256 | **Variante morte** du pilote de classification ; le vivant est `shapeClassificationModule.cpp`. `skullSolidThread.cpp`, qui porte le calcul du tenseur, est bien compile. |
| `PoreNetworkModelling/fullMorphoThread_current.cpp` (+ 4 autres variantes) | 3 012 | Le compile est `fullMorphoThread.cpp` (432 l.) |
| `Gui/Segment/segmentationwindow.cpp` | 580 | fenetre de segmentation jamais branchee |
| `Thread/Granulometry/corticalModule*.cpp` (copies) | 3 631 | masques par les versions de `Thread/Cortical/` |

Seul `nlmeans_lib.cpp` (IPOL, GPL) est compile dans `Thread/Segment/`. **La
segmentation avancee d'iMorph 3.2 se reduit donc au debruitage NL-means** — tout
le reste du dossier est mort.

Total a ne pas porter : **209 fichiers, 89 229 lignes** (120 doublons + 89 morts).
Voir [INVENTAIRE_IMORPH32.md](INVENTAIRE_IMORPH32.md), regenerable par
`python tools/inventory.py <racine iMorph> --markdown docs/INVENTAIRE_IMORPH32.md`.
