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
| `distance.distance_transform` | ⬜ P2 | `Thread/Granulometry/fastMarchManu.cpp::distFastMarching`, `calc_fdmapFast` | **`scipy.ndimage.distance_transform_edt` (exacte)**, pas le fast marching : la these mesure 2,77 voxels d'erreur max au 1er ordre (fig. 3.19). |
| `distance.travel_time` / `geodesic_distance` / `label_propagation` | ⬜ P2 | idem (1er/2nd ordre, champ de vitesse, `ManuLabelFastMarching`, `distFastIterativeMarching`) | `pykonal` (2nd ordre) ou `scikit-fmm` ; propagation etiquetee en Numba. |
| `granulometry.aperture_map` | ⬜ P3 | `Thread/Granulometry/morphology.cpp::calc_Aperture_Map3DFAHWithBall` (1 819 l.) | `porespy.filters.local_thickness` pour la carte. **La tolerance `apertureErrorPrecision` est commentee dans 3.2** : la reactiver (×10 en vitesse, ~2 % d'erreur, tableaux 2.4–2.5). |
| `granulometry.maximal_balls` / `cell_markers` | ⬜ P3 | `morphology.cpp` (image `id`), `utility.cpp::createBallsFromIdMap*`, `computeMaxBallsHistoFromIdMap` | **Sans equivalent** : l'image d'identifiants et le critere de boule quasi entiere (75 %) sont a ecrire (~400 l. Numba). |
| `segmentation.watershed_cells` | ⬜ P4 | `morphology.cpp::watershedBinarySearchTree` + `mostRepresenatedlabel` | **Sans equivalent** : priorites reelles (tas binaire) + collisions par label majoritaire. `skimage.segmentation.watershed` quantifie le relief → artefacts en marches d'escalier (fig. 3.4b vs 3.5b). |
| `segmentation.cell_morphometry` | ⬜ P4 | `Thread/Granulometry/morphometry.cpp` (2 198 l.) | `skimage.measure.regionprops` + `numpy.linalg.eigh`. **Supprimer `jacobi`/`eigsrt`/`svdcmp`** (Numerical Recipes, non redistribuable). |
| `segmentation.throats` / `connectivity` / `pore_network` | ⬜ P4 | `throatThread.cpp` (1 055 l.), `graph3D.cpp` (2 313 l.) | A arbitrer contre `porespy.networks.snow2` + OpenPNM apres comparaison numerique. |
| `skeleton.skeletonize` | ⬜ P5 | `Thread/Skeleton/thin3D.cpp` (3 680 l., classe `Doht`) | `skimage.morphology.skeletonize` (Lee 1994, meme LUT d'Euler) ou `kimimaro`. 3 680 lignes → un appel. |
| `skeleton.plateau_skeleton` | ⬜ P5 | `graph3D.cpp::Graph3D(waterInSolid, distInSolid, …)` | **Sans equivalent** (~200 l.) : voisinage 2×2×2, ≥ 4 labels ⇒ noeud, 3 labels ⇒ brin. |
| `skeleton.medial_axis_flux` | ⬜ P5 | `thin3D.cpp::gradientX/Y/Z` + `get_flux` | `numpy.gradient` + divergence (~60 l.). |
| `shape.local_shape_tensor` / `classify_solid` | ⬜ P5 | `Thread/ShapeClassif/skullSolidThread.cpp` (310 l.) + `shapeClassificationModule.cpp` (1 141 l.) | **Le cœur de valeur.** Covariance sur boule geodesique de rayon `expand_factor × ouverture locale`, `eigh`, a/b et b/c, seuil 1,6. Propagation au solide par `distance_transform_edt(return_indices=True)`. ~250 l. |
| `tortuosity.*` | ⬜ P6 | `Thread/Tortuosity/` (7 622 l.), `fastMarchManu.cpp::tortuosityFastMarch*` | `scipy.sparse.csgraph.dijkstra` pour le graphe ; fast marching avec champ de vitesse pour Poiseuille ; parallelepipedes inscrits pour la directionnelle. |

## Modules applicatifs — phases 7 a 9

| Module Python | Etat | Origine iMorph (compilee) | Strategie |
|---|:--:|---|---|
| `network.drainage` / `invasion_percolation` | ⬜ P7 | `PoreNetworkModelling/fullMorphoThread.cpp` (432 l., Hazlett + Hilpert), `invasionIPThread.cpp` (787 l.) | `porespy.filters.porosimetry` (= Hazlett), `porespy.simulations.drainage`, `porespy.filters.ibip`, OpenPNM. Essentiellement une couche d'adaptation. |
| `radiative.ray_trace` / `transmittance` / `reflectance` | ⬜ P8 | `iMorph_Rad/rayTracing.cpp` (1 054 l.), `rayTracingPileThread.cpp`, `utilityRayTracing.cpp` | `trimesh.ray` + `embreex`. Physique conservee (sphere integrante, reflexion speculaire, seuil a 1 %), moteur remplace par un BVH. |
| `cortical.*` | ⬜ P9 | `Thread/Cortical/` (5 491 l.) | numpy en coordonnees cylindriques + `scipy.spatial.Voronoi`. L'essentiel du volume C++ etait du QCustomPlot. |
| `viz.*` | ⬜ P11 | `Gui/` (35 081 l., dont 30 922 de tiers) | napari + matplotlib. **Jamais importe par le noyau** — verrouille par `tests/test_no_gui_imports.py`. |

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
