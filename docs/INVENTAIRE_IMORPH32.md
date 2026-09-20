# Inventaire iMorph 3.2 — ce qui est reellement compile

Genere par `tools/inventory.py` depuis `imorph_32.pro`.

- fichiers declares dans le `.pro` : **457**
- resolus dans l'arborescence : **457**
- introuvables : **0**
- doublons non compiles : **120** fichiers  66 527 lignes
- autres fichiers ni compiles ni inclus : **89** fichiers  22 702 lignes

## Lignes par domaine (code iMorph seul, tiers exclus)

| domaine | dossier | fichiers | lignes | dont tiers |
|---|---|---:|---:|---:|
| interface | `Gui` | 134 | 35 081 | 30 922 |
| types de donnees | `DataType` | 61 | 17 805 | 0 |
| filtres | `FilterModules` | 52 | 7 675 | 0 |
| radiatif | `PhysicalModules/iMorph_Rad` | 22 | 4 853 | 463 |
| reseau de pores | `PhysicalModules/PoreNetworkModelling` | 13 | 2 461 | 0 |
| cortical | `Thread/Cortical` | 10 | 5 491 | 0 |
| sections | `Thread/CrossSection` | 10 | 1 818 | 298 |
| granulometrie / distance / morphometrie | `Thread/Granulometry` | 97 | 28 222 | 0 |
| maillage | `Thread/Mesh` | 9 | 5 829 | 0 |
| segmentation avancee | `Thread/Segment` | 2 | 0 | 1 021 |
| classification de forme | `Thread/ShapeClassif` | 7 | 2 053 | 0 |
| squelette | `Thread/Skeleton` | 8 | 4 827 | 0 |
| tortuosite | `Thread/Tortuosity` | 33 | 7 622 | 0 |
| import | `Thread/Import` | 14 | 1 381 | 203 |
| export | `Thread/Export` | 2 | 312 | 0 |

**Total compile : 158 337 lignes**  dont 32 907 de bibliotheques tierces.

## Fichiers masques (meme nom dans deux dossiers du VPATH)

qmake retient la premiere occurrence dans l'ordre des `VPATH`. La colonne « ignore » est du code mort, a ne pas porter.

| nom | compile | ignore |
|---|---|---|
| `corticalModule.cpp` | `Sources/Thread/Cortical/corticalModule.cpp` | `Sources/Thread/Granulometry/corticalModule.cpp` |
| `corticalModuleTabAngularAper.cpp` | `Sources/Thread/Cortical/corticalModuleTabAngularAper.cpp` | `Sources/Thread/Granulometry/corticalModuleTabAngularAper.cpp` |
| `corticalModuleTabPorosity.cpp` | `Sources/Thread/Cortical/corticalModuleTabPorosity.cpp` | `Sources/Thread/Granulometry/corticalModuleTabPorosity.cpp` |
| `hysteresisThread.cpp` | `Sources/FilterModules/New/hysteresisThread.cpp` | `Sources/Thread/Granulometry/hysteresisThread.cpp` |
| `hysteresisThread.h` | `Sources/FilterModules/New/hysteresisThread.h` | `Sources/Thread/Granulometry/hysteresisThread.h` |
| `importFromBallFileThread.h` | `Sources/Thread/Import/importFromBallFileThread.h` | `Sources/DataType/importFromBallFileThread.h` |
| `skullSolidThread.cpp` | `Sources/Thread/ShapeClassif/skullSolidThread.cpp` | `Sources/Thread/Skeleton/skullSolidThread.cpp` |
| `skullSolidThread.h` | `Sources/Thread/ShapeClassif/skullSolidThread.h` | `Sources/Thread/Skeleton/skullSolidThread.h` |

## Fichiers ni compiles ni inclus — a ne pas porter

Ces fichiers ne figurent pas dans le `.pro` et aucun fichier compile ne les
inclut : ce sont des experiences abandonnees ou des variantes de travail.
Les plus volumineux d'abord.

| lignes | fichier |
|---:|---|
| 1585 | `Sources/Thread/Granulometry/corticalModuleTabPorosity.cpp` |
| 1536 | `Sources/Thread/Granulometry/corticalModuleTabAngularAper.cpp` |
| 1256 | `Sources/Thread/ShapeClassif/solidSegmentationThread.cpp` |
| 974 | `Sources/PhysicalModules/PoreNetworkModelling/fullMorphoThread_current.cpp` |
| 946 | `Sources/PhysicalModules/PoreNetworkModelling/fullMorphoThread_ObjetsPercolesOPTI.cpp` |
| 897 | `Sources/Thread/Simplificator/progmesh.cpp` |
| 752 | `Sources/Gui/Import/baseResolutionDialog.cpp` |
| 668 | `Sources/Thread/Segment/maxflow.cpp` |
| 665 | `Sources/Thread/Segment/powerwatershedthread.cpp` |
| 580 | `Sources/Gui/Segment/segmentationwindow.cpp` |
| 578 | `Sources/PhysicalModules/iMorph_Rad/exchangeFactor.cpp` |
| 552 | `Sources/Thread/Granulometry/heapVoxelArray.h` |
| 546 | `Sources/PhysicalModules/iMorph_Rad/ExchangeFactorPileThread.cpp` |
| 510 | `Sources/Thread/Granulometry/corticalModule.cpp` |
| 501 | `Sources/Thread/Segment/graphKolmo.h` |
| 441 | `Sources/Thread/Segment/graphsegmentationthread.cpp` |
| 417 | `Sources/Gui/2D/fluidSegmentationWindow.cpp` |
| 407 | `Sources/PhysicalModules/iMorph_Rad/exchangeFactorModule.cpp` |
| 364 | `Sources/Thread/Granulometry/fluidSegmentationThread.cpp` |
| 361 | `Sources/PhysicalModules/PoreNetworkModelling/fullMorphoThread_watershed.cpp` |
| 337 | `Sources/Thread/Granulometry/analyseThroatMorphology.h` |
| 321 | `Sources/Gui/MainWindow/sampleDescriberHeader.cpp` |
| 315 | `Sources/PhysicalModules/PoreNetworkModelling/fullMorphoThread_entire_dilate.cpp` |
| 305 | `Sources/Thread/Skeleton/skullSolidThread.cpp` |
| 305 | `Sources/PhysicalModules/PoreNetworkModelling/fullMorphoThread_parial_dilatation.cpp` |
| 305 | `Sources/Gui/Segment/fluidSegmentationImageViewer.cpp` |
| 291 | `Sources/Thread/Granulometry/heapVoxelLight.h` |
| 280 | `Sources/Gui/Import/baseResolutionDialog.h` |
| 268 | `Sources/Thread/Segment/block.h` |
| 247 | `Sources/Thread/Simplificator/simplificator.cpp` |
| 225 | `Sources/Thread/Granulometry/granuloModule.cpp` |
| 219 | `Sources/Thread/Granulometry/granuloThread.cpp` |
| 216 | `Sources/Thread/Simplificator/heapReductor.cpp` |
| 212 | `Sources/Thread/Granulometry/heapVoxel.cpp` |
| 194 | `Sources/Thread/Segment/watershedsegmentationthread.cpp` |
| 193 | `Sources/Thread/Granulometry/fileAttenteHierarchiqueCompact.h` |
| 193 | `Sources/Gui/3D/watershedWindow.cpp` |
| 192 | `Sources/Gui/Common/slices3dViewer.cpp` |
| 147 | `Sources/FilterModules/old_obsolete/filterSingleComponent.cpp` |
| 141 | `Sources/PhysicalModules/PoreNetworkModelling/graph3DSolid.cpp` |
| 136 | `Sources/Gui/MainWindow/tableSampleDescriber.cpp` |
| 132 | `Sources/FilterModules/old_obsolete/filterDilateBinary.cpp` |
| 129 | `Sources/Thread/Simplificator/listReductor.h` |
| 127 | `Sources/Thread/ShapeClassif/solidSegmentationThread.h` |
| 122 | `Sources/Thread/Segment/graphKolmo.cpp` |
| 121 | `Sources/FilterModules/old_obsolete/filterErodeBinary.cpp` |
| 116 | `Sources/FilterModules/old_obsolete/filterMedian.cpp` |
| 108 | `Sources/Thread/Granulometry/fluidSegmentationThread.h` |
| 107 | `Sources/Thread/Simplificator/vector.cpp` |
| 104 | `Sources/Thread/Cortical/corticalRoiModule.cpp` |
| 103 | `Sources/Gui/Segment/segmentationwindow.h` |
| 102 | `Sources/Gui/MainWindow/sampleViewer.cpp` |
| 100 | `Sources/Thread/Skeleton/skullSolidThread.h` |
| 83 | `Sources/Thread/Segment/arcs.cpp` |
| 83 | `Sources/FilterModules/old_obsolete/filterDilate.cpp` |
| 82 | `Sources/Gui/Common/slices3dViewer.h` |
| 82 | `Sources/FilterModules/old_obsolete/filterErode.cpp` |
| 79 | `Sources/Thread/Simplificator/progmesh.h` |
| 79 | `Sources/Thread/Granulometry/hysteresisThread.h` |
| 77 | `Sources/FilterModules/old_obsolete/filterMyFilter.cpp` |
| 76 | `Sources/Gui/2D/visualizer2D.h` |
| 72 | `Sources/Gui/3D/visualizer3D.cpp` |
| 71 | `Sources/DataType/voxelLabel.h` |
| 68 | `Sources/Thread/Simplificator/vector.h` |
| 68 | `Sources/Gui/2D/visualizer2D.cpp` |
| 68 | `Sources/DataType/mask2.h` |
| 68 | `Sources/DataType/importFromBallFileThread.h` |
| 63 | `Sources/DataType/mask2.cpp` |
| 54 | `Sources/Gui/MainWindow/tableSampleDescriber.h` |
| 54 | `Sources/DataType/voxelDist.cpp` |
| 53 | `Sources/Thread/Segment/graphsegmentationthread.h` |

Et 18 fichiers de moins de 50 lignes.
