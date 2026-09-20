# Harnais « oracle » — sorties de reference d'iMorph 3.2

But : produire, sur les memes volumes que la suite de tests Python, les sorties
numeriques de l'implementation C++ d'origine, pour une comparaison voxel a
voxel. C'est ce qui transforme le portage en exercice verifiable.

## Etude de faisabilite (phase 0) — faite

**Le plan naif ne marche pas.** La cloture des `#include` de seulement cinq
fichiers de calcul (`morphology.cpp`, `fastMarchManu.cpp`, `utility.cpp`,
`heapVoxel.cpp`, `ball.cpp`) tire **481 fichiers et 165 527 lignes**, dont
**139 fichiers d'interface (67 880 lignes)** et tout QtWidgets/OpenGL. La cause
est unique : `utility.h` inclut `dataBaseWindow.h`, qui inclut `centralWidget.h`,
qui inclut l'application entiere.

**Mais les algorithmes eux-memes ne dependent pas de l'interface.** En
inspectant les corps de fonctions :

| Fichier | Symboles Qt utilises | Appels a l'interface |
|---|---|---:|
| `morphology.cpp` (1 819 l.) | `QString`, `qint64` | **0** |
| `fastMarchManu.cpp` (3 718 l.) | `QString`, `qint64` | **0** |
| `skullSolidThread.cpp` (310 l.) | `QObject`, `QString` | 0 (c'est un `QThread`) |
| `graph3D.cpp` (2 313 l.) | `QFile`, `QTextStream`, `QColor`, `QRgb`… | 0 (E/S et rendu, separables) |

Le couplage est donc **syntaxique, pas semantique** : il vient d'un en-tete trop
gros, pas du code numerique.

## Strategie retenue

Compiler les fichiers de calcul contre **QtCore seul**, en remplacant
`utility.h` par un en-tete reduit qui ne declare que ce qu'ils appellent
reellement.

1. `oracle_compat.h` — reprend tels quels les vrais en-tetes de structures
   (`image3D.h`, `roi.h`, `voxel.h`, `point3d.h`, `outputImage3D.h`,
   `heapVoxel*.h`, `fileAttenteHierarchique*.h`, `ball.h`) et **ne les modifie
   pas** : c'est la condition pour que l'oracle reste fidele.
2. `utility_min.h` — declare uniquement les fonctions de `utility.cpp` appelees
   par les fichiers cibles (`range3D`, `rangeROI`, `floodLabel*`,
   `createBallsFromIdMap*`, `createBlocks`…), sans toucher a l'interface.
3. `oracle_main.cpp` — lit un RAW, appelle une routine, ecrit le resultat en
   RAW + un JSON de description que `tests/` relit en numpy.

Ce qui reste a faire : etablir la liste exacte des symboles de `utility.h`
requis (un `nm -u` sur les `.o` de la version 2.6c, qui portent les memes
symboles, la donne directement), puis iterer sur les erreurs de compilation.
Compter une a deux journees, en phase 12.

## Alternative si cela resiste

Compiler l'application complete avec Qt 5.15 et lui ajouter un point d'entree
`--batch` : tout se compile deja (le binaire de 2017 en atteste), il suffit
d'instancier `QApplication` en mode `offscreen`, de construire
`Sample`/`Resolution`/`Roi`/`Phase` depuis un fichier, puis d'appeler
`Module::launch()`. Plus lourd a mettre en place, mais sans risque de deriver
de l'implementation d'origine.

## Construire

Qt5 n'est pas installable dans la VM locale (pas de privileges). Utiliser un
conteneur :

```bash
docker run --rm -it -v "$PWD":/w -w /w ubuntu:24.04 bash -lc '
  apt-get update -qq && apt-get install -y -qq qtbase5-dev cmake g++
  cmake -S tools/oracle -B tools/oracle/build -DIMORPH_ROOT=/chemin/vers/iMorph3.2
  cmake --build tools/oracle/build -j
'
```

Puis, cote Python, les tests marques `needs_oracle` comparent les sorties :

```bash
pytest -m needs_oracle
```
