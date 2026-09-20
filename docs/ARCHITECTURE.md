# Architecture

## La regle : bibliotheque d'abord, interface ensuite

iMorph etait une application Qt dans laquelle des calculs etaient enfouis. Chaque
grandeur passait par un `CalcModule`, un `QThread`, une `Param*Window` et la
`CentralWidget` ; rien ne s'executait sans fenetre. C'est ce qui rend le logiciel
impossible a scripter, a tester et a faire tourner sur une machine de calcul.

morphanalyzer inverse la dependance :

```
                  numpy / scipy / skimage
                            |
    +-----------------------+------------------------+
    |                       |                        |
  core                   modules de calcul       phantoms
  Volume, Roi            distance, granulometry   verite terrain
  neighborhood           segmentation, skeleton
    |                    shape, tortuosity, ...
    +-----------------------+------------------------+
                            |
              +-------------+-------------+
              |                           |
          pipeline                    (optionnel) viz
          CLI, YAML                    napari, matplotlib
```

Les fleches ne remontent jamais. `viz` depend du noyau ; le noyau ignore `viz`.

### Ce que cela implique concretement

1. **Toute routine est une fonction sur des tableaux.** Elle accepte un
   `Volume` ou un `ndarray` nu (`core.volume.as_array` normalise), et rend un
   tableau ou une valeur. Pas d'objet a construire, pas d'etat global, pas de
   fenetre.
2. **Aucun import graphique dans le noyau.** `tests/test_no_gui_imports.py`
   lance un sous-processus, importe tous les modules du noyau et echoue si
   `napari`, `qtpy`, `PyQt*`, `PySide*`, `matplotlib`, `pyvista`, `vtk`,
   `vispy` ou `magicgui` apparaissent dans `sys.modules`.
3. **Les dependances lourdes sont des extras.** Le noyau tient sur numpy,
   scipy, scikit-image, tifffile et pandas. numba, porespy, trimesh, zarr,
   napari sont optionnels et importes paresseusement via `_deps.require()`,
   qui dit quoi installer quand ca manque.
4. **Trois facons d'appeler, une seule implementation.** L'API Python est la
   reference ; `pipeline` l'enchaine de facon declarative ; `cli` expose les
   operations courantes au shell. Ni le pipeline ni la CLI ne contiennent de
   logique de calcul.

## Conventions

| Sujet | Choix | Pourquoi |
|---|---|---|
| Ordre des axes | `(z, y, x)` = `(k, j, i)` | Ordre des piles tomographiques et de scikit-image. iMorph indexait `data[k][j*width+i]` : meme ordre. |
| Binarisation | `True` = solide | Convention iMorph. Le fluide s'ecrit `~solid`, visible dans les signatures. |
| Taille de voxel | triplet `(dz, dy, dx)`, micrometres par defaut | Gere l'anisotropie des reconstructions. Les grandeurs physiques en dependent explicitement. |
| Labels | `0` = fond, `1..n` = objets | Convention `scipy.ndimage` / `skimage`. |
| Resultats tabulaires | `pandas.DataFrame` → Parquet | Remplace la base XML et les `.txt` maison. |
| Volumes | TIFF, RAW + sidecar JSON, OME-Zarr | Formats lisibles ailleurs. Plus de `.bin` proprietaire. |

## Verite terrain

Deux sources, dans cet ordre.

**Les fantomes** (`morphanalyzer.phantoms`) donnent la reponse exacte du continu.
`voronoi_foam` est le plus utile : les brins sont les aretes du diagramme de
Voronoi et les noeuds ses sommets, donc **la loi de Plateau y est vraie par
construction** et la partition de Voronoi fournit la verite terrain des
cellules, des cols, des brins et des noeuds. Aucun tomogramme ne permet cela.
`sinusoidal_tube` donne une tortuosite geometrique connue analytiquement,
`sphere_pack` une granulometrie qui doit etre un Dirac.

**Le harnais oracle** (`tools/oracle/`, phase 0–12) compile les fichiers de
calcul d'iMorph sans la GUI et dumpe leurs sorties sur les memes fantomes, pour
une comparaison voxel a voxel. Il tranche les cas ou une difference de
convention (connexite, arrondi, traitement des bords) fait diverger deux
implementations pourtant correctes.

S'y ajoutent, en dernier recours, les tableaux 2.3 a 3.3 de la these.

## Performance

Pas d'optimisation avant qu'un fantome ne passe. Ensuite, dans l'ordre :

1. **Vectoriser** — la plupart des boucles voxel d'iMorph deviennent des
   operations numpy.
2. **Numba** (`@njit(parallel=True)`) pour ce qui reste irreductiblement
   sequentiel : file d'attente hierarchique, watershed a priorites reelles,
   propagation par blocs.
3. **Hors-memoire** (`zarr` + `dask`) plutot que sous-echantillonner. La these
   mesurait des brins de 8 voxels faute de resolution (tableau 3.3) ; traiter
   2048³ change la donne.
4. **GPU** (`cupy`, `cucim`, `taufactor`) en dernier, et seulement la ou le
   profil le justifie.

iMorph parallelisait par `QThread` en decoupant le volume en blocs. La meme
strategie reste valable, mais `numba.prange` la rend triviale.
