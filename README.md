# morphanalyzer

Analyse morphologique 3D de milieux cellulaires et poreux — portage Python
d'[iMorph](https://imorph.sourceforge.net/) (J. Vicente & E. Brun, IUSTI).

> **Etat : phase 0.** Le socle est en place (types, E/S, filtres, porosite,
> surface specifique, VER, fantomes, pipeline, CLI). Les modules de calcul
> specifiques — granulometrie, segmentation des cellules, squelettes,
> classification par tenseur d'inertie, tortuosites, reseau de pores, radiatif —
> exposent leur API mais ne sont pas encore implementes. Voir
> [docs/PORTING_MAP.md](docs/PORTING_MAP.md).

## Principe : bibliotheque d'abord

iMorph ne savait rien faire sans sa fenetre. Ici c'est l'inverse : **toutes les
routines sont des fonctions sur des tableaux NumPy**, appelables depuis un
script, un notebook ou un job de calcul, sans ecran. La visualisation est un
extra optionnel que le noyau n'importe jamais — un test le verifie a chaque
execution de la suite.

```python
import morphanalyzer as ma

vol = ma.io.read_stack("tomo/", voxel_size=7.46)  # ou read_raw(...)
bin_ = ma.filters.threshold_otsu(vol)
bin_ = ma.filters.keep_largest_component(bin_)

print(ma.metrics.porosity(bin_))
print(ma.metrics.specific_surface(bin_), "um^-1")
```

Ou en declaratif, pour un traitement par lot reproductible :

```python
from morphanalyzer.pipeline import Pipeline

ctx = (
    Pipeline()
    .step("threshold_otsu")
    .step("keep_largest_component")
    .step("porosity", out="phi")
    .step("specific_surface", out="Sv")
    .run(vol)
)
```

Ou en ligne de commande :

```bash
morphanalyzer porosity tomo/ --voxel-size 7.46
morphanalyzer run analyse.yaml tomo/
morphanalyzer steps           # liste les etapes disponibles
```

## Installation

```bash
uv pip install -e ".[dev]"          # noyau + outils de developpement
uv pip install -e ".[all]"          # tout sauf le GPU
```

Le noyau n'exige que numpy, scipy, scikit-image, tifffile et pandas. Le reste
est reparti en extras : `fast` (numba, cc3d, edt), `fmm`, `bigdata` (zarr,
dask), `mesh`, `network` (porespy, openpnm), `rays`, `viz` (napari), `cli`,
`gpu`. Une dependance manquante donne un message qui dit quoi installer.

## Validation

Pas de tomogramme de reference pour l'instant : la validation repose sur des
**fantomes a verite terrain analytique** (`morphanalyzer.phantoms`). Le plus
utile est `voronoi_foam` — par construction la loi de Plateau y est exacte, donc
la partition de Voronoi fournit la verite terrain des cellules, des cols, des
brins et des noeuds. Aucun tomogramme ne permet cela.

S'y ajoutent, en phase 12, les tableaux 2.3 a 3.3 de la these et les sorties du
harnais C++ de reference (`tools/oracle/`).

## Licence

A trancher — voir [LICENSE](LICENSE). iMorph est sous CeCILL, dont la clause de
reciprocite contraint la licence de toute oeuvre derivee.
