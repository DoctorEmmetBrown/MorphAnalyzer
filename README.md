# morphanalyzer

Analyse morphologique 3D de milieux cellulaires et poreux — portage Python
d'[iMorph](https://imorph.sourceforge.net/) (J. Vicente & E. Brun, IUSTI).

> **Etat : phase 0 faite, coeur de la phase 5 livre.** Socle complet (types,
> E/S, filtres, porosite, surface specifique, VER, fantomes, pipeline, CLI),
> plus la carte de distance exacte, la granulometrie (carte d'ouverture),
> la squelettisation et la **classification locale de forme par tenseur
> d'inertie** — l'algorithme original d'iMorph, valide contre des fantomes a
> verite terrain analytique. Restent la segmentation des cellules, les
> tortuosites, le reseau de pores, le radiatif et le cortical : leur API est
> declaree et leur appel dit a quelle phase elle arrive. Voir
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

## Classification de forme

L'algorithme original du logiciel : en chaque point du squelette, la matrice de
covariance du nuage de voxels atteints dans une boule geodesique de rayon
`expand_factor x ouverture locale`, dont les valeurs propres donnent les
demi-axes `a >= b >= c` de l'ellipsoide equivalent. Les rapports discriminent
noeud (a~b~c), brin (a>>b~c) et plaque (a~b>>c).

```python
st = ma.shape.local_shape_tensor(bin_)  # expand_factor=3, comme iMorph
cls = ma.shape.classify_solid(st, bin_.solid)  # seuil a/b = 1,6, comme iMorph
st.to_frame()  # une ligne par point de mesure
```

`python examples/shape_classification.py` reproduit la figure 3.35 de la these
et valide le seuil 1,6 sur une mousse de Voronoi, ou la loi de Plateau est
exacte par construction et fournit donc la verite terrain des brins et des
noeuds.

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
