# Sans interface graphique

C'est la contrainte d'architecture centrale du projet, pas une option.

## Trois façons d'appeler, une seule implémentation

### L'API Python — la référence

```python
import morphanalyzer as ma

vol = ma.io.read_stack("tomo/", voxel_size=7.46)
bin_ = ma.filters.threshold_otsu(vol)
bin_ = ma.filters.keep_largest_component(bin_)
phi = ma.metrics.porosity(bin_)
sv = ma.metrics.specific_surface(bin_)
```

### Le pipeline déclaratif — pour les traitements par lot

```python
from morphanalyzer.pipeline import Pipeline, available_steps

ctx = (
    Pipeline()
    .step("threshold_otsu")
    .step("keep_largest_component")
    .step("porosity", out="phi")
    .step("specific_surface", out="Sv")
    .run(vol)
)

print(ctx["phi"], ctx["Sv"])
print(available_steps())  # ce qui est enregistré
```

Le même pipeline en YAML, versionnable à côté des données :

```yaml
verbose: false
steps:
  - threshold_otsu
  - {keep_largest_component: {connectivity: 26}}
  - {porosity: {out: phi}}
  - {specific_surface: {out: Sv}}
  - {shape_classification: {out: classes, expand_factor: 3.0}}
```

```python
from morphanalyzer.pipeline import run_from_config

ctx = run_from_config("analyse.yaml", vol)
```

Chaque étape reçoit le volume courant en premier argument. Si elle rend un
volume, il devient le volume courant ; sinon le résultat est rangé dans le
contexte sous la clé `out`.

### La ligne de commande

```bash
morphanalyzer info tomo/ --voxel-size 7.46
morphanalyzer porosity tomo/ --voxel-size 7.46 --per-slice
morphanalyzer surface tomo/ --voxel-size 7.46
morphanalyzer phantom voronoi_foam --shape 128 --out foam.tif
morphanalyzer run analyse.yaml tomo/
morphanalyzer steps
```

Ni le pipeline ni la CLI ne contiennent de logique de calcul : ce sont des
enveloppes autour de l'API.

## Comment la contrainte est tenue

Le noyau n'importe jamais napari, Qt, matplotlib, pyvista, vtk, vispy ni magicgui.
`tests/test_no_gui_imports.py` lance un sous-processus, importe tous les modules
du noyau et échoue si l'un de ces modules apparaît dans `sys.modules`. Un second
travail de CI installe le paquet **sans aucun extra** et vérifie qu'il calcule.

La visualisation existe, mais il faut la demander :

```python
from morphanalyzer import viz  # explicite

viz.show(vol, cells=cells)  # napari
```

## Volumes qui ne tiennent pas en mémoire

`io.read_raw` renvoie par défaut un `numpy.memmap` : rien n'est chargé. Pour
aller plus loin, l'extra `bigdata` apporte zarr et dask.

```python
vol = ma.io.read_raw("scan.raw", shape=(2048, 2048, 2048), dtype="uint16")
sub = vol.crop(slice(0, 256), slice(0, 256), slice(0, 256))
```

La thèse mesurait des brins de 8 voxels d'épaisseur faute de résolution
(tableau 3.3), en sous-échantillonnant pour tenir en mémoire. Traiter le volume
natif change la donne.
