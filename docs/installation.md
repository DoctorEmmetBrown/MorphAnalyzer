# Installation

Python 3.10 ou plus récent.

```bash
git clone https://github.com/DoctorEmmetBrown/MorphAnalyzer.git
cd MorphAnalyzer
uv pip install -e ".[dev,fast]"
```

`uv` n'est pas obligatoire : `pip install -e ".[dev,fast]"` fonctionne aussi.

## Le noyau est volontairement léger

Il n'exige que **numpy, scipy, scikit-image, tifffile et pandas**. Tout le reste
est réparti en extras, importés paresseusement. Une dépendance manquante donne un
message qui dit quoi installer.

| Extra | Contenu | Utile pour |
|---|---|---|
| `fast` | numba, connected-components-3d, edt | le watershed fidèle, les boucles chaudes |
| `fmm` | scikit-fmm | les géodésiques à vitesse non uniforme (phase 6) |
| `bigdata` | zarr, dask, ome-zarr | les volumes hors mémoire (2048³) |
| `mesh` | trimesh, meshio | les exports STL / OBJ / PLY, la décimation |
| `network` | porespy, openpnm, networkx | le réseau de pores, le drainage |
| `rays` | trimesh, embreex | le lancer de rayons (phase 8) |
| `viz` | matplotlib, napari, pyvista | la visualisation |
| `cli` | typer, rich, pyyaml | la ligne de commande et les pipelines YAML |
| `gpu` | cupy | l'accélération GPU |
| `all` | tout sauf `gpu` | |
| `dev` | pytest, ruff, mypy, hypothesis | le développement |

!!! note "`fast` mérite d'être installé"
    Sans numba, `segmentation.watershed_cells` se replie sur
    `skimage.segmentation.watershed`, qui **quantifie le relief** et n'implémente
    pas la résolution de collisions par label majoritaire. Le résultat reste
    utilisable, mais ce n'est plus l'algorithme d'iMorph. Un avertissement le
    signale.

## Vérifier l'installation

```bash
pytest -q                 # la suite complète, ~30 s
python examples/shape_classification.py
python examples/full_chain.py
```

## Documentation locale

```bash
uv pip install -e ".[docs]"
mkdocs serve
```
