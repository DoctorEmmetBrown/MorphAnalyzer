# Installation

Python 3.10 ou plus récent.

```bash
git clone https://github.com/DoctorEmmetBrown/MorphAnalyzer.git
cd MorphAnalyzer/morphanalyzer
uv pip install -e ".[dev,fast,web]"
```

`uv` n'est pas obligatoire : `pip install -e ".[dev,fast,web]"` fonctionne aussi.

!!! warning "Après un `git pull`, réinstallez"
    Un `pip install -e .` fige les **métadonnées** du paquet — dont la liste des
    extras — au moment de l'installation. Si un extra apparaît après coup,
    `pip install "morphanalyzer[web]"` répond :

    ```
    WARNING: morphanalyzer 0.1.0.dev0 does not provide the extra 'web'
    ```

    Il ne s'agit pas d'une erreur de nom : c'est l'installation qui date d'avant.
    Relancer `pip install -e ".[dev,fast,web]"` depuis le dossier du paquet
    rafraîchit les métadonnées.

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
| `web` | fastapi, uvicorn, pillow | [l'interface](guide/interface.md) (`morphanalyzer serve`) |
| `gpu` | cupy | l'accélération GPU |
| `all` | tout sauf `gpu` | |
| `dev` | pytest, ruff, mypy, hypothesis | le développement |

!!! warning "`fast` n'est pas optionnel, en pratique"
    Sans numba, `segmentation.watershed_cells` se replie sur
    `skimage.segmentation.watershed`, qui **quantifie le relief** et ne résout
    pas les collisions par label majoritaire. Ce n'est plus l'algorithme de la
    thèse : c'est celui de Meyer, et il redonne exactement les frontières en
    « marches d'escalier » de la **figure 3.4** au lieu de la **figure 3.5**.

    Mesuré sur une mousse de Voronoï de 128³ (mêmes marqueurs, même carte de
    distance) :

    | | IoU médian | surface d'interface | voxels attribués autrement |
    |---|---:|---:|---:|
    | tas binaire (numba) | **0,908** | référence | — |
    | file hiérarchique (skimage) | 0,892 | **+15 %** | **9,1 %** |

    Les +15 % de surface d'interface *sont* les marches d'escalier. Un
    avertissement le signale à chaque appel, et l'interface le remonte dans le
    journal de la tâche et dans l'historique du projet.

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
