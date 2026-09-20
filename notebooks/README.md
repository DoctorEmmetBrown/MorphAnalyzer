# Notebooks

## `tutoriel.ipynb`

Le tutoriel complet, en 15 sections, des conventions de base a l'export de
maillage. Il **execute la validation** : chaque section affiche les chiffres qui
confrontent le portage a une verite terrain analytique, avec les figures
correspondantes.

Le notebook est **genere** par `tools/build_tutorial.py` plutot qu'edite a la
main : le source reste lisible dans une revue de code (pas de JSON), les
cellules ne peuvent pas porter de sorties perimees, et la regeneration est une
commande.

```bash
pip install -e ".[dev]" matplotlib jupyter
python tools/build_tutorial.py
jupyter nbconvert --execute --inplace --to notebook \
    --ExecutePreprocessor.timeout=1800 notebooks/tutoriel.ipynb
```

Comptez deux a trois minutes d'execution. Pour en tirer une page HTML
autonome :

```bash
jupyter nbconvert --to html --template lab --output tutoriel.html notebooks/tutoriel.ipynb
```

Les `.html` produits ne sont pas versionnes.
