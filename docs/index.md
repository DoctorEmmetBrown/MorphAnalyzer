# morphanalyzer

Analyse morphologique 3D de milieux cellulaires et poreux : mousses métalliques,
frittés, os trabéculaire, tissus. C'est le portage Python d'[iMorph][imorph],
développé au IUSTI par Jérôme Vicente et Emmanuel Brun.

[imorph]: https://imorph.sourceforge.net/

## Bibliothèque d'abord

iMorph ne savait rien faire sans sa fenêtre : chaque calcul était un
`CalcModule` couplé à un `QThread`, une fenêtre de paramètres et le widget
central. Impossible à scripter, à tester, ou à faire tourner sur une machine de
calcul.

Ici c'est l'inverse. **Toutes les routines sont des fonctions sur des tableaux
NumPy**, appelables depuis un script, un notebook ou un job de calcul, sans
écran. La visualisation est un extra optionnel que le noyau n'importe jamais — un
test le vérifie à chaque exécution de la suite.

```python
import morphanalyzer as ma

vol = ma.io.read_stack("tomo/", voxel_size=7.46)  # µm
bin_ = ma.filters.threshold_otsu(vol)

print(ma.metrics.porosity(bin_))
print(ma.metrics.specific_surface(bin_), "µm⁻¹")
```

## Ce qui est disponible

| Domaine | Fonctions | État |
|---|---|:--:|
| Entrées-sorties | `io.read_raw`, `read_stack`, `read_table` et leurs écritures | ✅ |
| Prétraitement | Otsu, hystérésis, médian, NL-means, morphologie sphérique, composante principale, filtres de crête par le Hessien | ✅ |
| Grandeurs macroscopiques | porosité totale / par coupe / ouverte, surface spécifique, VER | ✅ |
| Distance et propagation | transformée euclidienne exacte, propagation au plus proche germe, boule géodésique | ✅ |
| Granulométrie | carte d'ouverture, boules maximales, marqueurs de cellules | ✅ |
| Segmentation | watershed à priorités réelles, morphométrie des cellules, cols, connectivité, réseau de pores | ✅ |
| Squelettes | amincissement de Lee, crête de distance, **loi de Plateau** | ✅ |
| Forme locale | **tenseur d'inertie local**, classification nœud / brin / plaque, orientations | ✅ |
| [Tortuosités](guide/tortuosite.md) | point, plan, directionnelle, graphe, **Poiseuille** | ✅ |
| [Réseau et drainage](guide/drainage.md) | Hazlett, Hilpert, percolation d'invasion, Young-Laplace | ✅ |
| [Os cortical](guide/cortical.md) | profils angulaires et radiaux, connectivité empilée, Voronoï 2D | ✅ |
| [Maillage](guide/maillage.md) | marching cubes, aire et volume, STL / OBJ / PLY, décimation | ✅ |
| Transfert radiatif | lancer de rayons, facteurs d'échange | ⛔ hors périmètre |

Le transfert radiatif (phase 8) est écarté du portage à la demande. Son API reste
déclarée et l'appel le dit. Les rares fonctions encore à faire — élagage de
squelette, flux d'axe médian — lèvent une erreur qui nomme leur phase, pour qu'un
notebook écrit aujourd'hui ne casse pas demain. Voir la
[carte de portage](PORTING_MAP.md).

## Ce qui est original

Trois algorithmes n'ont pas d'équivalent en bibliothèque et justifient à eux
seuls ce portage.

La [**classification locale de forme par tenseur d'inertie**](guide/classification-forme.md)
distingue nœuds, brins et plaques en tout point du solide, à partir de la
covariance d'une boule géodésique de rayon adaptatif. Elle se généralise bien
au-delà des mousses : os trabéculaire, fibres, réseaux vasculaires.

Le [**squelette par loi de Plateau**](guide/plateau.md) ne fait aucun
amincissement : il propage les labels de cellules dans le solide et lit les
jonctions dans un voisinage 2×2×2. Il produit directement un graphe de nœuds et
de brins physiquement interprétable.

Les [**cols déformables**](guide/drainage.md) de la percolation d'invasion
élargissent les passages à mesure que la pression monte — un modèle de milieu
élastique. Le rayon effectif d'un col admet une forme close, ce qui évite le
balayage en pression de l'original.

## Par ou commencer

Le **[tutoriel](https://github.com/DoctorEmmetBrown/MorphAnalyzer/blob/main/notebooks/tutoriel.ipynb)**
(`notebooks/tutoriel.ipynb`) parcourt la bibliotheque de bout en bout, en 15
sections, et **execute la validation** : chaque etape affiche les chiffres qui la
confrontent a une verite terrain analytique, figures a l'appui. Il se regenere
par `python tools/build_tutorial.py`.

## Validation

Pas de tomogramme de référence : la validation repose sur des
[**fantômes à vérité terrain analytique**](validation.md). Le plus utile est la
mousse de Voronoï — par construction la loi de Plateau y est exacte, donc la
partition de Voronoï fournit la vérité terrain des cellules, des cols, des brins
et des nœuds. Aucun tomogramme ne permet cela.
