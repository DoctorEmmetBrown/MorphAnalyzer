"""Interface web : visionneuse de projet et lanceur de pipeline.

Phase de portage : 11 (livree, premiere etape).
Origine iMorph : `Gui/` (35 081 lignes, dont 30 922 de tiers — QCustomPlot pour
l'essentiel), `Gui/MainWindow/` (21 157 l.) pour l'arbre de projet et le
lancement des modules, `Gui/2D/` (4 541 l.) pour le slicer.

Le choix d'architecture est l'inverse de celui d'iMorph : **le serveur ne
calcule rien**. Il expose `Project`, le registre de `pipeline` et le rendu de
coupes, qui s'utilisent tous les trois sans lui. On peut donc faire a la souris
ce qu'on ferait en script, et l'historique du projet rejoue en lot ce qu'on a
fait a la souris.

    morphanalyzer serve mon_echantillon/

ou depuis Python :

    from morphanalyzer.webapp import serve
    serve("mon_echantillon/", port=8080)

Le rendu se fait **cote serveur** : une coupe composee part en PNG, jamais le
volume. Le cout ne depend donc pas de la taille du volume, et l'interface
marche telle quelle sur une machine de calcul distante, a travers un tunnel
SSH.

Aucun module du noyau n'importe ce paquet.
"""

from morphanalyzer.webapp.render import RAMPS, LayerView, colormap, label_colors, render_slice
from morphanalyzer.webapp.server import create_app, serve

__all__ = [
    "serve",
    "create_app",
    "render_slice",
    "LayerView",
    "colormap",
    "label_colors",
    "RAMPS",
]
