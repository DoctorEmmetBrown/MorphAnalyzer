"""Chaines types : ce que l'on enchaine presque toujours.

Ces chaines ne sont pas de la decoration d'interface. Elles encodent l'ordre
**correct** des etapes, et cet ordre a un piege : chaque etape recoit par defaut
la sortie de la precedente, alors que plusieurs d'entre elles ont besoin d'un
resultat plus ancien. Le cas qui fait mal :

    distance_transform  -> distance     le volume courant devient la distance
    cell_markers        -> marqueurs    il devient les marqueurs
    watershed_cells                     il lui faut la DISTANCE comme relief

Sans `input`, la derniere etape inonde l'image des marqueurs. Elle rend une
partition qui *ressemble* a des cellules — c'est a peu pres un Voronoi
euclidien des marqueurs — mais l'IoU contre la partition exacte tombe de 0,908 a
0,636. D'ou `input` explicite, et un test qui rejoue chaque chaine et verifie
son resultat contre la verite terrain.

Deuxieme piege, celui-la signale par l'usage : **une chaine s'execute sur une
phase**. La meme granulometrie a un sens sur le fluide (taille des pores) et sur
le solide (epaisseur des brins), et iMorph donnait les deux. Si les deux ecrivent
dans `distance`, la seconde ecrase la premiere et on ne peut plus les comparer.
Les chaines sont donc resolues **pour un projet et une phase** : le placeholder
`{phase}` devient le calque de la phase demandee — precede au besoin d'un
`complement` — et toutes les sorties portent le suffixe `_fluide` ou `_solide`.
Les deux jeux coexistent, calque par calque, courbe par courbe.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PRESETS", "build_presets", "PHASE", "OTHER", "TARGETS"]

#: Marqueur remplace par le nom reel du calque de la phase visee.
PHASE = "{phase}"

#: Marqueur remplace par le nom reel du calque de l'autre phase.
OTHER = "{autre}"

#: Les deux phases, dans l'ordre ou l'interface les propose.
TARGETS = ("fluid", "solid")

_SUFFIX = {"fluid": "_fluide", "solid": "_solide"}
_LABEL = {"fluid": "fluide", "solid": "solide"}

PRESETS: list[dict[str, Any]] = [
    {
        "id": "granulo",
        "label": "granulométrie",
        "summary": "Distance à la paroi, carte d'ouverture, distribution de taille de pore.",
        "summary_solid": "Distance à l'interface, carte d'ouverture, "
        "distribution d'épaisseur de brin.",
        "requires": [],
        "steps": [
            {"step": "distance_transform", "params": {"out": "distance"}, "input": PHASE},
            {"step": "aperture_map", "params": {"out": "ouverture", "n_radii": 24}, "input": PHASE},
            {
                "step": "pore_size_distribution",
                "params": {"out": "granulometrie", "bins": 30, "mask": "@" + PHASE},
                "input": "ouverture",
            },
            {
                # iMorph n'avait qu'un calcul de granulometrie, qui donnait du
                # meme coup les centres des boules. Ici ce sont deux algorithmes
                # differents — le balayage en rayons pour la carte, les h-maxima
                # pour les centres — mais la chaine rend les deux, avec la table
                # des boules et leur taux de remplissage.
                "step": "maximal_balls",
                "params": {"out": "boules", "distance": "@distance"},
                "input": PHASE,
            },
        ],
    },
    {
        "id": "cellules",
        "label": "cellules & cols",
        "summary": "Boules maximales, marqueurs, ligne de partage des eaux, morphométrie, cols.",
        "summary_solid": "Segmentation de la phase solide en grains : "
        "boules maximales, partage des eaux, morphométrie, contacts.",
        "requires": [],
        "steps": [
            {"step": "distance_transform", "params": {"out": "distance"}, "input": PHASE},
            {
                "step": "cell_markers",
                "params": {"out": "marqueurs", "distance": "@distance", "fill_ratio": 0.55},
                "input": PHASE,
            },
            {
                "step": "watershed_cells",
                "params": {
                    "out": "cellules",
                    "markers": "@marqueurs",
                    # le masque est la MEME phase que les marqueurs
                    "mask": "@" + PHASE,
                },
                # le relief est la DISTANCE, pas les marqueurs
                "input": "distance",
            },
            {"step": "cell_morphometry", "params": {"out": "morphometrie"}, "input": "cellules"},
            {"step": "throats", "params": {"out": "cols"}, "input": "cellules"},
        ],
    },
    {
        "id": "plateau",
        "label": "squelette de Plateau",
        "summary": "Nœuds et brins par la loi de Plateau, à partir des cellules segmentées.",
        "requires": ["cellules"],
        # la loi de Plateau decrit les parois d'une mousse : le squelette vit
        # dans le solide, et les cellules qui s'y rencontrent sont celles du
        # fluide. La chaine n'a donc de sens que dans un sens.
        "targets": ("fluid",),
        "steps": [
            {
                # le squelette vit dans le SOLIDE, les cellules sont celles du
                # fluide : c'est la seule etape qui touche les deux phases.
                "step": "plateau_skeleton",
                "params": {"out": "plateau", "cells": "@cellules"},
                "input": OTHER,
            },
        ],
    },
    {
        "id": "poiseuille",
        "label": "tortuosité de Poiseuille",
        "summary": "Chemins choisis par un profil parabolique, de la face z = 0 "
        "à la face opposée, comparés aux géodésiques.",
        "requires": [],
        # « Si le fluide est newtonien, lors d'un ecoulement laminaire le chemin
        # pris par le fluide ne sera pas forcement le chemin topologiquement le
        # plus court » (these, 3.2.3). On pousse le fluide, pas la matrice.
        "targets": ("fluid",),
        "steps": [
            {"step": "distance_transform", "params": {"out": "distance"}, "input": PHASE},
            # La carte d'ouverture est le R du profil `1 - (r/R)^2`. C'est
            # l'etape couteuse : la calculer ici et la passer evite que
            # `poiseuille_tortuosity` la refasse pour lui seul.
            {"step": "aperture_map", "params": {"out": "ouverture", "n_radii": 24}, "input": PHASE},
            {
                "step": "plane_tortuosity",
                "params": {"out": "tortuosite_plan", "face": 0},
                "input": PHASE,
            },
            {
                "step": "poiseuille_tortuosity",
                "params": {
                    "out": "poiseuille",
                    "face": 0,
                    "variant": "physical",
                    "distance": "@distance",
                    "aperture": "@ouverture",
                    "n_paths": 16,
                },
                "input": PHASE,
            },
        ],
    },
    {
        "id": "drainage",
        "label": "drainage",
        "summary": "Intrusion morphologique par la face z = 0 et courbe de rétention.",
        "requires": [],
        # on draine un espace poreux, pas une matrice.
        "targets": ("fluid",),
        "steps": [
            {
                "step": "drainage",
                "params": {
                    "out": "drainage",
                    "method": "hilpert",
                    "step": 0.5,
                    "surface_tension": 0.0728,
                },
                "input": PHASE,
            },
        ],
    },
]


def phase_layer(project, target: str) -> tuple[str, list[dict[str, Any]]]:
    """Nom du calque portant `target`, et les etapes a inserer pour l'obtenir.

    Le volume d'entree est d'une phase — `project.phase`, `"solid"` par defaut,
    convention `True = solide`. Demander l'autre phase coute un `complement`.
    """
    native = (project.phase or "solid").lower()
    if native not in TARGETS:
        native = "solid"
    if target == native:
        return "volume", []
    name = _LABEL[target]
    return name, [{"step": "complement", "params": {"out": name}, "input": "volume"}]


def build_presets(project, target: str = "fluid") -> list[dict[str, Any]]:
    """Resout les chaines types pour un projet et une phase.

    Rend une copie ou `{phase}` est remplace par le calque de la phase demandee
    (et `{autre}` par celui de l'autre phase), precedee au besoin d'un
    `complement`, et ou chaque sortie porte le suffixe de la phase. Chaque chaine
    porte `available` et `missing` pour que l'interface sache ce qu'elle peut
    lancer.
    """
    if target not in TARGETS:
        raise ValueError(f"phase inconnue : {target!r} (attendu : {' ou '.join(TARGETS)})")

    layers = set(project.layers)
    suffix = _SUFFIX[target]
    other = "solid" if target == "fluid" else "fluid"
    names = {t: phase_layer(project, t)[0] for t in TARGETS}
    makes = {t: phase_layer(project, t)[1] for t in TARGETS}

    out: list[dict[str, Any]] = []
    for preset in PRESETS:
        if target not in preset.get("targets", TARGETS):
            continue
        rename = {
            s["params"]["out"]: s["params"]["out"] + suffix
            for s in preset["steps"]
            if "out" in s.get("params", {})
        }
        # une chaine peut consommer les sorties d'une autre (plateau <- cellules)
        for req in preset["requires"]:
            rename.setdefault(req, req + suffix)

        steps = [_resolve(s, names[target], names[other], rename) for s in preset["steps"]]

        # une phase non native doit etre fabriquee avant d'etre lue
        prefix: list[dict[str, Any]] = []
        for t in (target, other):
            name = names[t]
            used = any(
                s.get("input") == name or "@" + name in str(s.get("params", {})) for s in steps
            )
            if used and name not in layers:
                prefix += [dict(s) for s in makes[t]]
        steps = prefix + steps

        missing = [rename[r] for r in preset["requires"] if rename[r] not in layers]
        summary = preset.get(f"summary_{target}") or preset["summary"]
        out.append(
            {
                **{
                    k: v
                    for k, v in preset.items()
                    if k not in ("steps", "requires", "summary", "summary_solid", "summary_fluid")
                },
                "summary": summary,
                "target": target,
                "phase": _LABEL[target],
                "requires": [rename[r] for r in preset["requires"]],
                "steps": steps,
                "missing": missing,
                "available": not missing,
            }
        )
    return out


def _resolve(
    step: dict[str, Any], phase_name: str, other_name: str, rename: dict[str, str]
) -> dict[str, Any]:
    def ref(value: Any) -> Any:
        # seules les references `@calque` sont renommees dans les parametres :
        # une valeur comme method="hilpert" n'est pas un nom de calque.
        if not isinstance(value, str):
            return value
        v = value.replace(PHASE, phase_name).replace(OTHER, other_name)
        if v.startswith("@"):
            return "@" + rename.get(v[1:], v[1:])
        return v

    params = {k: ref(v) for k, v in (step.get("params") or {}).items()}
    if "out" in params and isinstance(params["out"], str):
        params["out"] = rename.get(params["out"], params["out"])
    resolved: dict[str, Any] = {"step": step["step"], "params": params}
    if step.get("input"):
        src = step["input"].replace(PHASE, phase_name).replace(OTHER, other_name)
        resolved["input"] = rename.get(src, src)
    return resolved
